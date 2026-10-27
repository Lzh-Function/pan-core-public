"""DynamicBucketDistributedSampler keeps its order when its indices become tensors.

The per-bucket indices used to be Python lists built from list(range(...)) and
pickled whole into every checkpoint. They are now int64 tensors, and state_dict
stores the shuffle round instead of the indices. The order a run sees must not
change, and a resumed run must continue exactly where the saved one stopped.
"""

import itertools

import pytest
import torch

from pan_core.common.data_handler import DynamicBucketDistributedSampler

SIZES = [37, 23, 11]


def _sampler(rank=0, num_replicas=2, seed=7, sizes=SIZES):
    return DynamicBucketDistributedSampler(
        batch_sizes=[4, 2, 1], accumulate_steps=[1, 2, 3], num_replicas=num_replicas, rank=rank,
        max_epochs=3, total_optimizer_steps=60, num_data=sizes,
        start_sampling_probabilities=[0.5, 0.3, 0.2], end_sampling_probabilities=[0.2, 0.3, 0.5],
        sampling_update_last_epoch=3, seed=seed,
    )


def _list_based_rank_indices(sizes, bucket, shuffle_round, seed, rank, num_replicas):
    """The original construction, kept verbatim as the reference."""
    offset = sum(sizes[:bucket])
    all_indices = list(range(offset, offset + sizes[bucket]))
    g = torch.Generator()
    g.manual_seed(seed + shuffle_round)
    shuffled_order = torch.randperm(len(all_indices), generator=g).tolist()
    shuffled_indices = [all_indices[i] for i in shuffled_order]
    return shuffled_indices[rank::num_replicas]


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("bucket", [0, 1, 2])
@pytest.mark.parametrize("shuffle_round", [0, 1, 5])
def test_rank_indices_match_the_list_based_order(rank, bucket, shuffle_round):
    s = _sampler(rank=rank, num_replicas=3)
    got = s._rank_indices(bucket, shuffle_round).tolist()
    assert got == _list_based_rank_indices(SIZES, bucket, shuffle_round, 7, rank, 3)


def _epochs(sampler, epochs):
    out = []
    for epoch in epochs:
        sampler.set_epoch(epoch)
        out.extend(list(sampler))
    return out


def test_resume_continues_the_same_order():
    # Resumed at an epoch boundary: within an epoch the bucket draws restart from
    # the epoch seed, so that is the point a resume reproduces exactly. Epoch 0
    # spends 20 steps over buckets of 37/23/11 split two ways, so several buckets
    # are already past their first reshuffle when the state is taken.
    whole = _epochs(_sampler(), [0, 1, 2])

    first = _sampler()
    head = _epochs(first, [0])
    assert any(c > 1 for c in first.reshuffle_counters)
    state = first.state_dict()

    resumed = _sampler()
    resumed.load_state_dict(state)
    rest = _epochs(resumed, [1, 2])

    assert head + rest == whole


def test_old_list_format_still_loads():
    s = _sampler()
    s.set_epoch(0)
    it = iter(s)
    list(itertools.islice(it, 5))
    state = s.state_dict()
    # The original format: the indices themselves, as lists, and no layout fields.
    old = {k: state[k] for k in ("optimizer_steps_yielded", "current_position_in_bucket",
                                 "reshuffle_counters", "_is_setup")}
    old["persistent_indices_per_bucket_rank"] = [t.tolist() for t in s.persistent_indices_per_bucket_rank]

    from_new, from_old = _sampler(), _sampler()
    from_new.load_state_dict(state)
    from_old.load_state_dict(old)
    for a, b in zip(from_new.persistent_indices_per_bucket_rank, from_old.persistent_indices_per_bucket_rank,
                    strict=True):
        assert torch.equal(a, b)
    from_new.set_epoch(0)
    from_old.set_epoch(0)
    assert list(from_new) == list(from_old)


@pytest.mark.parametrize("change", [{"seed": 8}, {"rank": 1}, {"sizes": [37, 23, 12]}])
def test_state_from_a_different_split_is_refused(change):
    s = _sampler()
    s.set_epoch(0)
    list(itertools.islice(iter(s), 2))
    other = _sampler(**change)
    with pytest.raises(ValueError, match="sampler state was saved for"):
        other.load_state_dict(s.state_dict())


def test_state_dict_does_not_carry_the_indices():
    s = _sampler(sizes=[100_000, 50_000, 10_000])
    s.set_epoch(0)
    list(itertools.islice(iter(s), 3))
    state = s.state_dict()
    assert "persistent_indices_per_bucket_rank" not in state
    assert all(not isinstance(v, (list, torch.Tensor)) or len(v) <= 3 for v in state.values())
