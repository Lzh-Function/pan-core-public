"""Tests for analysis 07 (head ablation): head geometry, ablation plans and the paper's head sets."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tests.utils import load_script

_ABLATION = load_script("analysis/scripts/compute/07_head_ablation.py")

_PAPER_HEADS = {  # (jump-interval heads, final-checkpoint heads)
    "addonce": ([(6, 1), (6, 7), (7, 5)], [(6, 1), (6, 7), (7, 5)]),
    "xattn_adalnzero": ([(6, 5), (7, 1), (7, 4)], [(7, 1), (5, 7), (7, 0)]),
    "baseline": ([(7, 0), (7, 2), (7, 7)], [(7, 7), (7, 0), (1, 0)]),
}
_PAPER_RANDOM = {  # random control heads logged in the paper run (before, peak, after, final)
    "addonce": [[(7, 4), (6, 2), (6, 0)], [(7, 6), (6, 5), (6, 4)], [(6, 4), (6, 3), (6, 2)], [(7, 4), (7, 6), (7, 2)]],
    "xattn_adalnzero": [[(7, 5), (6, 1), (6, 0)], [(7, 6), (6, 4), (6, 3)], [(6, 3), (6, 2), (6, 1)], [(7, 5), (7, 6), (7, 2)]],
    "baseline": [[(7, 4), (6, 1), (6, 0)], [(7, 5), (6, 4), (6, 3)], [(6, 3), (6, 2), (6, 1)], [(7, 4), (7, 5), (7, 1)]],
}


# ── head geometry from checkpoint configs ────────────────────────────────────

def _write_checkpoint_config(path: Path, *, n_layer=4, n_head=2, embedding_dim=16):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "n_layer": n_layer,
        "n_head": n_head,
        "embedding_dim": embedding_dim,
    }))


def test_head_geometry_is_derived_from_checkpoint_configs(tmp_path):
    checkpoints = {}
    for label in ("before_jump", "jump_peak", "after_jump", "final"):
        path = tmp_path / label
        _write_checkpoint_config(path)
        checkpoints[label] = str(path)

    assert _ABLATION.load_head_geometry(checkpoints) == (4, 2, 8)
    assert _ABLATION.parse_chiral_heads([[2, 0], [3, 1]], 4, 2) == [(2, 0), (3, 1)]


def test_head_geometry_rejects_inconsistent_checkpoints(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_checkpoint_config(first, n_layer=4)
    _write_checkpoint_config(second, n_layer=6)

    with pytest.raises(ValueError, match="share one architecture"):
        _ABLATION.load_head_geometry({"before": first, "after": second})


# ── head sets ────────────────────────────────────────────────────────────────

def test_all_chiral_heads_are_ablated_only_in_their_own_layer():
    heads = [(6, 1), (6, 7), (7, 5)]
    assert _ABLATION.heads_by_layer(heads) == {6: [1, 7], 7: [5]}


def test_legacy_union_reproduces_the_arxiv_v1_head_sets():
    # pancore-addonce: heads {1, 5, 6, 7} of layers 6 and 7 (8 heads)
    assert _ABLATION.legacy_union_heads([(6, 1), (6, 7), (7, 5)], 8) == {6: [1, 5, 6, 7], 7: [1, 5, 6, 7]}
    # pancore-xattn-adalnzero: heads {1, 4, 5, 6, 7} of layers 6 and 7 (10 heads)
    assert _ABLATION.legacy_union_heads([(6, 5), (7, 1), (7, 4)], 8) == {6: [1, 4, 5, 6, 7], 7: [1, 4, 5, 6, 7]}
    # pancore-baseline: heads {0, 2, 7} of layer 7 (the intended 3)
    assert _ABLATION.legacy_union_heads([(7, 0), (7, 2), (7, 7)], 8) == {7: [0, 2, 7]}


@pytest.mark.parametrize(
    "chiral_heads, expected",
    [
        ([(7, 0), (7, 2), (7, 7)], [
            [(7, 4), (6, 1), (6, 0)], [(7, 5), (6, 4), (6, 3)], [(6, 3), (6, 2), (6, 1)], [(7, 4), (7, 5), (7, 1)],
        ]),
        ([(6, 1), (6, 7), (7, 5)], [
            [(7, 4), (6, 2), (6, 0)], [(7, 6), (6, 5), (6, 4)], [(6, 4), (6, 3), (6, 2)], [(7, 4), (7, 6), (7, 2)],
        ]),
        ([(6, 5), (7, 1), (7, 4)], [
            [(7, 5), (6, 1), (6, 0)], [(7, 6), (6, 4), (6, 3)], [(6, 3), (6, 2), (6, 1)], [(7, 5), (7, 6), (7, 2)],
        ]),
    ],
)
def test_random_control_heads_match_the_paper_run(chiral_heads, expected):
    """Same pool order and one Random(42) stream drawn per checkpoint label, as logged in the paper run."""
    pool = [(layer, head) for layer in (6, 7) for head in range(8) if (layer, head) not in chiral_heads]
    rng = random.Random(42)
    assert [rng.sample(pool, len(chiral_heads)) for _ in range(4)] == expected


def test_format_heads():
    assert _ABLATION.format_heads({7: [1, 0], 5: [7]}) == "L5H7;L7H1;L7H0"
    assert _ABLATION.format_heads(None) == ""


# ── ablation plans ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", list(_PAPER_HEADS))
def test_plan_uses_checkpoint_specific_heads_and_paper_random_sets(model):
    jump_heads, final_heads = _PAPER_HEADS[model]
    plans = _ABLATION.plan_ablations(jump_heads, final_heads, n_layer=8, num_heads=8, seed=42, legacy_union=False)

    assert list(plans) == ["before_jump", "jump_peak", "after_jump", "final"]
    for time_label, expected_random in zip(plans, _PAPER_RANDOM[model]):
        variants = dict(plans[time_label])
        heads = final_heads if time_label == "final" else jump_heads
        assert variants["no_ablation"] is None
        assert variants["all_chiral_heads"] == _ABLATION.heads_by_layer(heads)
        assert [label for label, _ in plans[time_label]][2:-1] == [f"head_{l}_{h}" for l, h in heads]
        assert variants["random_control_heads"] == _ABLATION.heads_by_layer(expected_random)


def test_plan_legacy_union_applies_to_the_heads_of_each_checkpoint():
    jump_heads, final_heads = _PAPER_HEADS["xattn_adalnzero"]
    plans = _ABLATION.plan_ablations(jump_heads, final_heads, n_layer=8, num_heads=8, seed=42, legacy_union=True)

    assert dict(plans["after_jump"])["all_chiral_heads"] == {6: [1, 4, 5, 6, 7], 7: [1, 4, 5, 6, 7]}
    assert dict(plans["final"])["all_chiral_heads"] == {5: [0, 1, 5, 7], 7: [0, 1, 5, 7]}


def test_plan_rejects_random_draw_that_hits_a_final_chiral_head():
    # Random(42) draws (7, 4) for the final label of this configuration; making it a final head must fail.
    with pytest.raises(ValueError, match="Random control heads for final include chiral heads"):
        _ABLATION.plan_ablations([(6, 1), (6, 7), (7, 5)], [(7, 4), (6, 1), (6, 7)], 8, 8, 42, False)
