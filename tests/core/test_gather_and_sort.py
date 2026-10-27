"""DDP validation gathers the token mask without deadlocking.

evaluate() gathers the per-token CE and the bool token mask across ranks with
gather_and_sort. A bool tensor used to deadlock multi-node validation; see
HF_Trainer.gather_and_sort. A single process cannot show the hang, so this checks
the part that causes it: the dtype the tensor is gathered in.
"""

from types import SimpleNamespace

from accelerate import Accelerator

from pan_core.train import HF_Trainer


def test_gather_and_sort_keeps_a_bool_mask_bool_and_gathers_it_as_uint8(monkeypatch):
    # pad_across_processes pads with new_zeros(...) + pad_index, which turns a bool
    # tensor into int64 on the ranks that need padding but leaves it bool on the
    # rank that does not; NCCL then all-gathers mismatched dtypes and hangs.
    import torch

    assert (torch.zeros(2, dtype=torch.bool) + 0).dtype == torch.int64   # the trap
    assert (torch.zeros(2, dtype=torch.uint8) + 0).dtype == torch.uint8  # the way out

    seen = []
    acc = Accelerator()  # earlier tests already initialised the shared state
    original_gather = acc.gather

    def recording_gather(t):
        seen.append(t.dtype)
        return original_gather(t)

    monkeypatch.setattr(acc, "gather", recording_gather)
    fake = SimpleNamespace(accelerator=acc, args=SimpleNamespace(device=acc.device))
    masks = [torch.tensor([[True, False], [True, True]]), torch.tensor([[False, True, True]])]
    idx = [torch.tensor([3, 1]), torch.tensor([2])]

    out = HF_Trainer.gather_and_sort(fake, masks, idx, is_seq=True, pad_val=0)

    assert seen[0] == torch.uint8
    assert out.dtype == torch.bool
    assert out.tolist() == [[True, True, False], [False, True, True], [True, False, False]]
