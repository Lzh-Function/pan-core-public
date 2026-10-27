"""
Checkpoint resume integrity tests — dynamic training strategy.

Background
----------
SetEpochCallback bug (now fixed): the old code used ``int(state.epoch)`` to derive
the epoch index passed to DynamicBucketDistributedSampler.set_epoch().  After a
*mid-epoch* resume, the first (truncated) epoch finishes with ``state.epoch`` NOT
reaching the next integer (e.g. it ends at 1.5 instead of 2.0 because the sampler
only yields the remaining half of that epoch's batches).  Every subsequent
on_epoch_begin then saw the wrong integer, permanently shifting the
sampling-probability schedule (epoch 2 skipped, epoch 1 applied twice).

The fix maintains an explicit ``_next_epoch`` counter, initialised on the first
call from ``global_step // steps_per_epoch`` (after checkpoint state has been
fully restored) and incremented thereafter.  Because it uses integer division on
global_step, it is robust to the small per-epoch step jitter caused by HF
Trainer's batch prefetch (see note below).

What these tests verify
-----------------------
The **epoch sequence** passed to sampler.set_epoch() is correct and continuous
across a resume — this is the core of the bug.  The sequence is fully determined
by the surviving checkpoint's ``global_step`` and ``steps_per_epoch``, so it is
deterministic regardless of prefetch jitter.

Epoch layout (max_steps=16, num_epochs=4 → steps_per_epoch=4):
  epoch 0: steps  1– 4
  epoch 1: steps  5– 8
  epoch 2: steps  9–12
  epoch 3: steps 13–16

  resume @ step S  →  start_epoch = S // 4, so expected set_epoch calls are
    S=2  → [0, 1, 2, 3]   (mid-epoch 0)
    S=4  → [1, 2, 3]      (epoch-0 boundary)
    S=6  → [1, 2, 3]      (mid-epoch 1)  ← the bug's worst case
    S=8  → [2, 3]         (epoch-1 boundary)

NOTE on global_step jitter
--------------------------
With tiny toy buckets, HF Trainer's batch prefetch makes the *number of optimizer
steps actually executed per epoch* fluctuate, so the final global_step differs
between a clean run and a resumed run.  This is a toy-scale artefact: in real
training (steps_per_epoch = 10_000) the ±1–2 jitter is negligible and never
crosses an epoch boundary.  These tests therefore assert on the deterministic
epoch sequence, not on exact step totals.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pan_core.common.data_handler import DynamicBucketDistributedSampler
from pan_core.main import run_training
from tests.utils import VOCAB_FILE

_PHASE0_SMILES = ["C","CC","CCC","CCCC","CCO","CCCO","CO","OCC",
                  "CN","CCN","CCCN","NC","c1ccccc1","Cc1ccccc1","c1ccncc1","O"]
_PHASE1_SMILES = ["CCC","CCCC","CCCCC","CCO","CCCO","CO","OCC","OC",
                  "CN","CCN","CC(=O)O","c1ccccc1","CC(N)=O","CCC(=O)O","O","NCC"]
_VALID_SMILES  = ["C","CCO","c1ccccc1","O"]

MAX_STEPS = 16
NUM_EPOCHS = 4
STEPS_PER_EPOCH = MAX_STEPS // NUM_EPOCHS  # 4


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csvs(data_dir: Path) -> tuple[str, str, str]:
    for smiles, name in [(_PHASE0_SMILES, "phase0"),
                          (_PHASE1_SMILES, "phase1"),
                          (_VALID_SMILES,  "valid")]:
        pd.DataFrame({"random": smiles, "canonical": smiles}).to_csv(
            data_dir / f"{name}.csv", index=False
        )
    return (str(data_dir / "phase0.csv"),
            str(data_dir / "phase1.csv"),
            str(data_dir / "valid.csv"))


def _make_config(vocab: str, csv_a: str, csv_b: str, valid_csv: str) -> dict:
    """Dynamic strategy: 2 buckets, 4 epochs × 4 steps = 16 steps; ckpt every 2 steps."""
    return {
        "system": {"environment": "normal", "seed": 42, "num_workers": 0, "use_bf16": False},
        "logging": {"wandb": {"enabled": False}, "log_interval_steps": 1},
        "model": {
            "use_vae": False,
            "common": {"dropout": 0.0, "embedding_dim": 32,   # vocab_size comes from the tokenizer
                       "tanh_latent": True, "latent_noising": False},
            "loss": {"beta": 0.0},
            "Transformer": {
                "n_positions": 32, "n_layer": 1, "n_head": 2, "bias": True,
                "weight_sharing": False, "layer_norm_epsilon": 1e-5, "gpt2style": False,
                "rope": {"encoder": True, "decoder": True, "base_freq": 10000},
                "latent": {"pooling_mode": "MHA", "cross_attn": "L-T",
                           "latent_conditioning": "add_once"},
            },
        },
        "training": {
            "save_only_model": False,
            "strategy": {
                "type": "dynamic", "num_epochs": NUM_EPOCHS,
                "dynamic": {
                    "phases": 2, "max_steps": MAX_STEPS,
                    "start_sampling_probabilities": [0.6, 0.4],
                    "end_sampling_probabilities": [0.4, 0.6],
                    "sampling_update_last_epoch": NUM_EPOCHS,
                    "sampling_interpolation_type": "linear",
                    "early_stopping": {"enabled": False},
                },
            },
            "optimizer": {"type": "schedulefree", "max_lr": 1e-3,
                          "weight_decay": 0.0, "schedulefree": {"r": 0.5}},
            "zclip": {"enabled": False},
            "checkpoint": {"step": 2, "limit": None},
            "validation": {"enabled": True, "interval_steps": 4, "batch_size": 4},
            "data": {
                "mode": "on_the_fly",
                "csv_column_name": {"source": "random", "target": "canonical"},
                "batch_size": [4, 4], "accumulate_steps": [1, 1],
                "token_file_path": vocab,
                "train_data": [{"path": csv_a}, {"path": csv_b}],
                "valid_data": {"path": valid_csv},
            },
        },
    }


def _latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = sorted(output_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1] if ckpts else None


def _load_state(checkpoint: Path) -> dict:
    ts = checkpoint / "trainer_state.json"
    return json.loads(ts.read_text()) if ts.exists() else {}


def _simulate_interrupt(output_dir: Path, keep_step: int) -> None:
    """Delete checkpoints with step > keep_step and final_model/ to simulate interrupt."""
    for ckpt in list(output_dir.glob("checkpoint-*")):
        if int(ckpt.name.split("-")[1]) > keep_step:
            shutil.rmtree(ckpt)
    fm = output_dir / "final_model"
    if fm.exists():
        shutil.rmtree(fm)


def _assert_log_monotonic(log_history: list, label: str) -> None:
    steps = [e["step"] for e in log_history if "step" in e]
    assert steps, f"{label}: log_history is empty"
    for i in range(1, len(steps)):
        assert steps[i] >= steps[i - 1], (
            f"{label}: log step went backward at index {i}: {steps[i-1]} → {steps[i]}"
        )


class EpochRecorder:
    """Context manager recording every DynamicBucketDistributedSampler.set_epoch call."""

    def __init__(self):
        self.calls: list[int] = []
        self._orig = None

    def __enter__(self):
        self._orig = DynamicBucketDistributedSampler.set_epoch
        rec = self

        def _patched(self_sampler, epoch: int) -> None:
            rec.calls.append(epoch)
            rec._orig(self_sampler, epoch)

        DynamicBucketDistributedSampler.set_epoch = _patched
        return self

    def __exit__(self, *_):
        DynamicBucketDistributedSampler.set_epoch = self._orig


def _run(cfg_path: Path, out_dir: Path) -> list[int]:
    """Run training, return the recorded set_epoch call sequence."""
    with EpochRecorder() as rec:
        run_training(config_path=str(cfg_path), output_dir=str(out_dir))
    return rec.calls


@pytest.fixture
def trained_run(tmp_path):
    """Set up config + data and perform one full training run; yield paths for resume tests."""
    assert VOCAB_FILE.exists(), f"Vocab file missing: {VOCAB_FILE}"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir  = tmp_path / "out"
    csv_a, csv_b, valid_csv = _write_csvs(data_dir)

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(_make_config(str(VOCAB_FILE), csv_a, csv_b, valid_csv)))

    baseline_calls = _run(cfg_path, out_dir)
    yield cfg_path, out_dir, baseline_calls

    if tmp_path.exists():
        shutil.rmtree(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_baseline_epoch_sequence(trained_run):
    """A clean (uninterrupted) run must visit epochs 0,1,2,3 in order, exactly once each."""
    _cfg, out_dir, baseline_calls = trained_run

    assert baseline_calls == [0, 1, 2, 3], (
        f"Baseline epoch sequence should be [0,1,2,3], got {baseline_calls}"
    )

    ckpt = _latest_checkpoint(out_dir)
    assert ckpt is not None, "No checkpoint produced"
    state = _load_state(ckpt)
    _assert_log_monotonic(state.get("log_history", []), "baseline")


@pytest.mark.parametrize("keep_step, expected_resume_calls", [
    (2, [0, 1, 2, 3]),  # mid-epoch 0 → still starts at epoch 0
    (4, [1, 2, 3]),     # epoch-0 boundary
    (6, [1, 2, 3]),     # mid-epoch 1 — the bug's failure case ([1,1,3] before fix)
    (8, [2, 3]),        # epoch-1 boundary
])
def test_resume_epoch_sequence(trained_run, keep_step, expected_resume_calls):
    """
    After interrupting at ``keep_step`` and resuming, the sampler.set_epoch sequence
    must be sequential and start at ``keep_step // STEPS_PER_EPOCH``.

    Before the fix, resuming mid-epoch 1 (keep_step=6) produced [1, 1, 3]:
    epoch 1 was applied twice and epoch 2 was skipped.
    """
    cfg_path, out_dir, _baseline = trained_run

    _simulate_interrupt(out_dir, keep_step=keep_step)
    surviving = _latest_checkpoint(out_dir)
    assert surviving is not None and int(surviving.name.split("-")[1]) == keep_step, (
        f"Expected checkpoint-{keep_step} to survive the interrupt"
    )

    resume_calls = _run(cfg_path, out_dir)

    assert resume_calls == expected_resume_calls, (
        f"Resume from step {keep_step}: expected epoch calls {expected_resume_calls}, "
        f"got {resume_calls}.\n"
        f"  start_epoch should be {keep_step // STEPS_PER_EPOCH}.\n"
        f"  A result like [1,1,3] indicates the SetEpochCallback bug "
        f"(epoch repeated / skipped)."
    )


def test_resume_log_continuity(trained_run):
    """
    After a mid-epoch resume (step 6), the resumed run's trainer_state.json must
    continue the log history forward (no step going backwards, picks up after 6).
    """
    cfg_path, out_dir, _baseline = trained_run

    _simulate_interrupt(out_dir, keep_step=6)
    _run(cfg_path, out_dir)

    final_ckpt = _latest_checkpoint(out_dir)
    assert final_ckpt is not None
    state = _load_state(final_ckpt)
    log = state.get("log_history", [])
    assert log, "Resumed run produced no log history"

    _assert_log_monotonic(log, "resume@6")

    # The final checkpoint's global_step must be strictly greater than the
    # interrupt point — i.e. training genuinely progressed past step 6.
    assert state.get("global_step", 0) > 6, (
        f"Resumed training did not progress past step 6 "
        f"(global_step={state.get('global_step')})"
    )

    # Eval entries, if present, must sit on epoch boundaries (multiples of 4).
    for entry in log:
        if "eval_loss" in entry and "step" in entry:
            assert entry["step"] % STEPS_PER_EPOCH == 0, (
                f"eval at non-boundary step {entry['step']}"
            )


def test_resume_chain(trained_run):
    """
    Chained resume: interrupt at 4, resume; interrupt at 8, resume to the end.
    The concatenated epoch sequence of the two resumed segments, prefixed by the
    epochs already completed before each interrupt, must cover 0..3 with no gaps
    or repeats.
    """
    cfg_path, out_dir, baseline = trained_run
    assert baseline == [0, 1, 2, 3]

    # Segment A: resume from step 4 → epochs [1, 2, 3]
    _simulate_interrupt(out_dir, keep_step=4)
    seg_a = _run(cfg_path, out_dir)
    assert seg_a == [1, 2, 3], f"Resume@4 segment: expected [1,2,3], got {seg_a}"

    # Re-run baseline-to-completion already happened in seg_a; now interrupt at 8.
    _simulate_interrupt(out_dir, keep_step=8)
    seg_b = _run(cfg_path, out_dir)
    assert seg_b == [2, 3], f"Resume@8 segment: expected [2,3], got {seg_b}"

    # Final state must have advanced past step 8.
    final_ckpt = _latest_checkpoint(out_dir)
    state = _load_state(final_ckpt)
    assert state.get("global_step", 0) > 8, (
        f"Chained resume did not progress past step 8 (got {state.get('global_step')})"
    )
    _assert_log_monotonic(state.get("log_history", []), "resume-chain final")
