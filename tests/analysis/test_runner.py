"""Unit tests for pan_core_analysis.runner (pure-pandas / tensor helpers; no model needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pan_core_analysis.runner import (
    check_resume_signature,
    compute_accuracy_from_results,
    compute_phase_metrics,
    pad_target_by_one,
    phase_bounds,
)


class TestComputeAccuracyFromResults:
    def test_all_correct(self):
        df = pd.DataFrame({
            "target":    ["CC", "CCO", "C[C@@H](N)O"],
            "generated": ["CC", "CCO", "C[C@@H](N)O"],
            "judge":     [True, True, True],
        })
        result = compute_accuracy_from_results(df)
        assert result["accuracy"] == 1.0
        assert result["chiral_accuracy"] == 1.0

    def test_all_wrong(self):
        df = pd.DataFrame({
            "target":    ["CC", "CC"],
            "generated": ["CCC", "CCC"],
            "judge":     [False, False],
        })
        assert compute_accuracy_from_results(df)["accuracy"] == 0.0

    def test_chiral_mistakes_ex_mark_chirality_swap(self):
        # Generated differs from target only in @ vs @@
        df = pd.DataFrame({
            "target":    ["C[C@@H](N)O", "C[C@H](N)O"],
            "generated": ["C[C@H](N)O",  "C[C@@H](N)O"],
            "judge":     [False, False],
        })
        result = compute_accuracy_from_results(df)
        assert result["chiral_accuracy"] == 0.0
        assert result["chiral_mistakes_ex_mark_accuracy"] == 1.0

    def test_no_chiral_targets_gives_nan_chiral_acc(self):
        df = pd.DataFrame({
            "target":    ["CC", "CCC"],
            "generated": ["CC", "CCC"],
            "judge":     [True, True],
        })
        result = compute_accuracy_from_results(df)
        assert np.isnan(result["chiral_accuracy"])

    def test_mixed_correct_and_wrong(self):
        df = pd.DataFrame({
            "target":    ["CC", "CCO", "CCC", "CCCC"],
            "generated": ["CC", "CC",  "CCC", "CCCC"],
            "judge":     [True, False, True, True],
        })
        result = compute_accuracy_from_results(df)
        assert abs(result["accuracy"] - 0.75) < 1e-6


def test_compute_phase_metrics_matches_original_definitions():
    df = pd.DataFrame({
        "target": ["C[C@H]O", "C[C@@H]O", "CCO", "CCN", "C[C@H]N", "CC", "CCC", "C[C@@H]C"],
        "generated": ["C[C@H]O", "C[C@H]O", "CCO", "CCN", "CCN", "CC", "CCC", "C[C@@H]C"],
    })
    df["judge"] = df["target"] == df["generated"]

    row = compute_phase_metrics(df, n_phases=2)

    assert row["phase1_accuracy"] == 0.75
    assert row["phase1_chiral_accuracy"] == 0.5
    assert row["phase1_chiral_mistakes_ex_mark"] == 1.0   # C[C@@H]O vs C[C@H]O differ only in '@'
    assert row["phase1_background_accuracy"] == 1.0
    assert row["phase1_n_chiral"] == 2
    assert row["phase2_chiral_accuracy"] == 0.5
    assert row["phase2_chiral_mistakes_ex_mark"] == 0.0   # C[C@H]N vs CCN differ beyond '@'
    assert row["phase2_n_background"] == 2
    with pytest.raises(ValueError):
        compute_phase_metrics(df.iloc[:7], n_phases=2)


def test_phase_bounds_matches_dataset_slicing():
    assert phase_bounds(1000, 1, 5) == (0, 200)
    assert phase_bounds(10000, 2, 5) == (2000, 4000)
    assert phase_bounds(7, None, None) == (0, 7)
    with pytest.raises(ValueError):
        phase_bounds(1001, 1, 5)
    with pytest.raises(ValueError):
        phase_bounds(1000, 1, None)


def test_pad_target_by_one_keeps_masked_positions_padding():
    batch = {
        "input_ids": torch.tensor([[1, 5, 2]]),
        "labels": torch.tensor([[1, 5, 5, 2], [1, 5, 2, 0]]),
        "decoder_attention_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
    }
    padded = pad_target_by_one(batch)

    assert padded["labels"].tolist() == [[1, 5, 5, 2, 0], [1, 5, 2, 0, 0]]
    assert padded["decoder_attention_mask"][:, -1].tolist() == [False, False]
    assert torch.equal(padded["input_ids"], batch["input_ids"])
    # The teacher-forced decoder input labels[:, :-1] now contains EOS of the longest target.
    assert padded["labels"][0, :-1].tolist() == [1, 5, 5, 2]


def test_resume_signature_rejects_changed_config_and_unknown_results(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("print('x')\n")
    signature = tmp_path / "out" / ".resume_signature.json"
    output = tmp_path / "out" / "result.csv"

    check_resume_signature(signature, script, {"a": 1, "output_dir": "x"}, [output])
    check_resume_signature(signature, script, {"a": 1, "output_dir": "y", "checkpoint_steps": [5]}, [output])
    with pytest.raises(ValueError):
        check_resume_signature(signature, script, {"a": 2}, [output])

    other_signature = tmp_path / "other" / ".resume_signature.json"
    stale = tmp_path / "other" / "result.csv"
    stale.parent.mkdir()
    stale.write_text("step\n")
    with pytest.raises(ValueError):
        check_resume_signature(other_signature, script, {"a": 1}, [stale])
