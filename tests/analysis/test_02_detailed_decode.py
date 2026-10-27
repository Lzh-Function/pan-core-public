"""Tests for analysis 02 (detailed decode): per-sample decodes and the per-bucket summary."""

from __future__ import annotations

import pandas as pd
import pytest

from pan_core.models import PanCoreTransformerModel
from tests.analysis.tiny_run import MAX_LEN, N_BUCKETS
from tests.utils import run_script

_SCRIPT = "analysis/scripts/compute/02_detailed_decode.py"


def _config(tiny_run, output_dir, **overrides):
    config = {
        "model_name": tiny_run.model_name,
        "checkpoint_dir": str(tiny_run.checkpoint_dir),
        "checkpoint_steps": [10, 60],
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(output_dir),
        "batch_size": 8,
        "max_length": MAX_LEN,
        "use_bf16": False,
        "n_phases": N_BUCKETS,
    }
    config.update(overrides)
    return config


@pytest.fixture(scope="module")
def decode_output(tiny_run, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("decode")
    run_script(_SCRIPT, _config(tiny_run, output_dir), output_dir / "config.yml")
    return output_dir


def test_one_decode_csv_per_requested_step(decode_output, tiny_run):
    eval_df = pd.read_csv(tiny_run.eval_csv)
    for step in (10, 60):
        decode = pd.read_csv(decode_output / f"decode_tiny_step{step}.csv", keep_default_na=False)
        assert list(decode.columns) == ["generated", "input", "target", "judge"]
        assert decode["target"].tolist() == eval_df["canonical"].tolist()
        assert decode["judge"].tolist() == (decode["generated"] == decode["target"]).tolist()
    assert not (decode_output / "decode_tiny_step20.csv").exists()


def test_summary_has_overall_and_per_bucket_accuracy(decode_output):
    summary = pd.read_csv(decode_output / "decode_summary_tiny.csv", index_col="step")

    assert summary.index.tolist() == [10, 60]
    for p in range(1, N_BUCKETS + 1):
        assert {f"phase{p}_accuracy", f"phase{p}_chiral_accuracy", f"phase{p}_n_chiral"} <= set(summary.columns)
        assert (summary[f"phase{p}_n_chiral"] == 2).all()


def test_rerun_reuses_existing_decodes(decode_output, tiny_run, monkeypatch):
    before = pd.read_csv(decode_output / "decode_summary_tiny.csv")

    def fail(*args, **kwargs):
        raise AssertionError("an existing decode must not be recomputed")

    monkeypatch.setattr(PanCoreTransformerModel, "from_pretrained", fail)
    run_script(_SCRIPT, _config(tiny_run, decode_output), decode_output / "config.yml")

    pd.testing.assert_frame_equal(pd.read_csv(decode_output / "decode_summary_tiny.csv"), before)


def test_missing_checkpoint_step_is_rejected(tiny_run, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"\[70\]"):
        run_script(_SCRIPT, _config(tiny_run, tmp_path / "out", checkpoint_steps=[10, 70]), tmp_path / "config.yml")
