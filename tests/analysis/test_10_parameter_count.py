"""Tests for analysis 10 (parameter count): trainable parameters from checkpoint configs."""

from __future__ import annotations

import pandas as pd
import pytest

from pan_core.models import PanCoreTransformerModel
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/compute/10_parameter_count.py"
_COUNT = load_script(_SCRIPT)


def test_count_matches_the_trainable_parameters_of_the_checkpoint(tiny_run):
    checkpoint = tiny_run.checkpoint_dir / "checkpoint-10"
    model = PanCoreTransformerModel.from_pretrained(checkpoint)
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert _COUNT.count_trainable_parameters(checkpoint / "config.json") == expected


def test_main_accepts_checkpoint_directories_and_config_files(tiny_run, tmp_path):
    checkpoint = tiny_run.checkpoint_dir / "checkpoint-10"
    config = {
        "output_dir": str(tmp_path / "out"),
        "models": {"from_dir": str(checkpoint), "from_json": str(checkpoint / "config.json")},
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    counts = pd.read_csv(tmp_path / "out" / "parameter_count.csv", index_col="model")
    assert counts.index.tolist() == ["from_dir", "from_json"]
    assert counts.loc["from_dir", "trainable_parameters"] == counts.loc["from_json", "trainable_parameters"]
    assert counts.loc["from_dir", "millions"] == round(counts.loc["from_dir", "trainable_parameters"] / 1e6, 3)


def test_main_rejects_a_missing_config(tmp_path):
    config = {"output_dir": str(tmp_path / "out"), "models": {"missing": str(tmp_path / "nowhere")}}
    with pytest.raises(FileNotFoundError, match="missing"):
        run_script(_SCRIPT, config, tmp_path / "config.yml")
