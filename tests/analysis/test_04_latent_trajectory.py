"""Tests for analysis 04 (latent trajectory): static / dynamic latent geometry and resume."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from tests.analysis.tiny_run import MAX_LEN
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/compute/04_latent_trajectory.py"
_LATENT = load_script(_SCRIPT)
_STATIC_COLUMNS = ["l2_norm_mean", "l2_norm_std", "active_units", "intrinsic_dim", "isotropy_entropy",
                   "radial_mean", "radial_cv"]
_DYNAMIC_COLUMNS = ["from_step", "mid_step", "linear_cka", "kernel_cka", "cosine_mean", "cosine_std"]


def _config(checkpoint_dir, tiny_run, output_dir, **overrides):
    config = {
        "model_name": tiny_run.model_name,
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(output_dir),
        "src_col": "random",
        "batch_size": 8,
        "max_len": MAX_LEN,
    }
    config.update(overrides)
    return config


def _read(output_dir, kind, group):
    return pd.read_csv(output_dir / "tiny" / f"latent_{kind}_{group}_trajectory.csv", index_col="step")


@pytest.fixture(scope="module")
def latent_output(tiny_run, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("latent")
    run_script(_SCRIPT, _config(tiny_run.checkpoint_dir, tiny_run, output_dir), output_dir / "config.yml")
    return output_dir


@pytest.mark.parametrize("group", ["chiral", "nchiral"])
def test_static_metrics_are_written_for_every_checkpoint(latent_output, tiny_run, group):
    static = _read(latent_output, "static", group)

    assert static.index.tolist() == list(tiny_run.steps)
    assert list(static.columns) == _STATIC_COLUMNS
    assert static[["l2_norm_mean", "radial_cv"]].notna().all().all()


@pytest.mark.parametrize("group", ["chiral", "nchiral"])
def test_dynamic_metrics_compare_consecutive_checkpoints(latent_output, tiny_run, group):
    dynamic = _read(latent_output, "dynamic", group)
    steps = list(tiny_run.steps)

    assert dynamic.index.tolist() == steps
    assert list(dynamic.columns) == _DYNAMIC_COLUMNS
    assert dynamic.loc[steps[0]].isna().all()                       # no predecessor
    assert dynamic.loc[steps[1:], "from_step"].tolist() == steps[:-1]
    assert dynamic.loc[steps[1:], "mid_step"].tolist() == [(a + b) / 2 for a, b in zip(steps, steps[1:])]


def test_identical_checkpoints_have_unit_similarity(tiny_run, tmp_path):
    series = tmp_path / "checkpoints"
    shutil.copytree(tiny_run.checkpoint_dir / "checkpoint-10", series / "checkpoint-10")
    shutil.copytree(tiny_run.checkpoint_dir / "checkpoint-10", series / "checkpoint-20")

    run_script(_SCRIPT, _config(series, tiny_run, tmp_path / "out"), tmp_path / "config.yml")

    row = _read(tmp_path / "out", "dynamic", "chiral").loc[20]
    assert row[["linear_cka", "kernel_cka", "cosine_mean"]].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)


def test_resumed_run_matches_a_single_run(latent_output, tiny_run, tmp_path):
    output_dir = tmp_path / "out"
    first = _config(tiny_run.checkpoint_dir, tiny_run, output_dir, checkpoint_steps=list(tiny_run.steps[:3]))
    run_script(_SCRIPT, first, tmp_path / "config.yml")
    run_script(_SCRIPT, _config(tiny_run.checkpoint_dir, tiny_run, output_dir), tmp_path / "config.yml")

    for kind in ("static", "dynamic"):
        for group in ("chiral", "nchiral"):
            pd.testing.assert_frame_equal(_read(output_dir, kind, group), _read(latent_output, kind, group))


def test_kernel_cka_is_one_for_identical_and_below_one_for_unrelated_inputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    assert _LATENT.compute_kernel_cka_gpu(X, X, torch.device("cpu")) == pytest.approx(1.0, abs=1e-5)
    assert _LATENT.compute_kernel_cka_gpu(X, rng.normal(size=(30, 8)), torch.device("cpu")) < 0.9
