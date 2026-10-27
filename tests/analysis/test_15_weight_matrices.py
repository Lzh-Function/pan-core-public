"""Tests for visualization 15 (weight matrices): figures from the output of analysis 08."""

from __future__ import annotations

import json

import pytest

from tests.analysis.tiny_run import write_jump_logits
from tests.utils import run_script

_SCRIPT = "analysis/scripts/visualize/15_weight_matrices.py"


@pytest.fixture(scope="module")
def weight_dir(tiny_run, tmp_path_factory):
    """Output of analysis 08 on the tiny run."""
    output_dir = tmp_path_factory.mktemp("weights")
    config = {"model_name": tiny_run.model_name, "checkpoint_dir": str(tiny_run.checkpoint_dir),
              "output_dir": str(output_dir)}
    run_script("analysis/scripts/compute/08_weight_matrix.py", config, output_dir / "config.yml")
    return output_dir


def test_main_writes_one_figure_per_weight_group_and_metric(weight_dir, tiny_run, tmp_path):
    config = {
        "input_dir": str(weight_dir),
        "output_dir": str(tmp_path / "figures"),
        "jump_logits_dir": str(write_jump_logits(tmp_path / "jumpup")),
        "jump_ignore_initial_n": 0,
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    data = json.loads((weight_dir / "weight_matrix_tiny.json").read_text())
    assert data["steps"] == list(tiny_run.steps)
    figure_dir = tmp_path / "figures" / tiny_run.model_name
    group = "encoder.h.*.attn.c_proj.weight"
    assert any(key.startswith("encoder.h.0.attn.c_proj") for key in data["weight_matrices"])
    for metric in ("l2_norm", "l2_diff", "cka", "cosine"):
        stem = f"weight_{group.replace('.', '_').replace('*', 'IDX')}_{metric}"
        for extension in ("png", "svg"):
            assert (figure_dir / f"{stem}.{extension}").is_file(), f"{stem}.{extension}"


def test_main_rejects_metric_lengths_that_do_not_match_the_steps(tmp_path):
    corrupt = {"steps": [10, 20, 30], "weight_matrices": {"encoder.h.0.attn.c_proj.weight": {"l2_norm": [1.0, 2.0]}}}
    (tmp_path / "weight_matrix_bad.json").write_text(json.dumps(corrupt))

    with pytest.raises(ValueError, match="2 values for 3 steps"):
        run_script(_SCRIPT, {"input_dir": str(tmp_path), "output_dir": str(tmp_path / "figures")}, tmp_path / "config.yml")
