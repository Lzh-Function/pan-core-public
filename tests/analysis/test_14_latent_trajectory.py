"""Tests for visualization 14 (latent trajectory): figures from the outputs of analysis 04."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.analysis.tiny_run import MAX_LEN, write_jump_logits
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/visualize/14_latent_trajectory.py"
_LATENT_VIS = load_script(_SCRIPT)


@pytest.fixture(scope="module")
def latent_dir(tiny_run, tmp_path_factory):
    """Outputs of analysis 04 on the tiny run."""
    output_dir = tmp_path_factory.mktemp("latent")
    config = {
        "model_name": tiny_run.model_name,
        "checkpoint_dir": str(tiny_run.checkpoint_dir),
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(output_dir),
        "batch_size": 8,
        "max_len": MAX_LEN,
    }
    run_script("analysis/scripts/compute/04_latent_trajectory.py", config, output_dir / "config.yml")
    return output_dir


def _config(latent_dir, tmp_path, **overrides):
    config = {
        "input_dir": str(latent_dir),
        "jump_logits_dir": str(write_jump_logits(tmp_path / "jumpup")),
        "output_dir": str(tmp_path / "figures"),
        "jump_ignore_initial_n": 0,
    }
    config.update(overrides)
    return config


def test_dynamic_series_is_indexed_by_mid_step_without_the_first_checkpoint():
    df = pd.DataFrame({"mid_step": [None, 15.0, 25.0], "cosine_mean": [None, 0.9, 0.8]}, index=[10, 20, 30])

    series = _LATENT_VIS._dynamic_series(df, "cosine_mean")

    assert series.index.tolist() == [15.0, 25.0]
    assert series.tolist() == [0.9, 0.8]


def test_main_writes_every_figure(latent_dir, tiny_run, tmp_path):
    run_script(_SCRIPT, _config(latent_dir, tmp_path), tmp_path / "config.yml")

    stems = ["latent_l2_norm", "latent_cosine", "latent_cka", "intrinsic_dim", "isotropy_entropy", "active_units"]
    for stem in stems:
        for extension in ("png", "svg"):
            assert (tmp_path / "figures" / tiny_run.model_name / f"{stem}.{extension}").is_file(), f"{stem}.{extension}"


def test_main_rejects_unknown_ylim_keys(latent_dir, tmp_path):
    with pytest.raises(ValueError, match="Unknown ylim keys"):
        run_script(_SCRIPT, _config(latent_dir, tmp_path, ylim={"perplexity": [0, 1]}), tmp_path / "config.yml")
