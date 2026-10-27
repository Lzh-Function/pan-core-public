"""Tests for visualization 13 (residual stream): figures from analyses 05 (L2 norm, layerwise) and 03 (cosine)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.analysis.tiny_run import MAX_LEN, N_BUCKETS, write_jump_logits
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/visualize/13_residual_stream.py"
_RESIDUAL_VIS = load_script(_SCRIPT)


@pytest.fixture(scope="module")
def jumpup_dir(tiny_run, tmp_path_factory):
    """Analysis-03 inputs of visualization 13: a perplexity trajectory with a jump and the
    checkpoint-to-checkpoint cosine trajectories (index ``step{a}-{b}``, one column per layer)."""
    directory = write_jump_logits(tmp_path_factory.mktemp("jumpup"))
    pairs = [f"step{a}-{b}" for a, b in zip(tiny_run.steps, tiny_run.steps[1:])]
    columns = ["embeddings", "Layer_0", "Layer_1"]
    for token in ("chiral", "background"):
        for side in ("enc", "dec"):
            pd.DataFrame(0.99, index=pairs, columns=columns).to_csv(
                directory / tiny_run.model_name / f"cos_{token}_{side}_trajectory.csv")
    return directory


@pytest.fixture(scope="module")
def residual_dir(tiny_run, tmp_path_factory):
    """Outputs of analysis 05 on the tiny run."""
    output_dir = tmp_path_factory.mktemp("residual")
    config = {
        "model_name": tiny_run.model_name,
        "checkpoint_dir": str(tiny_run.checkpoint_dir),
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(output_dir),
        "batch_size": 8,
        "max_len": MAX_LEN,
        "phase": 1,
        "max_phase": N_BUCKETS,
    }
    run_script("analysis/scripts/compute/05_residual_stream.py", config, output_dir / "config.yml")
    return output_dir


def test_layer_labels_follow_the_paper_legend():
    assert _RESIDUAL_VIS._layer_labels(["embeddings", "Layer_0", "Layer_7"]) == ["Emb", "L0", "L7"]
    with pytest.raises(ValueError, match="Unexpected layer column"):
        _RESIDUAL_VIS._layer_labels(["Emb-L0"])


def test_checkpoint_pairs_are_placed_at_the_mid_step():
    assert _RESIDUAL_VIS._mid_step_index(["step500-1000", "step1000-1500"]) == [750, 1250]


def test_main_writes_every_figure(residual_dir, jumpup_dir, tiny_run, tmp_path):
    config = {
        "input_dir": str(residual_dir),
        "jump_logits_dir": str(jumpup_dir),
        "output_dir": str(tmp_path / "figures"),
        "jump_ignore_initial_n": 0,
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    figure_dir = tmp_path / "figures" / tiny_run.model_name
    for token in ("chiral", "background"):
        for side in ("enc", "dec"):
            stems = [f"l2_norm_{token}_{side}_lineplot", f"cos_{token}_{side}_lineplot",
                     f"cos_{token}_{side}_layerwise_trajectory", f"cka_{token}_{side}_layerwise_trajectory"]
            for stem in stems:
                for extension in ("png", "svg"):
                    assert (figure_dir / f"{stem}.{extension}").is_file(), f"{stem}.{extension}"
