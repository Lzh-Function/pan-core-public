"""Tests for visualization 12 (jump-up metrics): top-N heads, the chiral-head ranking and output files."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tests.utils import load_script

_JUMPUP_VIS = load_script("analysis/scripts/visualize/12_jumpup_metrics.py")


def test_top_heads_rank_by_amplitude_inside_the_jump_interval():
    df = pd.DataFrame(
        {
            "L0_H0": [0.9, 0.9, 0.9, 0.9],   # largest mean, no change
            "L0_H1": [0.1, 0.2, 0.6, 0.6],   # largest change inside the interval
            "L1_H0": [0.0, 0.1, 0.2, 0.9],   # changes only after the interval
        },
        index=[10, 20, 30, 40],
    )
    assert _JUMPUP_VIS._top_n_heads(df, 1, (10, 30, 20)) == ["L0_H1"]
    assert _JUMPUP_VIS._heads_per_layer(df.columns) == 2


def test_head_ranking_reports_interval_amplitude_and_last_step_mass():
    df = pd.DataFrame(
        {"L0_H0": [0.1, 0.1, 0.45, 0.2], "L0_H1": [0.0, 0.3, 0.4, 0.9], "L1_H0": [0.2, 0.2, 0.2, 0.3]},
        index=[10, 20, 30, 40],
    )
    ranking = _JUMPUP_VIS.head_ranking(df, (10, 30, 20))

    assert list(ranking.index) == ["L0_H1", "L0_H0", "L1_H0"]
    assert ranking.loc["L0_H1", "interval_amplitude"] == pytest.approx(0.4)
    assert ranking.loc["L0_H0", "interval_amplitude"] == pytest.approx(0.35)
    assert ranking.loc["L1_H0", "interval_amplitude_rank"] == 3
    assert ranking["last_step"].unique().tolist() == [40]
    assert ranking.loc["L0_H1", "last_step_rank"] == 1
    assert ranking.loc["L0_H0", "last_step_rank"] == 3


def test_save_writes_png_and_svg(tmp_path):
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    _JUMPUP_VIS._save(fig, tmp_path / "figure")

    assert (tmp_path / "figure.png").is_file()
    assert (tmp_path / "figure.svg").is_file()
