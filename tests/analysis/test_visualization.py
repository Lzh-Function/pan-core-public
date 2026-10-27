"""Unit tests for pan_core_analysis.visualization."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pan_core_analysis.visualization import (
    detect_jump_interval,
    load_jump_interval,
    plot_heatmap_with_jumpup,
    plot_trajectory_with_jumpup,
)


class TestDetectJumpInterval:
    def test_clear_step_jump(self):
        steps = list(range(0, 100, 10))
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 1.5, 1.2, 1.1, 1.0]
        series = pd.Series(values, index=steps)
        start, end, peak = detect_jump_interval(series)
        assert start in steps and end in steps and peak in steps
        assert start <= peak <= end

    def test_constant_series_raises(self):
        series = pd.Series([1.0] * 10, index=list(range(10)))
        with pytest.raises(ValueError, match="no change"):
            detect_jump_interval(series)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_jump_interval(pd.Series([3.0, 1.0], index=[1, 2]), ignore_initial_n=10)

    def test_returns_original_index_values(self):
        steps = [100, 200, 300, 400, 500]
        series = pd.Series([5.0, 5.0, 1.0, 0.5, 0.5], index=steps)
        start, end, peak = detect_jump_interval(series)
        assert start in steps and end in steps and peak in steps

    def test_ignore_initial_n(self):
        values = [100.0, 1.0] + [5.0, 5.0, 5.0, 1.0, 0.9, 0.8, 0.8, 0.8]
        series = pd.Series(values, index=list(range(10)))
        start, end, peak = detect_jump_interval(series, ignore_initial_n=2)
        # Peak should be after the initial noise region
        assert series.index.get_loc(peak) >= 2


def test_load_jump_interval_uses_paper_parameters(tmp_path):
    steps = list(range(0, 200, 10))
    values = [30.0, 5.0] + [5.0] * 8 + [5.0, 4.9, 4.0, 2.0, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1]
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    pd.DataFrame({"perplexity": values}, index=steps).to_csv(model_dir / "logits_metrics_trajectory.csv")

    start, end, peak = load_jump_interval(tmp_path, "m")

    assert start <= peak <= end
    assert peak >= 100  # the early drop at step 10 is ignored (ignore_initial_n = 10)
    with pytest.raises(FileNotFoundError):
        load_jump_interval(tmp_path, "missing")


def test_cmthermal_colormap_is_registered():
    assert "cmthermal" in mpl.colormaps


class TestPlotTrajectoryWithJumpup:
    def test_returns_fig_ax(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}, index=[0, 1, 2])
        fig, ax = plot_trajectory_with_jumpup(df, jump_interval=(0, 2, 1))
        assert fig is not None and ax is not None

    def test_no_jump_interval(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 1.5]}, index=[10, 20, 30])
        fig, ax = plot_trajectory_with_jumpup(df)
        assert fig is not None

    def test_uses_provided_axes(self):
        _, provided_ax = plt.subplots()
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
        fig, ax = plot_trajectory_with_jumpup(df, ax=provided_ax)
        assert ax is provided_ax


class TestPlotHeatmapWithJumpup:
    def test_returns_fig_ax(self):
        df = pd.DataFrame(np.random.default_rng(0).random((5, 4)), index=range(5))
        fig, ax = plot_heatmap_with_jumpup(df, jump_interval=(1, 3, 2))
        assert fig is not None and ax is not None

    def test_missing_jump_index_fails_fast(self):
        df = pd.DataFrame(np.ones((4, 3)), index=[0, 1, 2, 3])
        with pytest.raises(ValueError, match="not on the heatmap"):
            plot_heatmap_with_jumpup(df, jump_interval=(99, 100, 99))
