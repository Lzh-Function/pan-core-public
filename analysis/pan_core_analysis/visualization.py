"""
Shared visualization utilities for chirality emergence analysis.

All plot functions accept matplotlib Axes or create their own Figure/Axes.
The `cmthermal` colormap is registered once when this module is imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Custom colormap (cmthermal)
# ---------------------------------------------------------------------------

_CMTHERMAL_COLORS = ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"]


def generate_cmap(colors: list[str], cmap_name: str = "custom_cmap") -> mcolors.LinearSegmentedColormap:
    """Create a LinearSegmentedColormap from a list of hex color strings."""
    values = range(len(colors))
    vmax = float(max(values))
    color_list = [(v / vmax, c) for v, c in zip(values, colors)]
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)


def _register_cmthermal():
    cmap = generate_cmap(_CMTHERMAL_COLORS, "cmthermal")
    try:
        mpl.colormaps.register(cmap)
    except ValueError:
        pass  # already registered


_register_cmthermal()


# ---------------------------------------------------------------------------
# Index normalization
# ---------------------------------------------------------------------------

def _normalize_step(val) -> int:
    """Normalize a step index value to int.

    Handles:
      - "step5000-10000" → 10000  (raw-code CSV format)
      - "10000" → 10000
      - 10000   → 10000
    """
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if "-" in s:
        return int(s.split("-")[-1])
    return int(s.replace("step", ""))


# ---------------------------------------------------------------------------
# Jump-up interval detection
# ---------------------------------------------------------------------------

def detect_jump_interval(
    series: pd.Series,
    threshold_ratio: float = 0.15,
    ignore_initial_n: int = 0,
) -> tuple:
    """Detect the continuous phase-transition interval around the maximum gradient.

    Args:
        series: 1-D time series (e.g., perplexity indexed by training step).
        threshold_ratio: Fraction of the peak gradient that defines the interval boundary.
        ignore_initial_n: Number of leading points to ignore (avoids early-training noise).
    Returns:
        (start_index, end_index, peak_index) — all values from ``series.index``.
    Raises:
        ValueError: if the series has no more than ``ignore_initial_n`` points or no change
            after them (no interval can be defined).
    """
    smoothed = series.rolling(window=3, min_periods=1, center=True).mean()
    grad = smoothed.diff().abs()

    start_search_idx = min(ignore_initial_n, len(grad) - 1)
    search_grad = grad.iloc[start_search_idx:]

    if len(series) <= ignore_initial_n or search_grad.dropna().empty:
        raise ValueError(
            f"Cannot detect a jump-up interval: {len(series)} points with ignore_initial_n={ignore_initial_n}"
        )

    max_grad = search_grad.max()
    if pd.isna(max_grad) or max_grad == 0:
        raise ValueError("Cannot detect a jump-up interval: the trajectory has no change after the ignored points")

    peak_idx = start_search_idx + int(search_grad.argmax())
    threshold = max_grad * threshold_ratio

    start_idx = peak_idx
    while start_idx > ignore_initial_n and grad.iloc[start_idx] >= threshold:
        start_idx -= 1

    end_idx = peak_idx
    while end_idx < len(grad) - 1 and grad.iloc[end_idx] >= threshold:
        end_idx += 1

    return series.index[start_idx], series.index[end_idx], series.index[peak_idx]


# Paper settings (Section 2.3): perplexity of analysis 03, ±15 % of the peak rate,
# first 10 analyzed checkpoints ignored.
PAPER_JUMP_THRESHOLD_RATIO = 0.15
PAPER_JUMP_IGNORE_INITIAL_N = 10


def load_jump_interval(
    jump_logits_dir,
    model_name: str,
    threshold_ratio: float = PAPER_JUMP_THRESHOLD_RATIO,
    ignore_initial_n: int = PAPER_JUMP_IGNORE_INITIAL_N,
) -> tuple[int, int, int]:
    """Detect the jump-up interval of one model from analysis 03's perplexity trajectory.

    Reads ``{jump_logits_dir}/{model_name}/logits_metrics_trajectory.csv`` (column
    ``perplexity``, integer step index) and applies ``detect_jump_interval`` with the
    given parameters — the same computation as ``run_pipeline.resolve_jump``.

    Raises:
        FileNotFoundError / ValueError: if the trajectory or its column is missing or invalid.
    """
    from pathlib import Path

    path = Path(jump_logits_dir) / model_name / "logits_metrics_trajectory.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Jump trajectory not found (run analysis 03 first): {path}")
    frame = pd.read_csv(path, index_col=0)
    if "perplexity" not in frame:
        raise ValueError(f"Column 'perplexity' missing in {path}")
    series = frame["perplexity"]
    series.index = [int(value) for value in series.index]
    if series.index.duplicated().any() or series.isna().any():
        raise ValueError(f"Jump trajectory has duplicate steps or missing values: {path}")
    start, end, peak = detect_jump_interval(
        series.sort_index(), threshold_ratio=threshold_ratio, ignore_initial_n=ignore_initial_n
    )
    return int(start), int(end), int(peak)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _add_jump_band(ax: Axes, jump_start, jump_end, jump_peak=None, alpha: float = 0.1):
    """Shade the jump-up interval (light red) and mark its peak with a red dashed line."""
    ax.axvspan(jump_start, jump_end, color="red", alpha=alpha)
    if jump_peak is not None:
        ax.axvline(jump_peak, color="red", linestyle="--", alpha=0.5)


def plot_trajectory_with_jumpup(
    df: pd.DataFrame,
    jump_interval: tuple | None = None,
    ax: Axes | None = None,
    title: str = "",
    xlabel: str = "Training step",
    ylabel: str = "",
    figsize: tuple[float, float] = (10, 5),
    **lineplot_kwargs,
) -> tuple[Figure, Axes]:
    """Line plot of a trajectory DataFrame with optional jump-up interval shading.

    Args:
        df: DataFrame indexed by training step; each column is a separate line.
        jump_interval: (start, end, peak) tuple from ``detect_jump_interval``, or None.
        ax: Existing Axes to draw on. A new Figure is created if None.
        title: Figure/axes title.
        xlabel, ylabel: Axis labels.
        figsize: Figure size when creating a new Figure.
        **lineplot_kwargs: Forwarded to ``ax.plot()``.
    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col), **lineplot_kwargs)

    if jump_interval is not None:
        _add_jump_band(ax, jump_interval[0], jump_interval[1], jump_interval[2])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(df.columns) > 1:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_heatmap_with_jumpup(
    df: pd.DataFrame,
    jump_interval: tuple | None = None,
    ax: Axes | None = None,
    title: str = "",
    xlabel: str = "Training step",
    ylabel: str = "Head / Layer",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (18, 10),
    vmin: float | None = None,
    vmax: float | None = None,
    n_heads_per_layer: int | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap (feature × step) with optional jump-up boundary lines.

    Args:
        df: DataFrame with rows = training steps and columns = features (e.g., L×H heads).
            The matrix is transposed internally so x-axis = steps, y-axis = features.
        jump_interval: (start, end, peak) from ``detect_jump_interval``, or None.
        ax: Existing Axes or None.
        title, xlabel, ylabel: Labels.
        cmap: Colormap name (default ``"Blues"``).
        figsize: Figure size (default (18, 10)).
        vmin, vmax: Colormap clipping values.
        n_heads_per_layer: If set, draw horizontal separator lines every n_heads_per_layer rows
            to visually mark layer boundaries (e.g., 8 for 8-head models).
    Returns:
        (fig, ax) tuple.
    """
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalize step index to int so "step5000-10000" strings are handled correctly
    norm_index = [_normalize_step(v) for v in df.index]
    df = df.copy()
    df.index = norm_index

    sns.heatmap(
        df.T,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels="auto",
        yticklabels=True,
    )

    if jump_interval is not None:
        js = _normalize_step(jump_interval[0])
        je = _normalize_step(jump_interval[1])
        if js not in norm_index or je not in norm_index:
            raise ValueError(f"Jump interval ({js}, {je}) is not on the heatmap's step axis")
        start_col = norm_index.index(js)
        end_col = norm_index.index(je)
        ax.axvline(start_col, color="red", linestyle="--", linewidth=3, alpha=0.7)
        ax.axvline(end_col + 1, color="red", linestyle="--", linewidth=3, alpha=0.7)

    if n_heads_per_layer is not None:
        n_features = df.shape[1]
        for i in range(n_heads_per_layer, n_features, n_heads_per_layer):
            ax.axhline(i, color="gray", linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax
