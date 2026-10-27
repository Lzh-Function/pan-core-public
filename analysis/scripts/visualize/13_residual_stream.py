"""
13_residual_stream.py

Visualizes residual-stream metrics.

Paper figures (line plots, one line per layer: Emb, L0, ..., L7):
  - l2_norm_{chiral|background}_{enc|dec}_lineplot
        per-layer mean L2 norm per checkpoint, from 05_residual_stream.py
        (Fig 6a,b encoder; Fig S7a,b decoder)
  - cos_{chiral|background}_{enc|dec}_lineplot
        cosine similarity of the same layer between consecutive checkpoints, from
        03_jumpup_metrics.py (cos_*_trajectory.csv), plotted at the mid step (a + b) // 2
        (Fig 6c,d encoder; Fig S7c,d decoder; Fig S6 = encoder on the 5,000-step series)

Supplementary heatmaps (not in the paper), from 05_residual_stream.py:
  - cos_/cka_{chiral|background}_{enc|dec}_layerwise_trajectory
        cosine / linear CKA between ADJACENT layers within one checkpoint

The jump-up interval is detected from 03's perplexity with the pipeline parameters and drawn
as a light red band with a dashed peak line (set show_jump: false for Fig S6).

Config YAML:
  input_dir: directory with {model_name}/ subdirs from analysis 05
  jump_logits_dir: directory with {model_name}/ subdirs from analysis 03
  output_dir:
  models: list (optional)
  jump_threshold_ratio (0.15), jump_ignore_initial_n (10), show_jump (true)
  xlim: [min, max] (optional; Fig S6: [0, 80000])
  cos_ylim: [min, max] (optional; e.g. Fig 6c,d pancore-addonce: [0.99, 1.0])
  l2_ylim: [min, max] (optional)
  figsize: [w, h] for heatmaps (default [18, 10]); figsize_line (default [10, 6])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis.visualization import (
    PAPER_JUMP_IGNORE_INITIAL_N,
    PAPER_JUMP_THRESHOLD_RATIO,
    _add_jump_band,
    load_jump_interval,
    plot_heatmap_with_jumpup,
)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save(fig, path_stem: Path, dpi: int = 300) -> None:
    """Save a figure as both PNG and SVG, then close it."""
    for extension in ("png", "svg"):
        fig.savefig(path_stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _layer_labels(columns) -> list[str]:
    """Map 'embeddings', 'Layer_i' column names to the paper legend labels 'Emb', 'Li'."""
    labels = []
    for col in columns:
        if col == "embeddings":
            labels.append("Emb")
        elif str(col).startswith("Layer_"):
            labels.append("L" + str(col).removeprefix("Layer_"))
        else:
            raise ValueError(f"Unexpected layer column: {col!r}")
    return labels


def _mid_step_index(index) -> list[int]:
    """'step{a}-{b}' → (a + b) // 2, the x position used in the paper."""
    mids = []
    for value in index:
        before, after = str(value).removeprefix("step").split("-")
        mids.append((int(before) + int(after)) // 2)
    return mids


def _layer_lineplot(df: pd.DataFrame, jump, ylabel: str, figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    for col, label in zip(df.columns, _layer_labels(df.columns)):
        ax.plot(df.index, df[col], label=label)
    if jump is not None:
        _add_jump_band(ax, jump[0], jump[1], jump[2])
    ax.set_xlabel("steps", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right")
    ax.grid()
    fig.tight_layout()
    return fig, ax


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path, index_col=0)


def main():
    parser = argparse.ArgumentParser(description="Visualize residual stream metrics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(cfg["input_dir"])
    jump_logits_dir = Path(cfg["jump_logits_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "models" in cfg:
        model_names = cfg["models"]
    else:
        model_names = [p.name for p in sorted(input_dir.iterdir()) if p.is_dir()]

    figsize = tuple(cfg.get("figsize", [18, 10]))
    figsize_line = tuple(cfg.get("figsize_line", [10, 6]))
    xlim = cfg.get("xlim")
    cos_ylim = cfg.get("cos_ylim")
    l2_ylim = cfg.get("l2_ylim")
    show_jump = cfg.get("show_jump", True)
    threshold_ratio = cfg.get("jump_threshold_ratio", PAPER_JUMP_THRESHOLD_RATIO)
    ignore_initial_n = cfg.get("jump_ignore_initial_n", PAPER_JUMP_IGNORE_INITIAL_N)

    for model_name in model_names:
        mdir = input_dir / model_name
        jdir = jump_logits_dir / model_name
        for directory in (mdir, jdir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing analysis output directory: {directory}")
        plot_dir = output_dir / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        jump = load_jump_interval(jump_logits_dir, model_name, threshold_ratio, ignore_initial_n)
        band = jump if show_jump else None

        for side in ("enc", "dec"):
            for token in ("chiral", "background"):
                # Fig 6a,b / S7a,b: per-layer L2 norm (analysis 05)
                l2 = _read_csv(mdir / f"l2_norm_{token}_{side}_trajectory.csv").sort_index()
                fig, _ = _layer_lineplot(l2, band, "l2 norm", figsize_line, xlim, l2_ylim)
                _save(fig, plot_dir / f"l2_norm_{token}_{side}_lineplot")

                # Fig 6c,d / S6 / S7c,d: checkpoint-to-checkpoint cosine (analysis 03)
                cos = _read_csv(jdir / f"cos_{token}_{side}_trajectory.csv")
                cos.index = _mid_step_index(cos.index)
                cos = cos.sort_index()
                fig, _ = _layer_lineplot(cos, band, "cosine similarity", figsize_line, xlim, cos_ylim)
                _save(fig, plot_dir / f"cos_{token}_{side}_lineplot")

                # Supplementary: adjacent-layer cosine / CKA within a checkpoint (analysis 05)
                for metric, label, limits in (("cos", "cosine", {}), ("cka", "linear CKA", {"vmin": 0.0, "vmax": 1.0})):
                    stem = f"{metric}_{token}_{side}_layerwise_trajectory"
                    df = _read_csv(mdir / f"{stem}.csv")
                    fig, _ = plot_heatmap_with_jumpup(
                        df=df,
                        jump_interval=band,
                        title=f"{model_name} {side} {token} — adjacent-layer {label} (within checkpoint)",
                        xlabel="Training step",
                        ylabel="Layer transition",
                        figsize=figsize,
                        **limits,
                    )
                    _save(fig, plot_dir / stem)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
