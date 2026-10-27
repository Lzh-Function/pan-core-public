"""
14_latent_trajectory.py

Visualizes latent vector geometry from 04_latent_trajectory.py output CSVs.

Paper figures (chi-SMILES red, non-chi-SMILES blue):
  - latent_l2_norm   mean latent L2 norm at each checkpoint step (Fig 7a)
  - latent_cosine    cosine similarity between consecutive checkpoints, at mid_step (Fig 7b, S8a)
  - latent_cka       linear CKA between consecutive checkpoints, at mid_step (Fig 7c, S8b)
Supplementary (not in the paper): intrinsic dimension, isotropy entropy, active units.

The jump-up interval is detected from 03's perplexity with the pipeline parameters and drawn
as a light red band with a dashed peak line. Fig S8 (5,000-step series) has no band
(show_jump: false) and xlim [0, 80000], ylim cosine [0.95, 1.0], cka [0.90, 1.0];
Fig 7 (500-step series) used ylim cosine [0.999, 1.0], cka [0.9975, 1.0].

Input layout (per model):
  {input_dir}/{model_name}/latent_static_chiral_trajectory.csv
  {input_dir}/{model_name}/latent_static_nchiral_trajectory.csv
  {input_dir}/{model_name}/latent_dynamic_chiral_trajectory.csv
  {input_dir}/{model_name}/latent_dynamic_nchiral_trajectory.csv

Config YAML:
  input_dir: directory containing {model_name}/ subdirectories
  output_dir:
  models: list (optional; default: auto-discover from subdirs)
  jump_logits_dir: dir where {model_name}/logits_metrics_trajectory.csv (analysis 03) lives
  jump_threshold_ratio (0.15), jump_ignore_initial_n (10), show_jump (true)
  xlim: [min, max] (optional)
  ylim: {l2_norm: [..], cosine: [..], cka: [..]} (optional, per panel)
  figsize: [w, h] (default [12, 8])
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
)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save(fig, path_stem: Path):
    for ext in ("png", "svg"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis 04 output: {path}")
    return pd.read_csv(path, index_col="step").sort_index()


def _pair_plot(chiral: pd.Series, non_chiral: pd.Series, ylabel: str, jump, figsize, xlim, ylim):
    """chi-SMILES vs non-chi-SMILES line plot in the paper style."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(chiral.index, chiral.values, label="chi-SMILES", color="red")
    ax.plot(non_chiral.index, non_chiral.values, label="non-chi-SMILES", color="blue")
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


def _dynamic_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Dynamic metric indexed by mid_step; the first checkpoint (no predecessor) is dropped."""
    rows = df.dropna(subset=["mid_step"])
    return pd.Series(rows[column].values, index=rows["mid_step"].values)


def main():
    parser = argparse.ArgumentParser(description="Visualize latent trajectory.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "models" in cfg:
        model_names = cfg["models"]
    else:
        model_names = sorted(d.name for d in input_dir.iterdir() if d.is_dir())

    figsize = tuple(cfg.get("figsize", [12, 8]))
    jump_logits_dir = Path(cfg["jump_logits_dir"])
    threshold_ratio = cfg.get("jump_threshold_ratio", PAPER_JUMP_THRESHOLD_RATIO)
    ignore_initial_n = cfg.get("jump_ignore_initial_n", PAPER_JUMP_IGNORE_INITIAL_N)
    show_jump = cfg.get("show_jump", True)
    xlim = cfg.get("xlim")
    ylims = cfg.get("ylim", {}) or {}
    unknown = set(ylims) - {"l2_norm", "cosine", "cka"}
    if unknown:
        raise ValueError(f"Unknown ylim keys: {sorted(unknown)}")

    for model_name in model_names:
        model_dir = input_dir / model_name
        static_chi = _read(model_dir / "latent_static_chiral_trajectory.csv")
        static_nchi = _read(model_dir / "latent_static_nchiral_trajectory.csv")
        dyn_chi = _read(model_dir / "latent_dynamic_chiral_trajectory.csv")
        dyn_nchi = _read(model_dir / "latent_dynamic_nchiral_trajectory.csv")

        plot_dir = output_dir / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        jump = load_jump_interval(jump_logits_dir, model_name, threshold_ratio, ignore_initial_n)
        band = jump if show_jump else None

        # --- Fig 7a: L2 norm at each checkpoint ---
        fig, _ = _pair_plot(static_chi["l2_norm_mean"], static_nchi["l2_norm_mean"], "l2 norm",
                            band, figsize, xlim, ylims.get("l2_norm"))
        _save(fig, plot_dir / "latent_l2_norm")

        # --- Fig 7b / S8a: cosine similarity, Fig 7c / S8b: linear CKA (at mid_step) ---
        for column, ylabel, key, stem in [
            ("cosine_mean", "cosine similarity", "cosine", "latent_cosine"),
            ("linear_cka", "cka", "cka", "latent_cka"),
        ]:
            fig, _ = _pair_plot(_dynamic_series(dyn_chi, column), _dynamic_series(dyn_nchi, column), ylabel,
                                band, figsize, xlim, ylims.get(key))
            _save(fig, plot_dir / stem)

        # --- Supplementary (not in the paper): intrinsic dim, isotropy, active units ---
        for stat_key, label in [
            ("intrinsic_dim",     "intrinsic dimension (TwoNN)"),
            ("isotropy_entropy",  "isotropy entropy"),
            ("active_units",      "active units"),
        ]:
            fig, _ = _pair_plot(static_chi[stat_key], static_nchi[stat_key], label, band, figsize, xlim, None)
            _save(fig, plot_dir / stat_key)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
