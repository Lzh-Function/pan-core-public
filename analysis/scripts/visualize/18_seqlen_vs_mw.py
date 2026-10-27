"""
18_seqlen_vs_mw.py

Paper Fig S2: tokenized sequence length vs molecular weight of the training sources, as
log-count hexbins with 90 / 99 / 99.9 % quantile lines of both variables.
  (a) ZINC20: full range, gridsize 500
  (b) PubChem: zoomed to sequence length 0-1000 and molecular weight 0-10000, gridsize 5000

Inputs are per-molecule summary CSVs with columns ``seqlen`` and ``mw`` (paper:
ZINC_summary.csv and pubchem_summary.csv, about 2 GB each; the code that produced them was
not preserved). Reading them needs several GB of memory.

Output: {output_dir}/zinc_seqlen_mw_hexbin.{png,svg}, {output_dir}/pubchem_seqlen_mw_hexbin_zoomed.{png,svg}

Config YAML:
  zinc_summary: /path/to/ZINC_summary.csv
  pubchem_summary: /path/to/pubchem_summary.csv
  output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

QUANTILES = [0.90, 0.99, 0.999]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read(path: str | Path):
    frame = pl.read_csv(path, columns=["seqlen", "mw"])
    return frame["seqlen"], frame["mw"]


def _label(q: float) -> str:
    return f"{q * 100:.1f}".rstrip("0").rstrip(".") + "%"


def plot_zinc(x, y, path_stem: Path) -> None:
    x_q, y_q = np.quantile(x, QUANTILES), np.quantile(y, QUANTILES)
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(x, y, gridsize=500, bins="log", mincnt=1, cmap="Blues")
    x_max, y_max = x.max(), y.max()
    for q, xv in zip(QUANTILES, x_q):
        plt.axvline(xv, linestyle="--", color="black", alpha=0.5, linewidth=1)
        plt.text(xv, y_max * 0.98, _label(q), rotation=90, ha="right", va="top",
                 fontweight="bold", color="red", alpha=0.8, fontsize=11,
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    for q, yv in zip(QUANTILES, y_q):
        plt.axhline(yv, linestyle=":", color="black", alpha=0.5, linewidth=1)
        plt.text(x_max * 0.98, yv, _label(q), ha="right", va="bottom",
                 fontweight="bold", color="red", alpha=0.8, fontsize=11,
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    plt.colorbar(hb, label="log10(count)")
    plt.xlabel("Sequence Length", fontsize=18)
    plt.ylabel("Molecular Weight", fontsize=18)
    plt.subplots_adjust(left=0.088, right=1.0, top=0.978, bottom=0.08)
    for ext in ("png", "svg"):
        plt.savefig(path_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close()


def plot_pubchem_zoomed(x, y, path_stem: Path) -> None:
    x_q, y_q = np.quantile(x, QUANTILES), np.quantile(y, QUANTILES)
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(x, y, gridsize=5000, bins="log", mincnt=1, cmap="Blues")
    for q, xv in zip(QUANTILES, x_q):
        plt.axvline(xv, linestyle="--", color="black", alpha=0.8, linewidth=1)
        plt.text(xv, 9800, _label(q), rotation=90, ha="right", va="top",
                 fontweight="bold", color="red", alpha=0.8,
                 bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1))
    for q, yv in zip(QUANTILES, y_q):
        plt.axhline(yv, linestyle=":", color="black", alpha=0.8, linewidth=1)
        plt.text(980, yv, _label(q), ha="right", va="bottom",
                 fontweight="bold", color="red", alpha=0.8,
                 bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1))
    plt.colorbar(hb, label="log10(count)")
    plt.xlim(0, 1000)
    plt.xlabel("Sequence Length", fontsize=18)
    plt.ylim(0, 10000)
    plt.ylabel("Molecular Weight", fontsize=18)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(path_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot sequence length vs molecular weight hexbins.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_zinc(*_read(cfg["zinc_summary"]), output_dir / "zinc_seqlen_mw_hexbin")
    plot_pubchem_zoomed(*_read(cfg["pubchem_summary"]), output_dir / "pubchem_seqlen_mw_hexbin_zoomed")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
