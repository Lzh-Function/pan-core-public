"""
12_jumpup_metrics.py

Visualizes jump-up metrics from 03_jumpup_metrics.py outputs.
  - Perplexity + CDR trajectory (paper Fig 4)
  - Attention mass heatmap (all heads) + top-N line plot (Fig 5)
  - Attention entropy heatmap + top-N line plot (Fig S4)
  - Graph distance from chiral tokens heatmap + top-N line plot (Fig S5)
  - SAR / NIT / graph distance to chiral tokens heatmaps (not in the paper)

The jump-up interval is detected from 03's perplexity trajectory with the same parameters
as the pipeline (``jump_threshold_ratio``, ``jump_ignore_initial_n``; paper: 0.15, 10) and
drawn as a light red band with a dashed peak line (line plots) or two dashed boundary lines
(heatmaps). Top-N heads are the N heads with the largest amplitude (max - min) inside the
interval (inclusive), as in the paper figures; their full trajectories are plotted.

Chiral-head selection for analysis 07: attention_mass_{enc,dec}_head_ranking.csv ranks every head
by its amplitude inside the jump-up interval (-> chiral_heads, used at the pre-, peak- and
post-transition checkpoints) and by its attention mass at the last analyzed step
(-> final_chiral_heads, when that step is the final checkpoint, e.g. step 700,000 of the
5,000-step series).

Note: in arXiv v1 the Fig 4b band was detected on the CDR trajectory itself
(43,000-52,000 for pancore-addonce); this script draws the perplexity interval on both
panels, as the caption states (corrected figure).

Config YAML:
  input_dir:    directory with {model_name}/ subdirs containing trajectory CSVs
  output_dir:   PNG/SVG output directory
  models:       list of model names (or auto-discovered)
  encoder_only: bool (default false — plot both enc and dec)
  top_n_heads:  int (default 3)
  jump_threshold_ratio:  float (default 0.15)
  jump_ignore_initial_n: int (default 10)
  figsize_heatmap: [w, h] (default [18, 10])
  figsize_line:    [w, h] (default [12, 6])
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
    load_jump_interval,
    plot_heatmap_with_jumpup,
    plot_trajectory_with_jumpup,
)

# metric -> (top-N line plot, y-axis label)
_HEAD_METRICS = {
    "attention_mass": (True, "attention mass"),
    "attention_entropy": (True, "attention entropy"),
    "graph_dist_from": (True, "graph distance from chiral token"),
    "sar": (False, "SAR"),
    "nit": (False, "NIT"),
    "graph_dist_to": (False, "graph distance to chiral token"),
}


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _top_n_heads(df: pd.DataFrame, n: int, jump: tuple[int, int, int]) -> list[str]:
    """Return the n heads with the largest amplitude (max - min) inside the jump interval."""
    start, end, _ = jump
    window = df.loc[start:end]
    if window.empty:
        raise ValueError(f"No steps inside the jump interval {start}-{end}")
    amplitude = window.max(axis=0) - window.min(axis=0)
    return amplitude.nlargest(n).index.tolist()


def head_ranking(df: pd.DataFrame, jump: tuple[int, int, int]) -> pd.DataFrame:
    """Attention-mass head ranking for the two chiral-head selection criteria of analysis 07.

    interval_amplitude: max - min inside the jump-up interval (chiral_heads, paper Fig 5)
    last_step_mass:     value at the last analyzed step (final_chiral_heads when that step is the
                        final checkpoint)
    Rank 1 = largest. Rows are sorted by interval_amplitude_rank.
    """
    start, end, _ = jump
    window = df.loc[start:end]
    if window.empty:
        raise ValueError(f"No steps inside the jump interval {start}-{end}")
    last_step = int(df.index.max())
    ranking = pd.DataFrame({
        "interval_amplitude": window.max(axis=0) - window.min(axis=0),
        "last_step_mass": df.loc[last_step],
    })
    if ranking.isna().any().any():
        raise ValueError("Attention mass contains missing values inside the jump interval or at the last step")
    ranking["interval_amplitude_rank"] = ranking["interval_amplitude"].rank(ascending=False, method="min").astype(int)
    ranking["last_step_rank"] = ranking["last_step_mass"].rank(ascending=False, method="min").astype(int)
    ranking["last_step"] = last_step
    ranking.index.name = "head"
    return ranking.sort_values("interval_amplitude_rank")[
        ["interval_amplitude", "interval_amplitude_rank", "last_step", "last_step_mass", "last_step_rank"]
    ]


def _heads_per_layer(columns) -> int:
    """Number of heads per layer from column names 'L{l}_H{h}'."""
    heads = {int(str(col).split("_H")[1]) for col in columns}
    return max(heads) + 1


def _save(fig, path_stem: Path, dpi: int = 300) -> None:
    """Save a figure as both PNG and SVG, then close it."""
    for extension in ("png", "svg"):
        fig.savefig(path_stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize jump-up metrics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "models" in cfg:
        model_names = cfg["models"]
    else:
        model_names = [p.name for p in sorted(input_dir.iterdir()) if p.is_dir()]

    encoder_only = cfg.get("encoder_only", False)
    top_n = cfg.get("top_n_heads", 3)
    threshold_ratio = cfg.get("jump_threshold_ratio", PAPER_JUMP_THRESHOLD_RATIO)
    ignore_initial_n = cfg.get("jump_ignore_initial_n", PAPER_JUMP_IGNORE_INITIAL_N)
    figsize_hm = tuple(cfg.get("figsize_heatmap", [18, 10]))
    figsize_ln = tuple(cfg.get("figsize_line", [12, 6]))

    sides = ["enc"] if encoder_only else ["enc", "dec"]

    for model_name in model_names:
        mdir = input_dir / model_name
        if not mdir.is_dir():
            raise FileNotFoundError(f"Missing analysis 03 output directory: {mdir}")

        plot_dir = output_dir / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        jump = load_jump_interval(input_dir, model_name, threshold_ratio, ignore_initial_n)
        logits_df = pd.read_csv(mdir / "logits_metrics_trajectory.csv", index_col=0).sort_index()

        # --- Perplexity + CDR (Fig 4) ---
        cols_to_plot = ["perplexity", "chiral_dilemma_rate"]
        missing = [c for c in cols_to_plot if c not in logits_df.columns]
        if missing:
            raise ValueError(f"Missing logits metric columns {missing} for {model_name}")
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(figsize_ln[0] * len(cols_to_plot), figsize_ln[1]))
        for ax, col in zip(axes, cols_to_plot):
            plot_trajectory_with_jumpup(
                df=logits_df[[col]],
                jump_interval=jump,
                ax=ax,
                title=f"{model_name} — {col}",
                ylabel=col,
                figsize=figsize_ln,
            )
        fig.tight_layout()
        _save(fig, plot_dir / "perplexity_cdr")

        for side in sides:
            for metric, (with_top_n, ylabel) in _HEAD_METRICS.items():
                csv_path = mdir / f"{metric}_{side}_trajectory.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"Missing analysis 03 output: {csv_path}")
                df = pd.read_csv(csv_path, index_col=0).sort_index()

                fig, ax = plot_heatmap_with_jumpup(
                    df=df,
                    jump_interval=jump,
                    title=f"{model_name} {side} — {metric}",
                    xlabel="Training step",
                    ylabel="Layer × Head",
                    figsize=figsize_hm,
                    n_heads_per_layer=_heads_per_layer(df.columns),
                )
                _save(fig, plot_dir / f"{metric}_{side}_heatmap")
                if metric == "attention_mass":
                    head_ranking(df, jump).to_csv(plot_dir / f"{metric}_{side}_head_ranking.csv")

                if with_top_n:
                    top_cols = _top_n_heads(df, top_n, jump)
                    fig, ax = plot_trajectory_with_jumpup(
                        df=df[top_cols],
                        jump_interval=jump,
                        title=f"{model_name} {side} — top-{top_n} heads by change in the jump-up interval",
                        ylabel=ylabel,
                        figsize=figsize_ln,
                    )
                    _save(fig, plot_dir / f"{metric}_{side}_top{top_n}")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
