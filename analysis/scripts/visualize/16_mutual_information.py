"""
16_mutual_information.py

Visualizes JSON trajectories produced by 09_mutual_information.py.

Figures mirror the v5 mutual-information experiment:
  1. R/S balanced accuracy and shuffled-corrected information score
     (right panel: R/S solid, canonical @/@@ dashed = paper Fig S9b)
  2. Per-layer mean-pool R/S probe heatmaps
  3. Per-layer chiral-window R/S probe heatmaps
  6. Linear versus nonlinear pooled-latent probes (paper Fig S9c)
  7. Absolute R/S versus canonical @/@@ probes, all models (paper Fig S9a)
  7d. figure7d_{model}: the same comparison for one model in the paper Fig 7d style

probe_summary_{model}.json holds the values quoted in the paper text, using the jump-up
interval (start, end, peak) of analysis 03: R/S and @/@@ balanced accuracy at the interval end
and at the last checkpoint, the R/S maximum and its step, the fraction of consecutive
checkpoint pairs from the interval end onward in which @/@@ increased, the number of
checkpoints from the interval end onward with R/S >= @/@@, and last-checkpoint information
scores, MLP / RF readouts and per-layer mean-pool / window probes, and, when 09 ran with
enable_token_identity_probe, the per-layer minimum over checkpoints of the input @/@@
control probe (token_identity_min_balanced_acc_by_layer; null otherwise).

Jump bands use the pipeline parameters (jump_threshold_ratio 0.15, jump_ignore_initial_n 10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis.visualization import (
    PAPER_JUMP_IGNORE_INITIAL_N,
    PAPER_JUMP_THRESHOLD_RATIO,
    load_jump_interval,
)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def _save(fig, output_path: Path) -> None:
    for extension in ("png", "svg"):
        fig.savefig(output_path.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _jump_interval(
    jump_logits_dir: Path | None, model_name: str, threshold_ratio: float, ignore_initial_n: int
) -> tuple | None:
    """Jump-up interval from analysis 03 (None only when no jump_logits_dir is configured)."""
    if jump_logits_dir is None:
        return None
    return load_jump_interval(jump_logits_dir, model_name, threshold_ratio, ignore_initial_n)


def _add_jump_band(ax, interval: tuple | None, color: str) -> None:
    if interval is not None:
        ax.axvspan(interval[0], interval[1], color=color, alpha=0.1)
        ax.axvline(interval[2], color=color, linestyle="--", alpha=0.5)


def _metric_values(metrics: list[dict], section: str, key: str) -> list[float]:
    return [
        (metric.get(section) or {}).get(key, float("nan"))
        for metric in metrics
    ]


def make_plots(
    all_results: dict[str, dict],
    output_dir: Path,
    *,
    jump_logits_dir: Path | None = None,
    colors: dict[str, str] | None = None,
    display_names: dict[str, str] | None = None,
    figsize: tuple[float, float] = (14, 5),
    threshold_ratio: float = PAPER_JUMP_THRESHOLD_RATIO,
    ignore_initial_n: int = PAPER_JUMP_IGNORE_INITIAL_N,
) -> None:
    """Create all v5 mutual-information figures from result dictionaries."""
    if not all_results:
        raise ValueError("all_results must contain at least one model")
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = colors or {}
    display_names = display_names or {}
    intervals = {
        model_name: _jump_interval(jump_logits_dir, model_name, threshold_ratio, ignore_initial_n)
        for model_name in all_results
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for model_name, result in all_results.items():
        steps = result["steps"]
        metrics = result["metrics"]
        if len(steps) != len(metrics):
            raise ValueError(f"Mismatched steps and metrics for {model_name}")
        color = colors.get(model_name)
        label = display_names.get(model_name, model_name)
        line_options = {"color": color} if color else {}
        axes[0].plot(
            steps,
            _metric_values(metrics, "rs_probe", "balanced_acc"),
            label=label,
            linewidth=1.5,
            **line_options,
        )
        axes[1].plot(
            steps,
            _metric_values(metrics, "rs_probe", "mi_lb_corrected"),
            label=f"{label} R/S",
            linewidth=1.5,
            **line_options,
        )
        band_color = color or axes[0].lines[-1].get_color()
        if any(metric.get("canonical_token_probe") is not None for metric in metrics):
            axes[1].plot(
                steps,
                _metric_values(metrics, "canonical_token_probe", "mi_lb_corrected"),
                label=f"{label} canonical @/@@",
                linestyle="--",
                linewidth=1.5,
                color=band_color,
            )
        interval = intervals[model_name]
        _add_jump_band(axes[0], interval, band_color)
        _add_jump_band(axes[1], interval, band_color)

    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    axes[0].set(
        xlabel="Training Step",
        ylabel="Balanced Accuracy",
        title="R/S Decodability from Latent z",
    )
    axes[0].set_ylim(0.35, 1.05)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Training Step",
        ylabel="Information score (nats)",
        title="Probe MI Lower Bound Minus Shuffle Baseline",
    )
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "figure1_rs_probe")

    for figure_number, section, title in (
        (2, "rs_probe_by_layer", "Mean Pool"),
        (3, "rs_window_probe_by_layer", "Chiral Window Pool"),
    ):
        for model_name, result in all_results.items():
            layers = result["probe_layers"]
            steps = result["steps"]
            matrix = np.full((len(layers), len(steps)), np.nan)
            for step_index, metric in enumerate(result["metrics"]):
                for layer_index, layer in enumerate(layers):
                    matrix[layer_index, step_index] = (
                        metric.get(section, {})
                        .get(str(layer), {})
                        .get("balanced_acc", float("nan"))
                    )

            fig, axis = plt.subplots(figsize=(max(10, min(len(steps) // 8 + 4, 22)), 4))
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.5, vmax=1.0)
            axis.set_xticks(range(len(steps)))
            axis.set_xticklabels(steps, rotation=45, ha="right")
            axis.set_yticks(range(len(layers)))
            axis.set_yticklabels([f"Layer {layer}" for layer in layers])
            axis.set(
                xlabel="Training Step",
                ylabel="Encoder Layer",
                title=f"{display_names.get(model_name, model_name)} — {title}",
            )
            fig.colorbar(image, ax=axis, label="Balanced Accuracy")
            fig.tight_layout()
            safe_name = model_name.replace("/", "_")
            _save(fig, output_dir / f"figure{figure_number}_rs_heatmap_{safe_name}")

    if any(
        metric.get("rs_nonlinear_probe") is not None
        for result in all_results.values()
        for metric in result["metrics"]
    ):
        fig, axis = plt.subplots(figsize=(10, 5))
        for model_name, result in all_results.items():
            steps = result["steps"]
            metrics = result["metrics"]
            color = colors.get(model_name)
            label = display_names.get(model_name, model_name)
            line_options = {"color": color} if color else {}
            axis.plot(
                steps,
                _metric_values(metrics, "rs_probe", "balanced_acc"),
                label=f"{label} linear",
                linewidth=1.5,
                **line_options,
            )
            axis.plot(
                steps,
                _metric_values(metrics, "rs_nonlinear_probe", "mlp_balanced_acc"),
                label=f"{label} MLP",
                linestyle="--",
                linewidth=1,
                **line_options,
            )
            axis.plot(
                steps,
                _metric_values(metrics, "rs_nonlinear_probe", "rf_balanced_acc"),
                label=f"{label} RF",
                linestyle=":",
                linewidth=1,
                **line_options,
            )
            _add_jump_band(axis, intervals[model_name], color or "gray")
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Training Step",
            ylabel="Balanced Accuracy",
            title="R/S Probe: Linear versus Nonlinear",
        )
        axis.set_ylim(0.35, 1.05)
        axis.legend(fontsize=8, ncol=2)
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "figure6_rs_linear_vs_nonlinear")

    if any(
        metric.get("canonical_token_probe") is not None
        for result in all_results.values()
        for metric in result["metrics"]
    ):
        fig, axis = plt.subplots(figsize=(10, 5))
        for model_name, result in all_results.items():
            steps = result["steps"]
            metrics = result["metrics"]
            color = colors.get(model_name)
            label = display_names.get(model_name, model_name)
            line_options = {"color": color} if color else {}
            axis.plot(
                steps,
                _metric_values(metrics, "rs_probe", "balanced_acc"),
                label=f"{label} R/S",
                linewidth=1.5,
                **line_options,
            )
            axis.plot(
                steps,
                _metric_values(metrics, "canonical_token_probe", "balanced_acc"),
                label=f"{label} canonical @/@@",
                linestyle="--",
                linewidth=1,
                **line_options,
            )
            _add_jump_band(axis, intervals[model_name], color or "gray")
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Training Step",
            ylabel="Balanced Accuracy",
            title="Stereo Probes: Absolute R/S versus Canonical @/@@",
        )
        axis.set_ylim(0.35, 1.05)
        axis.legend(fontsize=8, ncol=2)
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir / "figure7_stereo_probe_comparison")

        for model_name, result in all_results.items():
            if any(metric.get("canonical_token_probe") is None for metric in result["metrics"]):
                print(f"[warning] figure7d_{model_name} skipped: canonical_token_probe missing (enable_canonical_token_probe)")
                continue
            _plot_figure7d(result, intervals[model_name], output_dir / f"figure7d_{model_name.replace('/', '_')}")

    for model_name, result in all_results.items():
        if intervals[model_name] is None:
            print(f"[warning] probe_summary_{model_name}.json skipped: jump_logits_dir is not set")
        else:
            summary = probe_summary(result, intervals[model_name])
            with open(output_dir / f"probe_summary_{model_name.replace('/', '_')}.json", "w") as f:
                json.dump(summary, f, indent=2)


def _plot_figure7d(result: dict, interval: tuple | None, path_stem: Path) -> None:
    """One model, absolute R/S (red) vs canonical @/@@ (black dashed), paper Fig 7d style."""
    steps = result["steps"]
    fig, axis = plt.subplots(figsize=(12, 8))
    axis.plot(steps, _metric_values(result["metrics"], "rs_probe", "balanced_acc"),
              color="red", linewidth=2.2, label="absolute R/S")
    axis.plot(steps, _metric_values(result["metrics"], "canonical_token_probe", "balanced_acc"),
              color="black", linestyle="--", linewidth=2.2, label="canonical @/@@")
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="chance")
    if interval is not None:
        axis.axvspan(interval[0], interval[1], color="red", alpha=0.1, label="Jump Up")
        axis.axvline(interval[2], color="red", linestyle="--", alpha=0.5)
    axis.set_xlabel("steps", fontsize=20)
    axis.set_ylabel("balanced accuracy", fontsize=20)
    axis.tick_params(labelsize=14)
    axis.legend(loc="lower right", fontsize=13)
    axis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, path_stem)


def probe_summary(result: dict, interval: tuple) -> dict:
    """Values quoted in the paper text, from one model's probe trajectory and its jump interval."""
    steps = list(result["steps"])
    metrics = result["metrics"]
    start, end, peak = (int(value) for value in interval)
    if end not in steps:
        raise ValueError(f"Jump-up interval end {end} is not a probed checkpoint")
    rs = pd.Series(_metric_values(metrics, "rs_probe", "balanced_acc"), index=steps)
    at = pd.Series(_metric_values(metrics, "canonical_token_probe", "balanced_acc"), index=steps)
    last = steps[-1]
    post_rs, post_at = rs.loc[end:], at.loc[end:]
    at_diffs = post_at.diff().dropna()
    last_metrics = metrics[-1]

    def layer_values(section: str) -> dict:
        return {
            layer: values.get("balanced_acc", float("nan"))
            for layer, values in (last_metrics.get(section) or {}).items()
        }

    return {
        "jump_interval": {"start": start, "end": end, "peak": peak},
        "last_step": last,
        "rs_balanced_acc_at_end": float(rs.loc[end]),
        "rs_balanced_acc_at_last": float(rs.loc[last]),
        "rs_balanced_acc_max": float(rs.max()),
        "rs_balanced_acc_max_step": int(rs.idxmax()),
        "at_balanced_acc_at_end": float(at.loc[end]),
        "at_balanced_acc_at_last": float(at.loc[last]),
        "at_fraction_increasing_pairs_from_end": float((at_diffs > 0).mean()) if len(at_diffs) else float("nan"),
        "rs_ge_at_checkpoints_from_end": int((post_rs >= post_at).sum()),
        "n_checkpoints_from_end": int(len(post_rs)),
        "rs_information_score_at_last": (last_metrics.get("rs_probe") or {}).get("mi_lb_corrected", float("nan")),
        "at_information_score_at_last": (last_metrics.get("canonical_token_probe") or {}).get("mi_lb_corrected", float("nan")),
        "rs_mlp_balanced_acc_at_last": (last_metrics.get("rs_nonlinear_probe") or {}).get("mlp_balanced_acc", float("nan")),
        "rs_rf_balanced_acc_at_last": (last_metrics.get("rs_nonlinear_probe") or {}).get("rf_balanced_acc", float("nan")),
        "rs_mean_pool_by_layer_at_last": layer_values("rs_probe_by_layer"),
        "rs_window_pool_by_layer_at_last": layer_values("rs_window_probe_by_layer"),
        "token_identity_min_balanced_acc_by_layer": token_identity_minimum(metrics),
    }


def token_identity_minimum(metrics: list[dict]) -> dict | None:
    """Minimum over checkpoints of the input @/@@ control probe per layer; None if not computed."""
    sections = [metric.get("token_identity_probe") for metric in metrics]
    if all(section is None for section in sections):
        return None
    if any(section is None for section in sections):
        raise ValueError("token_identity_probe is present for only some checkpoints")
    layers = sorted({layer for section in sections for layer in section}, key=int)
    return {
        layer: float(np.min([section[layer]["balanced_acc"] for section in sections]))
        for layer in layers
    }


def _discover_results(input_dir: Path, model_names: list[str] | None) -> dict[str, dict]:
    if model_names is None:
        paths = sorted(input_dir.glob("*/mutual_information_*.json"))
    else:
        paths = [input_dir / name / f"mutual_information_{name}.json" for name in model_names]
    if not paths:
        raise FileNotFoundError(f"No mutual_information_*.json files found under {input_dir}")

    results: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing mutual-information result: {path}")
        with open(path) as f:
            result = json.load(f)
        model_name = result.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"Result is missing model_name: {path}")
        for key in ("probe_layers", "steps", "metrics"):
            if key not in result:
                raise ValueError(f"Result is missing {key!r}: {path}")
        results[model_name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize mutual-information probe trajectories.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    for key in ("input_dir", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing required config key: {key}")
    results = _discover_results(Path(config["input_dir"]), config.get("models"))
    make_plots(
        results,
        Path(config["output_dir"]),
        jump_logits_dir=Path(config["jump_logits_dir"]) if config.get("jump_logits_dir") else None,
        colors=config.get("colors"),
        display_names=config.get("display_names"),
        figsize=tuple(config.get("figsize", [14, 5])),
        threshold_ratio=config.get("jump_threshold_ratio", PAPER_JUMP_THRESHOLD_RATIO),
        ignore_initial_n=config.get("jump_ignore_initial_n", PAPER_JUMP_IGNORE_INITIAL_N),
    )


if __name__ == "__main__":
    main()
