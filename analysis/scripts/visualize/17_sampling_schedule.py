"""
17_sampling_schedule.py

Paper Fig S1: bucket sampling probability under delayed-sigmoid interpolation, for a grid
of delay factors and steepness values. Computed analytically; no inputs.

For bucket b of B buckets (0-based), training progress t in [0, 1], delay d and steepness k:
  offset = d * b / (B - 1)
  y(t) = 1 / (1 + exp(-k (t - offset))), y_start = y(0), y_end = y(1)
  t_delayed = clip((y(t) - y_start) / (y_end - y_start), 0, 1)
  p_b(t) = (1 - t_delayed) * start_b + t_delayed * end_b
This is the interpolation of DynamicBucketDistributedSampler (sampling_interpolation_type
"sigmoid"). The paper figure shows p_b(t) before normalization; the sampler additionally
divides by sum_b p_b(t) at every epoch (set normalize: true to plot that).

Output: {output_dir}/sampling_schedule.{png,svg}

Config YAML (paper values as defaults):
  output_dir
  start_probs: [0.4, 0.4, 0.1, 0.05, 0.05]
  end_probs: [0.2, 0.2, 0.2, 0.2, 0.2]
  delay_factors: [0.5, 0.8, 1.0]
  steepness_values: [5.0, 12.0, 15.0]
  resolution: 100
  normalize: false
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def delayed_sigmoid_probabilities(
    start_probs: list[float],
    end_probs: list[float],
    delay_factor: float,
    k: float,
    t_values: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """Bucket probabilities [len(t_values), n_buckets] along training progress t."""
    if len(start_probs) != len(end_probs) or not start_probs:
        raise ValueError("start_probs and end_probs must be non-empty and of equal length")
    num_buckets = len(start_probs)
    start = np.asarray(start_probs, dtype=float)
    end = np.asarray(end_probs, dtype=float)
    probabilities = np.empty((len(t_values), num_buckets))
    for row, t in enumerate(t_values):
        t_delayed = np.empty(num_buckets)
        for b in range(num_buckets):
            offset = delay_factor * (b / (num_buckets - 1)) if num_buckets > 1 else 0
            y = 1 / (1 + np.exp(-k * (t - offset)))
            y_start = 1 / (1 + np.exp(k * offset))
            y_end = 1 / (1 + np.exp(-k * (1 - offset)))
            if abs(y_end - y_start) < 1e-9:
                t_delayed[b] = 1.0 if t > offset else 0.0
            else:
                t_delayed[b] = (y - y_start) / (y_end - y_start)
        t_delayed = np.clip(t_delayed, 0, 1)
        probabilities[row] = (1 - t_delayed) * start + t_delayed * end
        if normalize:
            probabilities[row] /= probabilities[row].sum()
    return probabilities


def main():
    parser = argparse.ArgumentParser(description="Plot the delayed-sigmoid bucket sampling schedule.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_probs = cfg.get("start_probs", [0.4, 0.4, 0.1, 0.05, 0.05])
    end_probs = cfg.get("end_probs", [0.2, 0.2, 0.2, 0.2, 0.2])
    delay_factors = cfg.get("delay_factors", [0.5, 0.8, 1.0])
    steepness_values = cfg.get("steepness_values", [5.0, 12.0, 15.0])
    normalize = cfg.get("normalize", False)
    t_values = np.linspace(0, 1, cfg.get("resolution", 100))

    fig, axes = plt.subplots(
        len(delay_factors),
        len(steepness_values),
        figsize=(4 * len(steepness_values), 3 * len(delay_factors)),
        squeeze=False,
    )
    for i, delay_factor in enumerate(delay_factors):
        for j, k in enumerate(steepness_values):
            ax = axes[i][j]
            probabilities = delayed_sigmoid_probabilities(start_probs, end_probs, delay_factor, k, t_values, normalize)
            for b in range(len(start_probs)):
                ax.plot(t_values, probabilities[:, b], label=f"bucket {b + 1}")
            ax.set_title(f"delay={delay_factor}, k={k}")
            ax.set_xlabel("t")
            ax.set_ylabel("probability")
            ax.grid(True)
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"sampling_schedule.{ext}", dpi=300)
    plt.close(fig)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
