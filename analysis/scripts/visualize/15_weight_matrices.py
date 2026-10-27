"""
15_weight_matrices.py  (supplementary; not used in the paper)

Visualizes weight matrix evolution from 08_weight_matrix.py output JSON.
  - L2 norm trajectory per weight matrix (at each checkpoint step)
  - L2 diff (consecutive checkpoint Frobenius distance)
  - CKA / cosine similarity between consecutive checkpoints
    (dynamic metrics are plotted at the later step of each pair)

Config YAML:
  input_dir: directory containing weight_matrix_{model_name}.json files
  output_dir:
  models: list (optional)
  jump_logits_dir: (optional) dir with logits_metrics_trajectory.csv per model (analysis 03)
  jump_threshold_ratio (0.15), jump_ignore_initial_n (10)
  figsize: [w, h] (default [12, 5])
"""

from __future__ import annotations

import argparse
import json
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
    plot_trajectory_with_jumpup,
)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save(fig, path_stem: Path, dpi: int = 300) -> None:
    """Save a figure as both PNG and SVG, then close it."""
    for extension in ("png", "svg"):
        fig.savefig(path_stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize weight matrix evolution.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "models" in cfg:
        model_names = cfg["models"]
    else:
        model_names = [
            p.stem.removeprefix("weight_matrix_")
            for p in sorted(input_dir.glob("weight_matrix_*.json"))
        ]

    figsize = tuple(cfg.get("figsize", [12, 5]))
    jump_logits_dir = Path(cfg["jump_logits_dir"]) if "jump_logits_dir" in cfg else None

    for model_name in model_names:
        json_path = input_dir / f"weight_matrix_{model_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing analysis 08 output: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        steps = data["steps"]
        wm = data["weight_matrices"]
        plot_dir = output_dir / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        jump = None
        if jump_logits_dir is not None:
            jump = load_jump_interval(
                jump_logits_dir,
                model_name,
                cfg.get("jump_threshold_ratio", PAPER_JUMP_THRESHOLD_RATIO),
                cfg.get("jump_ignore_initial_n", PAPER_JUMP_IGNORE_INITIAL_N),
            )

        # Group keys by short prefix for cleaner plots
        # e.g., "encoder.h.0.attn.c_proj.weight" → group "encoder.h.*.attn.c_proj.weight"
        def _group_key(k: str) -> str:
            parts = k.split(".")
            new_parts = ["*" if p.isdigit() else p for p in parts]
            return ".".join(new_parts)

        groups: dict[str, list[str]] = {}
        for key in wm.keys():
            gk = _group_key(key)
            groups.setdefault(gk, []).append(key)

        for group_name, keys in groups.items():
            safe_name = group_name.replace(".", "_").replace("*", "IDX")

            for metric in ["l2_norm", "l2_diff", "cka", "cosine"]:
                series_dict = {}
                for key in keys:
                    vals = wm[key].get(metric)
                    key_steps = wm[key].get("steps", steps)
                    if vals:
                        # l2_norm has one value per step; dynamic metrics one per consecutive pair,
                        # placed at the later step of the pair.
                        index = key_steps if metric == "l2_norm" else key_steps[1:]
                        if len(vals) != len(index):
                            raise ValueError(f"{key} {metric}: {len(vals)} values for {len(index)} steps")
                        short_key = key.replace("encoder.", "enc.").replace("decoder.", "dec.")
                        series_dict[short_key] = pd.Series(vals, index=index)

                if not series_dict:
                    continue

                df = pd.DataFrame(series_dict)
                fig, ax = plot_trajectory_with_jumpup(
                    df=df,
                    jump_interval=jump,
                    title=f"{model_name} — {group_name} — {metric}",
                    ylabel=metric,
                    figsize=figsize,
                )
                _save(fig, plot_dir / f"weight_{safe_name}_{metric}")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
