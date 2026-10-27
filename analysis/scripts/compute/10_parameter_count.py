"""
10_parameter_count.py

Paper Table S3: number of trainable parameters of each model.

Builds every model from its checkpoint ``config.json`` (no weights are loaded) and counts
the elements of all parameters with ``requires_grad``; registered buffers (e.g. positional
tables, RoPE frequencies) are not parameters and are not counted.

Output: {output_dir}/parameter_count.csv  (index: model; columns: trainable_parameters, millions)

Config YAML:
  output_dir
  models: {label: /path/to/checkpoint-N or /path/to/config.json, ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from pan_core.models import PanCoreTransformerModel


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_trainable_parameters(config_path: Path) -> int:
    """Trainable parameter count of a model built from a checkpoint config.json."""
    config_class = PanCoreTransformerModel.config_class
    config = config_class.from_json_file(str(config_path))
    model = PanCoreTransformerModel(config)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Count trainable model parameters (Table S3).")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("models must map labels to checkpoint directories or config.json files")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, location in models.items():
        path = Path(location)
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found for {label}: {config_path}")
        count = count_trainable_parameters(config_path)
        rows.append({"model": label, "trainable_parameters": count, "millions": round(count / 1e6, 3)})
        print(f"{label}: {count:,} trainable parameters")

    out_csv = output_dir / "parameter_count.csv"
    pd.DataFrame(rows).set_index("model").to_csv(out_csv)
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
