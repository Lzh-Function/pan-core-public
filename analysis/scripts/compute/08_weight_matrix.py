"""
08_weight_matrix.py

Analyzes weight matrix evolution across checkpoints.

For each pair of consecutive checkpoints, computes:
  - L2 norm of each key weight matrix
  - L2 diff (||W_curr - W_prev||_F) between consecutive checkpoints
  - Linear CKA between consecutive checkpoints
  - Cosine similarity between flattened weight vectors

Output: weight_matrix_{model_name}.json

Config YAML:
  model_name: arbitrary label used in the output filename
  checkpoint_dir, checkpoint_steps (optional list), output_dir,
  seed (42)

Important weight matrix keys are derived from each checkpoint's config.json and
validated against its state dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis import metrics as M


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ARCHITECTURE_FIELDS = (
    "architectures",
    "embedding_dim",
    "n_layer",
    "n_head",
    "pooling_mode",
    "gpt2style",
    "cross_attn",
    "latent_conditioning",
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint_config(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    with open(config_path) as file:
        config = json.load(file)
    missing = [field for field in ARCHITECTURE_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Checkpoint config is missing architecture fields {missing}: {config_path}")
    if not isinstance(config["n_layer"], int) or config["n_layer"] < 1:
        raise ValueError(f"n_layer must be a positive integer: {config_path}")
    return config


def architecture_signature(config: dict) -> dict:
    return {field: config[field] for field in ARCHITECTURE_FIELDS}


def derive_important_keys(config: dict, available_keys: set[str]) -> list[str]:
    """Derive the original semantic weight subset from checkpoint structure."""
    n_layer = config["n_layer"]
    pooling_mode = str(config["pooling_mode"]).lower()
    if pooling_mode == "concat":
        keys = ["encoder.pooling.proj.weight"]
    elif pooling_mode == "mha":
        keys = ["encoder.pooling.queries"]
    else:
        raise ValueError(f"Unsupported pooling_mode for weight analysis: {config['pooling_mode']!r}")

    gpt2style = config["gpt2style"]
    if not isinstance(gpt2style, bool):
        raise ValueError("gpt2style must be boolean")
    mlp_weights = ("c_proj",) if gpt2style else ("v", "w1", "w2")
    for layer in range(n_layer):
        for side in ("encoder", "decoder"):
            keys.append(f"{side}.h.{layer}.attn.c_proj.weight")
            keys.extend(f"{side}.h.{layer}.mlp.{name}.weight" for name in mlp_weights)
        if config["cross_attn"] is not None:
            keys.extend(
                f"decoder.h.{layer}.cross_attn.{name}.weight"
                for name in ("q_proj", "kv_proj", "c_proj")
            )

    if str(config["latent_conditioning"]).lower() in {"adaln", "adaln-zero"}:
        keys.append("decoder.ada_ln_projection.weight")
    keys.extend(("decoder.output_fc.weight", "encoder.wte.weight", "decoder.wte.weight"))

    missing = [key for key in keys if key not in available_keys]
    if missing:
        raise ValueError(f"Derived weight keys are missing from checkpoint state dict: {missing}")
    return keys


def main():
    parser = argparse.ArgumentParser(description="Weight matrix trajectory analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty output label")

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "08_weight_matrix.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    target_steps: list[int] | None = cfg.get("checkpoint_steps")
    all_ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if target_steps is not None:
        tset = set(target_steps)
        all_ckpts = [p for p in all_ckpts if int(p.name.split("-")[1]) in tset]
    logger.info(f"Processing {len(all_ckpts)} checkpoints")
    if not all_ckpts:
        raise FileNotFoundError(f"No selected checkpoint-* directories found in {checkpoint_dir}")

    reference_config = load_checkpoint_config(all_ckpts[0])
    reference_signature = architecture_signature(reference_config)
    first_state_dict = load_file(all_ckpts[0] / "model.safetensors")
    keys = derive_important_keys(reference_config, set(first_state_dict))
    logger.info(f"Derived {len(keys)} weight keys from checkpoint architecture")

    # Storage: key → {step: value}
    history: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    steps_list: list[int] = []
    prev_weights: dict[str, np.ndarray] = {}

    for ckpt in all_ckpts:
        step = int(ckpt.name.split("-")[1])
        logger.info(f"→ {ckpt.name} (step {step})")

        safetensors_path = ckpt / "model.safetensors"
        if not safetensors_path.exists():
            logger.warning(f"No model.safetensors in {ckpt}, skipping.")
            continue

        state_dict = load_file(safetensors_path)
        current_signature = architecture_signature(load_checkpoint_config(ckpt))
        if current_signature != reference_signature:
            raise ValueError(
                f"Checkpoint architecture changed at {ckpt}: "
                f"expected {reference_signature}, got {current_signature}"
            )
        steps_list.append(step)

        for key in keys:
            if key not in state_dict:
                raise ValueError(f"Derived weight key {key!r} is missing from {ckpt}")
            W = state_dict[key].float().numpy()  # [out, in] or [...]
            W_flat = W.reshape(-1)

            # Static: L2 norm
            history[key]["steps"].append(step)
            history[key]["l2_norm"].append(float(np.linalg.norm(W_flat)))

            # Dynamic vs previous checkpoint
            if key in prev_weights:
                W_prev_flat = prev_weights[key]
                diff = W_flat - W_prev_flat

                # L2 diff (Frobenius)
                history[key]["l2_diff"].append(float(np.linalg.norm(diff)))

                # CKA: treat as 2-D matrices [out, in]
                if W.ndim >= 2:
                    W2d = W.reshape(W.shape[0], -1)
                    Wp2d = prev_weights[key + "_2d"]
                    cka = M.compute_linear_cka(Wp2d, W2d)
                else:
                    cka = M.compute_linear_cka(
                        W_prev_flat.reshape(-1, 1),
                        W_flat.reshape(-1, 1),
                    )
                history[key]["cka"].append(float(cka))

                # Cosine similarity between flattened vectors
                cos = float(np.dot(W_flat, W_prev_flat) / (
                    np.linalg.norm(W_flat) * np.linalg.norm(W_prev_flat) + 1e-10
                ))
                history[key]["cosine"].append(float(cos))

            prev_weights[key] = W_flat
            if W.ndim >= 2:
                prev_weights[key + "_2d"] = W.reshape(W.shape[0], -1)

    result = {
        "model_name": model_name,
        "architecture": reference_signature,
        "steps": steps_list,
        "weight_matrices": {k: dict(v) for k, v in history.items()},
    }

    out_file = output_dir / f"weight_matrix_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
