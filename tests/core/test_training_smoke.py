"""
Smoke test for the pan-core pre-training pipeline.

Steps:
  1. Generate toy SMILES CSVs.
  2. Run pre-training (4 optimizer steps).
  3. Verify checkpoint and final_model outputs exist.
  4. Delete all temporary files once every assertion passes.

On failure the files under tmp_path are left in place, so the path printed by
pytest can be inspected to debug the contents.

Usage:
  pytest tests/core/test_training_smoke.py -v -s
"""

import shutil
from pathlib import Path

import pandas as pd
import yaml

from pan_core.main import run_training
from tests.utils import VOCAB_FILE

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# 20 simple SMILES that tokenize cleanly with the standard vocab
_ALL_SMILES = [
    "C", "CC", "CCC", "CCCC", "CCCCC",
    "CCO", "CCCO", "CO", "OCC", "OC",
    "CN", "CCN", "CCCN", "NC", "NCC",
    "CC(=O)O", "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "O",
]
TRAIN_SMILES = _ALL_SMILES[:16]   # 16 molecules for training
VALID_SMILES = _ALL_SMILES[16:]   # 4 molecules for validation


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_pretrain_csv(path: Path, smiles: list[str]) -> None:
    """random/canonical pair CSV for on_the_fly pre-training."""
    pd.DataFrame({"random": smiles, "canonical": smiles}).to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def _base_config(vocab: str, train_csv: str, valid_csv: str, max_steps: int = 4) -> dict:
    """Minimal YAML config dict for pre-training."""
    return {
        "system": {
            "environment": "normal",
            "seed": 42,
            "num_workers": 0,
            "use_bf16": False,
        },
        "logging": {
            "wandb": {"enabled": False},
            "log_interval_steps": 1,
        },
        "model": {
            "use_vae": False,
            "common": {                 # vocab_size comes from the tokenizer
                "dropout": 0.0,
                "embedding_dim": 64,
                "tanh_latent": True,
                "latent_noising": False,
            },
            "loss": {"beta": 0.0},
            "Transformer": {
                "n_positions": 64,
                "n_layer": 1,
                "n_head": 2,        # must divide embedding_dim (64 / 2 = 32 per head)
                "bias": True,
                "weight_sharing": False,
                "layer_norm_epsilon": 1e-5,
                "gpt2style": False,
                "rope": {"encoder": True, "decoder": True, "base_freq": 10000},
                "latent": {
                    "pooling_mode": "MHA",
                    "cross_attn": "L-T",
                    "latent_conditioning": "add_once",
                },
            },
        },
        "training": {
            "save_only_model": False,
            "strategy": {
                "type": "normal",
                "num_epochs": 10000,    # large; max_steps will terminate first
                "normal": {
                    "max_steps": max_steps,
                    "early_stopping": {"enabled": False},
                },
            },
            "optimizer": {
                "type": "schedulefree",
                "max_lr": 1e-3,
                "weight_decay": 0.0,
                "schedulefree": {"r": 0.5},
            },
            "zclip": {"enabled": False},
            "checkpoint": {"step": 2, "limit": 2},
            "validation": {"enabled": True, "interval_steps": 2, "batch_size": 4},
            "data": {
                "mode": "on_the_fly",
                "csv_column_name": {"source": "random", "target": "canonical"},
                "batch_size": 4,
                "accumulate_steps": 1,
                "token_file_path": vocab,
                "train_data": [{"path": train_csv}],
                "valid_data":  {"path": valid_csv},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _find_model_weights(directory: Path) -> Path | None:
    """Return path to saved model weights (safetensors or pytorch_model.bin)."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = directory / name
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_pretrain(tmp_path):
    """
    Full pipeline smoke test.

    Pre-training (4 steps) → verify outputs.
    Deletes all temporary files only if every assertion passes.
    On failure, files remain in tmp_path for inspection.
    """
    # ── Sanity check ──────────────────────────────────────────────────────────
    assert VOCAB_FILE.exists(), f"Vocab file not found: {VOCAB_FILE}"

    vocab = str(VOCAB_FILE)
    work = tmp_path / "work"
    work.mkdir()

    pretrain_data = work / "pretrain_data"
    pretrain_data.mkdir()
    pretrain_out = work / "pretrain_out"

    train_csv = str(pretrain_data / "train.csv")
    valid_csv = str(pretrain_data / "valid.csv")
    _write_pretrain_csv(Path(train_csv), TRAIN_SMILES)
    _write_pretrain_csv(Path(valid_csv), VALID_SMILES)

    cfg_path = work / "pretrain_config.yml"
    with open(cfg_path, "w") as f:
        yaml.dump(_base_config(vocab, train_csv, valid_csv), f)

    print(f"\n[pretrain] output dir: {pretrain_out}")
    run_training(config_path=str(cfg_path), output_dir=str(pretrain_out))

    # ── Assertions ────────────────────────────────────────────────────────────
    final_model = pretrain_out / "final_model"
    assert final_model.is_dir(), (
        f"final_model/ directory was not created under {pretrain_out}"
    )
    assert (final_model / "config.json").exists(), (
        "config.json not found in final_model/"
    )
    weights = _find_model_weights(final_model)
    assert weights is not None, (
        f"Neither model.safetensors nor pytorch_model.bin found in {final_model}"
    )
    assert (final_model / "normal_tokens.txt").exists(), (
        "Vocabulary file (normal_tokens.txt) not saved alongside final_model"
    )

    checkpoints = sorted(pretrain_out.glob("checkpoint-*"))
    assert len(checkpoints) >= 1, (
        f"No checkpoint-* directories found under {pretrain_out}. "
        "Expected checkpoint-2 at minimum."
    )
    print(f"[pretrain] checkpoints found: {[c.name for c in checkpoints]}")
    print(f"[pretrain] model weights: {weights.name}")

    shutil.rmtree(work)
    print("\n[cleanup] all temp files deleted (tests passed)")
