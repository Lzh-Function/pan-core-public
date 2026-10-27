"""A tiny training run on disk (checkpoint series + evaluation CSV) for end-to-end script tests.

The analysis scripts read ``checkpoint-N/`` directories, a vocabulary file and an evaluation
CSV made of equal consecutive length buckets. These helpers write the smallest such run:
randomly initialised two-layer models (the analyses do not need trained weights) and a
5-bucket CSV whose every bucket holds chiral and non-chiral molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from pan_core.config import ModelConfig
from pan_core.models import PanCoreTransformerModel
from tests.utils import VOCAB_FILE

MODEL_NAME = "tiny"
STEPS = (10, 20, 30, 40, 50, 60)
N_BUCKETS = 5
MAX_LEN = 48   # padded encoder length and generation length used in the test configs

_CHIRAL = [
    "C[C@H](N)C(=O)O", "C[C@@H](O)CC", "N[C@@H](CO)C(=O)O", "C[C@H](Cl)CC", "O[C@H](C)c1ccccc1",
    "C[C@@H](N)CC", "CC[C@H](C)O", "N[C@H](C)CO", "C[C@@H](F)CO", "O[C@@H](CC)C",
]
_NON_CHIRAL = [
    "CCO", "CC(=O)O", "c1ccccc1O", "CCN", "CC(C)O", "OCCO", "c1ccncc1", "CCOC", "NCCO", "CC#N",
]


@dataclass(frozen=True)
class TinyRun:
    checkpoint_dir: Path
    eval_csv: Path
    vocab: Path
    model_name: str = MODEL_NAME
    steps: tuple[int, ...] = STEPS


def tiny_model_config(**overrides) -> ModelConfig:
    vocab_size = sum(1 for line in VOCAB_FILE.read_text().splitlines() if line)
    settings = dict(
        vocab_size=vocab_size, embedding_dim=16, n_layer=2, n_head=2, n_positions=64, tanh_latent=False,
        latent_noising=False, dropout=0.0, bias=True, weight_sharing=False, layer_norm_epsilon=1e-5,
        pooling_mode="MHA", cross_attn=None, latent_conditioning="add_once",
    )
    settings.update(overrides)
    return ModelConfig(**settings)


def write_checkpoint_series(directory: Path, steps=STEPS, **config_overrides) -> Path:
    """Save one randomly initialised model per step as ``directory/checkpoint-{step}``."""
    for index, step in enumerate(steps):
        torch.manual_seed(index)
        PanCoreTransformerModel(tiny_model_config(**config_overrides)).save_pretrained(directory / f"checkpoint-{step}")
    return directory


def write_eval_csv(path: Path, rows_per_bucket: int = 4) -> Path:
    """``random`` / ``canonical`` pairs; each bucket alternates chiral and non-chiral molecules."""
    smiles = [pair[i % 2] for pair in zip(_CHIRAL, _NON_CHIRAL) for i in range(2)]
    smiles = smiles[: N_BUCKETS * rows_per_bucket]
    if len(smiles) != N_BUCKETS * rows_per_bucket:
        raise ValueError(f"At most {len(_CHIRAL) * 2 // N_BUCKETS} rows per bucket are available")
    rows = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        rows.append({
            "random": Chem.MolToRandomSmilesVect(mol, 1, randomSeed=index + 1)[0],
            "canonical": Chem.MolToSmiles(mol),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_jump_logits(directory: Path, model_name: str = MODEL_NAME, steps=STEPS) -> Path:
    """Analysis-03 ``logits_metrics_trajectory.csv`` whose perplexity drops in the middle of ``steps``."""
    half = len(steps) // 2
    perplexity = [8.0] * half + [1.0] * (len(steps) - half)
    model_dir = directory / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"perplexity": perplexity, "chiral_dilemma_rate": [0.9] * len(steps)},
                 index=pd.Index(steps, name="step")).to_csv(model_dir / "logits_metrics_trajectory.csv")
    return directory
