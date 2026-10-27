"""Tiny PanCoreTransformerModel and synthetic batch shared by the model-based analysis tests."""

from __future__ import annotations

import numpy as np
import torch

from pan_core.config import ModelConfig

VOCAB    = 20
B        = 4   # batch size
L        = 8   # sequence length (encoder input)
E        = 16  # embedding dim
N_LAYERS = 2
N_HEADS  = 2
ATMARK   = 7   # chiral token id (@); 8 is @@
ATATMARK = 8
BOS, EOS, PAD = 1, 2, 0


def make_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        embedding_dim=E,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_positions=32,
        tanh_latent=True,
        latent_noising=False,
        dropout=0.0,
        bias=True,
        weight_sharing=False,
        layer_norm_epsilon=1e-5,
        pooling_mode="MHA",
        cross_attn=None,
        latent_conditioning="add_once",
    )


def make_batch():
    """Return (input_ids, enc_mask, labels, dec_mask) with a chiral token at pos 2.

    Surrounding tokens vary across samples so that after the first transformer
    block the hidden representation at the chiral position is non-degenerate
    (i.e., not identical for all samples).  This is needed for CKA to be well-
    defined (non-zero denominator) at all layers except the embedding layer,
    where token 7 always maps to the same embedding vector.
    """
    ids = torch.zeros((B, L), dtype=torch.long)
    ids[:, 0] = BOS
    ids[:, 4] = EOS
    ids[:, 2] = ATMARK   # chiral @ token, same for all samples

    # Vary surrounding tokens per sample so hidden states are non-degenerate
    surrounding = torch.tensor([[3, 5], [4, 9], [5, 11], [6, 13]])  # [B, 2] (pos1, pos3)
    ids[:, 1] = surrounding[:, 0]
    ids[:, 3] = surrounding[:, 1]
    # positions 5,6,7 remain PAD=0

    enc_mask = (ids != PAD).long()   # [B, L]
    labels   = ids.clone()
    dec_mask = enc_mask.clone()
    return ids, enc_mask, labels, dec_mask


def masks(token_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return [B, L] numpy bool (valid, chiral) masks from a [B, L] token id tensor."""
    ids_np = token_ids.numpy()
    valid  = ids_np != PAD
    chiral = (ids_np == ATMARK) | (ids_np == ATATMARK)
    return valid, chiral
