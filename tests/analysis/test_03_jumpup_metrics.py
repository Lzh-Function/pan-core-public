"""Tests for analysis 03 (jump-up metrics): the memory-efficient forward pass and transition metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pan_core.models import PanCoreTransformerModel
from tests.analysis.tiny_model import ATATMARK, ATMARK, B, L, N_HEADS, N_LAYERS, make_batch, make_config, masks
from tests.utils import load_script

_JUMPUP = load_script("analysis/scripts/compute/03_jumpup_metrics.py")


class _LookupTokenizer:
    """Return pre-tokenized IDs for the synthetic integration samples."""

    def __init__(self, encoded: dict[str, list[int]]):
        self.encoded = encoded

    def encode(self, text: str) -> list[int]:
        return self.encoded[text]


@pytest.fixture(scope="module")
def jumpup_output():
    """Run the memory-efficient Jumpup pipeline on one synthetic batch."""
    torch.manual_seed(0)
    model = PanCoreTransformerModel(make_config())
    model.eval()

    input_ids, enc_mask, labels, dec_mask = make_batch()
    vm_enc, cm_enc = masks(input_ids)
    # Decoder positions include the EOS-input position (analysis 03 pads the target by one).
    vm_dec, cm_dec = masks(labels)
    sample_names = [f"sample_{i}" for i in range(B)]
    tokenizer = _LookupTokenizer({
        name: labels[i].tolist() for i, name in enumerate(sample_names)
    })
    batch = {
        "input_ids": input_ids,
        "encoder_attention_mask": enc_mask,
        "labels": labels,
        "decoder_attention_mask": dec_mask,
    }

    return _JUMPUP.forward_pass(
        model=model,
        dataloader=[batch],
        device=torch.device("cpu"),
        vm_enc=vm_enc,
        cm_enc=cm_enc,
        vm_dec=vm_dec,
        cm_dec=cm_dec,
        dist_enc_arr=None,
        dist_dec_arr=None,
        N=B,
        batch_size=B,
        smiles_tgt=sample_names,
        tokenizer=tokenizer,
        max_len=L,
        chiral_ids=(ATMARK, ATATMARK),
    )


@pytest.fixture(scope="module")
def jumpup_transition(jumpup_output):
    """Compare a compact Jumpup output with itself."""
    return _JUMPUP.compute_transition_metrics(jumpup_output, jumpup_output)


class TestJumpupMetricsPipeline:
    """Verify the memory-efficient 03_jumpup_metrics.py pipeline."""

    def test_encoder_attn_stack_shape(self, pancore_output):
        out, *_ = pancore_output
        stack = np.stack(out.attentions["encoder"], axis=0)  # [num_layers, B, H, L, L]
        assert stack.shape == (N_LAYERS, B, N_HEADS, L, L)

    def test_static_metrics_output_keys(self, jumpup_output):
        result = jumpup_output["static"]["enc"]
        expected_keys = {
            "attention_mass", "attention_entropy", "sar",
            "graph_dist_from", "seq_dist_from",
            "graph_dist_to", "seq_dist_to", "nit",
        }
        assert set(result.keys()) == expected_keys

    def test_static_metrics_shape(self, jumpup_output):
        result = jumpup_output["static"]["enc"]
        # Each metric: [num_layers, num_heads]
        for key, arr in result.items():
            assert arr.shape == (N_LAYERS, N_HEADS), f"{key}: {arr.shape}"

    def test_attention_mass_non_negative(self, jumpup_output):
        result = jumpup_output["static"]["enc"]
        assert (result["attention_mass"] >= 0).all()

    def test_attention_entropy_non_negative(self, jumpup_output):
        result = jumpup_output["static"]["enc"]
        assert (result["attention_entropy"] >= 0).all()

    def test_transition_metrics_output_keys(self, jumpup_transition):
        """The compact transition result should expose every metric."""
        result = jumpup_transition["enc"]
        expected_keys = {"jsd", "cka_chiral", "cka_background", "cos_chiral", "cos_background"}
        assert set(result.keys()) == expected_keys

    def test_transition_jsd_shape(self, jumpup_transition):
        result = jumpup_transition["enc"]
        assert result["jsd"].shape == (N_LAYERS, N_HEADS)

    def test_transition_cka_shape(self, jumpup_transition):
        result = jumpup_transition["enc"]
        # encoder: N_LAYERS+1 hidden layers → cka has N_LAYERS+1 values
        assert len(result["cka_chiral"]) == N_LAYERS + 1

    def test_identical_checkpoints_give_cka_one(self, jumpup_transition):
        """When curr and prev are identical, CKA should be 1.0 for non-degenerate layers.

        The embedding layer (index 0) is degenerate for the chiral position because
        all samples share the same ATMARK token → identical embedding vectors →
        zero-variance matrix → CKA denominator = 0 → CKA = 0.  Post-embedding
        layers process different surrounding tokens through attention and produce
        distinct representations across samples → CKA = 1.0.

        The background CKA is 1.0 at all layers because background positions include
        tokens that vary across samples (positions 1 and 3).
        """
        result = jumpup_transition["enc"]
        assert result["cka_chiral"][0] == 0.0, "embedding CKA should be 0 (degenerate)"
        assert np.allclose(result["cka_chiral"][1:], 1.0, atol=1e-5)
        assert np.allclose(result["cka_background"], 1.0, atol=1e-5)

    def test_identical_checkpoints_give_cosine_one(self, jumpup_transition):
        result = jumpup_transition["enc"]
        assert np.allclose(result["cos_chiral"], 1.0, atol=1e-5)
        assert np.allclose(result["cos_background"], 1.0, atol=1e-5)

    def test_identical_checkpoints_give_jsd_zero(self, jumpup_transition):
        result = jumpup_transition["enc"]
        assert np.allclose(result["jsd"], 0.0, atol=1e-5)

    def test_decoder_transition_metrics_shape(self, jumpup_transition):
        """Decoder transition metrics must exclude the final logits layer."""
        result = jumpup_transition["dec"]
        assert len(result["cka_chiral"]) == N_LAYERS + 1
