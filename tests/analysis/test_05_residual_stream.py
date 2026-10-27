"""Tests for analysis 05 (residual stream): hidden states → compute_layerwise_metrics."""

from __future__ import annotations

import numpy as np

from tests.analysis.tiny_model import B, E, L, N_LAYERS, masks
from tests.utils import load_script

_RESIDUAL = load_script("analysis/scripts/compute/05_residual_stream.py")


class TestResidualStreamPipeline:
    """Verify the 05_residual_stream.py pipeline: hidden states → compute_layerwise_metrics."""

    def _prep_encoder(self, pancore_output):
        """Transpose encoder hidden states [L,B,E] → [B,L,E], exclude nothing."""
        out, input_ids, enc_mask, labels, dec_mask = pancore_output
        hs = [h.transpose(1, 0, 2) for h in out.hidden_states["encoder"]]  # [B,L,E]
        valid, chiral = masks(input_ids)    # [B, L_enc]
        return hs, valid, chiral

    def _prep_decoder(self, pancore_output):
        """Transpose decoder hidden states (excluding logits) [L,B,E] → [B,L,E]."""
        out, input_ids, enc_mask, labels, dec_mask = pancore_output
        dec_hs = out.hidden_states["decoder"][:-1]   # drop logits
        hs = [h.transpose(1, 0, 2) for h in dec_hs]  # [B,L-1,E]
        # decoder input = labels[:, :-1]
        dec_input_ids = labels[:, :-1]               # [B, L-1]
        valid, chiral = masks(dec_input_ids)
        return hs, valid, chiral

    def test_encoder_hidden_transposed_shape(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        for i, h in enumerate(hs):
            assert h.shape == (B, L, E), f"layer {i}: {h.shape}"

    def test_decoder_hidden_transposed_shape(self, pancore_output):
        hs, valid, chiral = self._prep_decoder(pancore_output)
        L_dec = L - 1
        for i, h in enumerate(hs):
            assert h.shape == (B, L_dec, E), f"layer {i}: {h.shape}"

    def test_layerwise_metrics_output_keys(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        expected_keys = {
            "l2_norm_chiral", "l2_norm_background",
            "cos_chiral", "cos_background",
            "cka_chiral", "cka_background",
        }
        assert set(result.keys()) == expected_keys

    def test_l2_norm_array_length(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        # l2_norm has one value per layer (including embedding)
        assert len(result["l2_norm_chiral"]) == N_LAYERS + 1

    def test_cosine_cka_array_length(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        # layer-to-layer transitions: N_LAYERS+1 layers → N_LAYERS transitions
        assert len(result["cos_chiral"]) == N_LAYERS
        assert len(result["cka_chiral"]) == N_LAYERS

    def test_l2_norm_positive(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        assert (result["l2_norm_chiral"] >= 0).all()
        assert (result["l2_norm_background"] >= 0).all()

    def test_cosine_in_range(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        assert (np.abs(result["cos_chiral"]) <= 1.0 + 1e-5).all()

    def test_cka_in_range(self, pancore_output):
        hs, valid, chiral = self._prep_encoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        assert (result["cka_chiral"] >= 0.0).all()
        assert (result["cka_chiral"] <= 1.0 + 1e-5).all()

    def test_decoder_layerwise_metrics_run(self, pancore_output):
        hs, valid, chiral = self._prep_decoder(pancore_output)
        result = _RESIDUAL.compute_layerwise_metrics(hs, valid, chiral)
        assert "l2_norm_chiral" in result
        assert len(result["l2_norm_chiral"]) == N_LAYERS + 1
