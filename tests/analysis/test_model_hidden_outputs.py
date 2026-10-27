"""Structure of model.forward(need_hidden=True) outputs consumed by the analysis scripts."""

from __future__ import annotations

import numpy as np

from pan_core_analysis.metrics import compute_logits_metrics
from tests.analysis.tiny_model import ATATMARK, ATMARK, B, E, L, N_HEADS, N_LAYERS, VOCAB


class TestModelForwardHiddenStateStructure:
    """Verify that model.forward(need_hidden=True) produces correctly structured outputs."""

    def test_hidden_states_is_dict(self, pancore_output):
        out, *_ = pancore_output
        assert isinstance(out.hidden_states, dict)
        assert set(out.hidden_states.keys()) == {"encoder", "decoder"}

    def test_attentions_is_dict(self, pancore_output):
        out, *_ = pancore_output
        assert isinstance(out.attentions, dict)
        assert set(out.attentions.keys()) == {"encoder", "decoder"}

    def test_encoder_hidden_count(self, pancore_output):
        out, *_ = pancore_output
        # embedding + n_layer blocks = N_LAYERS + 1
        assert len(out.hidden_states["encoder"]) == N_LAYERS + 1

    def test_decoder_hidden_count(self, pancore_output):
        out, *_ = pancore_output
        # embedding + n_layer blocks + logits = N_LAYERS + 2
        assert len(out.hidden_states["decoder"]) == N_LAYERS + 2

    def test_encoder_attn_count(self, pancore_output):
        out, *_ = pancore_output
        assert len(out.attentions["encoder"]) == N_LAYERS

    def test_decoder_attn_count(self, pancore_output):
        out, *_ = pancore_output
        assert len(out.attentions["decoder"]) == N_LAYERS

    def test_encoder_hidden_shapes(self, pancore_output):
        out, *_ = pancore_output
        for i, h in enumerate(out.hidden_states["encoder"]):
            assert isinstance(h, np.ndarray), f"layer {i}: expected ndarray"
            # shape: [L_enc, B, E]
            assert h.shape == (L, B, E), f"layer {i}: {h.shape}"

    def test_decoder_hidden_shapes_except_logits(self, pancore_output):
        out, *_ = pancore_output
        dec_hs = out.hidden_states["decoder"]
        L_dec = L - 1   # decoder input = labels[:-1]
        for i, h in enumerate(dec_hs[:-1]):
            assert h.shape == (L_dec, B, E), f"decoder layer {i}: {h.shape}"

    def test_decoder_logits_shape(self, pancore_output):
        out, *_ = pancore_output
        logits = out.hidden_states["decoder"][-1]
        L_dec = L - 1
        assert logits.shape == (L_dec, B, VOCAB)

    def test_encoder_attn_shapes(self, pancore_output):
        out, *_ = pancore_output
        for i, a in enumerate(out.attentions["encoder"]):
            assert isinstance(a, np.ndarray), f"attn layer {i}: expected ndarray"
            assert a.shape == (B, N_HEADS, L, L), f"attn layer {i}: {a.shape}"

    def test_decoder_attn_shapes(self, pancore_output):
        out, *_ = pancore_output
        L_dec = L - 1
        for i, a in enumerate(out.attentions["decoder"]):
            assert a.shape == (B, N_HEADS, L_dec, L_dec), f"dec attn layer {i}: {a.shape}"

    def test_attention_rows_sum_to_one(self, pancore_output):
        out, *_ = pancore_output
        a = out.attentions["encoder"][0]   # [B, H, L, L]
        row_sums = a.sum(axis=-1)          # [B, H, L]
        assert np.allclose(row_sums, 1.0, atol=1e-5), "attention rows must sum to 1"


class TestLogitsMetricsPipeline:
    """Verify extraction of logits from decoder hidden states and compute_logits_metrics."""

    @staticmethod
    def _logits_metrics(pancore_output):
        out, input_ids, enc_mask, labels, dec_mask = pancore_output
        # decoder all_hidden_states[-1] = logits [L-1, B, V] -> [B, L-1, V]
        logits_np = out.hidden_states["decoder"][-1].transpose(1, 0, 2)
        # Target = labels shifted by 1 (labels[:, 1:])
        tgt_ids = labels[:, 1:].numpy()          # [B, L-1]
        chiral_mask = (tgt_ids == ATMARK) | (tgt_ids == ATATMARK)
        return compute_logits_metrics(logits_np, tgt_ids, chiral_mask)

    def test_logits_metrics_from_decoder_output(self, pancore_output):
        result = self._logits_metrics(pancore_output)
        for key in ("perplexity", "chiral_dilemma_rate", "mean_rank", "mean_gap", "chiral_margin"):
            assert key in result

    def test_logits_perplexity_finite_positive(self, pancore_output):
        result = self._logits_metrics(pancore_output)
        assert np.isfinite(result["perplexity"]) and result["perplexity"] > 0

    def test_logits_cdr_in_range(self, pancore_output):
        result = self._logits_metrics(pancore_output)
        assert 0.0 <= result["chiral_dilemma_rate"] <= 1.0
