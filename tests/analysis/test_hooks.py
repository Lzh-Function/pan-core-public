"""Unit tests for pan_core_analysis.hooks (CPU-only, fake model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pan_core_analysis.hooks import HeadAblationHook, HiddenStateCaptureHook, LatentCaptureHook


class _FakeAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self._num_heads = num_heads

    def forward(self, x: torch.Tensor):
        out = self.c_proj(x)
        B, L, _ = x.shape
        attn = torch.ones(B, self._num_heads, L, L) / L
        return out, attn


class _FakeBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = _FakeAttn(embed_dim, num_heads)

    def forward(self, x: torch.Tensor):
        out, attn = self.attn(x)
        return (out, None, attn)


class _FakeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.wte = nn.Embedding(100, embed_dim)
        self.h = nn.ModuleList([_FakeBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.pooling = nn.Linear(embed_dim, embed_dim)

    def forward(self, ids: torch.Tensor):
        h = self.wte(ids)           # [B, L, E]
        for block in self.h:
            h, _, _ = block(h)
        pooled = self.pooling(h.mean(dim=1))
        return h, pooled


class _FakeDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.wte = nn.Embedding(100, embed_dim)
        self.h = nn.ModuleList([_FakeBlock(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, ids: torch.Tensor):
        h = self.wte(ids)
        for block in self.h:
            h, _, _ = block(h)
        return h


class _FakeModel(nn.Module):
    def __init__(self, embed_dim: int = 16, num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        self.encoder = _FakeEncoder(embed_dim, num_heads, num_layers)
        self.decoder = _FakeDecoder(embed_dim, num_heads, num_layers)

    def forward(self, enc_ids: torch.Tensor, dec_ids: torch.Tensor):
        enc_out, _ = self.encoder(enc_ids)
        dec_out = self.decoder(dec_ids)
        return enc_out, dec_out


@pytest.fixture
def fake_model():
    torch.manual_seed(0)
    m = _FakeModel(embed_dim=16, num_heads=2, num_layers=2)
    m.eval()
    return m


@pytest.fixture
def sample_ids():
    return torch.randint(0, 100, (3, 5)), torch.randint(0, 100, (3, 4))


class TestHiddenStateCaptureHook:
    def test_hidden_state_count(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        # 1 embedding + 2 blocks = 3 per side
        assert len(hook.hidden_states["encoder"]) == 3
        assert len(hook.hidden_states["decoder"]) == 3

    def test_attn_score_count(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        # 2 blocks per side
        assert len(hook.attn_scores["encoder"]) == 2
        assert len(hook.attn_scores["decoder"]) == 2

    def test_hidden_shape_batch_first(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        B, L_enc = enc_ids.shape
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        # Block output: [B, L, E]
        h = hook.hidden_states["encoder"][1]
        assert h.shape == (B, L_enc, 16)

    def test_attn_shape(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        B, L_enc = enc_ids.shape
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        a = hook.attn_scores["encoder"][0]
        assert a.shape == (B, 2, L_enc, L_enc)  # [B, H, L, L]

    def test_hooks_cleared_after_exit(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        assert hook._handles == []

    def test_hidden_states_are_numpy(self, fake_model, sample_ids):
        enc_ids, dec_ids = sample_ids
        with HiddenStateCaptureHook(fake_model) as hook:
            fake_model(enc_ids, dec_ids)
        for h in hook.hidden_states["encoder"]:
            assert isinstance(h, np.ndarray)


class TestLatentCaptureHook:
    def test_captures_pooling_output(self):
        torch.manual_seed(0)
        model = _FakeModel(embed_dim=16, num_heads=2, num_layers=1)
        model.eval()
        B, L = 4, 6
        enc_ids = torch.randint(0, 100, (B, L))
        dec_ids = torch.randint(0, 100, (B, 5))
        with LatentCaptureHook(model) as hook:
            model(enc_ids, dec_ids)
        assert hook.latent is not None
        assert hook.latent.shape == (B, 16)

    def test_latent_is_numpy(self):
        model = _FakeModel(embed_dim=16, num_heads=2, num_layers=1)
        model.eval()
        enc_ids = torch.randint(0, 100, (2, 4))
        dec_ids = torch.randint(0, 100, (2, 3))
        with LatentCaptureHook(model) as hook:
            model(enc_ids, dec_ids)
        assert isinstance(hook.latent, np.ndarray)

    def test_handle_removed_after_exit(self):
        model = _FakeModel()
        with LatentCaptureHook(model) as hook:
            pass
        assert hook._handle is None


class TestHeadAblationHook:
    def test_ablated_output_differs_from_baseline(self):
        torch.manual_seed(42)
        model = _FakeModel(embed_dim=16, num_heads=2, num_layers=2)
        model.eval()
        enc_ids = torch.randint(0, 100, (2, 5))
        dec_ids = torch.randint(0, 100, (2, 4))

        with torch.no_grad():
            baseline, _ = model(enc_ids, dec_ids)

        hook = HeadAblationHook(head_indices=[0], num_heads=2, head_dim=8)
        handles = hook.register(model, layer_indices=[0])
        with torch.no_grad():
            ablated, _ = model(enc_ids, dec_ids)
        HeadAblationHook.remove(handles)

        assert not torch.allclose(baseline, ablated)

    def test_register_returns_correct_handle_count(self):
        model = _FakeModel(embed_dim=16, num_heads=2, num_layers=2)
        hook = HeadAblationHook(head_indices=[0, 1], num_heads=2, head_dim=8)
        handles = hook.register(model, layer_indices=[0, 1])
        assert len(handles) == 2
        HeadAblationHook.remove(handles)

    def test_remove_restores_original_output(self):
        torch.manual_seed(0)
        model = _FakeModel(embed_dim=16, num_heads=2, num_layers=2)
        model.eval()
        enc_ids = torch.randint(0, 100, (2, 5))
        dec_ids = torch.randint(0, 100, (2, 4))

        with torch.no_grad():
            before, _ = model(enc_ids, dec_ids)

        hook = HeadAblationHook(head_indices=[1], num_heads=2, head_dim=8)
        handles = hook.register(model, layer_indices=[0])
        HeadAblationHook.remove(handles)

        with torch.no_grad():
            after, _ = model(enc_ids, dec_ids)

        assert torch.allclose(before, after)
