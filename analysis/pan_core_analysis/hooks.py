"""
PyTorch hook utilities for hidden-state capture and head ablation.

Usage pattern:
    with HiddenStateCaptureHook(model) as hook:
        _ = model(...)
        hidden_states = hook.hidden_states  # {"encoder": [...], "decoder": [...]}, [B, L, E] per layer
        attn_scores   = hook.attn_scores    # {"encoder": [...], "decoder": [...]}, [B, H, L, L] per layer

For ablation, register HeadAblationHook on the target module(s) and remove after.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hidden-state and attention capture
# ---------------------------------------------------------------------------

class HiddenStateCaptureHook:
    """Context manager that captures hidden states and attention weights from a model.

    The model must expose ``encoder.h`` and ``decoder.h`` as ``nn.ModuleList``
    of TransformerBlock modules, each producing ``(hidden, ?, attn, ...)`` tuples.

    Inside the context all hooks are active; on __exit__ they are removed.

    Attributes:
        hidden_states: dict with keys "encoder" / "decoder", each a list of
            numpy arrays [B, L, E] (one per layer, including embedding layer first).
        attn_scores: dict with keys "encoder" / "decoder", each a list of
            numpy arrays [B, H, L, L] (one per Transformer block).
        encoder_embeddings: numpy array [B, L, E] of embedding-layer output.
        decoder_embeddings: numpy array [B, L, E] of embedding-layer output.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self._model = model
        self._device = device
        self._handles: list = []
        self.hidden_states: dict[str, list[np.ndarray]] = {"encoder": [], "decoder": []}
        self.attn_scores: dict[str, list[np.ndarray]] = {"encoder": [], "decoder": []}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    def _make_block_hook(self, side: str):
        """Returns a forward hook that appends hidden state & attention from one block."""
        hidden_list = self.hidden_states[side]
        attn_list = self.attn_scores[side]

        def hook(module, inputs, outputs):
            # outputs is typically (hidden, present_kv, attn_weights, ...)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                hidden_out = outputs[0]  # [L, B, E] or [B, L, E]
                attn_out = outputs[2]    # [B, H, L, L]
                # Normalize hidden to [B, L, E]
                if hidden_out.shape[0] != attn_out.shape[0]:
                    hidden_out = hidden_out.permute(1, 0, 2)
                hidden_list.append(self._to_numpy(hidden_out))
                attn_list.append(self._to_numpy(attn_out))

        return hook

    def _make_emb_hook(self, side: str):
        """Returns a forward hook that captures the embedding output."""
        hidden_list = self.hidden_states[side]

        def hook(module, inputs, output):
            # output: [L, B, E]
            out = output
            if isinstance(out, (tuple, list)):
                out = out[0]
            # Normalize to [B, L, E]
            arr = self._to_numpy(out)
            if arr.ndim == 3 and arr.shape[0] != arr.shape[1]:
                # Heuristic: if seq_len < batch_size it's fine, but if seq_len > batch_size assume [L,B,E]
                # We just check against what we know about the model layout
                pass
            # Prepend embedding as layer-0
            hidden_list.insert(0, arr)

        return hook

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def _register_hooks(self):
        model = self._model
        for side, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
            # Embedding layer hook
            if hasattr(part, "wte"):
                h = part.wte.register_forward_hook(self._make_emb_hook(side))
                self._handles.append(h)
            # Per-block hooks
            if hasattr(part, "h"):
                for block in part.h:
                    h = block.register_forward_hook(self._make_block_hook(side))
                    self._handles.append(h)

    def __enter__(self):
        self.hidden_states = {"encoder": [], "decoder": []}
        self.attn_scores = {"encoder": [], "decoder": []}
        self._handles.clear()
        self._register_hooks()
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class LatentCaptureHook:
    """Captures the encoder pooling output (latent vector z).

    Usage:
        with LatentCaptureHook(model) as hook:
            _ = model(...)
            latent = hook.latent  # numpy array [B, E]
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._handle = None
        self.latent: np.ndarray | None = None

    def _hook(self, module, inputs, output):
        out = output
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.latent = out.detach().float().cpu().numpy()

    def __enter__(self):
        self.latent = None
        pooling = self._model.encoder.pooling
        self._handle = pooling.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Head ablation
# ---------------------------------------------------------------------------

class HeadAblationHook:
    """Pre-hook that zeros out specified attention head outputs before the output projection.

    Register this on ``model.encoder.h[layer_idx].attn.c_proj``.

    Args:
        head_indices: 0-based head indices to zero out.
        num_heads: Total number of attention heads (must match model config).
        head_dim: Per-head dimension (embedding_dim // num_heads).

    Example:
        hook = HeadAblationHook(head_indices=[0, 2], num_heads=8, head_dim=64)
        handles = hook.register(model, layer_indices=[7])
        # ... forward pass ...
        hook.remove(handles)
    """

    def __init__(self, head_indices: Sequence[int], num_heads: int, head_dim: int):
        self.head_indices = list(head_indices)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _pre_hook(self, module: nn.Module, args: tuple) -> tuple:
        hidden = args[0]  # [B, L, E]
        B, L, E = hidden.shape
        hidden = hidden.view(B, L, self.num_heads, self.head_dim)
        for h_idx in self.head_indices:
            hidden[:, :, h_idx, :] = 0.0
        hidden = hidden.view(B, L, E)
        return (hidden,)

    def register(self, model: nn.Module, layer_indices: Sequence[int]) -> list:
        """Register pre-hooks on encoder.h[layer_idx].attn.c_proj for each layer.

        Returns:
            List of hook handles (pass to ``remove``).
        """
        handles = []
        for layer_idx in layer_indices:
            module = model.encoder.h[layer_idx].attn.c_proj
            handle = module.register_forward_pre_hook(self._pre_hook)
            handles.append(handle)
        return handles

    @staticmethod
    def remove(handles: list):
        for h in handles:
            h.remove()
