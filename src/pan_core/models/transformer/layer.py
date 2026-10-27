"""
Created by Zehao Li (Takuho Ri)
Created on 2025-07-26 (Sat)  14:14:06 (+09:00)

model layer compartments for Transformer-style models
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import NewGELUActivation
from transformers.pytorch_utils import Conv1D

""" Positional Encoding """

class SinusoidalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, embedding_dim, max_len=500):
        """Initializes the PositionalEncoding layer.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            max_len (int, optional): The maximum sequence length. Defaults to 500.
        """
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)  # [maxlen, E]
        position = torch.arange(0, max_len).unsqueeze(1)  # [maxlen, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) *
                             -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [maxlen, 1, E]
        self.register_buffer("pe", pe)  # fixed parameters

    def forward(self, x):
        """Forward pass of the PositionalEncoding layer.

        Args:
            x (torch.Tensor): The input tensor of shape [L, B, E].

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0)]
        return x


def rotate_half(x: torch.Tensor):
    """
    Rotates half the hidden dimensions of the input tensor.
    (x+iy) * i = -y+ix
    (x, y) -> (-y, x)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies rotary positional embedding to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape [B, Num_H, L, Hdim].
        k (torch.Tensor): Key tensor of shape [B, Num_H, L, Hdim].
        cos (torch.Tensor): Cosine cache of shape [1, 1, L, Hdim].
        sin (torch.Tensor): Sine cache of shape [1, 1, L, Hdim].

    Returns:
        A tuple containing the rotated query and key tensors.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """
    def __init__(self, dim: int, n_positions: int = 2048, base: int = 10000, device: Optional[torch.device] = None):
        """
        Args:
            dim (int): Dimension of the model (must be even).
            n_positions (int, optional): Maximum sequence length. Defaults to 2048.
            base (int, optional): Base frequency. Defaults to 10000.
            device (torch.device, optional): Device for computations. Defaults to None.
        """
        super().__init__()
        self.dim = dim
        self.n_positions = n_positions
        self.base = base

        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for sine and cosine values
        self._set_cos_sin_cache(seq_len=n_positions, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        """
        Precompute and cache sine and cosine values up to a given sequence length.

        Args:
            seq_len (int): The sequence length to cache.
            device (torch.device): The device to store the caches on.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # Calculate frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Concatenate to handle both real and imaginary parts of complex numbers
        emb = torch.cat((freqs, freqs), dim=-1)

        # Store as non-persistent buffers
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the cached sine and cosine values for the given sequence length.

        Args:
            x (torch.Tensor): Input tensor to infer device and dtype.
            seq_len (int): The sequence length.

        Returns:
            A tuple containing the cosine and sine caches.
            - cos (torch.Tensor): Cosine cache of shape [1, 1, seq_len, dim].
            - sin (torch.Tensor): Sine cache of shape [1, 1, seq_len, dim].
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


_CAUSAL_MASK_CACHE: dict[torch.device, torch.Tensor] = {}


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular [1, 1, L, L] boolean mask, shared by every decoder block.

    Each block used to build its own with ``torch.tril(torch.ones(L, L))``, so one
    forward pass allocated and filled n_layer identical L-by-L tensors for a mask
    that depends on nothing but the length. One mask per device is kept here and
    grown on demand; the return value is a view of its top-left corner, holding
    exactly the values ``tril`` would have produced, so what reaches attention is
    unchanged down to the bit.

    The cache is skipped while tracing. Under ``torch.compile`` the dict write is
    a side effect on global state, and with ``mode="reduce-overhead"`` the tensor
    it stores is a CUDA-graph output that the next run overwrites; reading it back
    then raises "accessing tensor output of CUDAGraphs that has been overwritten".
    main.py's teacher-forcing path compiles exactly that way and fails on its
    second batch. Building the mask inline there restores the pre-cache behaviour
    for the compiled path, value for value.

    The returned tensor is READ-ONLY. When the requested length equals the cached
    one the slice is already contiguous, so a caller's ``.contiguous()`` hands
    back the cache itself; one in-place op on it, say ``logical_not_()``, would
    corrupt every later forward pass on that device. Combining it with another
    mask (``valid & causal``) allocates a fresh tensor and is safe. Anything that
    needs to write must ``clone()`` first.
    """
    if torch.compiler.is_compiling():
        return torch.tril(
            torch.ones(length, length, device=device, dtype=torch.bool)
        )[None, None]
    cached = _CAUSAL_MASK_CACHE.get(device)
    if cached is None or cached.shape[-1] < length:
        cached = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
        _CAUSAL_MASK_CACHE[device] = cached
    return cached[None, None, :length, :length]


def resolve_past_length(past_length: Optional[int],
                        input_pos: Optional[torch.Tensor]) -> int:
    """How many cached positions sit in front of this step's query, as a Python int.

    RoPE needs that offset on the host to slice its tables, and reading it off
    ``input_pos`` costs a device synchronisation inside every layer of every
    generated token. The decode loop already knows the number, so it passes it in
    as *past_length* and the ``.item()`` below never runs. The fallback stays for
    callers that hand over only the tensor, and returns the same value.
    """
    if past_length is not None:
        return past_length
    if input_pos is not None:
        return int(input_pos.item())
    return 0


# NOTE: the GPT2-style (non-RoPE) path below is retained only to load the
# older checkpoints that were trained with it; new configurations use RoPE.

""" Attention Layer """

def Vanilla_SDPA(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Vanilla scaled dot-product attention that also returns attention weights (for analysis).

    Args:
        query (torch.Tensor): Query tensor of shape [B, Num_H, L_q, Hdim].
        key (torch.Tensor): Key tensor of shape [B, Num_H, L_kv, Hdim].
        value (torch.Tensor): Value tensor of shape [B, Num_H, L_kv, Hdim].
        attn_mask (torch.Tensor, optional): Attention bias/mask of shape [B, Num_H, L_q, L_kv]. Defaults to None.
        dropout_p (float, optional): Dropout probability applied to attention weights. Defaults to 0.0.
        is_causal (bool, optional): If True, applies a causal mask (attn_mask must be None). Defaults to False.
        scale (float, optional): Scale factor for dot product. Defaults to 1/sqrt(Hdim).
        enable_gqa (bool, optional): If True, enables grouped-query attention by repeating K/V heads. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor of shape [B, Num_H, L_q, Hdim].
            - Attention weight tensor of shape [B, Num_H, L_q, L_kv].
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # Not logical_not_(): attn_mask can be causal_mask()'s cached tensor.
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


class Attention(nn.Module):
    """
    Custom Attention module (GPT2-style, using Conv1D projections). Legacy path,
    kept for checkpoints trained before RoPE; see the note above.
    Using F.scaled_dot_product_attention for optimized calculation and supporting KV caching.
    Accepts separate valid_token_mask and attn_mask like nn.MultiheadAttention.
    Uses Conv1D for projections like original GPT-2.
    This version can conditionally apply Rotary Positional Embedding (RoPE) based on a `use_rope` flag in the configuration.
    """

    def __init__(self, config, use_rope:bool, decoder:bool):
        """Initializes the Attention module.

        Args:
            config: The configuration object containing model parameters
                    (embedding_dim, n_head, dropout, n_positions).
            use_rope (bool): Whether to apply Rotary Positional Embedding.
            decoder (bool): Whether this attention block is used in the decoder
                            (enables causal masking during training).
        """
        super().__init__()
        nx = config.embedding_dim  # = E
        self.n_head = config.n_head
        self.split_size = nx  # Used in splitting QKV output of c_attn
        self.head_dim = nx // self.n_head
        if nx % self.n_head != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.c_attn = Conv1D(3*nx, nx)  # Combined QKV projection
        self.c_proj = Conv1D(nx, nx)  # Output projection
        self.attn_dropout = nn.Dropout(config.dropout)  # Used for F.sdpa dropout_p

        # RoPE Initialization
        self.use_rope = use_rope
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even to use RoPE.")
            self.rotary_emb = RotaryEmbedding(self.head_dim, n_positions=getattr(config, "n_positions", 2048))

        self.is_decoder = decoder

    def _split_heads(self, tensor, num_heads, head_dim):
        """
        Splits the last dimension of a tensor into (num_heads, head_dim).
        Input: [BatchSize, SeqLen, EmbedDim]
        Output: [BatchSize, NumHeads, SeqLen, HeadDim]
        """
        new_shape = (*tensor.size()[:-1], num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # [B, Num_H, L, Hdim]

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges the num_heads and attn_head_size dimensions.
        Input: [BatchSize, NumHeads, SeqLen, HeadDim]
        Output: [BatchSize, SeqLen, EmbedDim]
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()  # [B, L, Num_H, Hdim]
        new_shape = (*tensor.size()[:-2], num_heads * attn_head_size)
        return tensor.view(new_shape)  # [B, L, E]

    def forward(self,
                x: torch.Tensor,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                input_pos: Optional[torch.Tensor] = None,
                valid_token_mask: Optional[torch.Tensor] = None,
                need_hidden: bool = False,
                past_length: Optional[int] = None
                ):
        """
        Forward pass for the Attention module with KV caching.

        Args:
            x (torch.Tensor): Input tensor, shape [L, B, E] (SeqLen, BatchSize, EmbedDim).
            layer_past (Tuple[torch.Tensor, torch.Tensor], optional): Static KV cache tensors
                (K_cache, V_cache), shapes [B, Num_H, MaxLen, Hdim]. Written in-place during inference.
            input_pos (torch.Tensor, optional): Scalar tensor with the current position index
                for in-place KV cache updates. Required when `layer_past` is provided.
            valid_token_mask (torch.Tensor, optional): Padding mask, shape [B, L_kv].
                Boolean; False = padding position.
            need_hidden (bool, optional): If True, uses Vanilla_SDPA and returns attention weights.
                Defaults to False.
            past_length (int, optional): Number of cached positions before this step, on the
                host. Only RoPE reads it. Given, it saves the device synchronisation that
                reading `input_pos` would cost; left out, it is read off `input_pos`.

        Returns:
            If need_hidden is False:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output tensor, shape [L, B, E].
                - Present KV cache tuple, each tensor shape [B, Num_H, L_kv, Hdim].
            If need_hidden is True:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - Output tensor, shape [L, B, E].
                - Present KV cache tuple, each tensor shape [B, Num_H, L_kv, Hdim].
                - Attention weights, shape [B, Num_H, L_q, L_kv].
        """
        L = x.shape[0]
        x_transposed = x.transpose(0, 1).contiguous() # Shape: [B, L, E]

        # Calculate QKV using the Conv1D layer
        qkv_combined = self.c_attn(x_transposed)  # Output: [B, L, 3*E]

        # Split into query, key, value
        query, key, value = qkv_combined.split(self.split_size, dim=2)  # Shapes: [B, L, E]

        # Split heads
        # [B, Num_H, L_q, Hdim] (L_q = L)
        query_heads = self._split_heads(query, self.n_head, self.head_dim)
        # [B, Num_H, L_k, Hdim] (L_k = L)
        key_heads = self._split_heads(key, self.n_head, self.head_dim)
        # [B, Num_H, L_v, Hdim] (L_v = L)
        value_heads = self._split_heads(value, self.n_head, self.head_dim)

        # RoPE
        if self.use_rope:
            past_length = resolve_past_length(past_length, input_pos)

            cos, sin = self.rotary_emb(x=query_heads, seq_len=past_length + L)

            query_heads, key_heads = apply_rotary_pos_emb(
                query_heads, key_heads, cos[:, :, past_length:past_length + L, :], sin[:, :, past_length:past_length + L, :]
            )

        # KV Cache Handling (Only Inference)
        if layer_past is not None:
            # Retrieve past key and value
            past_key, past_value = layer_past  # [B, Num_H, L_past, Hdim]
            # Concatenate past K/V with current K/V
            # [B, Num_H, L_kv, Hdim] (L_kv = L_past + L)
            past_key.index_copy_(2, input_pos, key_heads)
            past_value.index_copy_(2, input_pos, value_heads)
            key_heads = past_key
            value_heads = past_value

        # Store the updated K/V pair
        present = (key_heads, value_heads)  # [B, Num_H, L_kv, Hdim]

        # Determine if causal masking is needed (only for training).
        is_causal = (L > 1) and self.is_decoder
        if is_causal:
            causal = causal_mask(L, query_heads.device)  # [1, 1, L, L]
            if valid_token_mask is not None:
                attn_mask = valid_token_mask & causal
            else:
                # The cached mask is a view into a possibly larger buffer. Attention
                # is handed the same contiguous tensor it always was.
                attn_mask = causal.contiguous()
        else:
            attn_mask = valid_token_mask

        # Calculate Attention
        if need_hidden:
            attn_output_heads, attn_weight = Vanilla_SDPA(
                query_heads, key_heads,value_heads,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_output_heads = F.scaled_dot_product_attention(
                query_heads, key_heads, value_heads,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False  # Causal masking is handled by attn_mask
            )  # [B, Num_H, L_q, Hdim]

        # Merge heads and apply output projection
        attn_output_merged = self._merge_heads(
            attn_output_heads, self.n_head, self.head_dim)  # [B, L_q, E]
        attn_output_projected = self.c_proj(attn_output_merged)

        # Return the attention output (transposed back to Batch_first=False) and the KV cache
        return (attn_output_projected.transpose(0, 1), present) if not need_hidden else (attn_output_projected.transpose(0, 1), present, attn_weight)


class GlobalAttention(nn.Module):
    """
    Custom GlobalAttention module.
    Using F.scaled_dot_product_attention for optimized calculation and supporting KV caching.
    Accepts separate valid_token_mask and attn_mask like nn.MultiheadAttention.
    This version can conditionally apply Rotary Positional Embedding (RoPE) based on a `use_rope` flag in the configuration.
    """

    def __init__(self, config, use_rope:bool, decoder:bool):
        """Initializes the GlobalAttention module.

        Args:
            config: The configuration object containing model parameters
                    (embedding_dim, n_head, dropout, n_positions, base_freq).
            use_rope (bool): Whether to apply Rotary Positional Embedding.
            decoder (bool): Whether this attention block is used in the decoder
                            (enables causal masking during training).
        """
        super().__init__()
        nx = config.embedding_dim  # = E
        self.n_head = config.n_head
        self.split_size = nx  # Used in splitting QKV output of c_attn
        self.head_dim = nx // self.n_head
        if nx % self.n_head != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.c_attn = nn.Linear(nx, 3*nx)  # Combined QKV projection
        self.c_proj = nn.Linear(nx, nx)  # Output projection
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(p=config.dropout)

        # RoPE Initialization
        self.use_rope = use_rope
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even to use RoPE.")
            self.rotary_emb = RotaryEmbedding(self.head_dim, n_positions=config.n_positions, base=config.base_freq)

        self.is_decoder = decoder

    def forward(self,
                x: torch.Tensor,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                input_pos: Optional[torch.Tensor] = None,
                valid_token_mask: Optional[torch.Tensor] = None,
                need_hidden: bool = False,
                past_length: Optional[int] = None
                ):
        """
        Forward pass for the GlobalAttention module with KV caching.

        Args:
            x (torch.Tensor): Input tensor, shape [L, B, E] (SeqLen, BatchSize, EmbedDim).
            layer_past (Tuple[torch.Tensor, torch.Tensor], optional): Static KV cache tensors
                (K_cache, V_cache), shapes [B, Num_H, MaxLen, Hdim]. Written in-place during inference.
            input_pos (torch.Tensor, optional): Scalar tensor with the current position index
                for in-place KV cache updates. Required when `layer_past` is provided.
            valid_token_mask (torch.Tensor, optional): Padding mask, shape [B, L_kv].
                Boolean; False = padding position.
            need_hidden (bool, optional): If True, uses Vanilla_SDPA and returns attention weights.
                Defaults to False.
            past_length (int, optional): Number of cached positions before this step, on the
                host. Only RoPE reads it. Given, it saves the device synchronisation that
                reading `input_pos` would cost; left out, it is read off `input_pos`.

        Returns:
            If need_hidden is False:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output tensor, shape [L, B, E].
                - Present KV cache tuple, each tensor shape [B, Num_H, L_kv, Hdim].
            If need_hidden is True:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - Output tensor, shape [L, B, E].
                - Present KV cache tuple, each tensor shape [B, Num_H, L_kv, Hdim].
                - Attention weights, shape [B, Num_H, L_q, L_kv].
        """
        L, B, E = x.shape
        x_transposed = x.transpose(0, 1).contiguous() # Shape: [B, L, E]

        # Calculate QKV using the Linear layer
        qkv_combined = self.c_attn(x_transposed)  # Output: [B, L, 3*E]

        # Split into query, key, value
        query, key, value = qkv_combined.split(self.split_size, dim=2)  # Shapes: [B, L, E]

        # Split heads
        # [B, Num_H, L_q, Hdim] (L_q = L)
        query_heads = query.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)
        # [B, Num_H, L_k, Hdim] (L_k = L)
        key_heads = key.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)
        # [B, Num_H, L_v, Hdim] (L_v = L)
        value_heads = value.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE
        if self.use_rope:
            past_length = resolve_past_length(past_length, input_pos)

            cos, sin = self.rotary_emb(x=query_heads, seq_len=past_length + L)

            query_heads, key_heads = apply_rotary_pos_emb(
                query_heads, key_heads, cos[:, :, past_length:past_length + L, :], sin[:, :, past_length:past_length + L, :]
            )

        # KV Cache Handling (Only Inference)
        if layer_past is not None:
            # Retrieve past key and value
            past_key, past_value = layer_past  # [B, Num_H, L_past, Hdim]
            # Concatenate past K/V with current K/V
            # [B, Num_H, L_kv, Hdim] (L_kv = L_past + L)
            past_key.index_copy_(2, input_pos, key_heads)
            past_value.index_copy_(2, input_pos, value_heads)
            key_heads = past_key
            value_heads = past_value

        # Store the updated K/V pair
        present = (key_heads, value_heads)  # [B, Num_H, L_kv, Hdim]

        # Determine if causal masking is needed (only for training).
        is_causal = (L > 1) and self.is_decoder
        if is_causal:
            causal = causal_mask(L, query_heads.device)  # [1, 1, L, L]
            if valid_token_mask is not None:
                attn_mask = valid_token_mask & causal
            else:
                # The cached mask is a view into a possibly larger buffer. Attention
                # is handed the same contiguous tensor it always was.
                attn_mask = causal.contiguous()
        else:
            attn_mask = valid_token_mask

        # Calculate Attention
        if need_hidden:
            attn_output_heads, attn_weight = Vanilla_SDPA(
                query_heads, key_heads, value_heads,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_output_heads = F.scaled_dot_product_attention(
                query_heads, key_heads, value_heads,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, Num_H, L_q, Hdim]

        # Merge heads and apply output projection
        attn_output_merged = attn_output_heads.transpose(1, 2).reshape(B, L, E)  # [B, L_q, E]
        attn_output_projected = self.c_proj(attn_output_merged)

        # Return the attention output (transposed back to Batch_first=False) and the KV cache
        return (self.dropout(attn_output_projected.transpose(0, 1)), present) if not need_hidden else (attn_output_projected.transpose(0, 1), present, attn_weight)

class GlobalCrossAttention(nn.Module):
    """
    Global Cross-Attention module.
    - Query is derived from the decoder's hidden state (`x`).
    - Key and Value are derived from the encoder's output (`memory`).
    """

    def __init__(self, config):
        """Initializes the Cross-Attention module.

        Args:
            config: The configuration object containing model parameters.
        """
        super().__init__()
        nx = config.embedding_dim  # = E
        self.n_head = config.n_head
        self.head_dim = nx // self.n_head
        if nx % self.n_head != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # separated projections for Q (from x) and K/V (from memory).
        self.q_proj = nn.Linear(nx, nx)
        self.kv_proj = nn.Linear(nx, 2*nx)

        self.c_proj = nn.Linear(nx, nx)  # Output projection
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                valid_token_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass for the Cross-Attention module.

        Args:
            x: Query tensor from the decoder. shape [L_q, B, E].
            memory: Key/Value tensor from the encoder. shape [L_kv, B, E]. (if memory is latent vector, then L_kv=1)
            valid_token_mask: Padding mask for the memory. Shape [B, L_kv]. Boolean; False = padding.

        Returns:
            - Output tensor shape [L_q, B, E].
        """
        L_q, B, E = x.shape
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        L_kv = memory.shape[0]

        # Transpose to [B, L, E] for linear layers
        q_in = x.transpose(0, 1).contiguous()
        memory_in = memory.transpose(0, 1).contiguous()

        # 1. Project Q from x, and K,V from memory
        query = self.q_proj(q_in)
        kv_combined = self.kv_proj(memory_in)
        key, value = kv_combined.split(E, dim=2)

        # 2. Split heads
        query_heads = query.reshape(B, L_q, self.n_head, self.head_dim).transpose(1, 2)  # [B, Num_H, L_q, Hdim]
        key_heads = key.reshape(B, L_kv, self.n_head, self.head_dim).transpose(1, 2)    # [B, Num_H, L_kv, Hdim]
        value_heads = value.reshape(B, L_kv, self.n_head, self.head_dim).transpose(1, 2)   # [B, Num_H, L_kv, Hdim]

        # 3. Calculate Attention
        attn_output_heads = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads,
            attn_mask=valid_token_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # Causal masking is never used in cross-attention
        )  # [B, Num_H, L_q, Hdim]

        # 4. Merge heads and apply output projection
        attn_output_merged = attn_output_heads.transpose(1, 2).reshape(B, L_q, E)
        attn_output_projected = self.c_proj(attn_output_merged)

        # Return the attention output (transposed back to Batch_first=False)
        return self.dropout(attn_output_projected.transpose(0, 1))

""" Pooling Layer """
class MaxPooling(nn.Module):
    """Pools encoder states into one latent vector by taking the per-dimension
    maximum over valid positions, then LayerNorm and dropout."""

    def __init__(self, embed_dim:int, dropout_p:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory:torch.Tensor, valid_mask:torch.Tensor):
        """
        Args:
            memory: Encoder output hidden states, shape [L, B, E]
            valid_mask: Padding mask (False=padding), shape [L, B, 1]

        Returns:
            Pooled vector, shape [B, E]
        """
        masked_for_max = memory.masked_fill(~valid_mask, -torch.inf)
        max_pooled = torch.max(masked_for_max, dim=0)[0] # [L, B, E] -> [B, E]
        max_pooled = self.norm(max_pooled)
        max_pooled = self.dropout(max_pooled)

        return max_pooled


class AveragePooling(nn.Module):
    """Pools encoder states into one latent vector by averaging over valid
    positions, then LayerNorm and dropout."""

    def __init__(self, embed_dim:int, dropout_p:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory:torch.Tensor, valid_mask:torch.Tensor):
        """
        Args:
            memory: Encoder output hidden states, shape [L, B, E]
            valid_mask: Padding mask (False=padding), shape [L, B, 1]

        Returns:
            Pooled vector, shape [B, E]
        """
        non_padding_mask = valid_mask
        sum_non_padded = torch.sum(
            memory * non_padding_mask.to(memory.dtype), dim=0)  # [L, B, E] -> [B, E]
        num_non_padded = torch.sum(non_padding_mask.to(memory.dtype), dim=0)  # [B, 1]
        # Avoid division by zero if a sequence is entirely padding
        num_non_padded = num_non_padded.clamp(min=1.0)
        ave_pooled = sum_non_padded / num_non_padded  # [B, E]
        ave_pooled = self.norm(ave_pooled)
        ave_pooled = self.dropout(ave_pooled)

        return ave_pooled


class CLSPooling(nn.Module):
    """Pools encoder states into one latent vector by taking the first position
    (the BOS/CLS token), then LayerNorm and dropout."""

    def __init__(self, embed_dim:int, dropout_p:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory:torch.Tensor):
        """
        Args:
            memory: Encoder output hidden states, shape [L, B, E]

        Returns:
            First token hidden states for each data, shape [B, E]
        """
        firsts = memory[0] # [B, E]
        firsts = self.norm(firsts)
        firsts = self.dropout(firsts)
        return firsts


class MaxAverageCLSConcatPooling(nn.Module):
    """Pools a sequence into a single vector using [maxpooling, averagepooling, firsttoken(CLS)] concatenated vectors."""
    def __init__(self, embed_dim:int, dropout_p:float):
        super().__init__()
        self.maxpool = MaxPooling(embed_dim, dropout_p)
        self.avepool = AveragePooling(embed_dim, dropout_p)
        self.cls = CLSPooling(embed_dim, dropout_p)
        self.proj = nn.Linear(3*embed_dim, embed_dim)

    def forward(self, memory:torch.Tensor, valid_mask:torch.Tensor):
        """
        Args:
            memory: Encoder output hidden states, shape [L, B, E]
            valid_mask: Padding mask (False=padding), shape [B, 1, 1, L]

        Returns:
            Pooled vector, shape [B, E]
        """
        valid_mask = valid_mask.transpose(0, -1).squeeze(1).squeeze(1).unsqueeze(-1) # [L, B, 1]
        concats = torch.cat([
            self.maxpool(memory, valid_mask),
            self.avepool(memory, valid_mask),
            self.cls(memory)
        ], dim=1) # [B, 3*E]
        latent = self.proj(concats) # [B, E]

        return latent


class MultiHeadAttentionPooling(nn.Module):
    """Pools a sequence into a single vector using multi-head attention from a learned global query"""
    def __init__(self, embed_dim:int, num_heads:int, num_queries:int, dropout_p:float=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p

        # === Learnable Parameters ===
        # 1. Base queries for Multi-Query Pooling
        self.queries = nn.Parameter(torch.empty(num_queries, 1, embed_dim))
        nn.init.normal_(self.queries, mean=0, std=0.02)

        # 2. Adapter for Adaptive Query Generation
        self.query_adapter = nn.Linear(embed_dim, num_queries * embed_dim)

        # === Sub-layer 1: Multi-Head Cross-Attention (Pre-LN) ===
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_memory = nn.LayerNorm(embed_dim)
        self.kv = nn.Linear(embed_dim, 2*embed_dim)
        self.proj_attn = nn.Linear(embed_dim, embed_dim)
        self.dropout_attn = nn.Dropout(p=dropout_p)

        # === Sub-layer 2: Feed-Forward Network (Pre-LN) ===
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, dropout_p)

        # === Aggregation ===
        if self.num_queries > 1:
            self.query_aggregator = SwiGLU(embed_dim, dropout_p, embed_dim//4, out_dim=1)
        self.gate_generator = SwiGLU(embed_dim, dropout_p, embed_dim//4, out_dim=1)

    def forward(self, memory: torch.Tensor, valid_mask: torch.Tensor):
        """
        Args:
            memory: Encoder output hidden states, shape [L, B, E].
            valid_mask: Padding mask (False=padding), shape [B, 1, 1, L].

        Returns:
            Pooled representation, shape [B, E].
        """
        memory = memory.transpose(0, 1) # [B, L, E]
        B, L, E = memory.shape

        # === 1. Generate Adaptive Multi-Query ===
        # Calculate global context from memory
        valid_token_mask = valid_mask.squeeze(1).squeeze(1).unsqueeze(-1) # [B, L, 1]
        masked_memory = memory * valid_token_mask
        sum_memory = masked_memory.sum(dim=1, keepdim=True)
        num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        memory_mean = sum_memory / num_valid_tokens # [B, 1, E]

        # Adapt base queries with the global context
        q_base = self.queries.expand(-1, B, -1).transpose(0, 1)  # [B, NumQueries, E]
        q_adapt = self.query_adapter(memory_mean.squeeze(1))  # [B, NumQueries*E]
        q_adapt = q_adapt.reshape(B, self.num_queries, E)  # [B, NumQueries, E]

        # The final queries used for attention
        q_residual = q_base + q_adapt # [B, NumQueries, E]

        # === 2. Multi-Head Attention Sub-layer (Pre-LN) ===
        q_norm = self.norm1_q(q_residual)
        memory_norm = self.norm1_memory(memory)

        kv_combined = self.kv(memory_norm) # [B, L, 2E]
        k, v = kv_combined.split(E, dim=2) # [B, L, E]

        # Reshape for MHA: [B, SeqLen, E] -> [B, NumHeads, SeqLen, HeadDim]
        q_mha = q_norm.reshape(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k_mha = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_mha = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        pooled_heads = F.scaled_dot_product_attention(
            query=q_mha,
            key=k_mha,
            value=v_mha,
            attn_mask=valid_mask,
            is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0
        ) # [B, NumHeads, 1, HeadDim]

        # Concatenate heads, project and residual connection (query)
        pooled_concat = pooled_heads.transpose(1, 2).reshape(B, self.num_queries, E)
        attn_output = self.dropout_attn(self.proj_attn(pooled_concat))
        q_updated = q_residual + attn_output  # [B, NumQueries, E]

        # === 3. Feed-Forward Network Sub-layer (Pre-LN) ===
        q_norm2 = self.norm2(q_updated)
        ffn_output = self.ffn(q_norm2)
        pooled_queries = q_updated + ffn_output # [B, NumQueries, E]

        # === 4. Final Aggregation and Residual Token Pooling ===
        # Aggregate the multiple query outputs into a single vector
        if self.num_queries == 1:
            final_pooled = pooled_queries.squeeze(1)
        else:
            aggregation_weights = self.query_aggregator(pooled_queries)  # [B, NumQueries, 1]
            aggregation_weights = F.softmax(aggregation_weights, dim=1)
            final_pooled = (pooled_queries * aggregation_weights).sum(dim=1)  # [B, E]
        # Add the gated residual connection from the original memory's mean
        gate_logits = self.gate_generator(final_pooled).squeeze(-1) # [B]
        gate = torch.sigmoid(gate_logits).unsqueeze(-1) # [B, 1]
        final_pooled = (1 - gate) * final_pooled + gate * memory_mean.squeeze(1)

        return final_pooled


""" FFN layer """

class GPT2MLP(nn.Module):
    """GPT2MLP (from HuggingFace transformers)"""
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = Conv1D(4*embed_dim, embed_dim)
        self.c_proj = Conv1D(embed_dim, 4*embed_dim)
        self.act = NewGELUActivation() # GPT2config default setting
        self.dropout = nn.Dropout(0.1) # GPT2config default resid_pdrop: 0.1

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        """Applies the feed-forward block: project up, activate, project down, drop."""
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit
    original paper: Noam Shazeer, 2020, arXiv (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, embed_dim, dropout_p, middle_dim=None, out_dim=None) -> None:
        super().__init__()
        middle_dim = int((2/3)* 4 * embed_dim) if middle_dim is None else middle_dim # default value at original paper
        self.w1 = nn.Linear(embed_dim, middle_dim)
        self.v = nn.Linear(embed_dim, middle_dim)
        self.act = nn.SiLU()
        self.w2 = nn.Linear(middle_dim, embed_dim if out_dim is None else out_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [..., embed_dim].

        Returns:
            torch.Tensor: Output tensor, shape [..., out_dim] (out_dim = embed_dim unless overridden).
        """
        return self.dropout(self.w2(self.act(self.w1(x)) * self.v(x)))


""" Transformer Block """

class TransformerBlock(nn.Module):
    """A single Transformer block with pre-LN."""
    def __init__(self, config, use_rope:bool, decoder=False):
        """Initializes the TransformerBlock.

        Args:
            config: The configuration object containing model parameters
                    (embedding_dim, n_head, dropout, layer_norm_epsilon, latent_conditioning,
                    cross_attn, gpt2style, n_positions, base_freq).
            use_rope (bool): Whether to apply Rotary Positional Embedding in the attention sub-layer.
            decoder (bool, optional): Whether this block is used in the decoder. Affects causal masking
                and whether LayerNorm uses learnable affine parameters. Defaults to False.
        """
        super().__init__()
        nx = config.embedding_dim
        self.ln_1 = nn.LayerNorm(nx, config.layer_norm_epsilon, elementwise_affine=True if not decoder else (config.latent_conditioning == "add_once"))
        self.attn = Attention(config, use_rope, decoder) if config.gpt2style else GlobalAttention(config, use_rope, decoder)
        if config.cross_attn is not None and decoder:
            self.cross_attn_type = config.cross_attn
            self.ln_cross = nn.LayerNorm(nx, config.layer_norm_epsilon, elementwise_affine=(config.latent_conditioning == "add_once"))
            self.cross_attn = GlobalCrossAttention(config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon, elementwise_affine=True if not decoder else (config.latent_conditioning == "add_once"))
        self.mlp = GPT2MLP(nx) if config.gpt2style else SwiGLU(nx, config.dropout)

    def forward(self,
                x: torch.Tensor,
                # [B, L_kv], boolean False=mask
                valid_token_mask: Optional[torch.Tensor] = None,
                gamma_self: torch.Tensor|None = None, beta_self: torch.Tensor|None = None, alpha_self: torch.Tensor|None = None,
                gamma_cross: torch.Tensor|None = None, beta_cross: torch.Tensor|None = None, alpha_cross: torch.Tensor|None = None,
                gamma_ffn: torch.Tensor|None = None, beta_ffn: torch.Tensor|None = None, alpha_ffn: torch.Tensor|None = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                input_pos: Optional[torch.Tensor] = None,
                past_length: Optional[int] = None,
                memory: torch.Tensor|None = None,
                memory_padding_mask: torch.Tensor|None = None,
                need_hidden: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Transformer block using the custom Attention layer. (Pre-LN)

        Args:
            x (torch.Tensor): Input tensor, shape [L, B, E].
            valid_token_mask (torch.Tensor, optional): Padding mask for self-attention,
                shape [B, L_kv]. Boolean; False = padding position.
            gamma_self (torch.Tensor, optional): AdaLN scale for self-attention pre-norm. Shape [B, 1, E].
            beta_self (torch.Tensor, optional): AdaLN shift for self-attention pre-norm. Shape [B, 1, E].
            alpha_self (torch.Tensor, optional): AdaLN gate for self-attention residual. Shape [B, 1, E].
            gamma_cross (torch.Tensor, optional): AdaLN scale for cross-attention pre-norm. Shape [B, 1, E].
            beta_cross (torch.Tensor, optional): AdaLN shift for cross-attention pre-norm. Shape [B, 1, E].
            alpha_cross (torch.Tensor, optional): AdaLN gate for cross-attention residual. Shape [B, 1, E].
            gamma_ffn (torch.Tensor, optional): AdaLN scale for FFN pre-norm. Shape [B, 1, E].
            beta_ffn (torch.Tensor, optional): AdaLN shift for FFN pre-norm. Shape [B, 1, E].
            alpha_ffn (torch.Tensor, optional): AdaLN gate for FFN residual. Shape [B, 1, E].
            layer_past (Tuple[torch.Tensor, torch.Tensor], optional): Static KV cache tensors
                (K_cache, V_cache) for this layer. Written in-place during inference.
            input_pos (torch.Tensor, optional): Scalar tensor with current position index for KV cache.
            past_length (int, optional): The same index on the host, for RoPE. Passing it keeps
                the attention layer from synchronising on `input_pos`; see `resolve_past_length`.
            memory (torch.Tensor, optional): Memory tensor for cross-attention.
                Shape [1, B, E] (latent) or [L_src, B, E] (source hidden states).
            memory_padding_mask (torch.Tensor, optional): Padding mask for the memory in S-T cross-attention.
                Shape [B, L_src]. Boolean; False = padding position.
            need_hidden (bool, optional): If True, returns attention weights. Defaults to False.

        Returns:
            If need_hidden is False:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output tensor, shape [L, B, E].
                - Updated KV cache tuple for this layer.
            If need_hidden is True:
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - Output tensor, shape [L, B, E].
                - Updated KV cache tuple for this layer.
                - Self-attention weights, shape [B, Num_H, L_q, L_kv].
        """
        # Store input for the 1st residual connection
        residual = x

        # Pre-Self-Attention Layer Normalization
        normed_x_self = self.ln_1(x)
        if gamma_self is not None and beta_self is not None:
            normed_x_self = gamma_self * normed_x_self + beta_self

        # Self-Attention
        attn_outputs = self.attn(
            normed_x_self,                 # Pass the normalized input
            layer_past=layer_past,
            input_pos=input_pos,
            valid_token_mask=valid_token_mask,
            need_hidden=need_hidden,
            past_length=past_length
        )

        # Unpack the attention output and the new cache
        attn_output = attn_outputs[0]  # [L, B, E]
        present = attn_outputs[1]      # Tuple (K_present, V_present)
        if need_hidden:
            attn_weight = attn_outputs[2] # [B, Num_head, L, L]

        # Post-Self-Attention Residual Connection & store for the next residual connection
        if alpha_self is not None:
            x = residual + alpha_self * attn_output
        else:
            x = residual + attn_output
        residual = x

        if hasattr(self, 'cross_attn'):
            # Cross-Attention with Latent Vector
            # Pre-Cross-Attention Layer Normalization
            normed_x_cross = self.ln_cross(x)
            if gamma_cross is not None and beta_cross is not None:
                normed_x_cross = gamma_cross * normed_x_cross + beta_cross

            # Cross-Attention
            cross_attn_output = self.cross_attn(
                x=normed_x_cross,
                memory=memory,
                valid_token_mask=memory_padding_mask if self.cross_attn_type == "S-T" else None
            )

            # Post-Cross-Attention Residual Connection & store for the next residual connection
            if alpha_cross is not None:
                x = residual + alpha_cross * cross_attn_output
            else:
                x = residual + cross_attn_output

            residual = x

        # Pre-FFN Layer Normalization
        normed_x2 = self.ln_2(x)
        if gamma_ffn is not None and beta_ffn is not None:
            normed_x2 = gamma_ffn * normed_x2 + beta_ffn

        # FFN
        mlp_output = self.mlp(normed_x2)

        # Post-FFN Residual Connection
        if alpha_ffn is not None:
            final_output = residual + alpha_ffn * mlp_output
        else:
            final_output = residual + mlp_output

        # Return the final output of the block and the cache state for this layer
        return (final_output, present) if not need_hidden else (final_output, present, attn_weight)


""" VAE sampling layer """

class Sampling(nn.Module):
    """Reparameterization trick for VAEs."""
    def __init__(self):
        """Initializes the Sampling module."""
        super().__init__()

    def forward(self, mu, log_var):
        """Reparameterization sampling: z = mu + sqrt(exp(log_var)) * epsilon.

        Args:
            mu (torch.Tensor): Mean of the latent distribution, shape [B, E].
            log_var (torch.Tensor): Log variance of the latent distribution, shape [B, E].

        Returns:
            torch.Tensor: Sampled latent vector, shape [B, E].
        """
        epsilon = torch.randn_like(mu)
        return mu + torch.sqrt(torch.exp(log_var)) * epsilon
