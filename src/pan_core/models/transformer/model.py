"""
Created by Zehao Li (Takuho Ri)
Created on 2025-07-26 (Sat)  15:46:50 (+09:00)

model compartment for Transformer-style models
"""

from dataclasses import dataclass
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from pan_core.common.utils import KLLoss, kl_per_dimension, move_to_cpu
from pan_core.config import ModelConfig

from .layer import (
    MaxAverageCLSConcatPooling,
    MultiHeadAttentionPooling,
    RotaryEmbedding,
    Sampling,
    SinusoidalEncoding,
    TransformerBlock,
)


class Encoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, config):
        """Initializes the Encoder.

        Args:
            config: The configuration object containing model parameters.
        """
        super().__init__()
        self.vae = config.vae
        self.cross_attn = config.cross_attn
        nx = config.embedding_dim
        # Before TransformerBlock
        self.wte = nn.Embedding(config.vocab_size, nx, padding_idx=0)
        if not getattr(config, "encoder_rope", True):
            self.wpe = SinusoidalEncoding(nx, max_len=config.n_positions)
        self.dropout_emb = nn.Dropout(p=config.dropout)
        self.pooling_mode = getattr(config, "pooling_mode", "concat")

        # TransformerBlock
        self.h = nn.ModuleList([TransformerBlock(config, config.encoder_rope, decoder=False) for _ in range(config.n_layer)])

        # LayerNorm
        self.ln_f = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

        if self.cross_attn != "S-T":
            if self.pooling_mode == "concat":
                self.pooling = MaxAverageCLSConcatPooling(nx, config.dropout)
            elif self.pooling_mode == "MHA":
                self.pooling = MultiHeadAttentionPooling(nx, config.n_head, config.n_head, config.dropout)

            if self.vae:
                self.mu = nn.Linear(nx, nx)
                self.log_var = nn.Linear(nx, nx)
            else:
                self.latent_noising = getattr(config, "latent_noising", False)
                self.tanh_latent = getattr(config, "tanh_latent", True)

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor, need_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass for the Encoder.

        Args:
            x: Input token IDs. Shape [L, B].
            valid_token_mask: Padding mask (True = valid token). Shape [B, 1, 1, L].
            need_hidden: If True, stores all hidden states and attention scores in
                         ``self.all_hidden_states`` / ``self.all_attn_score``.

        Returns:
            VAE mode     — Tuple[mu, log_var], each shape [B, E].
            S-T mode     — Tuple[hidden_states [L, B, E], valid_token_mask [B, 1, 1, L]].
            Default mode — latent vector, shape [B, E].
        """
        self.all_hidden_states = [] if need_hidden else None
        self.all_attn_score = [] if need_hidden else None

        # Get embeddings + positional encoding
        hidden_states = self.wte(x)  # [L, B, E]
        if need_hidden:
            self.all_hidden_states.append(move_to_cpu(hidden_states))
        if hasattr(self, "wpe"):
            hidden_states = self.wpe(hidden_states)  # [L, B, E]
        hidden_states = self.dropout_emb(hidden_states)

        # Pass through Transformer blocks
        for block in self.h:
            outputs = block(
                x=hidden_states,
                valid_token_mask=valid_token_mask,
                need_hidden=need_hidden
            )
            if need_hidden:
                self.all_hidden_states.append(move_to_cpu(outputs[0]))
                self.all_attn_score.append(move_to_cpu(outputs[2]),)
            # only need hidden states
            hidden_states = outputs[0]

        # Layer Normalization
        hidden_states = self.ln_f(hidden_states)  # [L, B, E]

        # Return hidden_states when using Source-Target CrossAttention in Decoder
        if self.cross_attn == "S-T":
            return hidden_states, valid_token_mask

        # Memory pooling
        latent = self.pooling(hidden_states, valid_token_mask) # [B, E]

        # Linear layer and activation
        if self.vae:
            mu = self.mu(latent) # [B, E]
            log_var = self.log_var(latent) # [B, E]
            return mu, log_var
        else:
            if self.latent_noising and self.training:
                latent = latent + torch.randn_like(latent) * 0.05
            return torch.tanh(latent) if self.tanh_latent else latent


class Decoder(nn.Module):
    """Transformer Decoder"""

    def __init__(self, config):
        """Initializes the Decoder.

        Args:
            config: The configuration object containing model parameters.
        """
        super().__init__()
        self.nx = config.embedding_dim
        self.n_layer = config.n_layer
        # Before TransformerBlock
        self.wte = nn.Embedding(config.vocab_size, self.nx, padding_idx=0)
        if not getattr(config, "decoder_rope", False):
            self.wpe = SinusoidalEncoding(self.nx, max_len=config.n_positions)
        self.dropout_emb = nn.Dropout(p=config.dropout)

        self.prepare_adaln_network(config)

        # TransformerBlock
        self.h = nn.ModuleList([TransformerBlock(config, config.decoder_rope, decoder=True) for _ in range(self.n_layer)])

        # LayerNorm
        self.ln_f = nn.LayerNorm(self.nx, eps=config.layer_norm_epsilon)

        # Linear layer for output (Embed -> Vocab)
        self.weight_sharing = config.weight_sharing
        self.output_fc = nn.Linear(self.nx, config.vocab_size, bias=not config.weight_sharing)
        self._tie_weights()

    def _tie_weights(self):
        """Points the output projection at this decoder's embedding table.

        HuggingFace calls this hook after loading a checkpoint. It is needed because
        ``_tied_weights_keys`` keeps the duplicate out of the saved file, so on the way
        back in ``output_fc.weight`` would otherwise stay at its random initialisation.

        The tie cannot be left to ``PreTrainedModel.tie_weights``: that one ties the
        output embeddings to ``get_input_embeddings()``, which is the *encoder's* table,
        and the encoder and decoder keep separate embeddings here.
        """
        if self.weight_sharing:
            self.output_fc.weight = self.wte.weight  # [vocab_size, E]

    def create_valid_token_mask(self, input_ids: torch.Tensor):
        """
        Creates valid_token_mask from input token IDs.

        Args:
            input_ids: Input token IDs, shape [L, B].

        Returns:
            valid_token_mask: Boolean mask [B, 1, 1, L], False indicates padding.
        """
        # input_ids: [L, B]
        valid_token_mask = (input_ids != 0).transpose(0, 1).unsqueeze(1).unsqueeze(2)
        return valid_token_mask.to(input_ids.device)

    def prepare_adaln_network(self, config):
        """
        prepare nn.Linear network for Adaptive-LayerNorm scaler parameters.
        """
        self.latent_conditioning = config.latent_conditioning
        self.cross_attn = config.cross_attn
        if self.cross_attn == "S-T" or self.latent_conditioning is None:
            return
        if self.latent_conditioning.startswith("adaln"):
            if self.cross_attn is None:
                num_param = 6 if self.latent_conditioning == "adaln-zero" else 4
            else:
                num_param = 9 if self.latent_conditioning == "adaln-zero" else 6
            self.ada_ln_projection = nn.Linear(self.nx, num_param*self.n_layer*self.nx)
            # initialize the weights and bias of alphas to zero
            if self.latent_conditioning == "adaln-zero":
                if self.cross_attn is None:
                    for i in range(self.n_layer):
                        start, end = i*6*self.nx + 2*self.nx, i*6*self.nx + 3*self.nx
                        nn.init.constant_(self.ada_ln_projection.weight[start:end, :], 0)
                        nn.init.constant_(self.ada_ln_projection.bias[start:end], 0)
                        start, end = i*6*self.nx + 5*self.nx, (i+1)*6*self.nx
                        nn.init.constant_(self.ada_ln_projection.weight[start:end, :], 0)
                        nn.init.constant_(self.ada_ln_projection.bias[start:end], 0)
                else:
                    for i in range(self.n_layer):
                        start, end = i*9*self.nx + 2*self.nx, i*9*self.nx + 3*self.nx
                        nn.init.constant_(self.ada_ln_projection.weight[start:end, :], 0)
                        nn.init.constant_(self.ada_ln_projection.bias[start:end], 0)
                        start, end = i*9*self.nx + 5*self.nx, i*9*self.nx + 6*self.nx
                        nn.init.constant_(self.ada_ln_projection.weight[start:end, :], 0)
                        nn.init.constant_(self.ada_ln_projection.bias[start:end], 0)
                        start, end = i*9*self.nx + 8*self.nx, (i+1)*9*self.nx
                        nn.init.constant_(self.ada_ln_projection.weight[start:end, :], 0)
                        nn.init.constant_(self.ada_ln_projection.bias[start:end], 0)

    def split_adaln_params(self, layer_idx:int, ada_params:torch.Tensor|None):
        """
        split adaptive layer norm params matrix to individual parameters
        """
        if ada_params is None:
            return (None,)*9
        E = ada_params.size()[-1]
        mult = E // self.nx
        assert (E % self.nx == 0) and (mult in (4, 6, 9)), "Invalid ada_params shape."
        (gamma_self, beta_self, alpha_self,
         gamma_cross, beta_cross, alpha_cross,
         gamma_ffn, beta_ffn, alpha_ffn) = (None,)*9

        if mult == 4:
            gamma_self = ada_params[:, layer_idx, :self.nx].unsqueeze(0)
            beta_self = ada_params[:, layer_idx, self.nx:2*self.nx].unsqueeze(0)
            gamma_ffn = ada_params[:, layer_idx, 2*self.nx:3*self.nx].unsqueeze(0)
            beta_ffn = ada_params[:, layer_idx, 3*self.nx:].unsqueeze(0)
        elif mult == 6:
            if self.cross_attn is None:
                gamma_self = ada_params[:, layer_idx, :self.nx].unsqueeze(0)
                beta_self = ada_params[:, layer_idx, self.nx:2*self.nx].unsqueeze(0)
                alpha_self = ada_params[:, layer_idx, 2*self.nx:3*self.nx].unsqueeze(0)
                gamma_ffn = ada_params[:, layer_idx, 3*self.nx:4*self.nx].unsqueeze(0)
                beta_ffn = ada_params[:, layer_idx, 4*self.nx:5*self.nx].unsqueeze(0)
                alpha_ffn = ada_params[:, layer_idx, 5*self.nx:].unsqueeze(0)
            else:
                gamma_self = ada_params[:, layer_idx, :self.nx].unsqueeze(0)
                beta_self = ada_params[:, layer_idx, self.nx:2*self.nx].unsqueeze(0)
                gamma_cross = ada_params[:, layer_idx, 2*self.nx:3*self.nx].unsqueeze(0)
                beta_cross = ada_params[:, layer_idx, 3*self.nx:4*self.nx].unsqueeze(0)
                gamma_ffn = ada_params[:, layer_idx, 4*self.nx:5*self.nx].unsqueeze(0)
                beta_ffn = ada_params[:, layer_idx, 5*self.nx:].unsqueeze(0)
        elif mult == 9:
            gamma_self = ada_params[:, layer_idx, :self.nx].unsqueeze(0)
            beta_self = ada_params[:, layer_idx, self.nx:2*self.nx].unsqueeze(0)
            alpha_self = ada_params[:, layer_idx, 2*self.nx:3*self.nx].unsqueeze(0)
            gamma_cross = ada_params[:, layer_idx, 3*self.nx:4*self.nx].unsqueeze(0)
            beta_cross = ada_params[:, layer_idx, 4*self.nx:5*self.nx].unsqueeze(0)
            alpha_cross = ada_params[:, layer_idx, 5*self.nx:6*self.nx].unsqueeze(0)
            gamma_ffn = ada_params[:, layer_idx, 6*self.nx:7*self.nx].unsqueeze(0)
            beta_ffn = ada_params[:, layer_idx, 7*self.nx:8*self.nx].unsqueeze(0)
            alpha_ffn = ada_params[:, layer_idx, 8*self.nx:].unsqueeze(0)

        return (gamma_self, beta_self, alpha_self,
                gamma_cross, beta_cross, alpha_cross,
                gamma_ffn, beta_ffn, alpha_ffn)

    def forward(self,
                x: torch.Tensor,  # Input token IDs
                latent: torch.Tensor|None = None,  # Latent condition vector [B, E]
                # KV Cache from previous steps
                past_key_values: Optional[List[Tuple[torch.Tensor,
                                                     torch.Tensor]]] = None,
                input_pos: Optional[torch.Tensor] = None,
                past_length: Optional[int] = None,
                valid_token_mask: Optional[torch.Tensor] = None,
                source:torch.Tensor|None = None,
                encoder_mask: torch.Tensor|None = None,
                need_hidden: bool = False,
                use_cache: bool = True
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Decoder. Handles both training (L > 1) and cached generation (L = 1).

        Args:
            x: Input token IDs. Shape [L, B].
            latent: Latent condition vector. Shape [B, E]. Used unless cross_attn == "S-T".
            past_key_values: List of per-layer KV-cache tuples ``(K, V)`` from the previous step,
                             each with shape [B, Num_H, maxlen, Hdim]. None during training.
            input_pos: Current position index (scalar tensor). Required when ``past_key_values`` is not None.
            past_length: The same index as a Python int. Optional, and equal to ``input_pos``
                whenever both are given. Reading it off the tensor instead costs one device
                synchronisation per layer per generated token, because RoPE and the learned
                positional table both slice on the host; the decode loop counts the position
                itself and passes it here so none of those synchronisations happen.
            valid_token_mask: Causal / padding mask passed to self-attention. Shape [B, 1, 1, L] or
                              [1, 1, 1, maxlen] during cached generation.
            source: Encoder hidden states used for S-T cross-attention. Shape [L_src, B, E].
            encoder_mask: Padding mask for the encoder source. Shape [B, 1, 1, L_src].
            need_hidden: If True, stores per-layer hidden states and attention scores in
                         ``self.all_hidden_states`` / ``self.all_attn_score``.
            use_cache: Collect and return each layer's (K, V). Generation needs it; a
                teacher-forced pass that only wants the loss does not, and keeping the
                cache holds every layer's K and V alive until the caller drops the output
                (K is float32 after RoPE). At validation batch 2048, length 402 that is
                most of the forward's peak.

        Returns:
            Tuple[logits, next_key_values]:
                - logits: Output logits. Shape [L, B, VocabSize].
                - next_key_values: Updated list of KV-cache tuples, same structure as
                  ``past_key_values``; None when ``use_cache`` is False.
        """
        L_in, B = x.shape  # generation mode: L_in should be 1 for using KV cache.

        # Determine sequence length based on input_pos for generation
        if past_key_values is None:
            # Training mode or first generation step, position starts from 0
            past_length = 0
        elif past_length is None:
            # Generation mode with static cache
            # The calling function MUST provide the current position
            assert input_pos is not None, "input_pos must be provided for static cache generation."
            past_length = int(input_pos.item())

        self.all_hidden_states = [] if need_hidden else None
        self.all_attn_score = [] if need_hidden else None

        # Embedding and Positional Encoding
        hidden_states = self.wte(x)  # [L_in, B, E]
        if need_hidden:
            self.all_hidden_states.append(move_to_cpu(hidden_states))
        if hasattr(self, "wpe"):
            pos_embeds = self.wpe.pe[:past_length + L_in, :, :]
            # generation mode: [1, B, E]
            current_pos_embeds = pos_embeds[past_length: past_length + L_in, :, :]
            hidden_states = hidden_states + current_pos_embeds
        hidden_states = self.dropout_emb(hidden_states)

        # latent conditioning
        if source is None:
            if self.latent_conditioning is None:
                ada_params = None
            elif self.latent_conditioning == "add_once":
                hidden_states = hidden_states + latent.unsqueeze(0)  # [L_in, B, E]
                ada_params = None
            elif self.latent_conditioning.startswith("adaln"):
                ada_params = self.ada_ln_projection(latent) # [B, n_layer*mult*E]
                if self.latent_conditioning == "adaln-zero":
                    ada_params = ada_params.view(B, self.n_layer, 6*self.nx if self.cross_attn is None else 9*self.nx) # [B, n_layer, (6or9)*E]
                elif self.latent_conditioning == "adaln":
                    ada_params = ada_params.view(B, self.n_layer, 4*self.nx if self.cross_attn is None else 6*self.nx) # [B, n_layer, (4or6)*E]
        else:
            ada_params = None

        # Transformer Blocks
        next_key_values = []  # Stores the kv cache from each block
        for i, block in enumerate(self.h):
            layer_past_i = past_key_values[i] if past_key_values is not None else None

            (gamma_self, beta_self, alpha_self,
             gamma_cross, beta_cross, alpha_cross,
             gamma_ffn, beta_ffn, alpha_ffn) = self.split_adaln_params(i, ada_params)

            if self.cross_attn is None:
                memory = None
            elif self.cross_attn == "L-T":
                memory = latent
            elif self.cross_attn == "S-T":
                memory = source
                assert source is not None

            block_outputs = block(
                x=hidden_states,
                valid_token_mask=valid_token_mask,
                gamma_self=gamma_self, beta_self=beta_self, alpha_self=alpha_self,
                gamma_cross=gamma_cross, beta_cross=beta_cross, alpha_cross=alpha_cross,
                gamma_ffn=gamma_ffn, beta_ffn=beta_ffn, alpha_ffn=alpha_ffn,
                layer_past=layer_past_i, input_pos=input_pos, past_length=past_length,
                memory=memory, memory_padding_mask=encoder_mask,
                need_hidden=need_hidden
            )
            if need_hidden:
                self.all_hidden_states.append(move_to_cpu(block_outputs[0]))
                self.all_attn_score.append(move_to_cpu(block_outputs[2]))
            hidden_states = block_outputs[0]  # Output tensor [L_in, B, E]
            if use_cache:
                next_key_values.append(block_outputs[1])  # Updated cache (K, V) for layer i

        # Final Layers
        hidden_states = self.ln_f(hidden_states)
        logits = self.output_fc(hidden_states)  # [L_in, B, VocabSize]

        if need_hidden:
            self.all_hidden_states.append(move_to_cpu(logits))

        # Return logits and the updated cache list
        return logits, (next_key_values if use_cache else None)



# === HuggingFace Model Wrapper ===
@dataclass
class PanCOREModelOutput(ModelOutput):
    """Return type of :meth:`PanCoreTransformerModel.forward`.

    Which fields are populated depends on the call: ``logits`` only when a target
    was given, the VAE fields only when ``config.vae`` is set, the ``*_raw`` fields
    only during validation (where the per-phase losses are needed unreduced), and
    the cache fields only when generation asks for them.
    """

    loss: Optional[torch.FloatTensor] = None
    latent: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

    # VAE outputs
    mu: Optional[torch.FloatTensor] = None
    log_var: Optional[torch.FloatTensor] = None
    celoss: Optional[torch.FloatTensor] = None
    klloss: Optional[torch.FloatTensor] = None

    # Validation outputs (used for calculate each phase loss separately)
    celoss_raw: Optional[torch.FloatTensor] = None
    klloss_raw: Optional[torch.FloatTensor] = None
    kl_per_dim: Optional[torch.FloatTensor] = None

    # cache for generation
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PanCOREPreTrainedModel(PreTrainedModel):
    """HuggingFace PreTrainedModel base class for the PanCORE Transformer model."""
    config_class = ModelConfig
    base_model_prefix = "pancore"

    def _init_weights(self, module):
        """
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        """
        # RoPE tables are a deterministic function of (dim, base, n_positions) and are
        # rebuilt here rather than relied on from __init__. transformers>=5 builds the
        # model on the meta device and only restores what the checkpoint carries, so a
        # buffer computed in __init__ and absent from the checkpoint is left as
        # uninitialised memory -- which surfaces as NaN latents. This hook is the one
        # place transformers calls for exactly those modules, on both 4.x and 5.x.
        if isinstance(module, RotaryEmbedding):
            with torch.no_grad():
                dim, base = module.dim, module.base
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                t = torch.arange(module.n_positions, dtype=torch.float32)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                module.inv_freq = inv_freq.to(module.inv_freq.dtype)
                module.cos_cached = emb.cos()[None, None, :, :]
                module.sin_cached = emb.sin()[None, None, :, :]
                module.max_seq_len_cached = module.n_positions
        return

class PanCoreTransformerModel(PanCOREPreTrainedModel):
    # Declared so safetensors will serialize the model when config.weight_sharing ties
    # the decoder's output projection to its embedding table. Without it, saving raises
    # "contained shared tensors ... that are mismatching the transformers base
    # configuration", and HF_Trainer.save_model logs that and carries on -- a run would
    # finish, report success and leave a final_model/ holding only config.json.
    # Listing a key that is not tied is harmless, so this needs no condition.
    _tied_weights_keys: ClassVar[list[str]] = ["decoder.output_fc.weight"]

    """HuggingFace wrapper for the PanCORE Transformer model."""
    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig instance specifying architecture and hyperparameters.
        """
        super().__init__(config)
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        if config.vae:
            self.sampling = Sampling()

        self.pad_token_id = config.pad_token_id
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.pad_token_id)

        self.post_init()
        # Built after post_init() on purpose: post_init() re-initializes every
        # submodule, which would overwrite the zero-initialized alpha gates that
        # adaln-zero depends on. Creating the projection afterwards preserves them.
        self.decoder.prepare_adaln_network(config)

        # Multiplies config.beta so the trainer can warm the KL term up. Written in
        # place every step, so a plain float would be a new graph constant each time.
        # Not persistent: it is a schedule position, not a learned value, and keeping
        # it out of the state_dict leaves existing checkpoints loadable.
        self.register_buffer("kl_beta_scale", torch.ones(()), persistent=False)

    def get_input_embeddings(self):
        """Returns the shared token-embedding table (HuggingFace API)."""
        return self.encoder.wte

    def set_input_embeddings(self, value):
        """Replaces the shared token-embedding table (HuggingFace API)."""
        self.encoder.wte = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        calculate_loss: bool = True,
        reduction: str = "mean",
        need_hidden: bool = False,
        return_dict: bool | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Union[Tuple, PanCOREModelOutput]:
        """
        Forward pass for the PanCoreTransformerModel.

        Args:
            input_ids: Encoder input token IDs. Shape [B, L]. Ignored when ``latents`` is provided.
            encoder_attention_mask: Encoder padding mask (1 = valid, 0 = pad). Shape [B, L].
            labels: Decoder target token IDs. Shape [B, L]. Required for loss computation.
            decoder_attention_mask: Decoder padding mask (1 = valid, 0 = pad). Shape [B, L].
            latents: Pre-computed latent vectors to condition the decoder. Shape [B, E].
                     When provided, the encoder is skipped entirely.
            calculate_loss: If False, skips loss computation and returns raw logits only.
            reduction: Loss reduction mode — "mean" | "sum" | "none".
            need_hidden: If True, collects per-layer hidden states and attention scores.
            return_dict: If True, returns a ``PanCOREModelOutput``; otherwise a plain tuple.
            use_cache: Return the decoder's KV cache as ``past_key_values``. Training and
                validation pass False: they only read the loss, and the cache would keep
                every layer's K and V alive for as long as the output is held.

        Returns:
            PanCOREModelOutput (or tuple) containing loss, latent, celoss, klloss,
            celoss_raw, klloss_raw, past_key_values, hidden_states, attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Prepare data
        src = input_ids.transpose(0, 1) if input_ids is not None else None  # [L, B]
        tgt = labels.transpose(0, 1) if labels is not None else None  # [L, B]

        valid_token_mask = None
        if encoder_attention_mask is not None and src is not None:
            # encoder_attention_mask: [B, L] -> encoder_valid_mask: [B, 1, 1, L]
            valid_token_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2).bool()

        mu, log_var = None, None
        latent_vector = None
        encoder_out = None # For S-T Cross-Attention

        # 2. Encoder / Latent
        if latents is not None:
            latent_vector = latents

        elif src is not None:
            # Seq2Seq mode
            if self.config.vae:
                mu, log_var = self.encoder(src, valid_token_mask=valid_token_mask, need_hidden=need_hidden) # [B, E]
                latent_vector = self.sampling(mu, log_var)
            elif self.config.cross_attn == "S-T":
                encoder_out, encoder_mask = self.encoder(src, valid_token_mask=valid_token_mask, need_hidden=need_hidden)
            else:
                latent_vector = self.encoder(src, valid_token_mask=valid_token_mask, need_hidden=need_hidden)  # [B, E]

        # 3. Decoder
        logits = None
        kvcache = None
        valid_token_mask = None
        if decoder_attention_mask is not None and tgt is not None:
            # decoder_attention_mask: [B, L] -> decoder_valid_mask: [B, 1, 1, L]
            sliced_mask = decoder_attention_mask[:, :-1]
            valid_token_mask = sliced_mask.unsqueeze(1).unsqueeze(2).bool()

        if tgt is not None:
            # Teacher forcing mode
            decoder_input = tgt[:-1, :]

            if self.config.cross_attn == "S-T":
                logits, kvcache = self.decoder(
                    x=decoder_input,
                    source=encoder_out,
                    encoder_mask=encoder_mask,
                    valid_token_mask=valid_token_mask,
                    need_hidden=need_hidden,
                    use_cache=use_cache,
                )
            else:
                logits, kvcache = self.decoder(
                    x=decoder_input,
                    latent=latent_vector,
                    valid_token_mask=valid_token_mask,
                    need_hidden=need_hidden,
                    use_cache=use_cache,
                )
        # logits shape: [L, Batch, Vocab]

        # 4. Compute Loss
        total_loss = None
        celoss = None
        klloss = None
        celoss_raw = None
        klloss_raw = None
        kl_per_dim = None

        if labels is not None and logits is not None and calculate_loss:
            shift_logits = logits.transpose(-2, -1)  # [L-1, Vocab, B]
            shift_labels = tgt[1:, :] # [L-1, B]
            valid_tokens = (shift_labels != self.pad_token_id).sum().clamp(min=1)

            self.loss_fct.reduction = reduction

            # Cross Entropy Loss
            if reduction == "mean":
                celoss = self.loss_fct(shift_logits, shift_labels)
                total_loss = celoss
            elif reduction == "none":
                # celoss(raw) shape: [L-1, B], zero at the ignored padding positions.
                celoss_raw = self.loss_fct(shift_logits, shift_labels)
                # Divide by the real tokens, not by every position. A plain .mean()
                # here counts the padded positions as zero-loss tokens, so the scalar
                # comes out below the "mean" branch's by the padding fraction -- and
                # that fraction differs per length bucket, which makes the number
                # incomparable across buckets as well as against the training loss.
                total_loss = celoss_raw.sum() / valid_tokens

            # VAE KL Loss
            if self.config.vae and mu is not None and log_var is not None:
                # Divide by the tokens a molecule actually has, not by the latent
                # dimension. The reconstruction term above is a per-token mean, the
                # KL is per molecule, so a constant divisor leaves the effective
                # weight on the KL proportional to sequence length: beta*L/E rather
                # than beta. Under the bucketed curriculum that swings the objective
                # by an order of magnitude between consecutive optimizer steps.
                # Dividing by the token count makes this loss exactly the per-molecule
                # ELBO divided by the number of target tokens, so `beta` means
                # "nats of KL allowed per token of reconstruction" at every length.
                tokens_per_mol = (valid_tokens / shift_labels.size(1)).clamp(min=1.0)
                beta = self.config.beta * self.kl_beta_scale
                if reduction == "mean":
                    klloss = KLLoss(mu, log_var)
                    total_loss = total_loss + beta * (klloss / tokens_per_mol)
                elif reduction == "none":
                    klloss_raw = KLLoss(mu, log_var, reduction="none")  # [B]
                    total_loss = total_loss + beta * (klloss_raw.mean() / tokens_per_mol)
                kl_per_dim = kl_per_dimension(mu, log_var)  # [E]

        # 5. Format Output
        if logits is not None:
            logits = logits.transpose(0, 1)  # [B, L, Vocab]

        # 6. hidden layer outputs
        all_layer_hidden_states = None
        all_layer_attentions = None

        if need_hidden:
            all_layer_hidden_states = {
                "encoder": self.encoder.all_hidden_states,
                "decoder": self.decoder.all_hidden_states
            }
            all_layer_attentions = {
                "encoder": self.encoder.all_attn_score,
                "decoder": self.decoder.all_attn_score
            }

        if not return_dict:
            output = (latent_vector, logits,)
            if total_loss is not None:
                output = (total_loss, *output)
            return output
        else:
            return PanCOREModelOutput(
            loss=total_loss,
            latent=latent_vector,
            logits=logits,
            mu=mu,
            log_var=log_var,
            celoss=celoss,
            klloss=klloss,
            celoss_raw=celoss_raw,
            klloss_raw=klloss_raw,
            kl_per_dim=kl_per_dim,
            past_key_values=kvcache,
            hidden_states=all_layer_hidden_states,
            attentions=all_layer_attentions,
            )

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = False,
    ) -> torch.FloatTensor:
        """
        Encoding method for the PanCoreTransformerModel.

        Args:
            input_ids: Input token IDs for the encoder. Shape [B, L]
            encoder_attention_mask: Attention mask for the encoder. Shape [B, L]
            use_bf16: Whether to use bfloat16 precision during encoding.
        Returns:
            torch.Tensor: Encoded latent vectors. Shape [B, E]
        """
        valid_token_mask = None
        if encoder_attention_mask is not None:
            # encoder_attention_mask: [B, L] -> encoder_valid_mask: [B, 1, 1, L]
            valid_token_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2).bool()

        x = input_ids.transpose(0, 1)  # [B, L] -> [L, B]
        device = x.device

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if self.config.vae:
                mu, _ = self.encoder(x, valid_token_mask=valid_token_mask)
                return mu  # [B, E]
            elif self.config.cross_attn == "S-T":
                raise ValueError("Encoding with S-T cross-attention is not supported in this method.")
            else:
                latent = self.encoder(x, valid_token_mask=valid_token_mask)
                return latent  # [B, E]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = False,
        need_logits: bool = False,
        **kwargs,
    ):
        """
        Autoregressive greedy generation.

        Args:
            input_ids: Encoder input token IDs. Shape [B, L].
                       Mutually exclusive with ``latents`` — provide one or the other.
            max_length: Maximum length of the generated sequence (including BOS).
            latents: Pre-computed latent vectors. Shape [B, E]. Skips encoding when provided.
            encoder_attention_mask: Encoder padding mask. Shape [B, L].
            use_bf16: If True, runs decoder forward passes under ``torch.bfloat16`` autocast.
            need_logits: If True, collects and returns per-step logits.

        Returns:
            Tuple[generated_ids, latents, logits]:
                - generated_ids: Shape [B, L_out].
                - latents: Encoded latent vectors [B, E], or None when ``latents`` was provided.
                - logits: Shape [B, L_out-1, Vocab] if ``need_logits`` is True, else None.
        """
        max_length = max_length if max_length is not None else self.config.max_position_embeddings
        valid_token_mask = None

        if latents is not None:
            mode="generate"
            x = latents  # [B, E]

        elif input_ids is not None:
            mode="convert"
            x = input_ids.transpose(0, 1)  # [B, L] -> [L, B]
            if encoder_attention_mask is not None:
                # encoder_attention_mask: [B, L] -> encoder_valid_mask: [B, 1, 1, L]
                valid_token_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2).bool()

        else:
            raise ValueError("Either input_ids or latents must be provided for generation.")

        token_ids, latents, logits = self._greedy_decode(mode, x, maxlen=max_length, use_bf16=use_bf16, need_logits=need_logits, valid_token_mask=valid_token_mask)

        return token_ids.transpose(0, 1), latents, logits.transpose(0, 1) if logits is not None else None  # [L, B] -> [B, L]

    def _greedy_decode(self, mode:Literal["convert", "generate"], x:torch.Tensor,  maxlen:int, use_bf16: bool, need_logits: bool, valid_token_mask = None):
        """
        Internal greedy decode loop shared by ``generate()``.

        Args:
            mode: ``"convert"`` — encodes ``x`` (token IDs [L, B]) then decodes.
                  ``"generate"`` — decodes directly from ``x`` (latent [B, E]).
            x: Token IDs [L, B] in convert mode; latent vectors [B, E] in generate mode.
            maxlen: Maximum number of tokens to generate (including BOS).
            use_bf16: Runs the decoder under bfloat16 autocast when True.
            need_logits: Accumulates and returns per-step logits when True.
            valid_token_mask: Encoder padding mask forwarded to the encoder. Shape [B, 1, 1, L].

        Returns:
            Tuple[token_ids, latent, all_logits]:
                - token_ids: Generated token IDs. Shape [maxlen (or shorter), B].
                - latent: Encoded latent [B, E] in convert mode, None in generate mode.
                - all_logits: Stacked logits [L_out-1, B, Vocab] if ``need_logits`` else None.
        """
        device = x.device

        # --- Encoding ---
        if mode == "convert":
            # x: [L, B]
            batch_size = x.size(1)
            if self.config.vae:
                latent, _ = self.encoder(x, valid_token_mask=valid_token_mask)
                source, encoder_mask = None, None
            elif self.config.cross_attn == "S-T":
                source, encoder_mask = self.encoder(x, valid_token_mask=valid_token_mask)
                latent = None
            else:
                latent = self.encoder(x, valid_token_mask=valid_token_mask)
                source, encoder_mask = None, None
        else:
            # x: [B, E] (Latent)
            batch_size = x.size(0)
            latent = x
            source, encoder_mask = None, None

        # --- Decoding Setup ---
        # Initialize output buffer [L, B]
        token_ids = torch.full((maxlen, batch_size), fill_value=self.config.pad_token_id, dtype=torch.long, device=device)
        token_ids[0, :] = self.config.bos_token_id

        # Static KV Cache Initialization
        num_layers = self.config.n_layer
        num_heads = self.config.n_head
        head_dim = self.config.embedding_dim // num_heads
        cache_shape = (batch_size, num_heads, maxlen, head_dim)

        # Use model's dtype for cache
        dtype = torch.bfloat16 if use_bf16 else self.dtype

        # Pre-allocate cache
        kv_cache = [
            (torch.zeros(cache_shape, dtype=dtype, device=device),
             torch.zeros(cache_shape, dtype=dtype, device=device))
            for _ in range(num_layers)
        ]

        # Precompute positions as a static tensor (no new tensor alloc in loop)
        positions = torch.arange(maxlen, device=device, dtype=torch.long)  # [maxlen]

        # causal mask
        causal_mask_all = torch.tril(torch.ones(maxlen, maxlen, device=device, dtype=torch.bool))

        # finished mask as int (0/1) to allow branchless arithmetic updates
        finished = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: running, 1: finished

        all_logits = [] if need_logits else None

        # Choose the step function once (closure) to avoid per-step branching
        if self.config.cross_attn == "S-T":
            def step_fn(curr_token, past_kv, pos, mask, past_len):
                """One decoder step cross-attending to the encoder states."""
                return self.decoder(x=curr_token, past_key_values=past_kv, input_pos=pos, past_length=past_len, valid_token_mask=mask, source=source, encoder_mask=encoder_mask)
        else:
            def step_fn(curr_token, past_kv, pos, mask, past_len):
                """One decoder step conditioned on the latent vector."""
                return self.decoder(x=curr_token, past_key_values=past_kv, input_pos=pos, past_length=past_len, valid_token_mask=mask, latent=latent)

        # --- Decoding Loop ---
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            for i in range(1, maxlen):
                input_pos = positions[i-1]

                # Current token [1, B]
                curr_token = token_ids[i-1:i, :]

                # Current causal mask (1, 1, 1, maxlen)
                step_mask = causal_mask_all[input_pos].view(1, 1, 1, maxlen)

                # Decoder Forward
                # i-1 is what `input_pos` holds, so handing it over keeps the decoder
                # from reading the tensor back and stalling on the device every layer.
                logits, kv_cache = step_fn(curr_token, kv_cache, input_pos, step_mask, i - 1)  # logits: [1, B, V]

                # Store logits if needed
                if need_logits:
                    all_logits.append(logits)

                # Greedy search
                # logits: [1, B, Vocab]
                next_token = logits[0].argmax(dim=-1) # [B]

                # Branchless update for finished sequences:
                # finished is 0 or 1; make new_finished = (next_token == eos).long()
                new_finished = ((next_token == self.config.eos_token_id) | (next_token == self.config.pad_token_id)).to(torch.long)
                # mask_running = 1 if running, 0 if finished
                mask_running = 1 - finished
                # uses broadcasting and integer arithmetic (no python branch)
                effective_next = next_token * mask_running + (self.config.pad_token_id * (1 - mask_running))
                token_ids[i, :] = effective_next

                # update finished: once finished -> stays 1
                finished = torch.maximum(finished, new_finished)

                # EarlyStop
                if bool(finished.all()):
                    token_ids = token_ids[:i+1, :]  # shorten for return
                    break

        if need_logits:
            all_logits = torch.cat(all_logits, dim=0)  # [L, B, Vocab]

        return token_ids, latent if mode == "convert" else None, all_logits
