"""Regression tests for core transformer model behavior."""

import pytest
import torch

from pan_core.config import ModelConfig
from pan_core.models.transformer.layer import Sampling
from pan_core.models.transformer.model import Encoder, PanCoreTransformerModel


def test_sampling_uses_input_device():
    """VAE sampling must create noise on the input tensor's device."""
    mu = torch.zeros(2, 3, device="meta")
    log_var = torch.zeros_like(mu)

    output = Sampling()(mu, log_var)

    assert output.device == mu.device
    assert output.shape == mu.shape


def test_latent_noising_uses_randn_like(monkeypatch):
    """Encoder latent noise must be derived from the latent tensor itself."""
    config = ModelConfig(
        vocab_size=20,
        embedding_dim=8,
        n_layer=1,
        n_head=2,
        n_positions=16,
        dropout=0.0,
        pooling_mode="MHA",
        cross_attn=None,
        latent_noising=True,
        tanh_latent=False,
    )
    encoder = Encoder(config)
    encoder.train()

    calls = []
    original_randn_like = torch.randn_like

    def tracked_randn_like(tensor):
        calls.append(tensor)
        return original_randn_like(tensor)

    monkeypatch.setattr(torch, "randn_like", tracked_randn_like)

    input_ids = torch.randint(1, config.vocab_size, (4, 2))
    valid_mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
    latent = encoder(input_ids, valid_mask)

    assert len(calls) == 1
    assert calls[0].device == latent.device
    assert calls[0].shape == latent.shape


def test_forward_without_labels_returns_no_cache():
    """Skipping the decoder must return a defined empty cache."""
    config = ModelConfig(
        vocab_size=20,
        embedding_dim=8,
        n_layer=1,
        n_head=2,
        n_positions=16,
        dropout=0.0,
        pooling_mode="MHA",
        cross_attn="L-T",
        latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config)
    model.eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 4))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

    assert output.past_key_values is None
    assert output.latent.shape == (2, config.embedding_dim)


def test_forward_with_labels_still_returns_cache():
    """The existing teacher-forcing path must continue to return its cache."""
    config = ModelConfig(
        vocab_size=20,
        embedding_dim=8,
        n_layer=1,
        n_head=2,
        n_positions=16,
        dropout=0.0,
        pooling_mode="MHA",
        cross_attn="L-T",
        latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config)
    model.eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 4))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            labels=input_ids,
            decoder_attention_mask=attention_mask,
            return_dict=True,
        )

    assert output.past_key_values is not None
    assert output.loss is not None


def test_forward_returns_logits_in_both_return_modes():
    """`return_dict=True` used to drop the logits it had just computed.

    `PanCOREModelOutput` carried no `logits` field, so `model(...).logits` raised
    AttributeError while the tuple path returned them.
    """
    config = ModelConfig(
        vocab_size=20,
        embedding_dim=8,
        n_layer=1,
        n_head=2,
        n_positions=16,
        dropout=0.0,
        pooling_mode="MHA",
        cross_attn="L-T",
        latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config)
    model.eval()
    seq_len, batch = 6, 2
    src = torch.randint(3, config.vocab_size, (seq_len, batch))
    src[0], src[-1] = 1, 2
    mask = torch.ones(batch, seq_len, dtype=torch.bool)

    with torch.no_grad():
        as_dict = model(input_ids=src.T, labels=src.T, encoder_attention_mask=mask,
                        decoder_attention_mask=mask, return_dict=True)
        as_tuple = model(input_ids=src.T, labels=src.T, encoder_attention_mask=mask,
                         decoder_attention_mask=mask, return_dict=False)

    assert as_dict.logits is not None
    assert as_dict.logits.shape == (batch, seq_len - 1, config.vocab_size)
    assert torch.equal(as_dict.logits, as_tuple[-1])
    # loss must stay the first element, which HuggingFace's Trainer relies on
    assert torch.equal(as_dict[0], as_dict.loss)


@pytest.mark.parametrize("real_len, alloc_len", [(16, 16), (16, 32), (16, 64), (16, 128)])
def test_reduction_none_matches_mean_regardless_of_padding(real_len, alloc_len):
    """The two reduction modes must report the same scalar loss.

    `reduction="none"` used to finish with `celoss_raw.mean()`, which counts the
    ignored padding positions as zero-loss tokens. The validation loss then came out
    below the training loss by the padding fraction -- and that fraction differs per
    length bucket, so the number was not comparable across buckets either. At 87.5%
    padding it understated the loss by 8.5x.
    """
    config = ModelConfig(
        vocab_size=20, embedding_dim=8, n_layer=1, n_head=2, n_positions=256,
        dropout=0.0, pooling_mode="MHA", cross_attn="L-T",
        latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config)
    model.eval()

    batch = 4
    torch.manual_seed(0)
    tokens = torch.full((alloc_len, batch), config.pad_token_id, dtype=torch.long)
    body = torch.randint(3, config.vocab_size, (real_len, batch))
    body[0], body[-1] = 1, 2
    tokens[:real_len] = body
    mask = torch.zeros(batch, alloc_len, dtype=torch.bool)
    mask[:, :real_len] = True

    with torch.no_grad():
        as_mean = model(input_ids=tokens.T, labels=tokens.T, encoder_attention_mask=mask,
                        decoder_attention_mask=mask, reduction="mean", return_dict=True)
        as_none = model(input_ids=tokens.T, labels=tokens.T, encoder_attention_mask=mask,
                        decoder_attention_mask=mask, reduction="none", return_dict=True)

    assert as_none.loss.item() == pytest.approx(as_mean.loss.item(), rel=1e-5)


@pytest.mark.parametrize("weight_sharing", [True, False])
def test_model_round_trips_through_save_pretrained(tmp_path, weight_sharing):
    """`weight_sharing: True` used to make the model unsaveable.

    Tying `decoder.output_fc.weight` to `decoder.wte.weight` without telling
    HuggingFace made safetensors refuse the whole state dict ("contained shared
    tensors ... mismatching the transformers base configuration"), and
    `HF_Trainer.save_model` logged that and carried on -- a run finished, reported
    success and left an output directory holding only config.json.
    """
    config = ModelConfig(
        vocab_size=40, embedding_dim=16, n_layer=1, n_head=2, n_positions=32,
        dropout=0.0, weight_sharing=weight_sharing, pooling_mode="MHA",
        cross_attn="L-T", latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config)
    model.eval()
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists(), "weights were not written"

    reloaded = PanCoreTransformerModel.from_pretrained(tmp_path)
    reloaded.eval()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), f"{name} differs"

    # The tie itself must survive the round trip, or the output projection silently
    # becomes an independent, randomly initialised matrix.
    shared = (reloaded.decoder.output_fc.weight.data_ptr()
              == reloaded.decoder.wte.weight.data_ptr())
    assert shared is weight_sharing

    seq_len, batch = 6, 2
    src = torch.randint(3, config.vocab_size, (seq_len, batch))
    src[0], src[-1] = 1, 2
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    with torch.no_grad():
        before = model(input_ids=src.T, labels=src.T, encoder_attention_mask=mask,
                       decoder_attention_mask=mask, return_dict=True)
        after = reloaded(input_ids=src.T, labels=src.T, encoder_attention_mask=mask,
                         decoder_attention_mask=mask, return_dict=True)
    assert torch.equal(before.logits, after.logits)


def test_use_cache_false_returns_no_cache_and_the_same_loss():
    """Training and validation drop the KV cache; the loss must not move."""
    config = ModelConfig(
        vocab_size=20, embedding_dim=8, n_layer=2, n_head=2, n_positions=16, dropout=0.0,
        pooling_mode="MHA", cross_attn="L-T", latent_conditioning="adaln-zero",
    )
    model = PanCoreTransformerModel(config).eval()
    ids = torch.randint(3, config.vocab_size, (3, 7))
    ids[:, 0] = 1
    mask = torch.ones_like(ids, dtype=torch.bool)
    kw = dict(input_ids=ids, labels=ids, encoder_attention_mask=mask,
              decoder_attention_mask=mask, return_dict=True)
    with torch.no_grad():
        cached = model(**kw)
        uncached = model(**kw, use_cache=False)

    assert cached.past_key_values is not None and len(cached.past_key_values) == config.n_layer
    assert uncached.past_key_values is None
    assert torch.equal(cached.loss, uncached.loss)
    assert torch.equal(cached.logits, uncached.logits)
