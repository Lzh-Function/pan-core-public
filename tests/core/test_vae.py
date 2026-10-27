"""VAE-specific behaviour: KL weighting, warm-up, and collapse diagnostics.

Nothing here touches the non-VAE path; every assertion is about code that only runs
when ``config.vae`` is set.
"""

import pytest
import torch

from pan_core.common.utils import KLLoss, kl_per_dimension
from pan_core.config import ModelConfig
from pan_core.models import PanCoreTransformerModel

# Deliberately far above a realistic value. These tests read the KL term as
# ``loss - celoss``, a difference of two float32 numbers around 5.4; at a realistic
# beta the difference is ~1e-4 and cancellation eats four digits. A large beta makes
# the term resolvable without changing what is being tested.
BETA = 10.0
LATENT_DIM = 64


def _model(**overrides):
    config = ModelConfig(
        vocab_size=185, embedding_dim=LATENT_DIM, n_layer=2, n_head=4,
        n_positions=512, dropout=0.0, vae=True, beta=BETA,
        pooling_mode="MHA", cross_attn="L-T", latent_conditioning="adaln",
        **overrides)
    model = PanCoreTransformerModel(config)
    model.eval()
    return model


def _forward(model, seq_len, batch=4):
    torch.manual_seed(1)
    src = torch.randint(3, model.config.vocab_size, (seq_len, batch))
    src[0], src[-1] = 1, 2
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    with torch.no_grad():
        return model(input_ids=src.T, labels=src.T, encoder_attention_mask=mask,
                     decoder_attention_mask=mask, return_dict=True)


@pytest.mark.parametrize("seq_len", [32, 66, 130, 258])
def test_effective_beta_does_not_depend_on_sequence_length(seq_len):
    """`beta` must mean the same thing at every length.

    The reconstruction term is a per-token mean and the KL is per molecule, so
    dividing the KL by a constant (the latent dimension, as it used to be) leaves the
    effective weight proportional to sequence length. Under the bucketed curriculum,
    where one optimizer step draws from a single length bucket, that swung the
    objective by an order of magnitude between consecutive steps.
    """
    torch.manual_seed(0)
    model = _model()
    out = _forward(model, seq_len)

    kl_contribution = out.loss.item() - out.celoss.item()
    tokens = seq_len - 1  # shift_labels length; this batch has no padding
    effective_beta = kl_contribution * tokens / out.klloss.item()
    assert effective_beta == pytest.approx(BETA, rel=1e-3)


def test_kl_warmup_scale_multiplies_the_weight():
    """The trainer writes the warm-up position into a buffer the loss reads."""
    torch.manual_seed(0)
    model = _model()
    assert float(model.kl_beta_scale) == 1.0

    full = _forward(model, 32)
    model.kl_beta_scale.fill_(0.0)
    none_ = _forward(model, 32)
    model.kl_beta_scale.fill_(0.5)
    half = _forward(model, 32)

    assert none_.loss.item() == pytest.approx(none_.celoss.item(), abs=1e-9)
    full_contribution = full.loss.item() - full.celoss.item()
    half_contribution = half.loss.item() - half.celoss.item()
    assert half_contribution == pytest.approx(full_contribution / 2, rel=1e-4)


def test_warmup_scale_stays_out_of_the_state_dict():
    """It is a schedule position, not a weight; existing checkpoints must still load."""
    model = _model()
    assert "kl_beta_scale" not in model.state_dict()


def test_kl_per_dim_is_reported_and_sums_to_the_total():
    torch.manual_seed(0)
    model = _model()
    out = _forward(model, 32)
    assert out.kl_per_dim is not None
    assert out.kl_per_dim.shape == (LATENT_DIM,)
    assert out.kl_per_dim.sum().item() == pytest.approx(out.klloss.item(), rel=1e-4)


def test_kl_per_dimension_detects_a_collapsed_dimension():
    """The summed KL cannot tell a spread posterior from a mostly collapsed one."""
    batch, dim = 128, 8
    mu = torch.zeros(batch, dim)
    log_var = torch.zeros(batch, dim)
    mu[:, 0] = torch.randn(batch) * 3.0  # one informative dimension
    per_dim = kl_per_dimension(mu, log_var)

    assert per_dim[0] > 1.0
    assert torch.all(per_dim[1:] < 1e-6)
    assert per_dim.sum().item() == pytest.approx(KLLoss(mu, log_var).item(), rel=1e-5)


def test_non_vae_model_has_no_kl_output():
    config = ModelConfig(vocab_size=185, embedding_dim=LATENT_DIM, n_layer=2, n_head=4,
                         n_positions=64, dropout=0.0, vae=False,
                         pooling_mode="MHA", cross_attn="L-T",
                         latent_conditioning="adaln")
    model = PanCoreTransformerModel(config)
    model.eval()
    out = _forward(model, 32)
    assert out.klloss is None and out.kl_per_dim is None
    assert out.loss.item() == pytest.approx(out.celoss.item(), abs=1e-9)
