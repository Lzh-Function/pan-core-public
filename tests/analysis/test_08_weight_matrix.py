"""Tests for analysis 08 (weight matrices): weight keys derived from the checkpoint structure."""

from __future__ import annotations

import pytest

from tests.utils import load_script

_WEIGHT_MATRIX = load_script("analysis/scripts/compute/08_weight_matrix.py")


def _candidate_state_keys(n_layer: int) -> set[str]:
    keys = {
        "encoder.pooling.proj.weight",
        "encoder.pooling.queries",
        "decoder.ada_ln_projection.weight",
        "decoder.output_fc.weight",
        "encoder.wte.weight",
        "decoder.wte.weight",
    }
    for layer in range(n_layer):
        for side in ("encoder", "decoder"):
            keys.add(f"{side}.h.{layer}.attn.c_proj.weight")
            for name in ("c_proj", "v", "w1", "w2"):
                keys.add(f"{side}.h.{layer}.mlp.{name}.weight")
        for name in ("q_proj", "kv_proj", "c_proj"):
            keys.add(f"decoder.h.{layer}.cross_attn.{name}.weight")
    return keys


def _architecture_config(**overrides):
    config = {
        "architectures": ["PanCoreTransformerModel"],
        "embedding_dim": 16,
        "n_layer": 2,
        "n_head": 2,
        "pooling_mode": "concat",
        "gpt2style": True,
        "cross_attn": None,
        "latent_conditioning": "add_once",
    }
    config.update(overrides)
    return config


def test_weight_keys_are_derived_for_gpt2style_checkpoint():
    config = _architecture_config()
    keys = _WEIGHT_MATRIX.derive_important_keys(config, _candidate_state_keys(2))

    assert "encoder.pooling.proj.weight" in keys
    assert "encoder.h.0.mlp.c_proj.weight" in keys
    assert "encoder.h.0.mlp.v.weight" not in keys
    assert not any("cross_attn" in key for key in keys)


def test_weight_keys_are_derived_for_cross_attention_checkpoint():
    config = _architecture_config(
        pooling_mode="MHA",
        gpt2style=False,
        cross_attn="L-T",
        latent_conditioning="adaln-zero",
    )
    keys = _WEIGHT_MATRIX.derive_important_keys(config, _candidate_state_keys(2))

    assert "encoder.pooling.queries" in keys
    assert "encoder.h.0.mlp.v.weight" in keys
    assert "decoder.h.0.cross_attn.q_proj.weight" in keys
    assert "decoder.ada_ln_projection.weight" in keys


def test_weight_key_derivation_fails_when_checkpoint_state_disagrees():
    with pytest.raises(ValueError, match="missing from checkpoint"):
        _WEIGHT_MATRIX.derive_important_keys(_architecture_config(), set())
