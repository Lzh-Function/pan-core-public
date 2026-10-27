"""Tests for YAML training configuration loading and validation."""

import re
from pathlib import Path

import pytest
import yaml

from pan_core.config import load_config
from tests.utils import REPO_ROOT


def test_sample_config_loads(tmp_path: Path):
    """sample/config.yml must load as shipped (the training data paths are blank).

    Two edits are made to the shipped file. ``use_bf16`` is switched off because
    Hugging Face rejects bf16 without a supporting GPU. ``token_file_path`` is made
    absolute because the shipped value is relative to the repository root, while
    this test writes the config into a temporary directory.
    """
    config = yaml.safe_load((REPO_ROOT / "sample" / "config.yml").read_text())
    config["system"]["use_bf16"] = False
    config["training"]["data"]["token_file_path"] = str(
        REPO_ROOT / "sample" / "data" / "normal_tokens.txt")
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    model_config, training_args = load_config(str(config_path), str(tmp_path / "output"))

    assert (model_config.cross_attn, model_config.latent_conditioning, model_config.pooling_mode) == (
        "L-T", "adaln-zero", "MHA")
    assert training_args.training_strategy_type == "dynamic"
    assert training_args.max_steps == 700000
    assert training_args.dynamic_start_probs == [0.4, 0.4, 0.1, 0.05, 0.05]
    assert training_args.eval_steps == 100 and training_args.per_device_eval_batch_size == 1024


@pytest.mark.parametrize("max_steps", [None, 0, -1, 1.5])
def test_dynamic_strategy_rejects_invalid_max_steps(tmp_path: Path, max_steps):
    """Dynamic sampling requires a positive integer optimizer-step budget."""
    dynamic = {} if max_steps is None else {"max_steps": max_steps}
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "system": {"environment": "normal"},
                "training": {
                    "strategy": {
                        "type": "dynamic",
                        "num_epochs": 4,
                        "dynamic": dynamic,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=re.escape("dynamic.max_steps must be a positive integer")):
        load_config(str(config_path), str(tmp_path / "output"))


def test_vae_with_source_target_cross_attention_is_rejected(tmp_path: Path):
    """S-T never pools a latent, so the VAE head has nothing to parameterise.

    Without this check the run dies several frames later on a tensor-shape mismatch
    that says nothing about the configuration.
    """
    config = yaml.safe_load((REPO_ROOT / "sample" / "config.yml").read_text())
    config["system"]["use_bf16"] = False
    config["training"]["data"]["token_file_path"] = str(
        REPO_ROOT / "sample" / "data" / "normal_tokens.txt")
    config["model"]["use_vae"] = True
    config["model"]["Transformer"]["latent"]["cross_attn"] = "S-T"
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        load_config(str(config_path), str(tmp_path / "output"))


def test_scalar_where_a_section_belongs_names_the_key(tmp_path: Path):
    """`rope: true` instead of `rope: {encoder: true}` used to surface as
    `AttributeError: 'bool' object has no attribute 'get'`, naming neither the key
    nor the file."""
    config = yaml.safe_load((REPO_ROOT / "sample" / "config.yml").read_text())
    config["system"]["use_bf16"] = False
    config["training"]["data"]["token_file_path"] = str(
        REPO_ROOT / "sample" / "data" / "normal_tokens.txt")
    config["model"]["Transformer"]["rope"] = True
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape("model.Transformer.rope must be a mapping")):
        load_config(str(config_path), str(tmp_path / "output"))


def _sample(tmp_path: Path, mutate=None):
    """sample/config.yml, runnable on CPU, with an optional edit applied."""
    config = yaml.safe_load((REPO_ROOT / "sample" / "config.yml").read_text())
    config["system"]["use_bf16"] = False
    config["training"]["data"]["token_file_path"] = str(
        REPO_ROOT / "sample" / "data" / "normal_tokens.txt")
    if mutate is not None:
        mutate(config)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_sample_config_has_no_unread_keys(tmp_path: Path):
    """Every key the shipped sample carries must be one the loader reads."""
    load_config(str(_sample(tmp_path)), str(tmp_path / "output"))


@pytest.mark.parametrize("section, correct, wrong", [
    (("model",), "Transformer", "transformer"),
    (("model",), "common", "COMMON"),
    (("model", "Transformer"), "latent", "Latent"),
    (("training",), "optimizer", "Optimizer"),
])
def test_section_keys_are_case_insensitive(tmp_path: Path, section, correct, wrong):
    """The schema mixes cases (`Transformer` beside `common`), and a wrong guess used
    to leave the whole section at its defaults without a word."""
    def mutate(config):
        # Values first: the rename below may be the very section they live in.
        config["model"]["Transformer"]["n_layer"] = 3
        config["model"]["common"]["embedding_dim"] = 64
        node = config
        for key in section:
            node = node[key]
        node[wrong] = node.pop(correct)

    model_config, _ = load_config(str(_sample(tmp_path, mutate)), str(tmp_path / "output"))
    assert (model_config.n_layer, model_config.embedding_dim) == (3, 64)


@pytest.mark.parametrize("section, correct, wrong, expected", [
    (("model", "common"), "embedding_dim", "embeding_dim", "embedding_dim"),
    (("model", "Transformer"), "n_layer", "n_layers", "n_layer"),
    (("training", "data"), "batch_size", "batchsize", "batch_size"),
    (("model", "loss"), "beta_warmup_steps", "beta_warmup", "beta_warmup_steps"),
])
def test_a_misspelled_key_is_rejected_with_a_suggestion(tmp_path, section, correct, wrong, expected):
    """An unread key is not harmless: the setting stays at its default and the run
    proceeds, so the mistake costs a whole training run before anyone notices."""
    def mutate(config):
        node = config
        for key in section:
            node = node[key]
        node[wrong] = node.pop(correct)

    with pytest.raises(ValueError, match=re.escape(f"{wrong} was never read")) as excinfo:
        load_config(str(_sample(tmp_path, mutate)), str(tmp_path / "output"))
    assert f"Did you mean {expected!r}?" in str(excinfo.value)


@pytest.mark.parametrize("strategy", ["normal", "phased", "dynamic"])
def test_inactive_strategy_blocks_are_not_flagged(tmp_path: Path, strategy):
    """All three blocks may be present; only the selected one is in use."""
    def mutate(config):
        config["training"]["strategy"]["type"] = strategy

    load_config(str(_sample(tmp_path, mutate)), str(tmp_path / "output"))
