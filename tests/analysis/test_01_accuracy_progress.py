"""Tests for analysis 01 (accuracy progress): free-running and teacher-forced token accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pan_core.common.data_handler import GeneralDataCollator, OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.config import ModelConfig
from pan_core.models import PanCoreTransformerModel
from tests.utils import load_script

_ACCURACY = load_script("analysis/scripts/compute/01_accuracy_progress.py")


class _CharTokenizer:
    def get_vocab(self):
        return {token: index for index, token in enumerate("CNO()=@")}

    def tokenize(self, text):
        return list(text)


def test_token_accuracy_is_position_aligned_over_the_shorter_sequence():
    result = _ACCURACY.compute_token_accuracy(["CCO", "CN"], ["CCN", "CNO"], _CharTokenizer())

    assert result["C"] == 1.0          # 3 compared C positions, all correct
    assert result["N"] == 0.5          # pair 1 position 2 wrong, pair 2 position 1 right
    assert np.isnan(result["O"])       # target O of pair 2 lies beyond the shorter generation
    assert np.isnan(result["@"])


# ── teacher-forced token accuracy ────────────────────────────────────────────

def _tiny_model_and_dataset(tmp_path, cross_attn, conditioning):
    vocab = ["<pad>", "<s>", "</s>", "<unk1>", "C", "N", "O", "@", "@@", "[", "]", "H", "(", ")"]
    vocab_path = tmp_path / "normal_tokens.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    tokenizer = SMILESTokenizer(vocab_path)
    pd.DataFrame({
        "random": ["CCO", "OCC", "N[C@H](C)O", "C[C@@H](N)O", "CCN", "NCC"],
        "canonical": ["CCO", "CCO", "C[C@@H](N)O", "C[C@@H](N)O", "CCN", "CCN"],
    }).to_csv(tmp_path / "eval.csv")
    dataset = OnTheFlyDataset(file_path_or_df=tmp_path / "eval.csv", tokenizer=tokenizer,
                              src_col_name="random", tgt_col_name="canonical", mode="eval")
    config = ModelConfig(
        vocab_size=len(vocab), embedding_dim=16, n_layer=2, n_head=2, n_positions=32, tanh_latent=True,
        latent_noising=False, dropout=0.0, bias=True, weight_sharing=False, layer_norm_epsilon=1e-5,
        pooling_mode="MHA", cross_attn=cross_attn, latent_conditioning=conditioning,
    )
    torch.manual_seed(0)
    return PanCoreTransformerModel(config).eval(), dataset, tokenizer


@pytest.mark.parametrize("cross_attn, conditioning", [(None, "add_once"), ("L-T", "adaln-zero")])
def test_teacher_forced_logits_match_the_model_loss(tmp_path, cross_attn, conditioning):
    model, dataset, _ = _tiny_model_and_dataset(tmp_path, cross_attn, conditioning)
    batch = GeneralDataCollator()([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        logits = _ACCURACY.teacher_forced_logits(model, batch)                    # [L-1, B, V]
        reference = model(input_ids=batch["input_ids"], encoder_attention_mask=batch["encoder_attention_mask"],
                          labels=batch["labels"], decoder_attention_mask=batch["decoder_attention_mask"]).loss
    target = batch["labels"].transpose(0, 1)[1:]
    loss = torch.nn.functional.cross_entropy(logits.transpose(-2, -1), target, ignore_index=0)
    assert torch.allclose(loss, reference, atol=1e-6)


def test_teacher_forced_token_accuracy_counts_every_target_token(tmp_path, monkeypatch):
    model, dataset, tokenizer = _tiny_model_and_dataset(tmp_path, None, "add_once")
    targets = pd.read_csv(tmp_path / "eval.csv")["canonical"].tolist()
    expected = _ACCURACY.target_token_counts(targets, tokenizer, n_phases=3)
    vocab = tokenizer.get_vocab()
    assert expected[1, vocab["@@"]] == 2 and expected[0, vocab["O"]] == 2

    def perfect(model, batch):  # logits whose argmax is the next target token
        target = batch["labels"].transpose(0, 1)[1:]
        return torch.nn.functional.one_hot(target, num_classes=len(vocab)).float()

    monkeypatch.setattr(_ACCURACY, "teacher_forced_logits", perfect)
    rows = _ACCURACY.teacher_forced_token_accuracy(model, dataset, tokenizer, torch.device("cpu"), expected,
                                                   batch_size=4)
    assert len(rows) == 3
    assert rows[1]["@@"] == 1.0 and rows[0]["C"] == 1.0
    assert np.isnan(rows[0]["@@"])            # no @@ in bucket 1 targets

    def shifted(model, batch):  # always predicts the token after the next one -> wrong almost everywhere
        target = batch["labels"].transpose(0, 1)[1:]
        return torch.nn.functional.one_hot(torch.roll(target, 1, dims=0), num_classes=len(vocab)).float()

    monkeypatch.setattr(_ACCURACY, "teacher_forced_logits", shifted)
    rows = _ACCURACY.teacher_forced_token_accuracy(model, dataset, tokenizer, torch.device("cpu"), expected,
                                                   batch_size=4)
    assert rows[1]["@@"] == 0.0


def test_teacher_forced_token_accuracy_rejects_count_mismatch(tmp_path):
    model, dataset, tokenizer = _tiny_model_and_dataset(tmp_path, None, "add_once")
    targets = pd.read_csv(tmp_path / "eval.csv")["canonical"].tolist()
    wrong = _ACCURACY.target_token_counts(targets, tokenizer, n_phases=3)
    wrong[0, 4] += 1
    with pytest.raises(ValueError, match="differ"):
        _ACCURACY.teacher_forced_token_accuracy(model, dataset, tokenizer, torch.device("cpu"), wrong, batch_size=4)
