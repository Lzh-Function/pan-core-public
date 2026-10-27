"""Tests for the mutual-information compute and visualization scripts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.utils import load_script

_COMPUTE = load_script("analysis/scripts/compute/09_mutual_information.py")
_VISUALIZE = load_script("analysis/scripts/visualize/16_mutual_information.py")


def test_linear_probe_detects_separable_labels():
    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], 20)
    features = rng.normal(scale=0.1, size=(40, 4))
    features[:, 0] += labels * 3.0

    result = _COMPUTE.compute_probe_metrics(
        features,
        labels,
        n_splits=2,
        min_samples=10,
        n_shuffle=2,
        seed=7,
        n_jobs=1,
    )

    assert result["balanced_acc"] > 0.95
    assert result["roc_auc"] > 0.95
    assert np.isfinite(result["probe_mi_lower_bound"])
    assert "mi_lb_corrected" in result


def test_probe_returns_nan_when_sample_requirement_is_not_met():
    result = _COMPUTE.compute_probe_metrics(
        np.ones((4, 2)),
        np.array([0, 0, 1, 1]),
        min_samples=5,
    )

    assert result["n_samples"] == 4
    assert np.isnan(result["balanced_acc"])


def test_probe_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same sample count"):
        _COMPUTE.compute_probe_metrics(np.ones((3, 2)), np.array([0, 1]))


def test_nonlinear_probe_runs_both_classifiers():
    rng = np.random.default_rng(11)
    labels = np.repeat([0, 1], 10)
    features = rng.normal(size=(20, 3))
    features[:, 0] += labels * 2.0

    result = _COMPUTE.compute_nonlinear_probe_metrics(
        features,
        labels,
        torch.device("cpu"),
        n_splits=2,
        min_samples=10,
        mlp_epochs=2,
        rf_estimators=5,
        seed=11,
        n_jobs=1,
    )

    assert set(result) == {
        "mlp_balanced_acc",
        "mlp_roc_auc",
        "rf_balanced_acc",
        "rf_roc_auc",
        "n_samples",
    }
    assert 0.0 <= result["mlp_balanced_acc"] <= 1.0
    assert 0.0 <= result["rf_balanced_acc"] <= 1.0


def test_mean_pool_hidden_excludes_padding():
    hidden = np.array([
        [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
        [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
    ])
    input_ids = np.array([[1, 2, 0], [1, 2, 3]])

    pooled = _COMPUTE.mean_pool_hidden(hidden, input_ids, pad_id=0)

    assert np.allclose(pooled[0], [2.0, 3.0])
    assert np.allclose(pooled[1], [6.0, 8.0])


def test_chiral_window_pool_uses_window_and_fallback():
    hidden = np.arange(8, dtype=np.float32).reshape(2, 4, 1)
    input_ids = np.array([
        [1, 7, 3, 0],
        [1, 4, 2, 0],
    ])

    pooled = _COMPUTE.chiral_window_pool(
        hidden,
        input_ids,
        np.array([0, 1]),
        atmark_id=7,
        atatmark_id=8,
        pad_id=0,
        window=0,
    )

    assert np.allclose(pooled[:, 0], [1.0, 5.0])


def test_discover_checkpoints_sorts_and_filters(tmp_path):
    for name in ("checkpoint-20", "checkpoint-5", "notes"):
        (tmp_path / name).mkdir()

    all_checkpoints = _COMPUTE.discover_checkpoints(tmp_path, None)
    selected = _COMPUTE.discover_checkpoints(tmp_path, [20])

    assert [path.name for path in all_checkpoints] == ["checkpoint-5", "checkpoint-20"]
    assert [path.name for path in selected] == ["checkpoint-20"]


def test_encode_all_collects_requested_encoder_layers(tmp_path):
    from pan_core.config import ModelConfig
    from pan_core.common.tokenizer import SMILESTokenizer
    from pan_core.models import PanCoreTransformerModel

    vocab = ["<pad>", "<s>", "</s>", "<unk1>", "C", "N", "@", "@@", "[", "]", "H", "(", ")"]
    vocab_path = tmp_path / "normal_tokens.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    tokenizer = SMILESTokenizer(vocab_path)
    config = ModelConfig(
        vocab_size=len(vocab),
        embedding_dim=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        tanh_latent=True,
        latent_noising=False,
        dropout=0.0,
        bias=True,
        weight_sharing=False,
        layer_norm_epsilon=1e-5,
        pooling_mode="MHA",
        cross_attn=None,
        latent_conditioning="add_once",
    )
    model = PanCoreTransformerModel(config).eval()

    latent, hidden_by_layer, input_ids = _COMPUTE.encode_all(
        ["C[C@H](N)C", "N[C@@H](C)C", "CC"],
        tokenizer,
        model,
        torch.device("cpu"),
        [0, 1],
        tokenizer.pad_token_id,
        batch_size=2,
        max_len=32,
    )

    assert latent.shape == (3, 16)
    assert set(hidden_by_layer) == {0, 1}
    assert hidden_by_layer[0].shape == (3, input_ids.shape[1], 16)
    assert hidden_by_layer[1].shape == (3, input_ids.shape[1], 16)


def test_make_plots_writes_all_v5_figures(tmp_path):
    def probe(value):
        return {"balanced_acc": value, "mi_lb_corrected": value - 0.5}

    metrics = []
    for value in (0.55, 0.8):
        metrics.append({
            "rs_probe": probe(value),
            "rs_nonlinear_probe": {
                "mlp_balanced_acc": value + 0.02,
                "rf_balanced_acc": value + 0.01,
            },
            "canonical_token_probe": probe(value - 0.02),
            "rs_probe_by_layer": {"0": probe(value), "2": probe(value + 0.01)},
            "rs_window_probe_by_layer": {"0": probe(value), "2": probe(value + 0.02)},
        })
    results = {
        "model": {
            "model_name": "model",
            "probe_layers": [0, 2],
            "steps": [10, 20],
            "metrics": metrics,
        }
    }

    _VISUALIZE.make_plots(results, tmp_path)

    expected_stems = {
        "figure1_rs_probe",
        "figure2_rs_heatmap_model",
        "figure3_rs_heatmap_model",
        "figure6_rs_linear_vs_nonlinear",
        "figure7_stereo_probe_comparison",
        "figure7d_model",
    }
    assert {path.stem for path in tmp_path.glob("*.png")} == expected_stems
    assert {path.stem for path in tmp_path.glob("*.svg")} == expected_stems


def test_probe_summary_reports_post_transition_values():
    def probe(value):
        return {"balanced_acc": value, "mi_lb_corrected": value - 0.5}

    rs_values = [0.50, 0.55, 0.61, 0.60, 0.59]
    at_values = [0.50, 0.60, 0.69, 0.72, 0.71]
    metrics = [
        {
            "rs_probe": probe(rs),
            "canonical_token_probe": probe(at),
            "rs_nonlinear_probe": {"mlp_balanced_acc": rs + 0.08, "rf_balanced_acc": rs - 0.02},
            "rs_probe_by_layer": {"7": probe(rs + 0.03)},
            "rs_window_probe_by_layer": {"7": probe(rs + 0.1)},
        }
        for rs, at in zip(rs_values, at_values)
    ]
    result = {"steps": [10, 20, 30, 40, 50], "metrics": metrics}

    summary = _VISUALIZE.probe_summary(result, (10, 30, 20))

    assert summary["at_balanced_acc_at_end"] == 0.69
    assert summary["at_balanced_acc_at_last"] == 0.71
    assert summary["rs_balanced_acc_max"] == 0.61
    assert summary["rs_balanced_acc_max_step"] == 30
    assert summary["at_fraction_increasing_pairs_from_end"] == 0.5
    assert summary["rs_ge_at_checkpoints_from_end"] == 0
    assert summary["n_checkpoints_from_end"] == 3
    assert summary["rs_window_pool_by_layer_at_last"]["7"] == pytest.approx(0.69)
    assert summary["token_identity_min_balanced_acc_by_layer"] is None


def test_find_chiral_token_positions_labels_and_groups():
    ids = np.array([[1, 5, 2, 6, 0], [1, 2, 3, 0, 0], [6, 6, 2, 0, 0]])

    pairs, labels, groups = _COMPUTE.find_chiral_token_positions(ids, atmark_id=5, atatmark_id=6)

    assert pairs == [(0, 1), (0, 3), (2, 0), (2, 1)]
    assert labels.tolist() == [0, 1, 1, 1]
    assert groups.tolist() == [0, 0, 2, 2]


def test_token_identity_probe_keeps_groups_within_one_fold(monkeypatch):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 2)
    labels = np.tile([0, 1], 30)
    features = rng.normal(scale=0.1, size=(60, 3))
    features[:, 0] += labels * 3.0
    seen_splits = []
    original = _COMPUTE._run_cv

    def spy(X, y, cv_splits, classes, n_jobs):
        seen_splits.extend(cv_splits)
        return original(X, y, cv_splits, classes, n_jobs)

    monkeypatch.setattr(_COMPUTE, "_run_cv", spy)
    result = _COMPUTE.compute_token_identity_probe(features, labels, groups, n_splits=5, min_samples=10, n_jobs=1)

    assert result["balanced_acc"] > 0.95
    assert result["n_unique_groups"] == 30
    assert len(seen_splits) == 5
    for train, test in seen_splits:
        assert not set(groups[train]) & set(groups[test])


def test_token_identity_probe_returns_nan_with_too_few_groups():
    result = _COMPUTE.compute_token_identity_probe(
        np.ones((60, 2)), np.tile([0, 1], 30), np.repeat(np.arange(3), 20), n_splits=5, min_samples=10
    )

    assert np.isnan(result["balanced_acc"])
    assert result["n_unique_groups"] == 3


def test_token_identity_minimum_over_checkpoints():
    metrics = [
        {"token_identity_probe": {"0": {"balanced_acc": 1.0}, "7": {"balanced_acc": 0.9999}}},
        {"token_identity_probe": {"0": {"balanced_acc": 0.9998}, "7": {"balanced_acc": 1.0}}},
    ]

    assert _VISUALIZE.token_identity_minimum(metrics) == {"0": 0.9998, "7": 0.9999}
    with pytest.raises(ValueError, match="only some checkpoints"):
        _VISUALIZE.token_identity_minimum(metrics + [{"token_identity_probe": None}])


def test_analyze_checkpoint_runs_token_identity_probe(tmp_path):
    from pan_core.config import ModelConfig
    from pan_core.common.tokenizer import SMILESTokenizer
    from pan_core.models import PanCoreTransformerModel

    vocab = ["<pad>", "<s>", "</s>", "<unk1>", "C", "N", "@", "@@", "[", "]", "H", "(", ")"]
    vocab_path = tmp_path / "normal_tokens.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    tokenizer = SMILESTokenizer(vocab_path)
    config = ModelConfig(
        vocab_size=len(vocab), embedding_dim=16, n_layer=2, n_head=2, n_positions=32,
        tanh_latent=True, latent_noising=False, dropout=0.0, bias=True, weight_sharing=False,
        layer_norm_epsilon=1e-5, pooling_mode="MHA", cross_attn=None, latent_conditioning="add_once",
    )
    torch.manual_seed(0)
    PanCoreTransformerModel(config).save_pretrained(tmp_path / "checkpoint-1")
    smiles = ["C[C@H](N)C", "N[C@@H](C)C"] * 10
    labels = np.tile([0, 1], 10)

    result = _COMPUTE.analyze_checkpoint(
        tmp_path / "checkpoint-1", smiles, tokenizer, labels, labels, [0, 1], torch.device("cpu"),
        {"n_splits": 2, "min_samples": 10, "n_probe_shuffle": 1, "n_jobs": 1,
         "enable_nonlinear_probe": False, "enable_token_identity_probe": True},
    )

    assert set(result["token_identity_probe"]) == {"0", "1"}
    assert result["token_identity_probe"]["0"]["n_samples"] == 20
    assert result["token_identity_probe"]["0"]["n_unique_groups"] == 20
    assert result["token_identity_probe"]["1"]["balanced_acc"] == pytest.approx(1.0)
