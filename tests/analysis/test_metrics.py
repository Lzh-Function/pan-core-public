"""Unit tests for pan_core_analysis.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pan_core_analysis.metrics import (
    compute_active_units,
    compute_cosine_similarity,
    compute_entropy,
    compute_intrinsic_dimension_twonn,
    compute_isotropy_entropy,
    compute_jsd,
    compute_l2_norm,
    compute_linear_cka,
    compute_logits_metrics,
)


class TestComputeLinearCKA:
    def test_identical_returns_one(self):
        X = np.random.default_rng(0).normal(size=(20, 16)).astype(np.float32)
        assert abs(compute_linear_cka(X, X) - 1.0) < 1e-5

    def test_range(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 16))
        Y = rng.normal(size=(30, 16))
        v = compute_linear_cka(X, Y)
        assert 0.0 <= v <= 1.0 + 1e-6

    def test_single_sample_returns_zero(self):
        assert compute_linear_cka(np.ones((1, 8)), np.ones((1, 8))) == 0.0

    def test_scaled_input_invariant(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 8))
        assert abs(compute_linear_cka(X, X * 5.0) - 1.0) < 1e-5


class TestComputeCosineSimilarity:
    def test_identical_returns_one(self):
        X = np.random.default_rng(0).normal(size=(10, 8))
        mean, std = compute_cosine_similarity(X, X)
        assert abs(mean - 1.0) < 1e-5
        assert std < 1e-5

    def test_opposite_returns_minus_one(self):
        X = np.random.default_rng(0).normal(size=(10, 8))
        mean, _ = compute_cosine_similarity(X, -X)
        assert abs(mean - (-1.0)) < 1e-5

    def test_orthogonal_near_zero(self):
        X = np.eye(8)  # 8 orthonormal vectors
        # Rotate by permutation → still orthogonal
        Y = np.roll(X, 1, axis=0)
        mean, _ = compute_cosine_similarity(X, Y)
        assert abs(mean) < 1e-5


class TestComputeEntropy:
    def test_uniform_distribution(self):
        p = np.full((1, 4), 0.25)
        h = compute_entropy(p)  # shape (1,)
        assert abs(float(h[0]) - np.log(4)) < 0.01

    def test_degenerate_distribution(self):
        p = np.array([[1.0, 0.0, 0.0, 0.0]])
        assert float(compute_entropy(p)[0]) < 0.01

    def test_batch_shape_preserved(self):
        p = np.full((5, 4), 0.25)
        h = compute_entropy(p)
        assert h.shape == (5,)


class TestComputeJSD:
    def test_identical_is_zero(self):
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        assert float(compute_jsd(p, p)[0]) < 1e-5

    def test_disjoint_is_log2(self):
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.0, 1.0]])
        assert abs(float(compute_jsd(p, q)[0]) - np.log(2)) < 0.01

    def test_batch_shape_preserved(self):
        p = np.full((3, 4), 0.25)
        q = np.full((3, 4), 0.25)
        assert compute_jsd(p, q).shape == (3,)


class TestComputeL2Norm:
    def test_unit_vectors(self):
        hidden = np.ones((2, 3, 4))  # norm = sqrt(4) = 2
        mask = np.ones((2, 3), dtype=bool)
        mean, std = compute_l2_norm(hidden, mask)
        assert abs(mean - 2.0) < 1e-5

    def test_seq_first_auto_transpose(self):
        hidden = np.ones((3, 2, 4))  # [L, B, E] — shape[0]=3 != mask.shape[0]=2
        mask = np.ones((2, 3), dtype=bool)
        mean, _ = compute_l2_norm(hidden, mask)
        assert abs(mean - 2.0) < 1e-5

    def test_empty_mask_returns_zeros(self):
        mean, std = compute_l2_norm(np.ones((2, 3, 4)), np.zeros((2, 3), dtype=bool))
        assert mean == 0.0 and std == 0.0


class TestComputeLogitsMetrics:
    def test_no_chiral_tokens_returns_nan(self):
        result = compute_logits_metrics(
            np.zeros((2, 5, 10)),
            np.zeros((2, 5), dtype=int),
            np.zeros((2, 5), dtype=bool),
        )
        assert np.isnan(result["perplexity"])

    def test_all_keys_present(self):
        rng = np.random.default_rng(0)
        B, L, V = 4, 6, 20
        logits = rng.normal(size=(B, L, V))
        labels = rng.integers(0, V, size=(B, L))
        chiral_mask = np.zeros((B, L), dtype=bool)
        chiral_mask[:, 2] = True
        result = compute_logits_metrics(logits, labels, chiral_mask)
        for key in ("perplexity", "chiral_dilemma_rate", "mean_rank", "mean_gap", "chiral_margin"):
            assert key in result

    def test_correct_prediction_has_positive_gap_and_rank_one(self):
        B, L, V = 2, 4, 10
        logits = np.zeros((B, L, V))
        logits[:, 1, 7] = 10.0  # token 7 (@) wins at position 1
        labels = np.full((B, L), 7, dtype=int)
        chiral_mask = np.zeros((B, L), dtype=bool)
        chiral_mask[:, 1] = True
        result = compute_logits_metrics(logits, labels, chiral_mask)
        assert result["mean_gap"] > 0
        assert result["mean_rank"] == 1.0

    def test_chiral_dilemma_rate_both_in_top2(self):
        B, L, V = 1, 3, 10
        logits = np.zeros((B, L, V))
        logits[0, 1, 7] = 5.0   # @ (id=7) top
        logits[0, 1, 8] = 4.9   # @@ (id=8) second
        labels = np.array([[0, 7, 0]])
        chiral_mask = np.zeros((B, L), dtype=bool)
        chiral_mask[0, 1] = True
        result = compute_logits_metrics(logits, labels, chiral_mask)
        assert result["chiral_dilemma_rate"] == 1.0

    def test_perplexity_is_mean_exp_entropy_of_predicted_distribution(self):
        logits = np.zeros((1, 2, 10))
        logits[0, 0, 7] = 2.0
        logits[0, 1, 8] = 1.0
        labels = np.array([[7, 8]])
        chiral = np.array([[True, True]])

        result = compute_logits_metrics(logits, labels, chiral, atmark_id=7, atatmark_id=8)

        probs = np.exp(logits[0]) / np.exp(logits[0]).sum(axis=1, keepdims=True)
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        assert result["perplexity"] == pytest.approx(np.exp(entropy).mean())


class TestComputeActiveUnits:
    def test_single_active_dim(self):
        latent = np.zeros((100, 32))
        latent[:, 0] = np.random.default_rng(0).normal(size=100)
        assert compute_active_units(latent) == 1

    def test_isotropic_uses_many_dims(self):
        latent = np.random.default_rng(0).normal(size=(200, 32))
        k = compute_active_units(latent, threshold_ratio=0.9)
        assert 5 <= k <= 32


class TestComputeIntrinsicDimension:
    def test_1d_random_line(self):
        # Randomly-placed points on a 1D line — TwoNN gives ~1/log(2) ≈ 1.44
        t = np.random.default_rng(0).uniform(0, 1, 500)[:, None]
        id_est = compute_intrinsic_dimension_twonn(t)
        assert 0.8 < id_est < 3.0

    def test_2d_higher_than_1d(self):
        rng = np.random.default_rng(1)
        data_1d = rng.uniform(0, 1, (500, 1))
        data_2d = rng.uniform(0, 1, (500, 2))
        assert compute_intrinsic_dimension_twonn(data_1d) < compute_intrinsic_dimension_twonn(data_2d)

    def test_too_few_samples_nan(self):
        assert np.isnan(compute_intrinsic_dimension_twonn(np.ones((2, 8))))


class TestComputeIsotropyEntropy:
    def test_isotropic_high_entropy(self):
        latent = np.random.default_rng(0).normal(size=(200, 16))
        h = compute_isotropy_entropy(latent)
        assert h > np.log(16) * 0.5

    def test_anisotropic_lower_than_isotropic(self):
        rng = np.random.default_rng(0)
        aniso = np.zeros((200, 16))
        aniso[:, 0] = rng.normal(size=200)
        h_aniso = compute_isotropy_entropy(aniso)
        h_iso = compute_isotropy_entropy(rng.normal(size=(200, 16)))
        assert h_aniso < h_iso
