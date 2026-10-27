"""
Metric computation functions for chirality emergence analysis.

All functions operate on numpy arrays unless explicitly noted.
Chiral tokens are defined as @ (token id 7) and @@ (token id 8) by default.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import paired_cosine_distances


# ---------------------------------------------------------------------------
# Basic similarity / divergence
# ---------------------------------------------------------------------------

def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (feature-space formulation).

    Args:
        X, Y: 2-D arrays of shape [N_samples, Hidden_dim].
    Returns:
        Scalar CKA score in [0, 1].
    """
    if X.shape[0] <= 1:
        return 0.0
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    C_XY = np.dot(X_c.T, Y_c)
    C_XX = np.dot(X_c.T, X_c)
    C_YY = np.dot(Y_c.T, Y_c)
    num = np.sum(C_XY ** 2)
    den = np.sqrt(np.sum(C_XX ** 2) * np.sum(C_YY ** 2))
    if den < 1e-10:
        return 0.0
    return float(num / den)


def compute_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Paired cosine similarity (mean, std) between corresponding rows of X and Y.

    Args:
        X, Y: 2-D arrays of shape [N, D].
    Returns:
        (mean_similarity, std_similarity) — similarity = 1 - cosine_distance.
    """
    distances = paired_cosine_distances(X, Y)  # [N]
    similarities = 1.0 - distances
    return float(similarities.mean()), float(similarities.std())


def compute_entropy(p: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Shannon entropy (base e) along the last axis.

    Args:
        p: Probability array. Last axis is the distribution dimension.
    Returns:
        Entropy values with the last axis reduced.
    """
    return -np.sum(p * np.log(p + epsilon), axis=-1)


def compute_jsd(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Jensen-Shannon Divergence between two distributions (row-wise).

    Args:
        p, q: Probability arrays normalized along the last axis.
    Returns:
        JSD values with the last axis reduced.
    """
    m = 0.5 * (p + q)
    kl_p = np.sum(p * (np.log(p + epsilon) - np.log(m + epsilon)), axis=-1)
    kl_q = np.sum(q * (np.log(q + epsilon) - np.log(m + epsilon)), axis=-1)
    return 0.5 * (kl_p + kl_q)


# ---------------------------------------------------------------------------
# Attention metrics
# ---------------------------------------------------------------------------

def compute_attention_mass(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention mass: how much attention other tokens pay to chiral tokens (key axis).

    Args:
        attn: [Batch, Heads, Query, Key].
        chiral_mask: [Batch, Seq_len] boolean — True for chiral token positions.
        valid_mask: [Batch, Seq_len] boolean — True for non-padding positions.
    Returns:
        (mean, std) each of shape [Heads].
    """
    # Mask padding
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]
    valid_v = valid_mask[valid_indices]
    valid_lengths = valid_v.sum(axis=-1)[:, np.newaxis]  # [N, 1]

    # Key axis: how much attention flows *toward* chiral tokens
    # attn_v: [N, H, Q, K] → transpose → [N, K, H, Q]
    attn_kt = attn_v.transpose(0, 3, 1, 2)
    attns_to_chiral = attn_kt[chiral_v]  # [N_chiral, H, Q]
    chiral_valid_len = valid_lengths[chiral_v.any(axis=-1)[valid_indices] if False else np.where(chiral_v)[0]]
    # Re-index: which batch items have chiral tokens?
    batch_idx = np.where(chiral_v)[0]
    valid_len_per_chiral = valid_lengths[batch_idx]  # [N_chiral, 1]
    mass = attns_to_chiral.sum(axis=-1) / (valid_len_per_chiral + 1e-10)  # [N_chiral, H]
    return mass.mean(axis=0), mass.std(axis=0)


def compute_attention_entropy(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention entropy: sharpness of chiral token's outgoing attention (query axis).

    Args:
        attn: [Batch, Heads, Query, Key].
        chiral_mask: [Batch, Seq_len] boolean.
        valid_mask: [Batch, Seq_len] boolean.
    Returns:
        (mean_entropy, std_entropy) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]

    # Query axis: attention *from* chiral tokens
    attn_qt = attn_v.transpose(0, 2, 1, 3)  # [N, Q, H, K]
    attns_from_chiral = attn_qt[chiral_v]   # [N_chiral, H, K]
    ent = compute_entropy(attns_from_chiral)  # [N_chiral, H]
    return ent.mean(axis=0), ent.std(axis=0)


def compute_sar(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Substituent-Focused Attention Ratio: fraction of chiral token's attention toward graph-distance-1 neighbors.

    Args:
        attn: [Batch, Heads, Query, Key].
        chiral_mask: [Batch, Seq_len] boolean.
        valid_mask: [Batch, Seq_len] boolean.
        dist_matrix: [Batch, Seq_len, Seq_len] token-level graph distances (-1 for non-atom).
    Returns:
        (mean_sar, std_sar) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]
    dist_v = dist_matrix[valid_indices]

    attn_qt = attn_v.transpose(0, 2, 1, 3)
    attns_from_chiral = attn_qt[chiral_v]     # [N_chiral, H, K]
    chiral_dists = dist_v[chiral_v]           # [N_chiral, K]
    neighbor_mask = (chiral_dists == 1.0)[:, np.newaxis, :]  # [N_chiral, 1, K]

    sar = (attns_from_chiral * neighbor_mask).sum(axis=-1)  # [N_chiral, H]
    return sar.mean(axis=0), sar.std(axis=0)


def compute_graph_distance_from_chiral(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected graph distance of tokens attended *from* chiral tokens.

    Returns:
        (mean, std) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]
    dist_v = dist_matrix[valid_indices]

    attn_qt = attn_v.transpose(0, 2, 1, 3)
    attns_from_chiral = attn_qt[chiral_v]
    chiral_dists = dist_v[chiral_v][:, np.newaxis, :]  # [N_chiral, 1, K]
    exp_dist = (attns_from_chiral * chiral_dists).sum(axis=-1)  # [N_chiral, H]
    return exp_dist.mean(axis=0), exp_dist.std(axis=0)


def compute_seq_distance_from_chiral(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected sequence (index) distance of tokens attended *from* chiral tokens.

    Returns:
        (mean, std) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]

    seq_len = attn.shape[-1]
    seq_idx = np.arange(seq_len)
    seq_dist_mat = np.abs(seq_idx[:, np.newaxis] - seq_idx[np.newaxis, :])  # [K, K]
    # Broadcast to [N, K, K] — we need per-chiral-token row
    n_valid = attn_v.shape[0]
    seq_dist_3d = np.broadcast_to(seq_dist_mat, (n_valid, seq_len, seq_len))

    attn_qt = attn_v.transpose(0, 2, 1, 3)
    attns_from_chiral = attn_qt[chiral_v]         # [N_chiral, H, K]
    chiral_seq_dists = seq_dist_3d[chiral_v][:, np.newaxis, :]  # [N_chiral, 1, K]
    exp_dist = (attns_from_chiral * chiral_seq_dists).sum(axis=-1)
    return exp_dist.mean(axis=0), exp_dist.std(axis=0)


def compute_graph_distance_to_chiral(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected graph distance of tokens that send attention *toward* chiral tokens.

    Returns:
        (mean, std) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]
    dist_v = dist_matrix[valid_indices]

    attn_kt = attn_v.transpose(0, 3, 1, 2)       # [N, K, H, Q]
    attns_to_chiral = attn_kt[chiral_v]            # [N_chiral, H, Q]
    chiral_dists = dist_v[chiral_v][:, np.newaxis, :]  # [N_chiral, 1, K]

    total_attn = attns_to_chiral.sum(axis=-1)      # [N_chiral, H]
    exp_dist = (attns_to_chiral * chiral_dists).sum(axis=-1) / (total_attn + 1e-10)
    return exp_dist.mean(axis=0), exp_dist.std(axis=0)


def compute_seq_distance_to_chiral(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected sequence distance of tokens that send attention *toward* chiral tokens.

    Returns:
        (mean, std) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]

    seq_len = attn.shape[-1]
    seq_idx = np.arange(seq_len)
    seq_dist_mat = np.abs(seq_idx[:, np.newaxis] - seq_idx[np.newaxis, :])
    n_valid = attn_v.shape[0]
    seq_dist_3d = np.broadcast_to(seq_dist_mat, (n_valid, seq_len, seq_len))

    attn_kt = attn_v.transpose(0, 3, 1, 2)
    attns_to_chiral = attn_kt[chiral_v]
    chiral_seq_dists = seq_dist_3d[chiral_v][:, np.newaxis, :]

    total_attn = attns_to_chiral.sum(axis=-1)
    exp_dist = (attns_to_chiral * chiral_seq_dists).sum(axis=-1) / (total_attn + 1e-10)
    return exp_dist.mean(axis=0), exp_dist.std(axis=0)


def compute_nit(
    attn: np.ndarray,
    chiral_mask: np.ndarray,
    valid_mask: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbor Influence on Target: fraction of neighbor attention directed at chiral token.

    Args:
        attn: [Batch, Heads, Query, Key].
        chiral_mask: [Batch, Seq_len] boolean.
        valid_mask: [Batch, Seq_len] boolean.
        dist_matrix: [Batch, Seq_len, Seq_len] graph distances.
    Returns:
        (mean_nit, std_nit) each of shape [Heads].
    """
    q_mask = valid_mask[:, np.newaxis, :, np.newaxis]
    k_mask = valid_mask[:, np.newaxis, np.newaxis, :]
    attn = attn * q_mask * k_mask

    valid_indices = valid_mask.any(axis=-1)
    attn_v = attn[valid_indices]
    chiral_v = chiral_mask[valid_indices]
    dist_v = dist_matrix[valid_indices]

    attn_kt = attn_v.transpose(0, 3, 1, 2)
    attns_to_chiral = attn_kt[chiral_v]                         # [N_chiral, H, Q]
    chiral_dists = dist_v[chiral_v][:, np.newaxis, :]           # [N_chiral, 1, K]
    neighbor_mask = (chiral_dists == 1.0)                        # [N_chiral, 1, K]

    nit = (attns_to_chiral * neighbor_mask).sum(axis=-1) / (attns_to_chiral.sum(axis=-1) + 1e-10)
    return nit.mean(axis=0), nit.std(axis=0)


# ---------------------------------------------------------------------------
# Logits / prediction metrics
# ---------------------------------------------------------------------------

def compute_logits_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    chiral_mask: np.ndarray,
    atmark_id: int = 7,
    atatmark_id: int = 8,
) -> dict:
    """Compute prediction-level chiral metrics.

    Args:
        logits: [Batch, Seq_len, Vocab_size].
        labels: [Batch, Seq_len] integer token ids (decoder target shifted by 1).
        chiral_mask: [Batch, Seq_len] boolean — True for chiral token positions in the label.
        atmark_id: Token id for '@'.
        atatmark_id: Token id for '@@'.
    Returns:
        dict with keys: perplexity, chiral_dilemma_rate, mean_rank, mean_gap, chiral_margin.
    """
    batch_idx, seq_idx = np.where(chiral_mask)
    if len(batch_idx) == 0:
        return dict(perplexity=float("nan"), chiral_dilemma_rate=float("nan"),
                    mean_rank=float("nan"), mean_gap=float("nan"), chiral_margin=float("nan"))

    tgt_logits = logits[batch_idx, seq_idx, :]  # [N_chiral, V]
    tgt_labels = labels[batch_idx, seq_idx]     # [N_chiral]

    # Logit gap
    true_lgts = np.take_along_axis(tgt_logits, tgt_labels[:, None], axis=1).squeeze(1)
    tmp = tgt_logits.copy()
    np.put_along_axis(tmp, tgt_labels[:, None], -np.inf, axis=1)
    gap = true_lgts - np.max(tmp, axis=1)

    # Mean perplexity at chiral positions (paper Eq. 1): exp of the entropy of the predicted
    # distribution at each target chiral position, averaged over positions.
    probs = softmax(tgt_logits, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ppl = float(np.exp(entropy).mean())

    # Rank
    sorted_idx = np.argsort(-tgt_logits, axis=1)
    ranks = np.where(sorted_idx == tgt_labels[:, None])[1] + 1

    # Chiral dilemma rate
    top2 = sorted_idx[:, :2]
    has_at = (top2 == atmark_id).any(axis=1)
    has_atat = (top2 == atatmark_id).any(axis=1)
    cdr = float((has_at & has_atat).mean())

    # Chiral discrimination margin
    at_l = tgt_logits[:, atmark_id]
    atat_l = tgt_logits[:, atatmark_id]
    is_at = tgt_labels == atmark_id
    is_atat = tgt_labels == atatmark_id
    margin = np.zeros(len(tgt_labels), dtype=float)
    if is_at.any():
        margin[is_at] = at_l[is_at] - atat_l[is_at]
    if is_atat.any():
        margin[is_atat] = atat_l[is_atat] - at_l[is_atat]
    valid_chiral = is_at | is_atat
    mean_margin = float(margin[valid_chiral].mean()) if valid_chiral.any() else float("nan")

    return dict(
        perplexity=ppl,
        chiral_dilemma_rate=cdr,
        mean_rank=float(ranks.mean()),
        mean_gap=float(gap.mean()),
        chiral_margin=mean_margin,
    )


# ---------------------------------------------------------------------------
# Hidden-state metrics
# ---------------------------------------------------------------------------

def compute_l2_norm(
    hidden: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Mean L2 norm of hidden states at selected positions.

    Args:
        hidden: [Batch, Seq_len, Hidden_dim] or [Seq_len, Batch, Hidden_dim].
        mask: [Batch, Seq_len] boolean — positions to include.
    Returns:
        (mean_norm, std_norm).
    """
    if hidden.shape[0] != mask.shape[0]:
        hidden = hidden.transpose(1, 0, 2)  # [Seq_len, Batch, D] → [Batch, Seq_len, D]
    vecs = hidden[mask]  # [N_selected, D]
    if len(vecs) == 0:
        return 0.0, 0.0
    norms = np.linalg.norm(vecs, axis=-1)
    return float(norms.mean()), float(norms.std())


# ---------------------------------------------------------------------------
# Latent vector geometry metrics
# ---------------------------------------------------------------------------

def compute_active_units(latent: np.ndarray, threshold_ratio: float = 0.9) -> int:
    """Number of dimensions explaining `threshold_ratio` of total variance (PCA-style).

    Args:
        latent: [N_samples, Hidden_dim].
    Returns:
        Integer count of effective dimensions.
    """
    variances = np.var(latent, axis=0)
    sorted_vars = np.sort(variances)[::-1]
    cumulative = np.cumsum(sorted_vars)
    total = cumulative[-1]
    if total < 1e-10:
        return 0
    idx = np.searchsorted(cumulative, total * threshold_ratio)
    return int(idx + 1)


def compute_intrinsic_dimension_twonn(latent: np.ndarray, n_neighbors: int = 2) -> float:
    """Two-Nearest-Neighbors intrinsic dimension estimator.

    Args:
        latent: [N_samples, Hidden_dim].
    Returns:
        Estimated intrinsic dimension (float).
    """
    from sklearn.neighbors import NearestNeighbors

    n = latent.shape[0]
    if n < 3:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto").fit(latent)
    distances, _ = nbrs.kneighbors(latent)

    r1 = distances[:, 1]  # 1st NN distance
    r2 = distances[:, 2]  # 2nd NN distance

    # Filter degenerate cases
    valid = (r1 > 1e-10) & (r2 > r1)
    if not valid.any():
        return float("nan")

    mu = r2[valid] / r1[valid]
    n_valid = valid.sum()
    # MLE estimator
    id_est = float(n_valid / np.sum(np.log(mu)))
    return id_est


def compute_isotropy_entropy(latent: np.ndarray) -> float:
    """Isotropy measured as normalized entropy of the eigenvalue spectrum of the covariance matrix.

    Args:
        latent: [N_samples, Hidden_dim].
    Returns:
        Scalar entropy value in [0, log(D)]; higher = more isotropic.
    """
    centered = latent - latent.mean(axis=0, keepdims=True)
    cov = np.dot(centered.T, centered) / max(latent.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total < 1e-10:
        return 0.0
    p = eigvals / total
    return float(-np.sum(p * np.log(p + 1e-10)))
