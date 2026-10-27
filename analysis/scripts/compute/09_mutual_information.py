"""
09_mutual_information.py

Tracks how well encoder representations predict molecular stereochemistry.

For every checkpoint, the script computes:
  - linear R/S probes on the pooled latent and selected encoder layers
  - nonlinear R/S probes (MLP and random forest) on the pooled latent
  - a canonical @/@@ probe on the pooled latent
  - selected-layer probes using sequence mean pooling and chiral-window pooling
  - optional (enable_token_identity_probe): per-layer input @/@@ token identity control probe
    on the hidden state at every @/@@ position of the input, with StratifiedGroupKFold over
    samples; the paper's "balanced accuracy >= 0.9999"
    control was computed on valid_10k.csv random SMILES for layers 0-7.

Output: {output_dir}/{model_name}/mutual_information_{model_name}.json

The JSON is updated after every checkpoint and can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.models import PanCoreTransformerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis.runner import check_resume_signature


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


PROBE_KEYS = (
    "acc",
    "balanced_acc",
    "roc_auc",
    "log_loss_val",
    "dummy_acc",
    "probe_mi_lower_bound",
    "probe_mi_lower_bound_clipped",
    "entropy_y",
    "balanced_acc_shuffled_mean",
    "balanced_acc_shuffled_std",
    "roc_auc_shuffled_mean",
    "roc_auc_shuffled_std",
    "mi_lb_shuffled_mean",
    "mi_lb_shuffled_std",
    "balanced_acc_corrected",
    "roc_auc_corrected",
    "mi_lb_corrected",
    "recall_class_0",
    "recall_class_1",
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def _validate_probe_inputs(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must have shape [N, D], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape [N], got {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same sample count, got {len(X)} and {len(y)}")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")


def _nan_probe_result(n: int) -> dict:
    return {key: float("nan") for key in PROBE_KEYS} | {"n_samples": int(n)}


def _empirical_entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def _make_balanced_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, class_weight="balanced"),
    )


def _make_probability_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, class_weight=None),
    )


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray,
    n_jobs: int,
) -> dict:
    balanced_probability = cross_val_predict(
        _make_balanced_classifier(),
        X,
        y,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    balanced_prediction = classes[balanced_probability.argmax(axis=1)]
    calibrated_probability = cross_val_predict(
        _make_probability_classifier(),
        X,
        y,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    dummy_prediction = cross_val_predict(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        cv=cv_splits,
        n_jobs=n_jobs,
    )

    balanced_acc = float(balanced_accuracy_score(y, balanced_prediction))
    if len(classes) == 2:
        roc_auc = float(roc_auc_score(y, balanced_probability[:, 1]))
    else:
        roc_auc = float("nan")
    nll = float(log_loss(y, calibrated_probability, labels=classes))
    entropy_y = _empirical_entropy(y)
    mi_lower_bound = entropy_y - nll
    matrix = confusion_matrix(y, balanced_prediction, labels=classes)
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)

    return {
        "acc": float(accuracy_score(y, balanced_prediction)),
        "balanced_acc": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss_val": nll,
        "dummy_acc": float(accuracy_score(y, dummy_prediction)),
        "entropy_y": entropy_y,
        "probe_mi_lower_bound": mi_lower_bound,
        "probe_mi_lower_bound_clipped": float(max(0.0, mi_lower_bound)),
        "recall_class_0": float(recalls[0]),
        "recall_class_1": float(recalls[1]) if len(recalls) > 1 else float("nan"),
    }


def compute_probe_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    min_samples: int = 50,
    n_shuffle: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Compute a linear probe and shuffled-label corrected baselines."""
    X = np.asarray(X)
    y = np.asarray(y)
    _validate_probe_inputs(X, y)
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be positive, got {min_samples}")
    if n_shuffle < 1:
        raise ValueError(f"n_shuffle must be positive, got {n_shuffle}")

    classes, counts = np.unique(y, return_counts=True)
    if len(y) < min_samples or len(classes) < 2 or counts.min() < n_splits:
        return _nan_probe_result(len(y))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = _run_cv(X, y, list(cv.split(X, y)), classes, n_jobs)

    shuffled_balanced_acc: list[float] = []
    shuffled_roc_auc: list[float] = []
    shuffled_mi: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(n_shuffle):
        y_shuffled = rng.permutation(y)
        shuffled_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(rng.integers(1 << 30)),
        )
        shuffled = _run_cv(
            X,
            y_shuffled,
            list(shuffled_cv.split(X, y_shuffled)),
            classes,
            n_jobs,
        )
        shuffled_balanced_acc.append(shuffled["balanced_acc"])
        shuffled_roc_auc.append(shuffled["roc_auc"])
        shuffled_mi.append(shuffled["probe_mi_lower_bound"])

    balanced_mean = float(np.nanmean(shuffled_balanced_acc))
    auc_mean = float(np.nanmean(shuffled_roc_auc))
    mi_mean = float(np.nanmean(shuffled_mi))
    return result | {
        "n_samples": int(len(y)),
        "class_counts": {str(value): int(count) for value, count in zip(classes, counts)},
        "balanced_acc_shuffled_mean": balanced_mean,
        "balanced_acc_shuffled_std": float(np.nanstd(shuffled_balanced_acc)),
        "roc_auc_shuffled_mean": auc_mean,
        "roc_auc_shuffled_std": float(np.nanstd(shuffled_roc_auc)),
        "mi_lb_shuffled_mean": mi_mean,
        "mi_lb_shuffled_std": float(np.nanstd(shuffled_mi)),
        "balanced_acc_corrected": result["balanced_acc"] - balanced_mean,
        "roc_auc_corrected": result["roc_auc"] - auc_mean,
        "mi_lb_corrected": result["probe_mi_lower_bound"] - mi_mean,
    }


def find_chiral_token_positions(
    input_ids: np.ndarray, atmark_id: int, atatmark_id: int
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """(sample, position) of every @ (label 0) / @@ (label 1) input token; groups = sample index."""
    pairs: list[tuple[int, int]] = []
    labels: list[int] = []
    for sample_index, ids in enumerate(input_ids):
        for position, token in enumerate(ids):
            if token == atmark_id:
                pairs.append((sample_index, position))
                labels.append(0)
            elif token == atatmark_id:
                pairs.append((sample_index, position))
                labels.append(1)
    return pairs, np.asarray(labels, dtype=np.int64), np.asarray([pair[0] for pair in pairs], dtype=np.int64)


def compute_token_identity_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    min_samples: int = 50,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Linear probe with StratifiedGroupKFold so tokens of one sample never span train and test.

    As in the computation behind the paper value, no shuffled-label baseline is
    computed for this control probe.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    _validate_probe_inputs(X, y)
    if groups.shape != y.shape:
        raise ValueError(f"groups must have shape {y.shape}, got {groups.shape}")
    classes, counts = np.unique(y, return_counts=True)
    n_groups = len(np.unique(groups))
    if len(y) < min_samples or len(classes) < 2 or counts.min() < n_splits or n_groups < n_splits:
        return _nan_probe_result(len(y)) | {"n_unique_groups": int(n_groups)}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = _run_cv(X, y, list(cv.split(X, y, groups)), classes, n_jobs)
    return result | {
        "n_samples": int(len(y)),
        "n_unique_groups": int(n_groups),
        "class_counts": {str(value): int(count) for value, count in zip(classes, counts)},
    }


def _nan_nonlinear_result(n: int) -> dict:
    return {
        "mlp_balanced_acc": float("nan"),
        "mlp_roc_auc": float("nan"),
        "rf_balanced_acc": float("nan"),
        "rf_roc_auc": float("nan"),
        "n_samples": int(n),
    }


class _MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _train_eval_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    n_classes: int,
    device: torch.device,
    n_epochs: int,
) -> np.ndarray:
    model = _MLP(X_train.shape[1], n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_features = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    train_labels = torch.as_tensor(y_train, dtype=torch.long, device=device)

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_features), train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation = torch.as_tensor(X_validation, dtype=torch.float32, device=device)
        return torch.softmax(model(validation), dim=-1).cpu().numpy()


def compute_nonlinear_probe_metrics(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    *,
    n_splits: int = 5,
    min_samples: int = 50,
    mlp_epochs: int = 200,
    rf_estimators: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Compute MLP and random-forest probes with stratified cross-validation."""
    X = np.asarray(X)
    y = np.asarray(y)
    _validate_probe_inputs(X, y)
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be positive, got {min_samples}")
    if mlp_epochs < 1 or rf_estimators < 1:
        raise ValueError("mlp_epochs and rf_estimators must be positive")

    classes, counts = np.unique(y, return_counts=True)
    if len(y) < min_samples or len(classes) < 2 or counts.min() < n_splits:
        return _nan_nonlinear_result(len(y))
    if len(classes) != 2:
        raise ValueError(f"Nonlinear R/S probe requires exactly two classes, got {classes.tolist()}")

    scaled = StandardScaler().fit_transform(X).astype(np.float32)
    encoded_labels = np.searchsorted(classes, y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(scaled, y))
    mlp_probabilities = np.zeros((len(y), len(classes)), dtype=np.float32)
    for fold, (train_indices, validation_indices) in enumerate(splits):
        torch.manual_seed(seed + fold)
        mlp_probabilities[validation_indices] = _train_eval_mlp_fold(
            scaled[train_indices],
            encoded_labels[train_indices],
            scaled[validation_indices],
            len(classes),
            device,
            mlp_epochs,
        )

    mlp_prediction = classes[mlp_probabilities.argmax(axis=1)]
    forest = RandomForestClassifier(
        n_estimators=rf_estimators,
        random_state=seed,
        n_jobs=n_jobs,
    )
    forest_probabilities = cross_val_predict(
        forest,
        scaled,
        y,
        cv=splits,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    forest_prediction = classes[forest_probabilities.argmax(axis=1)]
    return {
        "mlp_balanced_acc": float(balanced_accuracy_score(y, mlp_prediction)),
        "mlp_roc_auc": float(roc_auc_score(y, mlp_probabilities[:, 1])),
        "rf_balanced_acc": float(balanced_accuracy_score(y, forest_prediction)),
        "rf_roc_auc": float(roc_auc_score(y, forest_probabilities[:, 1])),
        "n_samples": int(len(y)),
    }


def mean_pool_hidden(hidden: np.ndarray, input_ids: np.ndarray, pad_id: int) -> np.ndarray:
    """Mean-pool hidden states over non-padding positions."""
    hidden = np.asarray(hidden)
    input_ids = np.asarray(input_ids)
    if hidden.ndim != 3 or input_ids.ndim != 2 or hidden.shape[:2] != input_ids.shape:
        raise ValueError(
            f"Expected hidden [N, L, E] aligned with input_ids [N, L], got "
            f"{hidden.shape} and {input_ids.shape}"
        )
    mask = input_ids != pad_id
    counts = mask.sum(axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("Cannot mean-pool a sequence containing only padding")
    return (hidden * mask[:, :, None]).sum(axis=1) / counts


def chiral_window_pool(
    hidden: np.ndarray,
    input_ids: np.ndarray,
    sample_indices: np.ndarray,
    atmark_id: int,
    atatmark_id: int,
    pad_id: int,
    window: int,
) -> np.ndarray:
    """Pool hidden states in a token window around each @ or @@ token."""
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    hidden = np.asarray(hidden)
    input_ids = np.asarray(input_ids)
    sample_indices = np.asarray(sample_indices)
    if hidden.ndim != 3 or input_ids.ndim != 2 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("hidden and input_ids must have aligned [N, L] dimensions")
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(hidden)):
        raise ValueError("sample_indices contains an out-of-range index")

    pooled: list[np.ndarray] = []
    sequence_length = input_ids.shape[1]
    for sample_index in sample_indices:
        ids = input_ids[sample_index]
        positions = np.flatnonzero((ids == atmark_id) | (ids == atatmark_id))
        vectors: list[np.ndarray] = []
        for position in positions:
            start = max(0, int(position) - window)
            end = min(sequence_length, int(position) + window + 1)
            valid = ids[start:end] != pad_id
            if valid.any():
                vectors.append(hidden[sample_index, start:end][valid].mean(axis=0))
        if vectors:
            pooled.append(np.mean(vectors, axis=0))
        else:
            valid = ids != pad_id
            if not valid.any():
                raise ValueError(f"Sample {sample_index} contains only padding")
            pooled.append(hidden[sample_index][valid].mean(axis=0))
    return np.asarray(pooled, dtype=np.float32)


def encode_all(
    smiles: list[str],
    tokenizer: SMILESTokenizer,
    model: PanCoreTransformerModel,
    device: torch.device,
    probe_layers: list[int],
    pad_id: int,
    *,
    batch_size: int = 512,
    max_len: int = 512,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Encode all samples and retain selected encoder block outputs."""
    if not smiles:
        raise ValueError("eval_data contains no samples")
    if batch_size < 1 or max_len < 1:
        raise ValueError("batch_size and max_len must be positive")
    if not probe_layers or min(probe_layers) < 0 or len(set(probe_layers)) != len(probe_layers):
        raise ValueError("probe_layers must contain unique non-negative layer indices")

    latent_batches: list[np.ndarray] = []
    hidden_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in probe_layers}
    input_id_batches: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            encoded = tokenizer(
                smiles[start:start + batch_size],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids_cpu = encoded["input_ids"]
            if input_ids_cpu.shape[1] > max_len:
                raise ValueError(
                    f"A probe SMILES has {input_ids_cpu.shape[1]} tokens, more than max_len={max_len}; "
                    "the paper run did not truncate inputs"
                )
            input_ids = input_ids_cpu.to(device)
            attention_mask = encoded["attention_mask"].to(device)
            latent = model.encoder(
                input_ids.transpose(0, 1),
                valid_token_mask=attention_mask.unsqueeze(1).unsqueeze(2).bool(),
                need_hidden=True,
            )
            if isinstance(latent, (tuple, list)):
                latent = latent[0]
            latent_batches.append(latent.float().cpu().numpy())

            block_outputs = model.encoder.all_hidden_states[1:]
            invalid_layers = [layer for layer in probe_layers if layer >= len(block_outputs)]
            if invalid_layers:
                raise ValueError(
                    f"probe_layers {invalid_layers} exceed encoder block range 0..{len(block_outputs) - 1}"
                )
            for layer in probe_layers:
                hidden = block_outputs[layer]
                hidden_batches[layer].append(np.transpose(hidden, (1, 0, 2)).astype(np.float16))
            input_id_batches.append(input_ids_cpu.numpy())

    global_length = max(ids.shape[1] for ids in input_id_batches)
    padded_ids = [
        np.pad(ids, ((0, 0), (0, global_length - ids.shape[1])), constant_values=pad_id)
        for ids in input_id_batches
    ]
    input_ids_all = np.concatenate(padded_ids, axis=0)

    hidden_by_layer: dict[int, np.ndarray] = {}
    for layer, batches in hidden_batches.items():
        padded = [
            np.pad(
                hidden,
                ((0, 0), (0, global_length - hidden.shape[1]), (0, 0)),
                constant_values=0,
            )
            for hidden in batches
        ]
        hidden_by_layer[layer] = np.concatenate(padded, axis=0)

    return np.concatenate(latent_batches, axis=0), hidden_by_layer, input_ids_all


def analyze_checkpoint(
    checkpoint_path: Path,
    smiles: list[str],
    tokenizer: SMILESTokenizer,
    rs_labels: np.ndarray,
    canonical_token_labels: np.ndarray,
    probe_layers: list[int],
    device: torch.device,
    config: dict,
) -> dict:
    """Compute every v5 probe for one checkpoint."""
    model = PanCoreTransformerModel.from_pretrained(checkpoint_path).to(device).eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad token")
    atmark_id = tokenizer.convert_tokens_to_ids("@")
    atatmark_id = tokenizer.convert_tokens_to_ids("@@")
    if atmark_id == tokenizer.unk_token_id or atatmark_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer vocabulary must contain both @ and @@ tokens")

    latent, hidden_by_layer, input_ids = encode_all(
        smiles,
        tokenizer,
        model,
        device,
        probe_layers,
        pad_id,
        batch_size=config.get("batch_size", 512),
        max_len=config.get("max_len", 512),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probe_options = {
        "n_splits": config.get("n_splits", 5),
        "min_samples": config.get("min_samples", 50),
        "n_shuffle": config.get("n_probe_shuffle", 5),
        "seed": config.get("seed", 42),
        "n_jobs": config.get("n_jobs", -1),
    }
    result = {
        "rs_probe": compute_probe_metrics(latent, rs_labels, **probe_options),
        "rs_nonlinear_probe": None,
        "canonical_token_probe": None,
        "rs_probe_by_layer": {},
        "rs_window_probe_by_layer": {},
        "token_identity_probe": None,
    }

    if config.get("enable_nonlinear_probe", True):
        result["rs_nonlinear_probe"] = compute_nonlinear_probe_metrics(
            latent,
            rs_labels,
            device,
            n_splits=probe_options["n_splits"],
            min_samples=probe_options["min_samples"],
            mlp_epochs=config.get("mlp_epochs", 200),
            rf_estimators=config.get("rf_estimators", 100),
            seed=probe_options["seed"],
            n_jobs=probe_options["n_jobs"],
        )

    if config.get("enable_canonical_token_probe", True):
        valid = canonical_token_labels != -1
        result["canonical_token_probe"] = compute_probe_metrics(
            latent[valid],
            canonical_token_labels[valid],
            **probe_options,
        )

    token_identity = config.get("enable_token_identity_probe", False)
    if token_identity:
        result["token_identity_probe"] = {}
        token_pairs, token_labels, token_groups = find_chiral_token_positions(input_ids, atmark_id, atatmark_id)
        if token_pairs:
            pair_samples = np.asarray([pair[0] for pair in token_pairs])
            pair_positions = np.asarray([pair[1] for pair in token_pairs])

    sample_indices = np.arange(len(rs_labels))
    for layer in probe_layers:
        hidden = hidden_by_layer[layer].astype(np.float32)
        mean_pooled = mean_pool_hidden(hidden, input_ids, pad_id)
        result["rs_probe_by_layer"][str(layer)] = compute_probe_metrics(
            mean_pooled,
            rs_labels,
            **probe_options,
        )
        window_pooled = chiral_window_pool(
            hidden,
            input_ids,
            sample_indices,
            atmark_id,
            atatmark_id,
            pad_id,
            config.get("chiral_window", 2),
        )
        result["rs_window_probe_by_layer"][str(layer)] = compute_probe_metrics(
            window_pooled,
            rs_labels,
            **probe_options,
        )
        if token_identity:
            result["token_identity_probe"][str(layer)] = (
                compute_token_identity_probe(
                    hidden[pair_samples, pair_positions],
                    token_labels,
                    token_groups,
                    n_splits=probe_options["n_splits"],
                    min_samples=probe_options["min_samples"],
                    seed=probe_options["seed"],
                    n_jobs=probe_options["n_jobs"],
                )
                if token_pairs else _nan_probe_result(0)
            )
    return result


def discover_checkpoints(checkpoint_dir: Path, target_steps: list[int] | None) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {path.name}")
        checkpoints.append((int(suffix), path))
    checkpoints.sort(key=lambda item: item[0])
    if target_steps is not None:
        requested = set(target_steps)
        checkpoints = [item for item in checkpoints if item[0] in requested]
        found = {step for step, _ in checkpoints}
        if found != requested:
            raise FileNotFoundError(f"Requested checkpoint steps not found: {sorted(requested - found)}")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {checkpoint_dir}")
    return [path for _, path in checkpoints]


def _load_eval_data(config: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(config["eval_data"])
    src_col = config.get("src_col", "random")
    rs_label_col = config.get("rs_label_col", "rs_label_id")
    canonical_label_col = config.get("canonical_token_label_col", "canonical_chiral_token_id")
    missing = [column for column in (src_col, rs_label_col, canonical_label_col) if column not in frame]
    if missing:
        raise ValueError(f"eval_data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("eval_data must contain at least one row")

    smiles = frame[src_col].tolist()
    if not all(isinstance(value, str) and value for value in smiles):
        raise ValueError(f"Column {src_col!r} must contain non-empty strings")
    rs_labels = frame[rs_label_col].to_numpy(dtype=np.int64)
    canonical_labels = frame[canonical_label_col].to_numpy(dtype=np.int64)
    if not set(np.unique(rs_labels)).issubset({0, 1}):
        raise ValueError(f"{rs_label_col} must contain only 0 and 1")
    if not set(np.unique(canonical_labels)).issubset({-1, 0, 1}):
        raise ValueError(f"{canonical_label_col} must contain only -1, 0, and 1")
    return smiles, rs_labels, canonical_labels


def _save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as f:
        json.dump(results, f, indent=2)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutual-information probe trajectory analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    required = ("model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir")
    missing = [key for key in required if not config.get(key) and not (key == "output_dir" and args.output_dir)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    model_name = str(config["model_name"])
    probe_layers = [int(layer) for layer in config.get("probe_layers", [0, 2, 5, 7])]
    output_dir = Path(args.output_dir or config["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "09_mutual_information.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    smiles, rs_labels, canonical_labels = _load_eval_data(config)
    tokenizer = SMILESTokenizer(config["tokenizer_path"])
    checkpoints = discover_checkpoints(
        Path(config["checkpoint_dir"]),
        config.get("checkpoint_steps"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {len(smiles)}, checkpoints: {len(checkpoints)}, layers: {probe_layers}")

    output_path = output_dir / f"mutual_information_{model_name}.json"
    check_resume_signature(output_dir / ".resume_signature.json", __file__, config, [output_path])
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        if results.get("model_name") != model_name or results.get("probe_layers") != probe_layers:
            raise ValueError(f"Existing output metadata does not match the current config: {output_path}")
    else:
        results = {"model_name": model_name, "probe_layers": probe_layers, "steps": [], "metrics": []}

    done_steps = set(results["steps"])
    for checkpoint in checkpoints:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        if step in done_steps:
            logger.info(f"Skipping step {step} (already done)")
            continue
        logger.info(f"Processing checkpoint-{step}")
        metrics = analyze_checkpoint(
            checkpoint,
            smiles,
            tokenizer,
            rs_labels,
            canonical_labels,
            probe_layers,
            device,
            config,
        )
        position = int(np.searchsorted(results["steps"], step))
        results["steps"].insert(position, step)
        results["metrics"].insert(position, metrics)
        _save_results(output_path, results)
        logger.info(f"Saved step {step} to {output_path}")


if __name__ == "__main__":
    main()
