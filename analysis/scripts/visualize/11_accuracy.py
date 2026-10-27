"""
11_accuracy.py

Visualizes accuracy progress from 01_accuracy_progress.py outputs (no jump band, as in the paper).

Per model:
  bucket_accuracy_{model}         exact-match accuracy per length bucket (Fig 2 lower panel)
  loss_acc_progress_{model}       Fig 2: train loss (gray points) + per-bucket eval cross-entropy
                                  (dashed, log scale) above the per-bucket accuracy;
                                  written only when trainer_states[model] is configured
  chiral_accuracy_{model}         Fig S3: chi-SMILES accuracy and mistranslated chi-SMILES
                                  accuracy without '@' for buckets 1 and 2
  token_accuracy_{model}_phase{p} Fig 3: token accuracy per target token type for bucket p
                                  (@, @@, /, \\ emphasized; tokens occurring <= 10 times in the
                                  bucket's targets drawn light gray; legend "token (n=count)")
  token_accuracy_teacher_forced_{model}_phase{p}
                                  Fig 3 with teacher-forced token accuracy (as the Methods describe);
                                  needs analysis 01 with teacher_forced_token_accuracy: true
  accuracy_{model}                pooled accuracy / chi-SMILES / non-chi-SMILES (summary)

Config YAML:
  input_dir: directory containing accuracy_{model}.csv and token_accuracy_{model}_phase{p}.csv
  output_dir: where to save PNG/SVG files
  models: list of model names (default: all accuracy_*.csv found in input_dir)
  tokenizer_path: vocabulary file (to count target tokens for the Fig 3 legend)
  eval_data: the eval CSV given to analysis 01; tgt_col (default "canonical")
  n_phases (5); token_phases (default [1, 2]); bucket_labels (default: the paper's five labels)
  trainer_states: {model: path to trainer_state.json of the final checkpoint} (optional)
  teacher_forced_token_accuracy (true): also plot the teacher-forced token accuracy
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from pan_core.common.tokenizer import SMILESTokenizer

PAPER_BUCKET_LABELS = ["ZINC (~100)", "PC (~100)", "PC (101~200)", "PC (201~300)", "PC (301~400)"]
BUCKET_COLORS = ["red", "orange", "green", "blue", "purple"]
TARGET_TOKENS = ["@", "@@", "/", "\\"]
LOW_FREQUENCY_COUNT = 10


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save(fig, path_stem: Path) -> None:
    for ext in ("png", "svg"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line(ax, series: pd.Series, **kwargs) -> None:
    """Plot a series with NaN rows dropped (lines connect across gaps, as seaborn did)."""
    series = series.dropna()
    ax.plot(series.index, series.values, **kwargs)


def token_counts_per_phase(eval_data: Path, tgt_col: str, tokenizer: SMILESTokenizer, n_phases: int) -> list[Counter]:
    """Occurrences of each token in the targets of each equal phase block."""
    targets = pd.read_csv(eval_data, usecols=[tgt_col])[tgt_col].tolist()
    if len(targets) % n_phases != 0:
        raise ValueError(f"{len(targets)} eval rows cannot be split into {n_phases} equal phase blocks")
    per_phase = len(targets) // n_phases
    return [
        Counter(token for smiles in targets[p * per_phase: (p + 1) * per_phase] for token in tokenizer.tokenize(smiles))
        for p in range(n_phases)
    ]


def read_trainer_state(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-loss rows (train_celoss) and per-bucket eval CE rows (eval_ce_phase*) of log_history."""
    with open(path) as f:
        log_history = json.load(f)["log_history"]
    train = pd.DataFrame([log for log in log_history if "train_celoss" in log])
    evaluation = pd.DataFrame([log for log in log_history if "eval_ce_phase1" in log])
    if train.empty or evaluation.empty:
        raise ValueError(f"No train_celoss / eval_ce_phase1 entries in {path}")
    return train, evaluation


def plot_bucket_accuracy(ax, df: pd.DataFrame, labels: list[str]) -> None:
    for p, (label, color) in enumerate(zip(labels, BUCKET_COLORS), start=1):
        _line(ax, df[f"phase{p}_accuracy"], color=color, linewidth=2.5, label=f"Acc: {label}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy", fontsize=20)
    ax.grid(True, axis="y", color="lightgray", linestyle="-", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=11, frameon=True)


def plot_token_accuracy(df: pd.DataFrame, counts: Counter):
    df = df.dropna(axis=1, how="all")
    tokens = list(df.columns)
    other_tokens = [t for t in tokens if t not in TARGET_TOKENS]
    color_map = dict(zip(other_tokens, sns.color_palette("husl", len(other_tokens))))
    fig, ax = plt.subplots(figsize=(12, 8))
    for token in tokens:
        if token not in counts:
            raise ValueError(f"Token {token!r} has an accuracy but does not occur in the targets")
        label = f"{token} (n={counts[token]})"
        if token in TARGET_TOKENS:
            _line(ax, df[token], label=label, linewidth=2.5, marker="o", markersize=2, zorder=100)
        elif counts[token] <= LOW_FREQUENCY_COUNT:
            _line(ax, df[token], label=label, color="lightgray", linewidth=0.5, alpha=0.3, zorder=1)
        else:
            _line(ax, df[token], label=label, color=color_map[token], linewidth=1.0, alpha=0.4, zorder=10)
    ax.set_xlabel("steps", fontsize=18)
    ax.set_ylabel("accuracy", fontsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False,
              ncol=2 if len(tokens) > 20 else 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize accuracy progress.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    n_phases = cfg.get("n_phases", 5)
    labels = cfg.get("bucket_labels", PAPER_BUCKET_LABELS)
    if len(labels) != n_phases or n_phases > len(BUCKET_COLORS):
        raise ValueError(f"bucket_labels must list {n_phases} labels (at most {len(BUCKET_COLORS)} buckets)")
    token_phases = cfg.get("token_phases", [1, 2])
    teacher_forced = cfg.get("teacher_forced_token_accuracy", True)
    if not isinstance(teacher_forced, bool):
        raise ValueError("teacher_forced_token_accuracy must be a boolean")
    token_kinds = ["token_accuracy"] + (["token_accuracy_teacher_forced"] if teacher_forced else [])
    trainer_states = cfg.get("trainer_states") or {}

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    counts = token_counts_per_phase(Path(cfg["eval_data"]), cfg.get("tgt_col", "canonical"), tokenizer, n_phases)

    if "models" in cfg:
        model_names = cfg["models"]
    else:
        model_names = [p.stem.removeprefix("accuracy_") for p in sorted(input_dir.glob("accuracy_*.csv"))]
    unknown = set(trainer_states) - set(model_names)
    if unknown:
        raise ValueError(f"trainer_states given for unknown models: {sorted(unknown)}")

    for model_name in model_names:
        csv_path = input_dir / f"accuracy_{model_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing analysis 01 output: {csv_path}")
        df = pd.read_csv(csv_path, index_col="step").sort_index()

        # --- Fig 2 lower panel alone ---
        fig, ax = plt.subplots(figsize=(16, 6))
        plot_bucket_accuracy(ax, df, labels)
        ax.set_xlabel("steps", fontsize=20)
        save(fig, output_dir / f"bucket_accuracy_{model_name}")

        # --- Fig 2: loss (log) above accuracy ---
        if model_name in trainer_states:
            train, evaluation = read_trainer_state(Path(trainer_states[model_name]))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
            ax1.set_yscale("log")
            ax1.scatter(train["step"], train["train_celoss"], color="gray", alpha=0.2, s=1,
                        rasterized=True, label="Train Loss", zorder=1)
            for p, (label, color) in enumerate(zip(labels, BUCKET_COLORS), start=1):
                ax1.plot(evaluation["step"], evaluation[f"eval_ce_phase{p}"], color=color, linestyle="--",
                         label=f"Eval: {label}", zorder=5)
            ax1.set_ylabel("cross entropy loss\n(per token)", fontsize=20)
            ax1.set_ylim(1e-7, 10)
            ax1.grid(True, which="both", axis="y", color="lightgray", linestyle="-", alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=11, frameon=True)
            plot_bucket_accuracy(ax2, df, labels)
            ax2.set_xlabel("steps", fontsize=20)
            last_step = max(df.index.max(), evaluation["step"].max())
            ax2.set_xlim(-last_step / 140, last_step * 141 / 140)  # paper: (-5000, 705000) for 700k steps
            fig.subplots_adjust(hspace=0.1)
            fig.tight_layout()
            save(fig, output_dir / f"loss_acc_progress_{model_name}")

        # --- Fig S3: chi-SMILES accuracy and mistranslated accuracy w/o '@', buckets 1-2 ---
        fig, ax = plt.subplots(figsize=(12, 8))
        for p, colors in ((1, ("blue", "cyan")), (2, ("red", "pink"))):
            _line(ax, df[f"phase{p}_chiral_accuracy"], color=colors[0], linewidth=2.5, marker="o",
                  markersize=2, label=f"{labels[p - 1]} chi-SMILES Acc.")
            _line(ax, df[f"phase{p}_chiral_mistakes_ex_mark"], color=colors[1], linewidth=2.5, marker="o",
                  markersize=2, label=f"{labels[p - 1]} Mistranslated chi-SMILES Acc. (w/o '@')")
        ax.set_xlabel("steps", fontsize=18)
        ax.set_ylabel("accuracy", fontsize=18)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        save(fig, output_dir / f"chiral_accuracy_{model_name}")

        # --- Fig 3: token accuracy per bucket ---
        for kind in token_kinds:
            for p in token_phases:
                token_path = input_dir / f"{kind}_{model_name}_phase{p}.csv"
                if not token_path.exists():
                    raise FileNotFoundError(f"Missing analysis 01 output: {token_path}")
                token_df = pd.read_csv(token_path, index_col="step").sort_index()
                save(plot_token_accuracy(token_df, counts[p - 1]), output_dir / f"{kind}_{model_name}_phase{p}")

        # --- Summary: pooled accuracy ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for column, label in (("accuracy", "all"), ("chiral_accuracy", "chi-SMILES"),
                              ("non_chiral_accuracy", "non-chi-SMILES")):
            _line(ax, df[column], label=label)
        ax.set_xlabel("steps")
        ax.set_ylabel("accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        save(fig, output_dir / f"accuracy_{model_name}")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
