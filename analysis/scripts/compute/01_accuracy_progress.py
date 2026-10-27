"""
01_accuracy_progress.py

Evaluates greedy-decoding accuracy across checkpoints for one model.

The eval CSV is made of n_phases equal consecutive length buckets (paper: valid_10k.csv,
5 x 2,000 rows: ZINC (~100), PubChem (~100), (101~200), (201~300), (301~400)).

Outputs in output_dir:
  accuracy_{model_name}.csv  (index: step)
    accuracy, chiral_accuracy, non_chiral_accuracy, chiral_mistakes_ex_mark_accuracy
        — pooled over all rows
    phase{p}_accuracy, phase{p}_chiral_accuracy, phase{p}_chiral_mistakes_ex_mark,
    phase{p}_background_accuracy, phase{p}_n_chiral, phase{p}_n_chiral_mistakes,
    phase{p}_n_background
        — per bucket (paper Fig 2 lower panels, Table 2, Fig S3)
  token_accuracy_{model_name}_phase{p}.csv  (index: step; one column per vocabulary token)
        — paper Fig 3: generated and target SMILES of the bucket are tokenized and compared
          position by position over the shorter of the two; accuracy of a token type =
          matches / compared positions whose target token is that type (NaN if none).
          This uses the free-running greedy outputs, as the code behind arXiv v1 Fig 3 did.
  token_accuracy_teacher_forced_{model_name}_phase{p}.csv  (same layout; teacher_forced_token_accuracy)
        — teacher-forced token accuracy as described in the paper Methods: the decoder receives
          the ground-truth canonical prefix and the argmax of its logits (fp32) at each target
          SMILES token (BOS and EOS excluded) is compared with that token; accuracy of a token
          type = correct / target occurrences. Target counts are checked against the tokenizer.

Config YAML:
  model_name, checkpoint_dir, checkpoint_steps (optional list), tokenizer_path, eval_data,
  src_col (default "random"), tgt_col (default "canonical"), output_dir, seed (42),
  batch_size (default 3584, the paper run), max_length (402), use_bf16 (true), n_phases (5),
  teacher_forced_token_accuracy (true), teacher_forced_batch_size (500)

Usage:
  python 01_accuracy_progress.py --config config.yml [--output_dir /path/to/out]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pan_core.common.data_handler import GeneralDataCollator, OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.models import PanCoreTransformerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis.runner import check_resume_signature, compute_phase_metrics, run_generation_evaluation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_token_accuracy(generated: list[str], target: list[str], tokenizer: SMILESTokenizer) -> dict[str, float]:
    """Position-aligned token accuracy per target token type (paper Fig 3).

    Each pair is tokenized and compared over ``zip`` of the two token lists; a compared position
    counts toward its target token type, and is correct if the generated token is identical.
    """
    counts = {token: [0, 0] for token in tokenizer.get_vocab()}  # token -> [correct, total]
    for gen, tgt in zip(generated, target):
        for g, t in zip(tokenizer.tokenize(gen), tokenizer.tokenize(tgt)):
            if t not in counts:
                raise ValueError(f"Target token {t!r} is not in the vocabulary")
            counts[t][1] += 1
            if g == t:
                counts[t][0] += 1
    return {token: (c / n if n > 0 else float("nan")) for token, (c, n) in counts.items()}


def teacher_forced_logits(model: PanCoreTransformerModel, batch: dict) -> torch.Tensor:
    """Decoder logits [L-1, B, V] under teacher forcing; mirrors PanCoreTransformerModel.forward."""
    if model.config.vae:
        raise ValueError("Teacher-forced token accuracy does not support VAE models")
    src = batch["input_ids"].transpose(0, 1)
    tgt = batch["labels"].transpose(0, 1)
    enc_mask = batch["encoder_attention_mask"].unsqueeze(1).unsqueeze(2).bool()
    dec_mask = batch["decoder_attention_mask"][:, :-1].unsqueeze(1).unsqueeze(2).bool()
    if model.config.cross_attn == "S-T":
        encoder_out, encoder_mask = model.encoder(src, valid_token_mask=enc_mask)
        logits, _ = model.decoder(x=tgt[:-1, :], source=encoder_out, encoder_mask=encoder_mask, valid_token_mask=dec_mask)
    else:
        latent = model.encoder(src, valid_token_mask=enc_mask)
        logits, _ = model.decoder(x=tgt[:-1, :], latent=latent, valid_token_mask=dec_mask)
    return logits


def target_token_counts(targets: list[str], tokenizer: SMILESTokenizer, n_phases: int) -> np.ndarray:
    """[n_phases, vocab_size] occurrences of every token in the target SMILES of each bucket."""
    vocab = tokenizer.get_vocab()
    counts = np.zeros((n_phases, max(vocab.values()) + 1), dtype=np.int64)
    per_phase = len(targets) // n_phases
    for index, smiles in enumerate(targets):
        for token in tokenizer.tokenize(smiles):
            if token not in vocab:
                raise ValueError(f"Target token {token!r} is not in the vocabulary")
            counts[index // per_phase, vocab[token]] += 1
    return counts


def teacher_forced_token_accuracy(
    model: PanCoreTransformerModel,
    dataset,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    expected_counts: np.ndarray,
    batch_size: int,
) -> list[dict[str, float]]:
    """Teacher-forced token accuracy per bucket (fp32); one {token: accuracy} dict per bucket."""
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1
    if sorted(vocab.values()) != list(range(vocab_size)):
        raise ValueError("Vocabulary ids must be contiguous")
    id_to_token = {index: token for token, index in vocab.items()}
    eos_id = vocab[tokenizer.eos_token]
    n_phases = expected_counts.shape[0]
    per_phase = len(dataset) // n_phases
    correct = np.zeros((n_phases, vocab_size), dtype=np.int64)
    total = np.zeros((n_phases, vocab_size), dtype=np.int64)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=GeneralDataCollator(), num_workers=2
    )
    model = model.to(device).eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            prediction = teacher_forced_logits(model, batch).float().argmax(dim=-1).transpose(0, 1)  # [B, L-1]
            target = batch["labels"][:, 1:]
            valid = batch["decoder_attention_mask"][:, 1:].bool() & (target != eos_id)
            hit = (prediction == target) & valid
            phase_of_row = (torch.arange(target.size(0), device=device) + offset) // per_phase
            for phase in range(n_phases):
                rows = phase_of_row == phase
                if rows.any():
                    total[phase] += np.bincount(target[rows][valid[rows]].cpu().numpy(), minlength=vocab_size)
                    correct[phase] += np.bincount(target[rows][hit[rows]].cpu().numpy(), minlength=vocab_size)
            offset += target.size(0)
    if offset != len(dataset):
        raise ValueError(f"Teacher-forced pass covered {offset} of {len(dataset)} rows")
    if not np.array_equal(total, expected_counts):
        raise ValueError("Teacher-forced target token counts differ from the tokenized targets (truncation?)")
    return [
        {id_to_token[i]: (correct[p, i] / total[p, i] if total[p, i] > 0 else float("nan")) for i in range(vocab_size)}
        for p in range(n_phases)
    ]


def evaluate_checkpoint(
    ckpt_path: Path,
    dataset,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    step: int,
    *,
    batch_size: int,
    max_length: int,
    use_bf16: bool,
    n_phases: int,
    teacher_forced_counts: np.ndarray | None = None,
    teacher_forced_batch_size: int = 500,
) -> tuple[dict, list[dict], list[dict] | None]:
    model = PanCoreTransformerModel.from_pretrained(ckpt_path)
    model.eval()

    results = run_generation_evaluation(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_bf16=use_bf16,
        output_file=None,
        logger=logger,
        compile_model=True,
    )

    # Overall accuracy
    acc = float(results["judge"].mean())

    # Chiral vs non-chiral
    chiral_mask = results["target"].str.contains("@", regex=False)
    chiral_df = results[chiral_mask]
    non_chiral_df = results[~chiral_mask]
    chiral_acc = float(chiral_df["judge"].mean()) if len(chiral_df) > 0 else float("nan")
    non_chiral_acc = float(non_chiral_df["judge"].mean()) if len(non_chiral_df) > 0 else float("nan")

    # Chiral mistakes: removing '@' still correct → only chirality was wrong
    mistakes = chiral_df[~chiral_df["judge"]]
    if len(mistakes) > 0:
        gen_ex = mistakes["generated"].str.replace("@", "", regex=False)
        tgt_ex = mistakes["target"].str.replace("@", "", regex=False)
        chiral_mistakes_ex = float((gen_ex == tgt_ex).mean())
    else:
        chiral_mistakes_ex = float("nan")

    phase_metrics = compute_phase_metrics(results, n_phases=n_phases)
    per_phase = len(results) // n_phases
    token_rows = []
    for p in range(n_phases):
        block = results.iloc[p * per_phase: (p + 1) * per_phase]
        token_rows.append(
            {"step": step, **compute_token_accuracy(block["generated"].tolist(), block["target"].tolist(), tokenizer)}
        )

    teacher_forced_rows = None
    if teacher_forced_counts is not None:
        teacher_forced_rows = [
            {"step": step, **accuracy}
            for accuracy in teacher_forced_token_accuracy(
                model, dataset, tokenizer, device, teacher_forced_counts, teacher_forced_batch_size
            )
        ]

    del model
    torch.cuda.empty_cache()

    row = dict(
        step=step,
        accuracy=acc,
        chiral_accuracy=chiral_acc,
        non_chiral_accuracy=non_chiral_acc,
        chiral_mistakes_ex_mark_accuracy=chiral_mistakes_ex,
        **phase_metrics,
    )
    return row, token_rows, teacher_forced_rows


def _append_atomic(path: Path, row: dict) -> None:
    """Append one row (keyed by step) to a CSV via a temporary file."""
    row_df = pd.DataFrame([row]).set_index("step")
    if path.exists():
        row_df = pd.concat([pd.read_csv(path, index_col="step"), row_df])
        row_df = row_df[~row_df.index.duplicated(keep="last")].sort_index()
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_df.to_csv(temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Accuracy progress across checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty output label")

    fh = logging.FileHandler(output_dir / f"01_accuracy_progress_{model_name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    batch_size = cfg.get("batch_size", 3584)
    max_length = cfg.get("max_length", 402)
    use_bf16 = cfg.get("use_bf16", True)
    n_phases = cfg.get("n_phases", 5)
    if not isinstance(use_bf16, bool):
        raise ValueError("use_bf16 must be a boolean")
    teacher_forced = cfg.get("teacher_forced_token_accuracy", True)
    teacher_forced_batch_size = cfg.get("teacher_forced_batch_size", 500)
    if not isinstance(teacher_forced, bool):
        raise ValueError("teacher_forced_token_accuracy must be a boolean")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])

    eval_data_path = Path(cfg["eval_data"])
    dataset = OnTheFlyDataset(
        file_path_or_df=eval_data_path,
        tokenizer=tokenizer,
        src_col_name=cfg.get("src_col", "random"),
        tgt_col_name=cfg.get("tgt_col", "canonical"),
        mode="eval",
    )
    if not dataset.has_target:
        raise ValueError(f"Column '{cfg.get('tgt_col', 'canonical')}' not found in {eval_data_path}")
    if len(dataset) % n_phases != 0:
        raise ValueError(f"{len(dataset)} eval rows cannot be split into {n_phases} equal phase blocks")
    logger.info(f"Eval dataset: {len(dataset)} samples from {eval_data_path}")
    teacher_forced_counts = None
    if teacher_forced:
        targets = pd.read_csv(eval_data_path)[cfg.get("tgt_col", "canonical")].tolist()
        if len(targets) != len(dataset):
            raise ValueError(f"{eval_data_path}: {len(targets)} targets but {len(dataset)} dataset rows")
        teacher_forced_counts = target_token_counts(targets, tokenizer, n_phases)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoints = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    target_steps = cfg.get("checkpoint_steps")
    if target_steps is not None:
        wanted = {int(s) for s in target_steps}
        checkpoints = [p for p in checkpoints if int(p.name.split("-")[1]) in wanted]
        missing = sorted(wanted - {int(p.name.split("-")[1]) for p in checkpoints})
        if missing:
            raise FileNotFoundError(f"checkpoint_steps not found in {checkpoint_dir}: {missing}")
    logger.info(f"Evaluating {len(checkpoints)} checkpoints in {checkpoint_dir}")

    out_csv = output_dir / f"accuracy_{model_name}.csv"
    token_csvs = [output_dir / f"token_accuracy_{model_name}_phase{p + 1}.csv" for p in range(n_phases)]
    teacher_forced_csvs = (
        [output_dir / f"token_accuracy_teacher_forced_{model_name}_phase{p + 1}.csv" for p in range(n_phases)]
        if teacher_forced else []
    )
    check_resume_signature(
        output_dir / f".resume_signature_01_{model_name}.json", __file__, cfg, [out_csv, *token_csvs, *teacher_forced_csvs]
    )

    # Resume: load already-completed steps (all outputs must agree)
    done_steps: set[int] = set()
    if out_csv.exists():
        done_steps = set(int(s) for s in pd.read_csv(out_csv, index_col="step").index)
        for path in [*token_csvs, *teacher_forced_csvs]:
            # A token CSV may hold one extra step written just before a crash; it is overwritten.
            if not path.exists() or not done_steps <= set(int(s) for s in pd.read_csv(path, index_col="step").index):
                raise ValueError(f"{path} is missing or out of sync with {out_csv}; delete the outputs and rerun")
        logger.info(f"Resuming: {len(done_steps)} steps already in {out_csv}")

    for ckpt in checkpoints:
        step = int(ckpt.name.split("-")[1])
        if step in done_steps:
            logger.info(f"Skipping checkpoint {ckpt.name} (already done)")
            continue
        logger.info(f"Evaluating checkpoint {ckpt.name} (step {step})...")
        row, token_rows, teacher_forced_rows = evaluate_checkpoint(
            ckpt, dataset, tokenizer, device, step,
            batch_size=batch_size, max_length=max_length, use_bf16=use_bf16, n_phases=n_phases,
            teacher_forced_counts=teacher_forced_counts, teacher_forced_batch_size=teacher_forced_batch_size,
        )
        logger.info(
            f"  step={step}  acc={row['accuracy']:.4f}  phase1_chiral_acc={row['phase1_chiral_accuracy']:.4f}"
        )
        # Token CSVs first: the accuracy CSV defines which steps are complete.
        for path, token_row in zip(token_csvs, token_rows):
            _append_atomic(path, token_row)
        for path, token_row in zip(teacher_forced_csvs, teacher_forced_rows or []):
            _append_atomic(path, token_row)
        _append_atomic(out_csv, row)

    logger.info(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
