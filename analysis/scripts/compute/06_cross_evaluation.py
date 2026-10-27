"""
06_cross_evaluation.py

Cross-evaluates pre-jump and post-jump encoder/decoder combinations.

Runs 4 combinations:
  Pre→Pre, Post→Pre, Pre→Post, Post→Post

For each: computes overall accuracy and, per phase bucket (the eval CSV is made of
n_phases equal consecutive blocks), accuracy, chiral accuracy, chiral mistakes (sans '@'),
and background (non-chiral) accuracy — the quantities of Table 3 / Table S4.

The paper run used fp32 encoding and decoding (use_bf16: false) with batch_size 1024,
and the jump-up interval start / end checkpoints as pre / post.

Output: cross_evaluation_{model_name}.csv

Config YAML:
  model_name, tokenizer_path, eval_data, src_col, tgt_col, output_dir, seed,
  batch_size (default 1024), max_length (default 402), use_bf16 (default false),
  n_phases (default 5),
  pre_jump_checkpoint: str
  post_jump_checkpoint: str
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
from pan_core_analysis.runner import compute_phase_metrics


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


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cross_eval(
    encoder_model: PanCoreTransformerModel,
    decoder_model: PanCoreTransformerModel,
    dataloader,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    max_length: int,
    use_bf16: bool,
) -> pd.DataFrame:
    """Run generation using encoder from one model and decoder from another."""
    from tqdm import tqdm

    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="cross-eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            labels = batch["labels"]

            # Encoder of one checkpoint → latent (mu for VAE models)
            latent = encoder_model.encode(input_ids, enc_mask, use_bf16=use_bf16)

            # Decode with the other checkpoint's decoder
            generated_ids, _, _ = decoder_model.generate(
                latents=latent,
                max_length=max_length,
                use_bf16=use_bf16,
                need_logits=False,
            )

            for i, ids in enumerate(generated_ids.cpu().tolist()):
                gen = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                tgt = tokenizer.decode(labels[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                records.append({"generated": gen, "target": tgt, "judge": tgt == gen})

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluation of encoder/decoder combinations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name", "model")
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "06_cross_evaluation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    src_col = cfg.get("src_col", "random")
    tgt_col = cfg.get("tgt_col", "canonical")
    batch_size = cfg.get("batch_size", 1024)
    max_length = cfg.get("max_length", 402)
    use_bf16 = cfg.get("use_bf16", False)
    n_phases = cfg.get("n_phases", 5)
    if not isinstance(use_bf16, bool):
        raise ValueError("use_bf16 must be a boolean")

    dataset = OnTheFlyDataset(
        file_path_or_df=Path(cfg["eval_data"]),
        tokenizer=tokenizer,
        src_col_name=src_col,
        tgt_col_name=tgt_col,
        mode="eval",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GeneralDataCollator(),
        num_workers=4,
    )

    pre_path = Path(cfg["pre_jump_checkpoint"])
    post_path = Path(cfg["post_jump_checkpoint"])

    logger.info(f"Loading pre-jump model from {pre_path}")
    pre_model = PanCoreTransformerModel.from_pretrained(pre_path).to(device).eval()
    logger.info(f"Loading post-jump model from {post_path}")
    post_model = PanCoreTransformerModel.from_pretrained(post_path).to(device).eval()

    combos = [
        ("pre", "pre", pre_model, pre_model),
        ("post", "pre", post_model, pre_model),
        ("pre", "post", pre_model, post_model),
        ("post", "post", post_model, post_model),
    ]

    rows = []
    for enc_label, dec_label, enc_m, dec_m in combos:
        label = f"{enc_label}enc_{dec_label}dec"
        logger.info(f"Running: {label}")
        results = run_cross_eval(enc_m, dec_m, dataloader, tokenizer, device, max_length, use_bf16)
        accuracy = float(results["judge"].mean())
        phase_metrics = compute_phase_metrics(results, n_phases=n_phases)
        rows.append({"combination": label, "accuracy": accuracy, **phase_metrics})
        results.to_csv(output_dir / f"decode_{model_name}_{label}.csv", index=False)
        logger.info(
            f"  acc={accuracy:.4f}  phase1 acc={phase_metrics['phase1_accuracy']:.4f}"
            f"  phase1 chiral_acc={phase_metrics['phase1_chiral_accuracy']:.4f}"
        )

    out_csv = output_dir / f"cross_evaluation_{model_name}.csv"
    pd.DataFrame(rows).set_index("combination").to_csv(out_csv)
    logger.info(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
