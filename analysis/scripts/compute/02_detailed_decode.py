"""
02_detailed_decode.py

Runs greedy decoding for a specified list of checkpoints and saves full
per-sample (input / target / generated / judge) CSVs, plus per-bucket accuracy.

Outputs in output_dir:
  decode_{model_name}_step{N}.csv   one per checkpoint step
  decode_summary_{model_name}.csv   index: step; overall accuracy and, per bucket p
                                    (n_phases equal consecutive blocks of eval_data),
                                    phase{p}_accuracy, _chiral_accuracy, _chiral_mistakes_ex_mark,
                                    _background_accuracy and counts
                                    (paper Table 2 "train" rows: eval_data = sampled_100k.csv,
                                    5 x 20,000 rows, checkpoint_steps: [700000])

Usage:
  python 02_detailed_decode.py --config config.yml [--output_dir /path/to/out]

Config YAML keys:
  model_name:       str
  checkpoint_dir:   str  # directory containing checkpoint-N subdirs
  checkpoint_steps: list[int]  # only evaluate these steps; null = all
  tokenizer_path:   str
  eval_data:        str
  src_col:          str (default: "random")
  tgt_col:          str (default: "canonical")
  output_dir:       str
  seed:             int (default: 42)
  batch_size:       int (default: 3584, the paper run)
  max_length:       int (default: 402)
  use_bf16:         bool (default: true)
  n_phases:         int (default: 5)
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

from pan_core.common.data_handler import OnTheFlyDataset
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


def main():
    parser = argparse.ArgumentParser(description="Detailed decode results per checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "02_detailed_decode.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])

    dataset = OnTheFlyDataset(
        file_path_or_df=Path(cfg["eval_data"]),
        tokenizer=tokenizer,
        src_col_name=cfg.get("src_col", "random"),
        tgt_col_name=cfg.get("tgt_col", "canonical"),
        mode="eval",
    )
    if not dataset.has_target:
        raise ValueError(f"Column '{cfg.get('tgt_col', 'canonical')}' not found in {cfg['eval_data']}")
    logger.info(f"Eval dataset: {len(dataset)} samples")

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    target_steps: list[int] | None = cfg.get("checkpoint_steps")

    all_ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if target_steps is not None:
        target_steps_set = {int(s) for s in target_steps}
        all_ckpts = [p for p in all_ckpts if int(p.name.split("-")[1]) in target_steps_set]
        missing = sorted(target_steps_set - {int(p.name.split("-")[1]) for p in all_ckpts})
        if missing:
            raise FileNotFoundError(f"checkpoint_steps not found in {checkpoint_dir}: {missing}")

    logger.info(f"Evaluating {len(all_ckpts)} checkpoints")

    model_name = cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty output label")
    batch_size = cfg.get("batch_size", 3584)
    max_length = cfg.get("max_length", 402)
    use_bf16 = cfg.get("use_bf16", True)
    n_phases = cfg.get("n_phases", 5)
    if not isinstance(use_bf16, bool):
        raise ValueError("use_bf16 must be a boolean")

    check_resume_signature(
        output_dir / f".resume_signature_02_{model_name}.json", __file__, cfg,
        list(output_dir.glob(f"decode_{model_name}_step*.csv")),
    )

    summary_rows = []
    for ckpt in all_ckpts:
        step = int(ckpt.name.split("-")[1])
        out_file = output_dir / f"decode_{model_name}_step{step}.csv"
        if out_file.exists():
            logger.info(f"Reusing step {step} — {out_file.name} already exists")
            results = pd.read_csv(out_file, keep_default_na=False)  # empty generations stay ""
            if results["judge"].dtype != bool:
                raise ValueError(f"Unexpected judge column in {out_file}")
        else:
            logger.info(f"→ checkpoint {ckpt.name}  (step {step})")

            model = PanCoreTransformerModel.from_pretrained(ckpt)
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
            temporary = out_file.with_suffix(".csv.tmp")
            results.to_csv(temporary, index=False)
            temporary.replace(out_file)

            del model
            torch.cuda.empty_cache()
            logger.info(f"  Saved → {out_file}")

        if len(results) != len(dataset):
            raise ValueError(f"{out_file} has {len(results)} rows, expected {len(dataset)}")
        summary_rows.append({
            "step": step,
            "accuracy": float(results["judge"].mean()),
            **compute_phase_metrics(results, n_phases=n_phases),
        })

    if summary_rows:
        summary_path = output_dir / f"decode_summary_{model_name}.csv"
        pd.DataFrame(summary_rows).set_index("step").sort_index().to_csv(summary_path)
        logger.info(f"Saved → {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
