"""
04_latent_trajectory.py

Computes latent vector geometry metrics across checkpoints.

Static metrics (per checkpoint):
  L2 norm (paper Fig 7a), active units, intrinsic dimension (TwoNN), isotropy entropy,
  radial mean (mean L2 norm), radial CV (std/mean)

Dynamic metrics (between consecutive checkpoints):
  Linear CKA (Fig 7c / S8b), Kernel CKA, cosine similarity (Fig 7b / S8a)

Computed separately for chiral SMILES ('@' in the tgt_col string) and non-chiral SMILES.
The encoder input is src_col; the paper (Fig 7, S8) encoded the canonical SMILES
(src_col: canonical) of bucket 1 of valid_10k.csv (phase 1 of 5, 2,000 rows).

Outputs to {output_dir}/{model_name}/:
  latent_static_chiral_trajectory.csv
  latent_static_nchiral_trajectory.csv
  latent_dynamic_chiral_trajectory.csv   (indexed by the later step; columns from_step and
  latent_dynamic_nchiral_trajectory.csv   mid_step = (from_step + step) / 2, the paper's x position)

Resume: skips already-computed steps; re-runs last completed step on restart
to restore prev_latent for dynamic metrics.

Config YAML:
  model_name, checkpoint_dir, checkpoint_steps (optional list),
  tokenizer_path, eval_data, src_col (default "canonical"), tgt_col (default "canonical"),
  phase / max_phase (optional; analyse only that 1-based block of max_phase equal blocks),
  output_dir, seed (default 42), batch_size (default 256), max_len (default 512)
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
from tqdm import tqdm

from pan_core.common.data_handler import GeneralDataCollator, OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.models import PanCoreTransformerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis import metrics as M
from pan_core_analysis.runner import check_resume_signature, phase_bounds


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


def collect_latents(
    model: PanCoreTransformerModel,
    dataloader,
    device: torch.device,
) -> np.ndarray:
    """Run the encoder (fp32) and collect latent vectors [N, E] (mu for VAE models)."""
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="latent", leave=False):
            input_ids = batch["input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            latent = model.encode(input_ids, enc_mask)  # [B, E]; raises for S-T cross-attention
            parts.append(latent.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def compute_kernel_cka_gpu(X: np.ndarray, Y: np.ndarray, device: torch.device) -> float:
    """Kernel CKA with RBF kernel (median heuristic sigma), computed on GPU."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)

    def rbf_kernel(A):
        d2 = torch.cdist(A, A, p=2) ** 2
        sigma = torch.sqrt(torch.median(d2[d2 > 0]) + 1e-10)
        return torch.exp(-d2 / (2 * sigma ** 2))

    def center(K):
        row_mean = K.mean(dim=0, keepdim=True)
        col_mean = K.mean(dim=1, keepdim=True)
        grand_mean = K.mean()
        return K - row_mean - col_mean + grand_mean

    Kc = center(rbf_kernel(Xt))
    Lc = center(rbf_kernel(Yt))
    hsic_kl = (Kc * Lc).sum()
    hsic_kk = Kc.norm(p="fro") ** 2
    hsic_ll = Lc.norm(p="fro") ** 2
    score = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return float(score.item())


def _append_row(csv_path: Path, step: int, row: dict):
    """Append one row (keyed by step) to a CSV, creating the file on first call."""
    df_new = pd.DataFrame([row], index=pd.Index([step], name="step"))
    if csv_path.exists():
        existing = pd.read_csv(csv_path, index_col="step")
        df_new = pd.concat([existing, df_new]).sort_index()
    df_new.to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Latent vector geometry trajectory.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name", "model")
    output_dir = Path(args.output_dir or cfg["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "04_latent_trajectory.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    src_col = cfg.get("src_col", "canonical")
    tgt_col = cfg.get("tgt_col", "canonical")
    max_len = cfg.get("max_len", 512)
    batch_size = cfg.get("batch_size", 256)
    phase = cfg.get("phase")
    max_phase = cfg.get("max_phase")

    import polars as pl
    eval_df = pl.read_csv(cfg["eval_data"])
    for col in (src_col, tgt_col):
        if col not in eval_df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg['eval_data']}")
    row_start, row_end = phase_bounds(len(eval_df), phase, max_phase)
    canonical = eval_df[tgt_col].to_list()[row_start:row_end]
    is_chiral = np.array(["@" in s for s in canonical])
    logger.info(f"Eval rows {row_start}..{row_end - 1}; encoder input column '{src_col}'")
    logger.info(f"Chiral: {is_chiral.sum()}, Non-chiral: {(~is_chiral).sum()}")

    dataset = OnTheFlyDataset(
        file_path_or_df=Path(cfg["eval_data"]),
        tokenizer=tokenizer,
        src_col_name=src_col,
        tgt_col_name=tgt_col,
        mode="eval",
        maxlen=max_len,
        phase=phase,
        max_phase=max_phase,
    )
    if len(dataset) != len(canonical):
        raise ValueError(f"Dataset size {len(dataset)} does not match the selected rows ({len(canonical)})")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GeneralDataCollator(),
        num_workers=4,
    )

    # Discover checkpoints
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    target_steps: list[int] | None = cfg.get("checkpoint_steps")
    all_ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if target_steps is not None:
        tset = set(target_steps)
        all_ckpts = [p for p in all_ckpts if int(p.name.split("-")[1]) in tset]
    logger.info(f"Processing {len(all_ckpts)} checkpoints")

    # CSV paths
    csv_static_chi  = output_dir / "latent_static_chiral_trajectory.csv"
    csv_static_nchi = output_dir / "latent_static_nchiral_trajectory.csv"
    csv_dyn_chi     = output_dir / "latent_dynamic_chiral_trajectory.csv"
    csv_dyn_nchi    = output_dir / "latent_dynamic_nchiral_trajectory.csv"

    # Resume: determine already-done steps (all four CSVs must agree)
    csv_paths = [csv_static_chi, csv_static_nchi, csv_dyn_chi, csv_dyn_nchi]
    check_resume_signature(output_dir / ".resume_signature.json", __file__, cfg, csv_paths)
    done_steps: set[int] = set()
    if any(path.exists() for path in csv_paths):
        step_sets = []
        for path in csv_paths:
            if not path.exists():
                raise FileNotFoundError(f"Incomplete previous run: {path} is missing; delete {output_dir} and rerun")
            step_sets.append(set(int(s) for s in pd.read_csv(path, index_col="step").index))
        if any(steps != step_sets[0] for steps in step_sets):
            raise ValueError(f"Previous run left CSVs with different steps in {output_dir}; delete it and rerun")
        done_steps = step_sets[0]
        logger.info(f"Resuming: {len(done_steps)} steps already done")

    # Restore prev_latent from the last completed checkpoint
    prev_latent_chi: np.ndarray | None = None
    prev_latent_nchi: np.ndarray | None = None
    prev_step: int | None = None
    pending = [int(p.name.split("-")[1]) for p in all_ckpts if int(p.name.split("-")[1]) not in done_steps]
    if done_steps and pending:
        last_done = max(done_steps)
        if min(pending) < last_done:
            raise ValueError(
                f"Pending checkpoint {min(pending)} precedes the last completed step {last_done}; "
                "dynamic metrics would use the wrong predecessor. Delete the outputs and rerun."
            )
        last_ckpt = checkpoint_dir / f"checkpoint-{last_done}"
        if not last_ckpt.is_dir():
            raise FileNotFoundError(f"Cannot resume: last completed checkpoint is missing: {last_ckpt}")
        logger.info(f"Restoring prev_latent from checkpoint-{last_done}...")
        model = PanCoreTransformerModel.from_pretrained(last_ckpt)
        model.to(device).eval()
        latent_all = collect_latents(model, dataloader, device)
        del model
        torch.cuda.empty_cache()
        prev_latent_chi  = latent_all[is_chiral]
        prev_latent_nchi = latent_all[~is_chiral]
        prev_step = last_done

    for ckpt in all_ckpts:
        step = int(ckpt.name.split("-")[1])
        if step in done_steps:
            logger.info(f"Skipping step {step} (already done)")
            continue

        logger.info(f"→ {ckpt.name} (step {step})")
        model = PanCoreTransformerModel.from_pretrained(ckpt)
        model.to(device).eval()
        latent_all = collect_latents(model, dataloader, device)
        del model
        torch.cuda.empty_cache()

        latent_chi  = latent_all[is_chiral]
        latent_nchi = latent_all[~is_chiral]

        # Static metrics
        for latent, csv_path in [(latent_chi, csv_static_chi), (latent_nchi, csv_static_nchi)]:
            if len(latent) < 2:
                row: dict = {}
            else:
                norms = np.linalg.norm(latent, axis=1)
                row = dict(
                    l2_norm_mean=float(norms.mean()),
                    l2_norm_std=float(norms.std()),
                    active_units=M.compute_active_units(latent),
                    intrinsic_dim=M.compute_intrinsic_dimension_twonn(latent),
                    isotropy_entropy=M.compute_isotropy_entropy(latent),
                    radial_mean=float(norms.mean()),
                    radial_cv=float(norms.std() / (norms.mean() + 1e-10)),
                )
            _append_row(csv_path, step, row)

        # Dynamic metrics vs previous checkpoint
        for latent_curr, latent_prev, csv_path in [
            (latent_chi,  prev_latent_chi,  csv_dyn_chi),
            (latent_nchi, prev_latent_nchi, csv_dyn_nchi),
        ]:
            if latent_prev is not None and len(latent_curr) >= 2 and len(latent_prev) >= 2:
                cosine_mean, cosine_std = M.compute_cosine_similarity(latent_prev, latent_curr)
                row = dict(
                    from_step=prev_step,
                    mid_step=(prev_step + step) / 2,
                    linear_cka=M.compute_linear_cka(latent_prev, latent_curr),
                    kernel_cka=compute_kernel_cka_gpu(latent_prev, latent_curr, device),
                    cosine_mean=cosine_mean,
                    cosine_std=cosine_std,
                )
            else:
                row = {}
            _append_row(csv_path, step, row)

        prev_latent_chi  = latent_chi
        prev_latent_nchi = latent_nchi
        prev_step = step
        logger.info(f"  step {step} saved")

    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
