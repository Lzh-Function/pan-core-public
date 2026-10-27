"""
05_residual_stream.py

Computes per-layer residual-stream metrics across checkpoints.

For each checkpoint:
  1. Per-layer L2 norm (chiral and background tokens; paper Fig 6a,b / S7a,b)
     → l2_norm_{chiral|background}_{enc|dec}_trajectory.csv
       rows: training step, columns: embeddings, Layer_0, ..., Layer_7

  2. Adjacent-layer cosine similarity within the SAME checkpoint (depth axis; not a paper figure):
     → cos_{chiral|background}_{enc|dec}_layerwise_trajectory.csv
       rows: training step, columns: Emb-L0, L0-L1, ..., L6-L7

  3. Adjacent-layer Linear CKA within the SAME checkpoint (depth axis; not a paper figure):
     → cka_{chiral|background}_{enc|dec}_layerwise_trajectory.csv
       rows: training step, columns: Emb-L0, L0-L1, ..., L6-L7

NOTE: L2 shift (||h_{L+1} - h_L||) is abolished.
NOTE: the paper's checkpoint-to-checkpoint cosine (Fig 6c,d / S6 / S7c,d) is computed by
      03_jumpup_metrics.py (cos_*_trajectory.csv) and plotted by 13_residual_stream.py.

Encoder tokens come from src_col, decoder tokens from tgt_col (teacher forcing); chiral =
'@'/'@@' tokens, background = all other tokens incl. BOS/EOS.

Config YAML:
  model_name, checkpoint_dir, checkpoint_steps (optional list),
  tokenizer_path, eval_data, src_col ("random"), tgt_col ("canonical"), output_dir,
  phase / max_phase (optional; analyse only that 1-based block of max_phase equal blocks),
  seed (42), batch_size (128), max_len (512)

Paper settings: eval_data = sampled_valid_1k_seed_42.csv, phase 1 of 5 (ZINC20, 200 rows).
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
from pan_core_analysis.runner import (
    check_resume_signature,
    chiral_token_ids,
    pad_target_by_one,
    phase_bounds,
)


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


def _build_masks(
    smiles_list: list[str],
    tokenizer: SMILESTokenizer,
    seq_len: int,
    chiral_ids: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    actual_len = max(seq_len, 1)
    B = len(smiles_list)
    valid_mask = np.zeros((B, actual_len), dtype=bool)
    chiral_mask = np.zeros((B, actual_len), dtype=bool)
    for i, smi in enumerate(smiles_list):
        ids = tokenizer.encode(smi)
        length = min(len(ids), actual_len)
        valid_mask[i, :length] = True
        for pos, tid in enumerate(ids[:length]):
            if tid in chiral_ids:
                chiral_mask[i, pos] = True
    return valid_mask, chiral_mask


def collect_hidden_states(
    model: PanCoreTransformerModel,
    dataloader,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    """Collect all layer hidden states. Returns {"encoder": [...], "decoder": [...]}
    Each list element: [N_total, L, E] (batch dim first, transposed from model's [L, B, E]).
    Decoder list excludes the final logits entry.
    """
    result: dict[str, list[np.ndarray]] = {"encoder": [], "decoder": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="forward", leave=False):
            batch = pad_target_by_one(batch)  # decoder also sees EOS as input (paper: fixed-length padding)
            input_ids = batch.get("input_ids").to(device)
            enc_mask = batch.get("encoder_attention_mask")
            labels = batch.get("labels")
            dec_mask = batch.get("decoder_attention_mask")
            if enc_mask is not None:
                enc_mask = enc_mask.to(device)
            if labels is not None:
                labels = labels.to(device)
            if dec_mask is not None:
                dec_mask = dec_mask.to(device)

            outputs = model.forward(
                input_ids=input_ids,
                encoder_attention_mask=enc_mask,
                labels=labels,
                decoder_attention_mask=dec_mask,
                need_hidden=True,
                calculate_loss=False,
                return_dict=True,
            )

            for side in ("encoder", "decoder"):
                hs_list = outputs.hidden_states[side]  # list of [L, B, E] numpy arrays
                # Decoder: last element is logits [L, B, V] — exclude
                if side == "decoder":
                    hs_list = hs_list[:-1]
                if len(result[side]) == 0:
                    result[side] = [h.transpose(1, 0, 2) for h in hs_list]  # [B, L, E]
                else:
                    for i, h in enumerate(hs_list):
                        h_t = h.transpose(1, 0, 2)  # [B, L_new, E]
                        acc = result[side][i]        # [N_prev, L_prev, E]
                        L_prev, L_new = acc.shape[1], h_t.shape[1]
                        if L_prev < L_new:
                            acc = np.concatenate(
                                [acc, np.zeros((acc.shape[0], L_new - L_prev, acc.shape[2]), dtype=acc.dtype)], axis=1
                            )
                        elif L_new < L_prev:
                            h_t = np.concatenate(
                                [h_t, np.zeros((h_t.shape[0], L_prev - L_new, h_t.shape[2]), dtype=h_t.dtype)], axis=1
                            )
                        result[side][i] = np.concatenate([acc, h_t], axis=0)

    return result


def compute_layerwise_metrics(
    hidden_states: list[np.ndarray],
    valid_mask: np.ndarray,
    chiral_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute L2 norm and layer-to-layer cosine/CKA within a single checkpoint.

    Args:
        hidden_states: List of [N, L, E] arrays, one per layer (including embedding).
        valid_mask: [N, L] boolean.
        chiral_mask: [N, L] boolean.
    Returns:
        dict with keys: l2_norm_chiral, l2_norm_background,
                        cos_chiral, cos_background,
                        cka_chiral, cka_background.
        Each value is a 1-D array with length = num_layers (for L2 norm)
        or num_layers - 1 (for layer-to-layer metrics).
    """
    num_layers = len(hidden_states)
    background_mask = valid_mask & (~chiral_mask)

    l2_norm_chiral = np.zeros(num_layers)
    l2_norm_bg = np.zeros(num_layers)

    for i, h in enumerate(hidden_states):
        # h: [N, L, E]
        l2_norm_chiral[i], _ = M.compute_l2_norm(h, chiral_mask)
        l2_norm_bg[i], _ = M.compute_l2_norm(h, background_mask)

    # Layer-to-layer (adjacent layers within this checkpoint)
    num_transitions = num_layers - 1
    cos_chiral = np.zeros(num_transitions)
    cos_bg = np.zeros(num_transitions)
    cka_chiral = np.zeros(num_transitions)
    cka_bg = np.zeros(num_transitions)

    for t in range(num_transitions):
        h1 = hidden_states[t]    # [N, L, E]
        h2 = hidden_states[t + 1]

        vecs1_chi = h1[chiral_mask]     # [N_chiral_tokens, E]
        vecs2_chi = h2[chiral_mask]
        vecs1_bg = h1[background_mask]
        vecs2_bg = h2[background_mask]

        if len(vecs1_chi) >= 2:
            cos_chiral[t], _ = M.compute_cosine_similarity(vecs1_chi, vecs2_chi)
            cka_chiral[t] = M.compute_linear_cka(vecs1_chi, vecs2_chi)
        if len(vecs1_bg) >= 2:
            cos_bg[t], _ = M.compute_cosine_similarity(vecs1_bg, vecs2_bg)
            cka_bg[t] = M.compute_linear_cka(vecs1_bg, vecs2_bg)

    return dict(
        l2_norm_chiral=l2_norm_chiral,
        l2_norm_background=l2_norm_bg,
        cos_chiral=cos_chiral,
        cos_background=cos_bg,
        cka_chiral=cka_chiral,
        cka_background=cka_bg,
    )


def _layer_cols(num_layers: int) -> list[str]:
    return ["embeddings"] + [f"Layer_{i}" for i in range(num_layers - 1)]


def _transition_cols(num_layers: int) -> list[str]:
    """Columns for layer-to-layer transitions: Emb-L0, L0-L1, ..., L(N-2)-L(N-1)."""
    cols = ["Emb-L0"]
    for i in range(num_layers - 2):
        cols.append(f"L{i}-L{i+1}")
    return cols


def main():
    parser = argparse.ArgumentParser(description="Residual stream L2 norm and layer-wise CKA/cosine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name", "model")
    output_dir = Path(args.output_dir or cfg["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "05_residual_stream.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    src_col = cfg.get("src_col", "random")
    tgt_col = cfg.get("tgt_col", "canonical")
    max_len = cfg.get("max_len", 512)
    batch_size = cfg.get("batch_size", 128)
    phase = cfg.get("phase")
    max_phase = cfg.get("max_phase")
    chiral_ids = chiral_token_ids(tokenizer)

    import polars as pl
    eval_df = pl.read_csv(cfg["eval_data"])
    for col in (src_col, tgt_col):
        if col not in eval_df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg['eval_data']}")
    row_start, row_end = phase_bounds(len(eval_df), phase, max_phase)
    smiles_src = eval_df[src_col].to_list()[row_start:row_end]
    smiles_tgt = eval_df[tgt_col].to_list()[row_start:row_end]
    logger.info(f"Eval set: rows {row_start}..{row_end - 1} ({len(smiles_src)} samples)")

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
    if len(dataset) != len(smiles_src):
        raise ValueError(f"Dataset size {len(dataset)} does not match the selected rows ({len(smiles_src)})")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GeneralDataCollator(),
        num_workers=4,
    )

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

    # Resume: load already-completed steps from existing CSVs
    _csv_names = [
        "l2_norm_chiral_enc_trajectory", "l2_norm_background_enc_trajectory",
        "l2_norm_chiral_dec_trajectory", "l2_norm_background_dec_trajectory",
        "cos_chiral_enc_layerwise_trajectory", "cos_background_enc_layerwise_trajectory",
        "cos_chiral_dec_layerwise_trajectory", "cos_background_dec_layerwise_trajectory",
        "cka_chiral_enc_layerwise_trajectory", "cka_background_enc_layerwise_trajectory",
        "cka_chiral_dec_layerwise_trajectory", "cka_background_dec_layerwise_trajectory",
    ]
    check_resume_signature(
        output_dir / ".resume_signature.json", __file__, cfg,
        [output_dir / f"{name}.csv" for name in _csv_names],
    )
    done_steps: set[int] = set()
    existing_dfs: dict[str, pd.DataFrame] = {}
    if any((output_dir / f"{name}.csv").exists() for name in _csv_names):
        step_sets = []
        for _name in _csv_names:
            _p = output_dir / f"{_name}.csv"
            if not _p.exists():
                raise FileNotFoundError(f"Incomplete previous run: {_p} is missing; delete {output_dir} and rerun")
            existing_dfs[_name] = pd.read_csv(_p, index_col=0)
            step_sets.append(set(int(s) for s in existing_dfs[_name].index))
        if any(steps != step_sets[0] for steps in step_sets):
            raise ValueError(f"Previous run left CSVs with different steps in {output_dir}; delete it and rerun")
        done_steps = step_sets[0]
        logger.info(f"Resuming: {len(done_steps)} steps already done")

    def _save(data_list, columns, name, path):
        new_df = pd.DataFrame(data_list, index=steps_list, columns=columns)
        if name in existing_dfs:
            combined = pd.concat([existing_dfs[name], new_df]).sort_index()
        else:
            combined = new_df
        combined.to_csv(path)

    # Precompute masks from data (seq_len = global max token length in eval set)
    seq_len_enc = max(min(len(tokenizer.encode(s)), max_len) for s in smiles_src)
    seq_len_dec = max(min(len(tokenizer.encode(s)), max_len) for s in smiles_tgt)
    vm_enc, cm_enc = _build_masks(smiles_src, tokenizer, seq_len_enc, chiral_ids)
    vm_dec, cm_dec = _build_masks(smiles_tgt, tokenizer, seq_len_dec, chiral_ids)

    # History (new checkpoints only)
    steps_list: list[int] = []
    l2_chiral_enc: list[np.ndarray] = []
    l2_bg_enc: list[np.ndarray] = []
    l2_chiral_dec: list[np.ndarray] = []
    l2_bg_dec: list[np.ndarray] = []
    cos_chiral_enc: list[np.ndarray] = []
    cos_bg_enc: list[np.ndarray] = []
    cka_chiral_enc: list[np.ndarray] = []
    cka_bg_enc: list[np.ndarray] = []
    cos_chiral_dec: list[np.ndarray] = []
    cos_bg_dec: list[np.ndarray] = []
    cka_chiral_dec: list[np.ndarray] = []
    cka_bg_dec: list[np.ndarray] = []

    for ckpt in all_ckpts:
        step = int(ckpt.name.split("-")[1])
        if step in done_steps:
            logger.info(f"Skipping {ckpt.name} (already done)")
            continue
        logger.info(f"→ {ckpt.name} (step {step})")

        model = PanCoreTransformerModel.from_pretrained(ckpt)
        model.to(device).eval()

        hs = collect_hidden_states(model, dataloader, device)
        del model
        torch.cuda.empty_cache()

        # Adjust masks if accumulated seq_len differs from precomputed
        _sl_enc = hs["encoder"][0].shape[1]
        _sl_dec = hs["decoder"][0].shape[1]
        _vm_enc = vm_enc[:, :_sl_enc] if _sl_enc <= vm_enc.shape[1] else np.pad(
            vm_enc, ((0, 0), (0, _sl_enc - vm_enc.shape[1])))
        _cm_enc = cm_enc[:, :_sl_enc] if _sl_enc <= cm_enc.shape[1] else np.pad(
            cm_enc, ((0, 0), (0, _sl_enc - cm_enc.shape[1])))
        _vm_dec = vm_dec[:, :_sl_dec] if _sl_dec <= vm_dec.shape[1] else np.pad(
            vm_dec, ((0, 0), (0, _sl_dec - vm_dec.shape[1])))
        _cm_dec = cm_dec[:, :_sl_dec] if _sl_dec <= cm_dec.shape[1] else np.pad(
            cm_dec, ((0, 0), (0, _sl_dec - cm_dec.shape[1])))

        steps_list.append(step)

        for side, _vm, _cm, l2c, l2b, cosc, cosb, ckac, ckab in [
            ("encoder", _vm_enc, _cm_enc, l2_chiral_enc, l2_bg_enc,
             cos_chiral_enc, cos_bg_enc, cka_chiral_enc, cka_bg_enc),
            ("decoder", _vm_dec, _cm_dec, l2_chiral_dec, l2_bg_dec,
             cos_chiral_dec, cos_bg_dec, cka_chiral_dec, cka_bg_dec),
        ]:
            m = compute_layerwise_metrics(hs[side], _vm, _cm)
            l2c.append(m["l2_norm_chiral"])
            l2b.append(m["l2_norm_background"])
            cosc.append(m["cos_chiral"])
            cosb.append(m["cos_background"])
            ckac.append(m["cka_chiral"])
            ckab.append(m["cka_background"])

        # Flush incremental CSV after each checkpoint for crash safety
        _nl_enc = len(l2_chiral_enc[-1])
        _nl_dec = len(l2_chiral_dec[-1])
        _lc_enc = _layer_cols(_nl_enc)
        _lc_dec = _layer_cols(_nl_dec)
        _tc_enc = _transition_cols(_nl_enc)
        _tc_dec = _transition_cols(_nl_dec)
        _save(l2_chiral_enc, _lc_enc, "l2_norm_chiral_enc_trajectory",
              output_dir / "l2_norm_chiral_enc_trajectory.csv")
        _save(l2_bg_enc, _lc_enc, "l2_norm_background_enc_trajectory",
              output_dir / "l2_norm_background_enc_trajectory.csv")
        _save(l2_chiral_dec, _lc_dec, "l2_norm_chiral_dec_trajectory",
              output_dir / "l2_norm_chiral_dec_trajectory.csv")
        _save(l2_bg_dec, _lc_dec, "l2_norm_background_dec_trajectory",
              output_dir / "l2_norm_background_dec_trajectory.csv")
        _save(cos_chiral_enc, _tc_enc, "cos_chiral_enc_layerwise_trajectory",
              output_dir / "cos_chiral_enc_layerwise_trajectory.csv")
        _save(cos_bg_enc, _tc_enc, "cos_background_enc_layerwise_trajectory",
              output_dir / "cos_background_enc_layerwise_trajectory.csv")
        _save(cos_chiral_dec, _tc_dec, "cos_chiral_dec_layerwise_trajectory",
              output_dir / "cos_chiral_dec_layerwise_trajectory.csv")
        _save(cos_bg_dec, _tc_dec, "cos_background_dec_layerwise_trajectory",
              output_dir / "cos_background_dec_layerwise_trajectory.csv")
        _save(cka_chiral_enc, _tc_enc, "cka_chiral_enc_layerwise_trajectory",
              output_dir / "cka_chiral_enc_layerwise_trajectory.csv")
        _save(cka_bg_enc, _tc_enc, "cka_background_enc_layerwise_trajectory",
              output_dir / "cka_background_enc_layerwise_trajectory.csv")
        _save(cka_chiral_dec, _tc_dec, "cka_chiral_dec_layerwise_trajectory",
              output_dir / "cka_chiral_dec_layerwise_trajectory.csv")
        _save(cka_bg_dec, _tc_dec, "cka_background_dec_layerwise_trajectory",
              output_dir / "cka_background_dec_layerwise_trajectory.csv")

    # ---------------------------------------------------------------------------
    # Export (merge new rows with any existing data from resume)
    # ---------------------------------------------------------------------------
    logger.info("Saving CSVs...")

    if not steps_list:
        logger.info("No new checkpoints processed; existing CSVs are up-to-date.")
        return

    num_layers_enc = len(l2_chiral_enc[0]) if l2_chiral_enc else 9
    num_layers_dec = len(l2_chiral_dec[0]) if l2_chiral_dec else 9
    layer_cols_enc = _layer_cols(num_layers_enc)
    layer_cols_dec = _layer_cols(num_layers_dec)
    trans_cols_enc = _transition_cols(num_layers_enc)
    trans_cols_dec = _transition_cols(num_layers_dec)

    _save(l2_chiral_enc, layer_cols_enc, "l2_norm_chiral_enc_trajectory",
          output_dir / "l2_norm_chiral_enc_trajectory.csv")
    _save(l2_bg_enc, layer_cols_enc, "l2_norm_background_enc_trajectory",
          output_dir / "l2_norm_background_enc_trajectory.csv")
    _save(l2_chiral_dec, layer_cols_dec, "l2_norm_chiral_dec_trajectory",
          output_dir / "l2_norm_chiral_dec_trajectory.csv")
    _save(l2_bg_dec, layer_cols_dec, "l2_norm_background_dec_trajectory",
          output_dir / "l2_norm_background_dec_trajectory.csv")

    _save(cos_chiral_enc, trans_cols_enc, "cos_chiral_enc_layerwise_trajectory",
          output_dir / "cos_chiral_enc_layerwise_trajectory.csv")
    _save(cos_bg_enc, trans_cols_enc, "cos_background_enc_layerwise_trajectory",
          output_dir / "cos_background_enc_layerwise_trajectory.csv")
    _save(cos_chiral_dec, trans_cols_dec, "cos_chiral_dec_layerwise_trajectory",
          output_dir / "cos_chiral_dec_layerwise_trajectory.csv")
    _save(cos_bg_dec, trans_cols_dec, "cos_background_dec_layerwise_trajectory",
          output_dir / "cos_background_dec_layerwise_trajectory.csv")

    _save(cka_chiral_enc, trans_cols_enc, "cka_chiral_enc_layerwise_trajectory",
          output_dir / "cka_chiral_enc_layerwise_trajectory.csv")
    _save(cka_bg_enc, trans_cols_enc, "cka_background_enc_layerwise_trajectory",
          output_dir / "cka_background_enc_layerwise_trajectory.csv")
    _save(cka_chiral_dec, trans_cols_dec, "cka_chiral_dec_layerwise_trajectory",
          output_dir / "cka_chiral_dec_layerwise_trajectory.csv")
    _save(cka_bg_dec, trans_cols_dec, "cka_background_dec_layerwise_trajectory",
          output_dir / "cka_background_dec_layerwise_trajectory.csv")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
