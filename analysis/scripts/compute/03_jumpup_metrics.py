"""
03_jumpup_metrics.py

Computes all jump-up metrics across checkpoints for one model.

Head-level (per encoder/decoder layer × head):
  attention_mass, attention_entropy, SAR, graph_dist_from/to, seq_dist_from/to, NIT
Between consecutive checkpoints (head-level):
  JSD
Between consecutive checkpoints (layer-level):
  CKA (chiral/background), cosine similarity (chiral/background)
Global (logits-based):
  perplexity, CDR, mean_rank, mean_gap, chiral_margin

L2 shift (checkpoint-to-checkpoint) is NOT computed.

Outputs in {output_dir}/{model_name}/:
  attention_mass_{enc|dec}_trajectory.csv          [step × L_H]
  attention_entropy_{enc|dec}_trajectory.csv
  sar_{enc|dec}_trajectory.csv
  graph_dist_from_{enc|dec}_trajectory.csv
  seq_dist_from_{enc|dec}_trajectory.csv
  graph_dist_to_{enc|dec}_trajectory.csv
  seq_dist_to_{enc|dec}_trajectory.csv
  nit_{enc|dec}_trajectory.csv
  jsd_{enc|dec}_trajectory.csv                     [transition × L_H]
  cka_chiral_{enc|dec}_trajectory.csv              [transition × layer]
  cka_background_{enc|dec}_trajectory.csv
  cos_chiral_{enc|dec}_trajectory.csv
  cos_background_{enc|dec}_trajectory.csv
  logits_metrics_trajectory.csv                    [step × metric]

Head-level metrics are averaged over all chiral tokens of the evaluated rows (each
chiral token has equal weight, independent of batching).

Config YAML:
  model_name, checkpoint_dir, checkpoint_steps (optional list), tokenizer_path,
  eval_data, src_col (default "random"), tgt_col (default "canonical"),
  phase / max_phase (optional; analyse only that 1-based block of max_phase equal blocks),
  output_dir, seed (default 42), batch_size (default 128), max_len (default 512),
  compute_graph_metrics (default true)

Paper settings: eval_data = sampled_valid_1k_seed_42.csv, phase 1 of 5 (ZINC20, 200 rows).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from pan_core.common.data_handler import GeneralDataCollator, OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.common.utils import move_to_cpu
from pan_core.models import PanCoreTransformerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis import metrics as M
from pan_core_analysis.graph import create_token_graph_distance_matrix, collate_distance_matrices
from pan_core_analysis.runner import check_resume_signature, chiral_token_ids, pad_target_by_one, phase_bounds


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


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def _build_masks(
    smiles_list: list[str],
    tokenizer: SMILESTokenizer,
    seq_len: int,
    chiral_ids: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return [N, seq_len] valid_mask and chiral_mask."""
    N = len(smiles_list)
    valid_mask = np.zeros((N, seq_len), dtype=bool)
    chiral_mask = np.zeros((N, seq_len), dtype=bool)
    for i, smi in enumerate(smiles_list):
        ids = tokenizer.encode(smi)
        length = min(len(ids), seq_len)
        valid_mask[i, :length] = True
        for pos, tid in enumerate(ids[:length]):
            if tid in chiral_ids:
                chiral_mask[i, pos] = True
    return valid_mask, chiral_mask


# ---------------------------------------------------------------------------
# Forward pass — memory-efficient: NO full attention accumulation
# ---------------------------------------------------------------------------

def forward_pass(
    model: PanCoreTransformerModel,
    dataloader,
    device: torch.device,
    vm_enc: np.ndarray,       # [N, seq_len_enc]
    cm_enc: np.ndarray,       # [N, seq_len_enc]
    vm_dec: np.ndarray,       # [N, seq_len_dec]
    cm_dec: np.ndarray,       # [N, seq_len_dec]
    dist_enc_arr: np.ndarray | None,   # [N, seq_len_enc, seq_len_enc]
    dist_dec_arr: np.ndarray | None,   # [N, seq_len_dec, seq_len_dec]
    N: int,
    batch_size: int,
    smiles_tgt: list[str],
    tokenizer: SMILESTokenizer,
    max_len: int,
    chiral_ids: tuple[int, int],
) -> dict:
    """
    Run one full forward pass and return a compact result dict:

    {
      "static": {"enc": {key: [L, H]}, "dec": {...}},
      "jsd_data": {"enc": list[L] of [N_chiral, H, K], "dec": ...},
      "hidden_chiral": {"enc": list[L+1] of [N_chiral, E], "dec": ...},
      "hidden_bg": {"enc": list[L+1] of [N_bg, E], "dec": ...},
      "logits_metrics": dict,
    }

    Memory strategy:
    - Attention matrices are NEVER accumulated globally.
    - Static metrics (mass, entropy, etc.) are computed per batch and combined with
      per-batch chiral-token counts as weights (= mean over all chiral tokens).
    - Only attention AT CHIRAL QUERY POSITIONS is kept (for JSD later).
    - Only hidden states at chiral/background token positions are kept (for CKA).
    - Logits are accumulated (small: [N, L, vocab_size]).
    """
    # ------------------------------------------------------------------
    # Running accumulators
    # ------------------------------------------------------------------
    # Static metrics: chiral-token-weighted sums + chiral-token counts per side
    #   After filling in first batch we know shapes → then allocate np arrays
    static_sum: dict[str, dict[str, np.ndarray]] = {"enc": {}, "dec": {}}
    static_cnt: dict[str, int] = {"enc": 0, "dec": 0}

    # JSD data: attention at chiral query positions, per layer
    #   jsd_data[side][layer] = list of [N_chiral_b, H, K] batches
    jsd_data: dict[str, list[list]] = {"enc": [], "dec": []}

    # Hidden states at chiral / background positions, per layer (incl. embedding)
    #   hidden_chiral[side][layer] = list of [N_chiral_b, E] batches
    hidden_chiral: dict[str, list[list]] = {"enc": [], "dec": []}
    hidden_bg: dict[str, list[list]] = {"enc": [], "dec": []}

    # Logits accumulator: list of [B, L, V] arrays
    logits_batches: list[np.ndarray] = []

    # For logits metric labels (build once)
    max_logit_len = max_len
    labels_arr = np.zeros((N, max_logit_len), dtype=int)
    chiral_logits = np.zeros((N, max_logit_len), dtype=bool)
    for i, smi in enumerate(smiles_tgt):
        ids = tokenizer.encode(smi)
        ids_shifted = ids[1:]  # skip BOS
        length = min(len(ids_shifted), max_logit_len)
        labels_arr[i, :length] = ids_shifted[:length]
        for pos in range(length):
            if labels_arr[i, pos] in chiral_ids:
                chiral_logits[i, pos] = True

    sides_info = [
        ("enc", vm_enc, cm_enc, dist_enc_arr),
        ("dec", vm_dec, cm_dec, dist_dec_arr),
    ]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="forward", leave=False)):
            batch = pad_target_by_one(batch)  # decoder also sees EOS as input (paper: fixed-length padding)
            b_start = batch_idx * batch_size
            b_end = min(b_start + batch_size, N)

            input_ids = batch.get("input_ids")
            encoder_attention_mask = batch.get("encoder_attention_mask")
            labels = batch.get("labels")
            decoder_attention_mask = batch.get("decoder_attention_mask")

            if input_ids is not None:
                input_ids = input_ids.to(device)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(device)

            outputs = model.forward(
                input_ids=input_ids,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                need_hidden=True,
                calculate_loss=False,
                return_dict=True,
            )

            # hs_side: list of [L, B, E]  → use outputs.hidden_states
            # at_side: list of [B, H, Q, K] → use outputs.attentions
            hs = {"enc": outputs.hidden_states["encoder"],
                  "dec": outputs.hidden_states["decoder"]}
            at = {"enc": outputs.attentions["encoder"],
                  "dec": outputs.attentions["decoder"]}

            # Initialize layer-level lists on first batch
            if batch_idx == 0:
                for side in ("enc", "dec"):
                    n_layers = len(at[side])
                    n_hs = len(hs[side])
                    jsd_data[side] = [[] for _ in range(n_layers)]
                    hidden_chiral[side] = [[] for _ in range(n_hs)]
                    hidden_bg[side] = [[] for _ in range(n_hs)]

            for side, vm, cm, dm in sides_info:
                vm_b_full = vm[b_start:b_end]  # [B, global_seq_len]
                cm_b_full = cm[b_start:b_end]
                dm_b_full = dm[b_start:b_end] if dm is not None else None
                bg_b_full = vm_b_full & ~cm_b_full

                attn_list = at[side]  # list of [B, H, Q, K]

                # ---- Static metrics (per batch, averaged later) ----
                # Truncate masks to batch's actual attention size
                K = attn_list[0].shape[-1]
                vm_b = vm_b_full[:, :K]
                cm_b = cm_b_full[:, :K]
                dm_b = dm_b_full[:, :K, :K] if dm_b_full is not None else None
                bg_b = vm_b & ~cm_b

                num_layers = len(attn_list)
                num_heads = attn_list[0].shape[1]
                n_chiral_b = int(cm_b.sum())

                def _zeros():
                    return np.zeros((num_layers, num_heads))

                mass = _zeros(); ent = _zeros(); sar = _zeros()
                gd_from = _zeros(); sd_from = _zeros()
                gd_to = _zeros(); sd_to = _zeros(); nit = _zeros()

                for l, a in enumerate(attn_list):
                    if n_chiral_b == 0:
                        break  # no chiral token on this side of the batch: contributes nothing
                    # a: [B, H, Q, K]
                    mass[l], _ = M.compute_attention_mass(a, cm_b, vm_b)
                    ent[l], _ = M.compute_attention_entropy(a, cm_b, vm_b)
                    sd_from[l], _ = M.compute_seq_distance_from_chiral(a, cm_b, vm_b)
                    sd_to[l], _ = M.compute_seq_distance_to_chiral(a, cm_b, vm_b)
                    if dm_b is not None:
                        sar[l], _ = M.compute_sar(a, cm_b, vm_b, dm_b)
                        gd_from[l], _ = M.compute_graph_distance_from_chiral(a, cm_b, vm_b, dm_b)
                        gd_to[l], _ = M.compute_graph_distance_to_chiral(a, cm_b, vm_b, dm_b)
                        nit[l], _ = M.compute_nit(a, cm_b, vm_b, dm_b)

                batch_static = dict(attention_mass=mass, attention_entropy=ent, sar=sar,
                                    graph_dist_from=gd_from, seq_dist_from=sd_from,
                                    graph_dist_to=gd_to, seq_dist_to=sd_to, nit=nit)

                if not static_sum[side]:
                    for k, v in batch_static.items():
                        static_sum[side][k] = v * n_chiral_b
                else:
                    for k, v in batch_static.items():
                        static_sum[side][k] += v * n_chiral_b
                static_cnt[side] += n_chiral_b

                # ---- JSD data: attention at CHIRAL QUERY positions only ----
                for l, a in enumerate(attn_list):
                    B_, H, Q, K_ = a.shape
                    # Transpose to [B, Q, H, K] then flatten B*Q
                    a_bqhk = a.transpose(0, 2, 1, 3).reshape(B_ * Q, H, K_)
                    a_chiral = a_bqhk[cm_b_full[:, :Q].reshape(B_ * Q)]  # [N_chiral_b, H, K]
                    jsd_data[side][l].append(a_chiral)

                # ---- Hidden states at chiral / background positions ----
                hs_list = hs[side]  # list of [L, B, E] (L=seq_len here)
                for layer_i, h in enumerate(hs_list):
                    h_t = h.transpose(1, 0, 2)  # [B, L_h, E]
                    B_, L_h, E = h_t.shape
                    cm_h = cm_b_full[:, :L_h].reshape(B_ * L_h)
                    bg_h = bg_b_full[:, :L_h].reshape(B_ * L_h)
                    h_flat = h_t.reshape(B_ * L_h, E)
                    hidden_chiral[side][layer_i].append(h_flat[cm_h])   # [N_chiral_b, E]
                    hidden_bg[side][layer_i].append(h_flat[bg_h])       # [N_bg_b, E]

            # ---- Logits ----
            dec_logits = model.decoder.all_hidden_states[-1]  # [L, B, V]
            lgts_np = dec_logits.transpose(1, 0, 2)          # [B, L, V]
            logits_batches.append(lgts_np)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    static_out: dict[str, dict[str, np.ndarray]] = {}
    for side in ("enc", "dec"):
        if static_cnt[side] == 0:
            raise ValueError(f"No chiral token found on the {side} side of the evaluation set")
        static_out[side] = {k: v / static_cnt[side] for k, v in static_sum[side].items()}

    jsd_out: dict[str, list[np.ndarray]] = {}
    for side in ("enc", "dec"):
        jsd_out[side] = []
        for fragments in jsd_data[side]:
            if not fragments:
                jsd_out[side].append(np.zeros((0, 0, 0)))
                continue
            # K (key dimension) may vary across batches due to dynamic padding;
            # pad to max_K so fragments can be concatenated along axis 0.
            max_K = max(f.shape[-1] for f in fragments)
            padded = []
            for f in fragments:
                if f.shape[-1] < max_K:
                    pad = np.zeros((f.shape[0], f.shape[1], max_K - f.shape[-1]), dtype=f.dtype)
                    f = np.concatenate([f, pad], axis=-1)
                padded.append(f)
            jsd_out[side].append(np.concatenate(padded, axis=0))

    hidden_chiral_out: dict[str, list[np.ndarray]] = {}
    hidden_bg_out: dict[str, list[np.ndarray]] = {}
    for side in ("enc", "dec"):
        hidden_chiral_out[side] = [
            np.concatenate(fragments) if fragments else np.zeros((0, 0))
            for fragments in hidden_chiral[side]
        ]
        hidden_bg_out[side] = [
            np.concatenate(fragments) if fragments else np.zeros((0, 0))
            for fragments in hidden_bg[side]
        ]

    # Logits metrics
    logits_metrics: dict = {}
    if logits_batches:
        max_l = max(a.shape[1] for a in logits_batches)
        padded = []
        for a in logits_batches:
            pad = max_l - a.shape[1]
            if pad > 0:
                a = np.concatenate([a, np.zeros((a.shape[0], pad, a.shape[2]))], axis=1)
            padded.append(a)
        all_logits = np.concatenate(padded, axis=0)  # [N, L, V]
        L_logits = all_logits.shape[1]
        la = labels_arr[:, :L_logits]
        cl = chiral_logits[:, :L_logits]
        logits_metrics = M.compute_logits_metrics(
            all_logits, la, cl, atmark_id=chiral_ids[0], atatmark_id=chiral_ids[1]
        )

    return {
        "static": static_out,
        "jsd_data": jsd_out,
        "hidden_chiral": hidden_chiral_out,
        "hidden_bg": hidden_bg_out,
        "logits_metrics": logits_metrics,
    }


# ---------------------------------------------------------------------------
# Transition metrics between consecutive checkpoints
# ---------------------------------------------------------------------------

def compute_transition_metrics(curr: dict, prev: dict) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute JSD (head-level) and CKA/cosine (layer-level) between curr and prev.

    curr/prev have keys: jsd_data, hidden_chiral, hidden_bg.

    Returns:
        {"enc": {"jsd": [L, H], "cka_chiral": [n_hs], ...}, "dec": {...}}
    """
    result: dict[str, dict[str, np.ndarray]] = {}

    for side in ("enc", "dec"):
        # JSD: attention at chiral positions
        curr_jsd = curr["jsd_data"][side]   # list of [N_chiral, H, K] per layer
        prev_jsd = prev["jsd_data"][side]

        num_layers_attn = len(curr_jsd)
        num_heads = curr_jsd[0].shape[1] if curr_jsd and curr_jsd[0].ndim == 3 else 8
        jsd = np.zeros((num_layers_attn, num_heads))

        for l in range(num_layers_attn):
            ac = curr_jsd[l]  # [N_chiral, H, K]
            ap = prev_jsd[l]
            if ac.shape[0] == 0 or ap.shape[0] == 0:
                continue
            # Align K dimension (may differ across checkpoints due to padding)
            K = min(ac.shape[-1], ap.shape[-1])
            ac = ac[:, :, :K]
            ap = ap[:, :, :K]
            # Normalize rows (should already be valid distributions but ensure)
            ac_sum = ac.sum(axis=-1, keepdims=True)
            ap_sum = ap.sum(axis=-1, keepdims=True)
            ac = np.where(ac_sum > 0, ac / np.maximum(ac_sum, 1e-12), 0)
            ap = np.where(ap_sum > 0, ap / np.maximum(ap_sum, 1e-12), 0)
            jsd[l] = M.compute_jsd(ac, ap).mean(axis=0)  # mean over N_chiral → [H]

        # CKA / cosine: hidden states at chiral/bg positions
        curr_ch = curr["hidden_chiral"][side]  # list of [N_chiral, E] per layer
        prev_ch = prev["hidden_chiral"][side]
        curr_bg = curr["hidden_bg"][side]
        prev_bg = prev["hidden_bg"][side]

        # Exclude last element from decoder (it's the logits layer)
        if side == "dec":
            curr_ch = curr_ch[:-1]
            prev_ch = prev_ch[:-1]
            curr_bg = curr_bg[:-1]
            prev_bg = prev_bg[:-1]

        num_hidden = len(curr_ch)
        cka_chiral = np.zeros(num_hidden)
        cka_bg = np.zeros(num_hidden)
        cos_chiral = np.zeros(num_hidden)
        cos_bg = np.zeros(num_hidden)

        for h_idx in range(num_hidden):
            chc = curr_ch[h_idx]  # [N_chiral, E]
            chp = prev_ch[h_idx]
            bgc = curr_bg[h_idx]  # [N_bg, E]
            bgp = prev_bg[h_idx]

            if chc.shape[0] > 0 and chp.shape[0] > 0:
                cka_chiral[h_idx] = M.compute_linear_cka(chp, chc)
                cos_chiral[h_idx], _ = M.compute_cosine_similarity(chp, chc)
            if bgc.shape[0] > 0 and bgp.shape[0] > 0:
                cka_bg[h_idx] = M.compute_linear_cka(bgp, bgc)
                cos_bg[h_idx], _ = M.compute_cosine_similarity(bgp, bgc)

        result[side] = dict(
            jsd=jsd,
            cka_chiral=cka_chiral,
            cka_background=cka_bg,
            cos_chiral=cos_chiral,
            cos_background=cos_bg,
        )

    return result


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _head_df(data_list: list[np.ndarray], steps: list[int]) -> pd.DataFrame:
    rows = []
    for arr in data_list:
        nl, nh = arr.shape
        rows.append({f"L{l}_H{h}": arr[l, h] for l in range(nl) for h in range(nh)})
    return pd.DataFrame(rows, index=steps)


def _head_transition_df(data_list: list[np.ndarray], steps: list[int]) -> pd.DataFrame:
    rows = []
    idx = []
    for i, arr in enumerate(data_list):
        nl, nh = arr.shape
        rows.append({f"L{l}_H{h}": arr[l, h] for l in range(nl) for h in range(nh)})
        idx.append(f"step{steps[i]}-{steps[i+1]}")
    return pd.DataFrame(rows, index=idx)


def _layer_transition_df(data_list: list[np.ndarray], steps: list[int]) -> pd.DataFrame:
    rows = []
    idx = []
    for i, arr in enumerate(data_list):
        n = len(arr)
        cols = ["embeddings"] + [f"Layer_{l}" for l in range(n - 1)]
        rows.append(dict(zip(cols, arr)))
        idx.append(f"step{steps[i]}-{steps[i+1]}")
    return pd.DataFrame(rows, index=idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute jump-up metrics across checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name", "model")
    output_dir = Path(args.output_dir or cfg["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "03_jumpup_metrics.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    compute_graph = cfg.get("compute_graph_metrics", True)
    batch_size = cfg.get("batch_size", 128)
    max_len = cfg.get("max_len", 512)
    src_col = cfg.get("src_col", "random")
    tgt_col = cfg.get("tgt_col", "canonical")
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
    N = len(smiles_src)
    logger.info(f"Eval set: rows {row_start}..{row_end - 1} ({N} samples)")

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
    if len(dataset) != N:
        raise ValueError(f"Dataset size {len(dataset)} does not match the selected rows ({N})")
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

    # Metric key lists
    static_keys = [
        "attention_mass", "attention_entropy", "sar",
        "graph_dist_from", "seq_dist_from",
        "graph_dist_to", "seq_dist_to", "nit",
    ]
    transition_head_keys = ["jsd"]
    transition_layer_keys = ["cka_chiral", "cka_background", "cos_chiral", "cos_background"]

    # ---------------------------------------------------------------------------
    # Resume: detect done steps and load existing CSVs
    # ---------------------------------------------------------------------------
    logits_path = output_dir / "logits_metrics_trajectory.csv"
    check_resume_signature(
        output_dir / ".resume_signature.json", __file__, cfg, list(output_dir.glob("*_trajectory.csv"))
    )
    done_steps: set[int] = set()
    ex_static: dict[str, dict[str, pd.DataFrame]] = {"enc": {}, "dec": {}}
    ex_trans_head: dict[str, dict[str, pd.DataFrame]] = {"enc": {}, "dec": {}}
    ex_trans_layer: dict[str, dict[str, pd.DataFrame]] = {"enc": {}, "dec": {}}
    ex_logits: pd.DataFrame | None = None

    if logits_path.exists():
        ex_logits = pd.read_csv(logits_path, index_col=0)
        done_steps = set(int(x) for x in ex_logits.index.tolist())
        if target_steps is not None and not done_steps <= set(int(s) for s in target_steps):
            raise ValueError(
                f"{logits_path} holds steps outside checkpoint_steps "
                f"({sorted(done_steps - set(int(s) for s in target_steps))}); the jump interval of the "
                "visualizations would differ from the one used for analyses 06/07. Use a new output_dir."
            )
        logger.info(f"Resuming: {len(done_steps)} steps already done: {sorted(done_steps)}")
        for side in ("enc", "dec"):
            for key in static_keys:
                p = output_dir / f"{key}_{side}_trajectory.csv"
                if p.exists():
                    ex_static[side][key] = pd.read_csv(p, index_col=0)
            for key in transition_head_keys:
                p = output_dir / f"{key}_{side}_trajectory.csv"
                if p.exists():
                    ex_trans_head[side][key] = pd.read_csv(p, index_col=0)
            for key in transition_layer_keys:
                p = output_dir / f"{key}_{side}_trajectory.csv"
                if p.exists():
                    ex_trans_layer[side][key] = pd.read_csv(p, index_col=0)

    # Pre-build masks from tokenizer (before main loop, once)
    seq_len_enc = max(min(len(tokenizer.encode(s)), max_len) for s in smiles_src)
    seq_len_dec = max(min(len(tokenizer.encode(s)), max_len) for s in smiles_tgt)
    logger.info(f"Precomputing masks: enc_len={seq_len_enc}, dec_len={seq_len_dec}")
    vm_enc, cm_enc = _build_masks(smiles_src, tokenizer, seq_len_enc, chiral_ids)
    vm_dec, cm_dec = _build_masks(smiles_tgt, tokenizer, seq_len_dec, chiral_ids)

    # Pre-build graph distance matrices (expensive, do once)
    dist_enc_arr: np.ndarray | None = None
    dist_dec_arr: np.ndarray | None = None
    if compute_graph:
        logger.info("Building graph distance matrices...")
        dist_enc_arr = collate_distance_matrices(
            [create_token_graph_distance_matrix(s, tokenizer) for s in smiles_src],
            maxlen=seq_len_enc,
        )
        dist_dec_arr = collate_distance_matrices(
            [create_token_graph_distance_matrix(s, tokenizer) for s in smiles_tgt],
            maxlen=seq_len_dec,
        )

    # History (new steps only)
    last_done_step: int | None = max(done_steps) if done_steps else None
    trans_steps: list[int] = [last_done_step] if last_done_step is not None else []
    steps_list: list[int] = []
    hist: dict[str, dict[str, list]] = {
        "enc": defaultdict(list),
        "dec": defaultdict(list),
    }
    logits_rows: list[dict] = []
    prev_data: dict | None = None

    # Restore prev_data from last done checkpoint for transition continuity
    pending_steps = [int(p.name.split("-")[1]) for p in all_ckpts if int(p.name.split("-")[1]) not in done_steps]
    if last_done_step is not None and pending_steps:
        if min(pending_steps) < last_done_step:
            raise ValueError(
                f"Pending checkpoint {min(pending_steps)} precedes the last completed step {last_done_step}; "
                "transitions would be paired with the wrong predecessor. Remove the outputs and rerun."
            )
        last_ckpt = checkpoint_dir / f"checkpoint-{last_done_step}"
        if not last_ckpt.is_dir():
            raise FileNotFoundError(f"Cannot resume: last completed checkpoint is missing: {last_ckpt}")
        logger.info(f"Restoring prev_data from checkpoint-{last_done_step}...")
        restore_model = PanCoreTransformerModel.from_pretrained(last_ckpt)
        restore_model.to(device).eval()
        prev_data = forward_pass(
            restore_model, dataloader, device,
            vm_enc, cm_enc, vm_dec, cm_dec,
            dist_enc_arr, dist_dec_arr,
            N, batch_size, smiles_tgt, tokenizer, max_len, chiral_ids,
        )
        del restore_model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Incremental flush
    # ---------------------------------------------------------------------------
    def _flush_all() -> None:
        if not steps_list:
            return
        for side in ("enc", "dec"):
            for key in static_keys:
                if hist[side][key]:
                    new_df = _head_df(hist[side][key], steps_list)
                    ex = ex_static[side].get(key)
                    df = pd.concat([ex, new_df]) if ex is not None else new_df
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    df.to_csv(output_dir / f"{key}_{side}_trajectory.csv")
            for key in transition_head_keys:
                if hist[side][key]:
                    new_df = _head_transition_df(hist[side][key], trans_steps)
                    ex = ex_trans_head[side].get(key)
                    df = pd.concat([ex, new_df]) if ex is not None else new_df
                    df.to_csv(output_dir / f"{key}_{side}_trajectory.csv")
            for key in transition_layer_keys:
                if hist[side][key]:
                    new_df = _layer_transition_df(hist[side][key], trans_steps)
                    ex = ex_trans_layer[side].get(key)
                    df = pd.concat([ex, new_df]) if ex is not None else new_df
                    df.to_csv(output_dir / f"{key}_{side}_trajectory.csv")
        if logits_rows:
            new_logits = pd.DataFrame(logits_rows).set_index("step").sort_index()
            if ex_logits is not None:
                df = pd.concat([ex_logits, new_logits])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            else:
                df = new_logits
            df.to_csv(logits_path)

    # ---------------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------------
    for ckpt in all_ckpts:
        step = int(ckpt.name.split("-")[1])

        if step in done_steps:
            logger.info(f"Skipping {ckpt.name} (already done)")
            continue

        logger.info(f"→ {ckpt.name} (step {step})")

        model = PanCoreTransformerModel.from_pretrained(ckpt)
        model.to(device).eval()

        curr = forward_pass(
            model, dataloader, device,
            vm_enc, cm_enc, vm_dec, cm_dec,
            dist_enc_arr, dist_dec_arr,
            N, batch_size, smiles_tgt, tokenizer, max_len, chiral_ids,
        )

        del model
        torch.cuda.empty_cache()

        steps_list.append(step)

        # Static metrics
        for side in ("enc", "dec"):
            for key in static_keys:
                if key in curr["static"][side]:
                    hist[side][key].append(curr["static"][side][key])

        # Logits metrics
        if curr["logits_metrics"]:
            logits_rows.append({"step": step, **curr["logits_metrics"]})

        # Transition metrics
        if prev_data is not None:
            tm = compute_transition_metrics(curr, prev_data)
            for side in ("enc", "dec"):
                for k, v in tm[side].items():
                    hist[side][k].append(v)

        trans_steps.append(step)
        prev_data = curr

        _flush_all()

    # Final save (idempotent)
    logger.info("Saving CSVs...")
    _flush_all()
    logger.info(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
