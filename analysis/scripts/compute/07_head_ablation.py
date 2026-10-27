"""
07_head_ablation.py

Head ablation analysis for chiral-sensitive attention heads.

Chiral heads are the heads of the checkpoint being ablated:
  - before_jump, jump_peak, after_jump: ``chiral_heads`` = the encoder heads with the largest
    change in attention mass on chiral tokens within the jump-up interval (visualization 12,
    ``attention_mass_enc_head_ranking.csv``, column ``interval_amplitude``; paper Fig 5)
  - final: ``final_chiral_heads`` = the encoder heads with the largest attention mass on chiral
    tokens at the final checkpoint (``attention_mass_enc_head_ranking.csv`` of the series whose
    last checkpoint is the final one, column ``last_step_mass``)
Paper heads (jump / final): pancore-addonce L6H1, L6H7, L7H5 / same; pancore-xattn-adalnzero
L6H5, L7H1, L7H4 / L7H1, L5H7, L7H0; pancore-baseline L7H0, L7H2, L7H7 / L7H7, L7H0, L1H0.

For each checkpoint label {before_jump, jump_peak, after_jump, final} × ablation variant:
  - all_chiral_heads:  ablate all chiral heads of that checkpoint simultaneously
  - head_{layer}_{idx}: ablate each chiral head of that checkpoint individually
  - random_control_heads: ablate a size-matched random set from the final two layers,
                          drawn anew for every checkpoint label (in the order above)
                          from one ``random.Random(seed)`` stream; the pool excludes
                          ``chiral_heads`` (as in the paper run, so the published random
                          sets are reproduced). A draw that hits a ``final_chiral_heads``
                          entry raises.
  - no_ablation:       baseline without any ablation

Every head is zeroed only in its own layer. Generation uses bf16 autocast (as in the
paper run). Accuracy is computed per phase bucket (5 phases × 2000 samples = 10000 total
eval samples). bf16 greedy outputs depend slightly on library versions: the paper numbers
were computed with torch 2.10, and other versions were observed to change the ZINC20 /
PubChem (~100) metrics by up to ~0.015.

Paper (arXiv v1) note: the original Table 4 / S5 run registered one hook holding all
(layer, head) tuples on every chiral layer; indexing the head axis with a tuple zeroed
every number appearing in the tuples, in every chiral layer (pancore-addonce: heads
{1, 5, 6, 7} of layers 6 and 7, i.e. 8 heads; pancore-xattn-adalnzero: heads {1, 4, 5, 6, 7}
of layers 6 and 7, i.e. 10 heads; pancore-baseline: heads {0, 2, 7} of layer 7, i.e. the
intended 3), and used the jump-interval heads at all four checkpoints.
``legacy_all_heads_union: true`` applies that union to ``all_chiral_heads``; with
``final_chiral_heads`` equal to ``chiral_heads`` it reproduces the arXiv v1 run. The default
ablates exactly the configured heads. Single-head and random variants are identical in
both modes.

Output: {output_dir}/{model_name}/head_ablation_{model_name}.csv
        (index: time, variant; column ablated_heads lists the zeroed heads)

Config YAML:
  model_name: arbitrary label used in output paths
  tokenizer_path, eval_data (10k), src_col, tgt_col, output_dir,
  seed (42), batch_size (2048), max_length (402), n_phases (5),
  legacy_all_heads_union (false),
  chiral_heads: [[layer_idx, head_idx], ...]         (before_jump, jump_peak, after_jump)
  final_chiral_heads: [[layer_idx, head_idx], ...]   (final)
  checkpoints:
    before_jump: /path/to/ckpt
    jump_peak:   /path/to/ckpt
    after_jump:  /path/to/ckpt
    final:       /path/to/ckpt
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
import yaml

from pan_core.common.data_handler import OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.models import PanCoreTransformerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # analysis/, for pan_core_analysis
from pan_core_analysis.hooks import HeadAblationHook
from pan_core_analysis.runner import compute_phase_metrics, run_generation_evaluation


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


def load_head_geometry(checkpoints: dict[str, str | Path]) -> tuple[int, int, int]:
    """Load consistent (n_layer, n_head, head_dim) values from checkpoint configs."""
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("checkpoints must be a non-empty mapping")
    signatures: dict[str, tuple[int, int, int]] = {}
    for label, checkpoint in checkpoints.items():
        config_path = Path(checkpoint) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
        with open(config_path) as file:
            config = json.load(file)
        try:
            n_layer = int(config["n_layer"])
            n_head = int(config["n_head"])
            embedding_dim = int(config["embedding_dim"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint config must define integer n_layer, n_head, and embedding_dim: {config_path}"
            ) from error
        if n_layer < 1 or n_head < 1 or embedding_dim < 1 or embedding_dim % n_head != 0:
            raise ValueError(f"Invalid transformer dimensions in {config_path}")
        signatures[label] = (n_layer, n_head, embedding_dim // n_head)
    unique = set(signatures.values())
    if len(unique) != 1:
        raise ValueError(f"All ablation checkpoints must share one architecture, got: {signatures}")
    return unique.pop()


def parse_chiral_heads(value, n_layer: int, n_head: int) -> list[tuple[int, int]]:
    """Validate explicitly selected chiral heads against checkpoint dimensions."""
    if not isinstance(value, list) or not value:
        raise ValueError("chiral_heads must be a non-empty list of [layer_idx, head_idx] pairs")
    heads: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid chiral head entry: {item!r}")
        layer_idx, head_idx = item
        if not isinstance(layer_idx, int) or not isinstance(head_idx, int):
            raise ValueError(f"Chiral head indices must be integers: {item!r}")
        if not 0 <= layer_idx < n_layer or not 0 <= head_idx < n_head:
            raise ValueError(
                f"Chiral head {item!r} is outside layer range 0..{n_layer - 1} "
                f"or head range 0..{n_head - 1}"
            )
        heads.append((layer_idx, head_idx))
    if len(set(heads)) != len(heads):
        raise ValueError("chiral_heads contains duplicate entries")
    return heads


def heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Group (layer_idx, head_idx) pairs into {layer_idx: [head_idx, ...]}."""
    grouped: dict[int, list[int]] = {}
    for layer_idx, head_idx in heads:
        grouped.setdefault(layer_idx, []).append(head_idx)
    return grouped


def legacy_union_heads(heads: list[tuple[int, int]], num_heads: int) -> dict[int, list[int]]:
    """Head set zeroed by the original Table 4 / S5 ``all_chiral_heads`` run (see module docstring)."""
    union = sorted({index for pair in heads for index in pair})
    if union[-1] >= num_heads:
        raise ValueError(f"legacy_all_heads_union needs every layer index < num_heads, got {union}")
    return {layer_idx: union for layer_idx in sorted({layer_idx for layer_idx, _ in heads})}


TIME_LABELS = ("before_jump", "jump_peak", "after_jump", "final")


def format_heads(layer_heads: dict[int, list[int]] | None) -> str:
    """'L6H1;L6H7;L7H5' for {6: [1, 7], 7: [5]}; '' for no ablation."""
    if layer_heads is None:
        return ""
    return ";".join(f"L{layer}H{head}" for layer, heads in sorted(layer_heads.items()) for head in heads)


def plan_ablations(
    chiral_heads: list[tuple[int, int]],
    final_chiral_heads: list[tuple[int, int]],
    n_layer: int,
    num_heads: int,
    seed: int,
    legacy_union: bool,
) -> dict[str, list[tuple[str, dict[int, list[int]] | None]]]:
    """Ablation variants per checkpoint label: [(variant_label, {layer: [heads]} or None), ...]."""
    control_layers = list(range(max(0, n_layer - 2), n_layer))
    pool = [(li, hi) for li in control_layers for hi in range(num_heads) if (li, hi) not in chiral_heads]
    rng = random.Random(seed)
    plans = {}
    for time_label in TIME_LABELS:
        heads = final_chiral_heads if time_label == "final" else chiral_heads
        if len(pool) < len(heads):
            raise ValueError("Not enough non-chiral heads for a size-matched random control")
        # Random control heads are redrawn for every checkpoint label, as in the paper run.
        random_heads = rng.sample(pool, len(heads))
        overlap = set(random_heads) & set(heads)
        if overlap:
            raise ValueError(f"Random control heads for {time_label} include chiral heads {sorted(overlap)}")
        all_heads = legacy_union_heads(heads, num_heads) if legacy_union else heads_by_layer(heads)
        plans[time_label] = [
            ("no_ablation", None),
            ("all_chiral_heads", all_heads),
            *[(f"head_{li}_{hi}", {li: [hi]}) for li, hi in heads],
            ("random_control_heads", heads_by_layer(random_heads)),
        ]
    return plans


def register_ablation(model, layer_heads: dict[int, list[int]], num_heads: int, head_dim: int) -> list:
    """Register one hook per layer that zeros only that layer's listed heads."""
    handles = []
    for layer_idx, head_indices in layer_heads.items():
        hook = HeadAblationHook(head_indices=head_indices, num_heads=num_heads, head_dim=head_dim)
        handles.extend(hook.register(model, [layer_idx]))
    return handles


def run_with_ablation(
    model: PanCoreTransformerModel,
    dataset,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    layer_heads: dict[int, list[int]] | None,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    max_length: int,
    output_file: Path,
) -> pd.DataFrame:
    """Run generation with optional per-layer head ablation and save CSV."""
    handles = []
    if layer_heads is not None:
        handles = register_ablation(model, layer_heads, num_heads, head_dim)

    try:
        results = run_generation_evaluation(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            use_bf16=True,
            output_file=output_file,
            logger=logger,
            compile_model=False,  # hooks are incompatible with fullgraph compile
        )
    finally:
        HeadAblationHook.remove(handles)

    return results


def main():
    parser = argparse.ArgumentParser(description="Head ablation analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty output label")

    required_checkpoint_labels = {"before_jump", "jump_peak", "after_jump", "final"}
    checkpoints = cfg.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != required_checkpoint_labels:
        raise ValueError(f"checkpoints must contain exactly {sorted(required_checkpoint_labels)}")
    n_layer, num_heads, head_dim = load_head_geometry(checkpoints)
    chiral_heads = parse_chiral_heads(cfg.get("chiral_heads"), n_layer, num_heads)
    if "final_chiral_heads" not in cfg:
        raise ValueError(
            "final_chiral_heads is required: the chiral heads of the final checkpoint "
            "(equal to chiral_heads when they did not change)"
        )
    final_chiral_heads = parse_chiral_heads(cfg["final_chiral_heads"], n_layer, num_heads)

    output_dir = Path(args.output_dir or cfg["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "07_head_ablation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SMILESTokenizer(cfg["tokenizer_path"])
    src_col = cfg.get("src_col", "random")
    tgt_col = cfg.get("tgt_col", "canonical")
    batch_size = cfg.get("batch_size", 2048)
    max_length = cfg.get("max_length", 402)
    n_phases = cfg.get("n_phases", 5)
    legacy_union = cfg.get("legacy_all_heads_union", False)
    if not isinstance(legacy_union, bool):
        raise ValueError("legacy_all_heads_union must be a boolean")

    dataset = OnTheFlyDataset(
        file_path_or_df=Path(cfg["eval_data"]),
        tokenizer=tokenizer,
        src_col_name=src_col,
        tgt_col_name=tgt_col,
        mode="eval",
    )

    # Validates every head set and random draw before any GPU work starts.
    plans = plan_ablations(chiral_heads, final_chiral_heads, n_layer, num_heads, cfg.get("seed", 42), legacy_union)
    logger.info(f"Chiral heads (jump checkpoints): {chiral_heads}; final: {final_chiral_heads}")
    for time_label, variants in plans.items():
        logger.info(f"{time_label}: " + ", ".join(f"{label}=[{format_heads(heads)}]" for label, heads in variants))

    all_rows = []

    for time_label in TIME_LABELS:
        ckpt_path = Path(cfg["checkpoints"][time_label])
        logger.info(f"Loading {time_label} from {ckpt_path}")
        model = PanCoreTransformerModel.from_pretrained(ckpt_path).to(device).eval()

        for variant_label, layer_heads in plans[time_label]:
            logger.info(f"  {time_label} / {variant_label}")
            out_file = output_dir / f"{time_label}_{variant_label}_results.csv"

            results = run_with_ablation(
                model=model,
                dataset=dataset,
                tokenizer=tokenizer,
                device=device,
                layer_heads=layer_heads,
                num_heads=num_heads,
                head_dim=head_dim,
                batch_size=batch_size,
                max_length=max_length,
                output_file=out_file,
            )

            phase_metrics = compute_phase_metrics(results, n_phases=n_phases)
            all_rows.append({
                "time": time_label, "variant": variant_label, "ablated_heads": format_heads(layer_heads), **phase_metrics,
            })

        del model
        torch.cuda.empty_cache()

    out_csv = output_dir / f"head_ablation_{model_name}.csv"
    df = pd.DataFrame(all_rows).set_index(["time", "variant"])
    df.to_csv(out_csv)
    logger.info(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
