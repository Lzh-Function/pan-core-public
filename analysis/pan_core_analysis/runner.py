"""
Greedy decoding evaluation loop for analysis scripts.

Designed as a thin, reusable wrapper around model.generate() that accumulates
results across batches and optionally saves them to a CSV file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pan_core.common.data_handler import GeneralDataCollator
from pan_core.common.utils import move_to_cpu


def run_generation_evaluation(
    dataset,
    model,
    tokenizer,
    device: torch.device | str,
    *,
    batch_size: int = 512,
    max_length: int = 402,
    use_bf16: bool = True,
    output_file: Path | str | None = None,
    logger: logging.Logger | None = None,
    compile_model: bool = True,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Run greedy decoding over a dataset and return a results DataFrame.

    For each sample the DataFrame contains columns:
    - ``input``:     decoded input SMILES string.
    - ``target``:    decoded target SMILES string.
    - ``generated``: model-generated SMILES string.
    - ``judge``:     bool, True when generated == target.

    Args:
        dataset: A Dataset that yields batches compatible with GeneralDataCollator.
        model: PanCoreTransformerModel instance.
        tokenizer: SMILESTokenizer instance.
        device: Torch device.
        batch_size: Dataloader batch size.
        max_length: Maximum generation length in tokens.
        use_bf16: Use bfloat16 during generation.
        output_file: If provided, save results CSV to this path.
        logger: Optional logger; prints to stdout if None.
        compile_model: Whether to torch.compile the model (disable when hooks are active).
        num_workers: DataLoader worker count.
    Returns:
        DataFrame indexed 0..N-1.
    """
    _log = logger.info if logger else print

    model = model.to(device)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
    model.eval()

    collate_fn = GeneralDataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    records: list[dict] = []

    _log("Starting greedy decoding evaluation...")
    with torch.no_grad():
        with tqdm(total=len(dataset), desc="Generating", unit="sample") as pbar:
            for batch in dataloader:
                latents = _move(batch.get("latents"), device)
                input_ids = _move(batch.get("input_ids"), device)
                encoder_attention_mask = _move(batch.get("encoder_attention_mask"), device)
                labels = _move(batch.get("labels"), device)

                current_bs = (
                    input_ids.size(0) if input_ids is not None else latents.size(0)
                )

                generated_ids, _, _ = model.generate(
                    input_ids=input_ids,
                    encoder_attention_mask=encoder_attention_mask,
                    latents=latents,
                    max_length=max_length,
                    use_bf16=use_bf16,
                    need_logits=False,
                )

                for i, ids in enumerate(generated_ids.cpu().tolist()):
                    gen_text = tokenizer.decode(
                        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    rec: dict = {"generated": gen_text}
                    if input_ids is not None:
                        rec["input"] = tokenizer.decode(
                            input_ids[i].cpu().tolist(),
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    if labels is not None:
                        tgt = tokenizer.decode(
                            labels[i].tolist(),
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        rec["target"] = tgt
                        rec["judge"] = tgt == gen_text
                    records.append(rec)

                pbar.update(current_bs)

    results = pd.DataFrame(records)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        _log(f"Results saved to {output_file}")

    _log("Evaluation complete.")
    return results


def compute_accuracy_from_results(
    df: pd.DataFrame,
) -> dict[str, float]:
    """Compute accuracy metrics from a results DataFrame produced by run_generation_evaluation.

    Returns a dict with keys:
    - ``accuracy``:                         overall exact-match accuracy.
    - ``chiral_accuracy``:                  accuracy on samples whose target contains '@'.
    - ``chiral_mistakes_ex_mark_accuracy``: among chiral mistakes, fraction where removing '@'/'@@'
                                            makes target == generated (wrong chirality only).
    """
    acc = float(df["judge"].mean()) if "judge" in df.columns else float("nan")

    chiral_mask = df["target"].str.contains("@") if "target" in df.columns else pd.Series(False, index=df.index)
    chiral_df = df[chiral_mask]
    chiral_acc = float(chiral_df["judge"].mean()) if len(chiral_df) > 0 else float("nan")

    mistakes_df = chiral_df[~chiral_df["judge"]] if len(chiral_df) > 0 else pd.DataFrame()
    if len(mistakes_df) > 0:
        gen_ex = mistakes_df["generated"].str.replace("@", "", regex=False)
        tgt_ex = mistakes_df["target"].str.replace("@", "", regex=False)
        chiral_mistakes_ex = float((gen_ex == tgt_ex).mean())
    else:
        chiral_mistakes_ex = float("nan")

    return dict(
        accuracy=acc,
        chiral_accuracy=chiral_acc,
        chiral_mistakes_ex_mark_accuracy=chiral_mistakes_ex,
    )


def check_resume_signature(
    signature_path: Path | str,
    script_path: Path | str,
    config: dict,
    existing_outputs: list[Path],
) -> None:
    """Refuse to resume into outputs produced by a different config or code version.

    ``signature_path`` stores a hash of ``config`` (minus ``output_dir`` and ``checkpoint_steps``),
    the compute script and every ``pan_core_analysis`` module. If it exists with a different hash, or if any of
    ``existing_outputs`` exists without a signature (results of unknown provenance), this
    raises so stale rows are never mixed with new ones.
    """
    import hashlib
    import json

    path = Path(signature_path)
    digest = hashlib.sha256()
    # checkpoint_steps only selects which rows exist; resume itself handles adding steps.
    digest.update(json.dumps({k: v for k, v in config.items() if k not in ("output_dir", "checkpoint_steps")},
                             sort_keys=True, default=str).encode())
    sources = [Path(script_path)] + sorted(Path(__file__).parent.glob("*.py"))
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    signature = {"hash": digest.hexdigest(), "script": str(script_path)}
    stale_message = (
        f"Existing results next to {path} come from a different config or code version "
        "(or an unknown one). Delete them (or choose another output_dir) and rerun."
    )
    if path.exists():
        if json.loads(path.read_text()).get("hash") != signature["hash"]:
            raise ValueError(stale_message)
        return
    if any(Path(output).exists() for output in existing_outputs):
        raise ValueError(stale_message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signature, indent=2))


def phase_bounds(n: int, phase: int | None, max_phase: int | None) -> tuple[int, int]:
    """Row range [start, end) of a 1-based phase block of an eval set with equal phase blocks.

    Returns (0, n) when both ``phase`` and ``max_phase`` are None. Matches the slicing of
    ``OnTheFlyDataset(phase=..., max_phase=...)``; ``n`` must be a multiple of ``max_phase``.
    """
    if phase is None and max_phase is None:
        return 0, n
    if not isinstance(phase, int) or not isinstance(max_phase, int):
        raise ValueError("phase and max_phase must both be integers (or both null)")
    if not 1 <= phase <= max_phase:
        raise ValueError(f"phase must be in 1..{max_phase}, got {phase}")
    if n % max_phase != 0:
        raise ValueError(f"{n} rows cannot be split into {max_phase} equal phase blocks")
    per_phase = n // max_phase
    return (phase - 1) * per_phase, phase * per_phase


def pad_target_by_one(batch: dict, pad_token_id: int = 0) -> dict:
    """Append one padding column to ``labels`` / ``decoder_attention_mask``.

    ``GeneralDataCollator`` trims each batch to its longest target, so the teacher-forced
    decoder input (``labels[:-1]``) never contains the EOS of the longest targets. The paper
    runs padded every sequence to a fixed length, so the decoder also produced a hidden state
    with EOS as input for every row. One extra pad column restores that position; states at
    all other positions are unchanged (causal attention, padded keys masked).
    """
    batch = dict(batch)
    labels = batch["labels"]
    mask = batch["decoder_attention_mask"]
    pad_column = torch.full((labels.size(0), 1), pad_token_id, dtype=labels.dtype, device=labels.device)
    batch["labels"] = torch.cat([labels, pad_column], dim=1)
    batch["decoder_attention_mask"] = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
    return batch


def chiral_token_ids(tokenizer) -> tuple[int, int]:
    """Token ids of '@' and '@@' in the tokenizer vocabulary."""
    try:
        return tokenizer.token2id["@"], tokenizer.token2id["@@"]
    except KeyError as error:
        raise ValueError("Tokenizer vocabulary must contain '@' and '@@'") from error


def compute_phase_metrics(df: pd.DataFrame, n_phases: int = 5) -> dict[str, float]:
    """Per-phase accuracy metrics for an evaluation set made of equal consecutive phase blocks.

    The evaluation CSV stores ``n_phases`` length buckets as equal consecutive blocks
    (e.g. 5 x 2,000 rows). For each block ``p`` (1-based) the returned dict contains:

    - ``phase{p}_accuracy``:                exact-match accuracy.
    - ``phase{p}_chiral_accuracy``:         accuracy on targets containing '@'.
    - ``phase{p}_chiral_mistakes_ex_mark``: among mistranslated chiral targets, fraction that
                                            match once every '@' is removed from both strings.
    - ``phase{p}_background_accuracy``:     accuracy on targets without '@'.
    - ``phase{p}_n_chiral``, ``phase{p}_n_chiral_mistakes``, ``phase{p}_n_background``: counts.

    Raises:
        ValueError: if the number of rows is not a positive multiple of ``n_phases``.
    """
    n = len(df)
    if n == 0 or n % n_phases != 0:
        raise ValueError(f"{n} results cannot be split into {n_phases} equal phase blocks")
    per_phase = n // n_phases
    row: dict[str, float] = {}
    for p in range(n_phases):
        phase_df = df.iloc[p * per_phase: (p + 1) * per_phase]
        chiral_mask = phase_df["target"].str.contains("@", regex=False)
        chiral_df = phase_df[chiral_mask]
        background_df = phase_df[~chiral_mask]
        mistakes = chiral_df[~chiral_df["judge"]]
        if len(mistakes) > 0:
            gen_ex = mistakes["generated"].str.replace("@", "", regex=False)
            tgt_ex = mistakes["target"].str.replace("@", "", regex=False)
            ex_mark = float((gen_ex == tgt_ex).mean())
        else:
            ex_mark = float("nan")
        row[f"phase{p+1}_accuracy"] = float(phase_df["judge"].mean())
        row[f"phase{p+1}_chiral_accuracy"] = float(chiral_df["judge"].mean()) if len(chiral_df) else float("nan")
        row[f"phase{p+1}_chiral_mistakes_ex_mark"] = ex_mark
        row[f"phase{p+1}_background_accuracy"] = (
            float(background_df["judge"].mean()) if len(background_df) else float("nan")
        )
        row[f"phase{p+1}_n_chiral"] = len(chiral_df)
        row[f"phase{p+1}_n_chiral_mistakes"] = len(mistakes)
        row[f"phase{p+1}_n_background"] = len(background_df)
    return row


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _move(tensor: torch.Tensor | None, device) -> torch.Tensor | None:
    return tensor.to(device) if tensor is not None else None
