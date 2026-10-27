"""Tests for analysis 06 (cross-evaluation): encoder / decoder swaps of two checkpoints."""

from __future__ import annotations

import pandas as pd
import torch

from pan_core.common.data_handler import GeneralDataCollator, OnTheFlyDataset
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.models import PanCoreTransformerModel
from pan_core_analysis.runner import run_generation_evaluation
from tests.analysis.tiny_run import MAX_LEN, N_BUCKETS
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/compute/06_cross_evaluation.py"
_CROSS = load_script(_SCRIPT)


def test_main_writes_the_four_combinations(tiny_run, tmp_path):
    config = {
        "model_name": tiny_run.model_name,
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(tmp_path / "out"),
        "pre_jump_checkpoint": str(tiny_run.checkpoint_dir / "checkpoint-10"),
        "post_jump_checkpoint": str(tiny_run.checkpoint_dir / "checkpoint-60"),
        "batch_size": 8,
        "max_length": MAX_LEN,
        "n_phases": N_BUCKETS,
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    result = pd.read_csv(tmp_path / "out" / "cross_evaluation_tiny.csv", index_col="combination")
    assert result.index.tolist() == ["preenc_predec", "postenc_predec", "preenc_postdec", "postenc_postdec"]
    assert {"accuracy", "phase1_accuracy", f"phase{N_BUCKETS}_chiral_mistakes_ex_mark"} <= set(result.columns)
    for label in result.index:
        decode = pd.read_csv(tmp_path / "out" / f"decode_tiny_{label}.csv", keep_default_na=False)
        assert result.loc[label, "accuracy"] == decode["judge"].mean()


def test_same_checkpoint_on_both_sides_matches_plain_generation(tiny_run):
    """Decoding from the encoder's latent must reproduce model.generate on the input ids (fp32)."""
    tokenizer = SMILESTokenizer(tiny_run.vocab)
    dataset = OnTheFlyDataset(file_path_or_df=tiny_run.eval_csv, tokenizer=tokenizer,
                              src_col_name="random", tgt_col_name="canonical", mode="eval")
    model = PanCoreTransformerModel.from_pretrained(tiny_run.checkpoint_dir / "checkpoint-30").eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=GeneralDataCollator())

    swapped = _CROSS.run_cross_eval(model, model, loader, tokenizer, torch.device("cpu"), MAX_LEN, use_bf16=False)
    plain = run_generation_evaluation(dataset, model, tokenizer, torch.device("cpu"), batch_size=8,
                                      max_length=MAX_LEN, use_bf16=False, compile_model=False, num_workers=0)

    assert swapped["generated"].tolist() == plain["generated"].tolist()
    assert swapped["target"].tolist() == plain["target"].tolist()
