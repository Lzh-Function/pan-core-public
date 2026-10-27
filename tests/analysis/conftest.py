"""Fixtures shared by the model-based analysis tests."""

from __future__ import annotations

import pytest
import torch

from pan_core.models import PanCoreTransformerModel
from tests.analysis.tiny_model import make_batch, make_config
from tests.analysis.tiny_run import TinyRun, write_checkpoint_series, write_eval_csv
from tests.utils import VOCAB_FILE


@pytest.fixture(scope="session", autouse=True)
def _eager_generation():
    """Run greedy decoding eagerly: torch.compile of the decoding loop is slow on CPU and not under test."""
    patch = pytest.MonkeyPatch()
    patch.setattr(torch, "compile", lambda model, **kwargs: model)
    yield
    patch.undo()


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory) -> TinyRun:
    """Checkpoint series, 5-bucket evaluation CSV and vocabulary of a tiny run (see tiny_run.py)."""
    root = tmp_path_factory.mktemp("tiny_run")
    return TinyRun(
        checkpoint_dir=write_checkpoint_series(root / "checkpoints"),
        eval_csv=write_eval_csv(root / "eval.csv"),
        vocab=VOCAB_FILE,
    )


@pytest.fixture(scope="module")
def pancore_output():
    """Run one forward pass with need_hidden=True and return all relevant tensors."""
    torch.manual_seed(0)
    model = PanCoreTransformerModel(make_config())
    model.eval()

    input_ids, enc_mask, labels, dec_mask = make_batch()
    with torch.no_grad():
        out = model.forward(
            input_ids=input_ids,
            encoder_attention_mask=enc_mask,
            labels=labels,
            decoder_attention_mask=dec_mask,
            need_hidden=True,
            calculate_loss=False,
            return_dict=True,
        )
    return out, input_ids, enc_mask, labels, dec_mask
