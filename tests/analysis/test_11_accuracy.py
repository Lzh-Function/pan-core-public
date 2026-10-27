"""Tests for visualization 11 (accuracy): figures from the outputs of analysis 01."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

from pan_core.common.tokenizer import SMILESTokenizer
from tests.analysis.tiny_run import MAX_LEN, N_BUCKETS
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/visualize/11_accuracy.py"
_ACCURACY_VIS = load_script(_SCRIPT)


@pytest.fixture(scope="module")
def accuracy_output(tiny_run, tmp_path_factory):
    """Outputs of analysis 01 on the tiny run (input of visualization 11)."""
    output_dir = tmp_path_factory.mktemp("accuracy")
    config = {
        "model_name": tiny_run.model_name,
        "checkpoint_dir": str(tiny_run.checkpoint_dir),
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "output_dir": str(output_dir),
        "batch_size": 8,
        "max_length": MAX_LEN,
        "use_bf16": False,
        "n_phases": N_BUCKETS,
        "teacher_forced_batch_size": 8,
    }
    run_script("analysis/scripts/compute/01_accuracy_progress.py", config, output_dir / "config.yml")
    return output_dir


def _trainer_state(path, steps):
    log_history = [{"step": s, "train_celoss": 1.0 / s} for s in steps]
    log_history += [{"step": s, **{f"eval_ce_phase{p}": p / s for p in range(1, N_BUCKETS + 1)}} for s in steps]
    path.write_text(json.dumps({"log_history": log_history}))
    return path


def test_token_counts_per_phase_counts_target_tokens_per_bucket(tmp_path, tiny_run):
    eval_csv = tmp_path / "eval.csv"
    pd.DataFrame({"canonical": ["CCO", "CC", "C[C@H](N)O", "CN"]}).to_csv(eval_csv, index=False)

    counts = _ACCURACY_VIS.token_counts_per_phase(eval_csv, "canonical", SMILESTokenizer(tiny_run.vocab), n_phases=2)

    assert counts[0] == Counter({"C": 4, "O": 1})
    assert counts[1]["@"] == 1 and counts[1]["N"] == 2
    with pytest.raises(ValueError, match="equal phase blocks"):
        _ACCURACY_VIS.token_counts_per_phase(eval_csv, "canonical", SMILESTokenizer(tiny_run.vocab), n_phases=3)


def test_read_trainer_state_separates_train_and_eval_rows(tmp_path):
    train, evaluation = _ACCURACY_VIS.read_trainer_state(_trainer_state(tmp_path / "state.json", [100, 200]))

    assert train["step"].tolist() == [100, 200]
    assert evaluation["eval_ce_phase1"].tolist() == [0.01, 0.005]
    (tmp_path / "empty.json").write_text(json.dumps({"log_history": [{"step": 1, "loss": 1.0}]}))
    with pytest.raises(ValueError, match="No train_celoss"):
        _ACCURACY_VIS.read_trainer_state(tmp_path / "empty.json")


def test_token_accuracy_plot_rejects_tokens_absent_from_the_targets():
    df = pd.DataFrame({"C": [0.5, 1.0], "Br": [0.0, 1.0]}, index=pd.Index([10, 20], name="step"))
    with pytest.raises(ValueError, match="Br"):
        _ACCURACY_VIS.plot_token_accuracy(df, Counter({"C": 4}))


def test_main_writes_every_figure(accuracy_output, tiny_run, tmp_path):
    config = {
        "input_dir": str(accuracy_output),
        "output_dir": str(tmp_path / "figures"),
        "tokenizer_path": str(tiny_run.vocab),
        "eval_data": str(tiny_run.eval_csv),
        "trainer_states": {"tiny": str(_trainer_state(tmp_path / "state.json", tiny_run.steps))},
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    stems = ["bucket_accuracy_tiny", "loss_acc_progress_tiny", "chiral_accuracy_tiny", "accuracy_tiny"]
    stems += [f"{kind}_tiny_phase{p}" for kind in ("token_accuracy", "token_accuracy_teacher_forced") for p in (1, 2)]
    for stem in stems:
        for extension in ("png", "svg"):
            assert (tmp_path / "figures" / f"{stem}.{extension}").is_file(), f"{stem}.{extension}"
