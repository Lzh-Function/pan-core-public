"""Gradient accumulation all-reduces once per optimizer step, not once per micro-batch.

training_step runs its own accumulation loop with the Trainer's
gradient_accumulation_steps pinned to 1, so Accelerate never skips a sync on its
own. Every micro-batch but the last has to sit inside accelerator.no_sync,
forward included, or DDP all-reduces the whole gradient acc_step times.

A single process cannot show the all-reduce itself (no_sync is a null context
there), so these tests count the contexts training_step enters and check the
DataLoader worker settings it builds.
"""

import contextlib
from types import SimpleNamespace

import pytest
import yaml
from accelerate import Accelerator

from pan_core.main import run_training
from pan_core.train import HF_Trainer
from tests.core.test_checkpoint_resume import _make_config, _write_csvs
from tests.utils import VOCAB_FILE


def test_no_sync_wraps_every_micro_batch_but_the_last(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_a, csv_b, valid = _write_csvs(data_dir)
    cfg = _make_config(str(VOCAB_FILE), csv_a, csv_b, valid)
    cfg["training"]["data"]["batch_size"] = [2, 1]
    cfg["training"]["data"]["accumulate_steps"] = [2, 4]
    cfg["training"]["strategy"]["num_epochs"] = 1
    cfg["training"]["strategy"]["dynamic"]["max_steps"] = 4
    cfg["training"]["strategy"]["dynamic"]["sampling_update_last_epoch"] = 1
    cfg["training"]["checkpoint"]["step"] = 1000
    cfg["training"]["validation"]["enabled"] = False
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    entered = []
    acc_per_step = []

    @contextlib.contextmanager
    def counting_no_sync(self, model):
        entered.append(model)
        yield

    original_step = HF_Trainer.training_step

    def recording_step(self, model, inputs, num_items_in_batch=None):
        acc_per_step.append(int(inputs.get("accumulate_step", 1)))
        return original_step(self, model, inputs, num_items_in_batch)

    monkeypatch.setattr(Accelerator, "no_sync", counting_no_sync)
    monkeypatch.setattr(HF_Trainer, "training_step", recording_step)

    run_training(config_path=str(cfg_path), output_dir=str(tmp_path / "out"))

    assert len(acc_per_step) == 4
    assert len(entered) == sum(acc - 1 for acc in acc_per_step)


@pytest.mark.parametrize(
    ("workers", "persistent", "prefetch", "expected"),
    [
        (0, True, 4, {"num_workers": 0, "pin_memory": True}),
        (16, False, None, {"num_workers": 16, "pin_memory": True, "persistent_workers": False}),
        (16, True, 4, {"num_workers": 16, "pin_memory": True, "persistent_workers": True,
                       "prefetch_factor": 4}),
    ],
)
def test_worker_settings_reach_the_curriculum_loader(workers, persistent, prefetch, expected):
    fake = SimpleNamespace(args=SimpleNamespace(
        dataloader_num_workers=workers,
        dataloader_persistent_workers=persistent,
        dataloader_prefetch_factor=prefetch,
    ))
    assert HF_Trainer._worker_kwargs(fake) == expected
