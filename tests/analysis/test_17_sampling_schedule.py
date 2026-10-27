"""Tests for visualization 17 (sampling schedule): agreement with the training sampler."""

from __future__ import annotations

import numpy as np
import pytest

from pan_core.common.data_handler import DynamicBucketDistributedSampler
from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/visualize/17_sampling_schedule.py"
_SCHEDULE = load_script(_SCRIPT)

_START = [0.4, 0.4, 0.1, 0.05, 0.05]
_END = [0.2, 0.2, 0.2, 0.2, 0.2]


def _sampler_probabilities(delay, steepness, update_last_epoch, max_epochs):
    """Per-epoch probabilities exactly as DynamicBucketDistributedSampler precalculates them."""
    sampler = DynamicBucketDistributedSampler.__new__(DynamicBucketDistributedSampler)
    sampler.sampling_interpolation_type = "sigmoid"
    sampler.sampling_interpolation_delay_factor = delay
    sampler.sampling_interpolation_steepness = steepness
    sampler._precalculate_probabilities_and_steps(_START, _END, update_last_epoch, max_epochs, total_optimizer_steps=700)
    return np.array(sampler.dynamic_sampling_probabilities)


@pytest.mark.parametrize("delay, steepness", [(0.8, 12.0), (0.5, 5.0), (1.0, 15.0)])
def test_normalized_schedule_matches_the_training_sampler(delay, steepness):
    update_last_epoch = 70
    t_values = np.arange(update_last_epoch) / (update_last_epoch - 1)

    expected = _sampler_probabilities(delay, steepness, update_last_epoch, max_epochs=update_last_epoch)
    actual = _SCHEDULE.delayed_sigmoid_probabilities(_START, _END, delay, steepness, t_values, normalize=True)

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_schedule_runs_from_the_start_to_the_end_probabilities():
    probabilities = _SCHEDULE.delayed_sigmoid_probabilities(_START, _END, 0.8, 12.0, np.array([0.0, 1.0]))

    np.testing.assert_allclose(probabilities, [_START, _END])


def test_schedule_rejects_mismatched_bucket_lists():
    with pytest.raises(ValueError, match="equal length"):
        _SCHEDULE.delayed_sigmoid_probabilities(_START, _END[:4], 0.8, 12.0, np.array([0.5]))


def test_main_writes_the_figure(tmp_path):
    run_script(_SCRIPT, {"output_dir": str(tmp_path / "figures"), "resolution": 11}, tmp_path / "config.yml")

    for extension in ("png", "svg"):
        assert (tmp_path / "figures" / f"sampling_schedule.{extension}").is_file()
