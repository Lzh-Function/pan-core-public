"""Tests for the single-manifest end-to-end analysis runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.utils import load_script

_PIPELINE = load_script("analysis/run_pipeline.py")


@pytest.fixture
def pipeline_manifest(tmp_path):
    tokenizer_path = tmp_path / "normal_tokens.txt"
    tokenizer_path.write_text("<pad>\n<s>\n</s>\n<unk1>\nC\n@\n@@\n")
    eval_path = tmp_path / "eval.csv"
    eval_path.write_text("random,canonical\nC,C\n")
    probe_path = tmp_path / "probe.csv"
    probe_path.write_text(
        "random,rs_label_id,canonical_chiral_token_id\nC,0,-1\n"
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in range(0, 120, 10):
        (checkpoint_dir / f"checkpoint-{step}").mkdir()

    output_root = tmp_path / "output"
    manifest = {
        "version": 1,
        "output_root": str(output_root),
        "shared": {
            "tokenizer_path": str(tokenizer_path),
            "eval_data": str(eval_path),
            "src_col": "random",
            "tgt_col": "canonical",
            "seed": 7,
        },
        "models": [{
            "name": "experiment_a",
            "checkpoint_dir": str(checkpoint_dir),
            "display_name": "baseline",
            "color": "tab:blue",
            "chiral_heads": [[6, 1], [7, 5]],
            "final_chiral_heads": [[7, 1], [5, 7]],
        }],
        "mutual_information": {
            "eval_data": str(probe_path),
            "src_col": "random",
        },
        "jump_detection": {
            "source": "perplexity",
            "threshold_ratio": 0.15,
            "ignore_initial_n": 1,
        },
        "analyses": {
            "enabled": [3, 6, 7, 8, 9],
            "visualize": [11, 12, 16],
        },
    }
    manifest_path = tmp_path / "pipeline.yml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path, output_root, checkpoint_dir, probe_path


def test_pipeline_resolves_compute_configs_from_one_manifest(pipeline_manifest):
    manifest_path, output_root, checkpoint_dir, probe_path = pipeline_manifest
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path, dry_run=True)

    jumpup = pipeline.compute_config("experiment_a", "03")
    mutual_information = pipeline.compute_config("experiment_a", "09")

    assert jumpup["checkpoint_dir"] == str(checkpoint_dir)
    assert jumpup["output_dir"] == str(output_root / "results/jumpup_metrics")
    assert mutual_information["eval_data"] == str(probe_path)
    assert mutual_information["output_dir"] == str(output_root / "results/mutual_information")


def test_jump_resolution_generates_dependent_checkpoint_configs(pipeline_manifest):
    manifest_path, output_root, checkpoint_dir, _ = pipeline_manifest
    trajectory_dir = output_root / "results/jumpup_metrics/experiment_a"
    trajectory_dir.mkdir(parents=True)
    steps = list(range(0, 120, 10))
    values = [10.0, 10.0, 10.0, 9.8, 9.0, 6.0, 3.0, 2.0, 1.8, 1.7, 1.6, 1.5]
    pd.DataFrame({"perplexity": values}, index=steps).to_csv(
        trajectory_dir / "logits_metrics_trajectory.csv"
    )
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)

    jump = pipeline.resolve_jump("experiment_a")
    cross_evaluation = pipeline.compute_config("experiment_a", "06", jump)
    head_ablation = pipeline.compute_config("experiment_a", "07", jump)

    # As in the paper: the interval start / peak / end checkpoints themselves.
    assert jump["selected"]["before_jump"]["step"] == jump["detected"]["start"]
    assert jump["selected"]["jump_peak"]["step"] == jump["detected"]["peak"]
    assert jump["selected"]["after_jump"]["step"] == jump["detected"]["end"]
    assert jump["detected"]["start"] < jump["detected"]["peak"] < jump["detected"]["end"]
    assert jump["selected"]["final"]["step"] == 110
    assert cross_evaluation["pre_jump_checkpoint"] == jump["selected"]["before_jump"]["path"]
    assert cross_evaluation["post_jump_checkpoint"] == jump["selected"]["after_jump"]["path"]
    assert Path(cross_evaluation["pre_jump_checkpoint"]).name.startswith("checkpoint-")
    assert Path(cross_evaluation["post_jump_checkpoint"]).name.startswith("checkpoint-")
    assert set(head_ablation["checkpoints"]) == {
        "before_jump", "jump_peak", "after_jump", "final"
    }
    assert (output_root / ".pipeline/jumps/experiment_a.json").is_file()
    assert head_ablation["model_name"] == "experiment_a"
    assert head_ablation["chiral_heads"] == [[6, 1], [7, 5]]
    assert head_ablation["final_chiral_heads"] == [[7, 1], [5, 7]]


def test_final_checkpoint_override_is_used_for_head_ablation(pipeline_manifest, tmp_path):
    manifest_path, output_root, _, _ = pipeline_manifest
    final_dir = tmp_path / "rough" / "checkpoint-700000"
    final_dir.mkdir(parents=True)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["models"][0]["final_checkpoint"] = str(final_dir)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    trajectory_dir = output_root / "results/jumpup_metrics/experiment_a"
    trajectory_dir.mkdir(parents=True)
    values = [10.0, 10.0, 10.0, 9.8, 9.0, 6.0, 3.0, 2.0, 1.8, 1.7, 1.6, 1.5]
    pd.DataFrame({"perplexity": values}, index=list(range(0, 120, 10))).to_csv(
        trajectory_dir / "logits_metrics_trajectory.csv"
    )
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)

    jump = pipeline.resolve_jump("experiment_a")
    head_ablation = pipeline.compute_config("experiment_a", "07", jump)

    assert jump["selected"]["final"] == {"step": 700000, "path": str(final_dir.resolve())}
    assert head_ablation["checkpoints"]["final"] == str(final_dir.resolve())
    assert "final_checkpoint" not in head_ablation


def test_pipeline_state_skips_unchanged_task_and_detects_new_checkpoint(
    pipeline_manifest, monkeypatch
):
    manifest_path, _, checkpoint_dir, _ = pipeline_manifest
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(_PIPELINE.subprocess, "run", fake_run)
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)
    config = pipeline.compute_config("experiment_a", "08")
    assert config["model_name"] == "experiment_a"

    pipeline._run_task("experiment_a:08", "08", config)
    pipeline._run_task("experiment_a:08", "08", config)
    assert len(calls) == 1

    (checkpoint_dir / "checkpoint-120").mkdir()
    pipeline._run_task("experiment_a:08", "08", config)
    assert len(calls) == 2


def test_visualization_fails_fast_when_compute_output_is_missing(pipeline_manifest):
    manifest_path, _, _, _ = pipeline_manifest
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)

    with pytest.raises(FileNotFoundError, match="Visualization 11"):
        pipeline._validate_visual_inputs("11")


def test_jump_resolution_rejects_incomplete_discovery_output(pipeline_manifest):
    manifest_path, output_root, _, _ = pipeline_manifest
    trajectory_dir = output_root / "results/jumpup_metrics/experiment_a"
    trajectory_dir.mkdir(parents=True)
    pd.DataFrame(
        {"perplexity": [10.0, 5.0]},
        index=[0, 10],
    ).to_csv(trajectory_dir / "logits_metrics_trajectory.csv")
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)

    with pytest.raises(ValueError, match="incomplete"):
        pipeline.resolve_jump("experiment_a")


def test_all_stage_dry_run_does_not_create_pipeline_files(pipeline_manifest, capsys):
    manifest_path, output_root, _, _ = pipeline_manifest
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path, dry_run=True)

    pipeline.run("all")

    output = capsys.readouterr().out
    assert "[dry-run]" in output
    assert "remaining analyses deferred" in output
    assert not output_root.exists()


def test_all_stops_after_selection_phase_when_heads_are_missing(
    pipeline_manifest, monkeypatch, capsys
):
    manifest_path, output_root, _, _ = pipeline_manifest
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["models"][0]["chiral_heads"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)
    calls = []
    monkeypatch.setattr(pipeline, "run_selection", lambda: calls.append("selection"))
    monkeypatch.setattr(pipeline, "run_remaining", lambda: calls.append("remaining"))

    pipeline.run("all")

    assert calls == ["selection"]
    assert "[awaiting-selection]" in capsys.readouterr().out
    selection = yaml.safe_load((output_root / ".pipeline/head_selection.json").read_text())
    assert selection["status"] == "awaiting_selection"
    assert selection["models"]["experiment_a"]["chiral_heads"] is None


def test_all_resumes_after_heads_are_added(pipeline_manifest, monkeypatch):
    manifest_path, output_root, _, _ = pipeline_manifest
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)
    calls = []
    monkeypatch.setattr(pipeline, "run_selection", lambda: calls.append("selection"))
    monkeypatch.setattr(pipeline, "run_remaining", lambda: calls.append("remaining"))

    pipeline.run("all")

    assert calls == ["selection", "remaining"]
    selection = yaml.safe_load((output_root / ".pipeline/head_selection.json").read_text())
    assert selection["status"] == "complete"
    assert selection["models"]["experiment_a"]["chiral_heads"] == [[6, 1], [7, 5]]
    assert selection["models"]["experiment_a"]["final_chiral_heads"] == [[7, 1], [5, 7]]


def test_remaining_phase_rejects_missing_head_selection(pipeline_manifest):
    manifest_path, _, _, _ = pipeline_manifest
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["models"][0]["chiral_heads"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)

    with pytest.raises(ValueError, match="awaiting manual selection"):
        pipeline.run("remaining")


def test_missing_final_chiral_heads_keeps_the_pipeline_waiting(pipeline_manifest, monkeypatch, capsys):
    manifest_path, output_root, _, _ = pipeline_manifest
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["models"][0]["final_chiral_heads"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)
    calls = []
    monkeypatch.setattr(pipeline, "run_selection", lambda: calls.append("selection"))
    monkeypatch.setattr(pipeline, "run_remaining", lambda: calls.append("remaining"))

    pipeline.run("all")

    assert calls == ["selection"]
    assert "experiment_a.final_chiral_heads" in capsys.readouterr().out
    selection = yaml.safe_load((output_root / ".pipeline/head_selection.json").read_text())
    assert selection["models"]["experiment_a"]["status"] == "awaiting_selection"
    assert selection["models"]["experiment_a"]["chiral_heads"] == [[6, 1], [7, 5]]
    assert selection["models"]["experiment_a"]["final_chiral_heads"] is None


def test_unknown_analysis_id_is_rejected(pipeline_manifest):
    manifest_path, _, _, _ = pipeline_manifest
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["analyses"]["enabled"] = [10]
    manifest_path.write_text(yaml.safe_dump(manifest))

    with pytest.raises(ValueError, match="Unsupported analysis ID"):
        _PIPELINE.AnalysisPipeline(manifest_path)


def test_task_hash_changes_when_library_code_changes(pipeline_manifest, monkeypatch, tmp_path):
    manifest_path, _, _, _ = pipeline_manifest
    pipeline = _PIPELINE.AnalysisPipeline(manifest_path)
    library = tmp_path / "lib.py"
    library.write_text("x = 1\n")
    monkeypatch.setattr(_PIPELINE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_PIPELINE, "LIBRARY_FILES", lambda: [library])
    script = _PIPELINE.SCRIPT_PATHS["07"]
    before = pipeline._task_hash(script, {"a": 1})
    library.write_text("x = 2\n")
    assert pipeline._task_hash(script, {"a": 1}) != before
