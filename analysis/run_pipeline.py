"""Run the complete PanCore analysis suite from one manifest.

The runner keeps the numbered compute and visualization scripts independently
usable. It resolves their individual YAML configs under ``.pipeline/configs``,
records resumable task state, and pauses after generating the head-selection
heatmaps when analysis 07 still needs human-selected ``chiral_heads`` /
``final_chiral_heads`` values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pan_core_analysis.visualization import detect_jump_interval


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPUTE_IDS = tuple(f"{value:02d}" for value in range(1, 10))
VISUALIZE_IDS = tuple(str(value) for value in range(11, 17))
DISCOVERY_IDS = ("01", "03")
DEPENDENT_IDS = ("06", "07")
HEAD_SELECTION_KEYS = ("chiral_heads", "final_chiral_heads")


def LIBRARY_FILES() -> list[Path]:
    """Library sources whose changes invalidate completed pipeline tasks."""
    files = sorted((REPO_ROOT / "analysis/pan_core_analysis").glob("*.py"))
    files += sorted((REPO_ROOT / "src/pan_core/models").rglob("*.py"))
    files += sorted((REPO_ROOT / "src/pan_core/common").glob("*.py"))
    if not files:
        raise FileNotFoundError(f"No library sources found under {REPO_ROOT}")
    return files

SCRIPT_PATHS = {
    **{
        analysis_id: REPO_ROOT / "analysis" / "scripts" / "compute" / f"{analysis_id}_{name}.py"
        for analysis_id, name in {
            "01": "accuracy_progress",
            "02": "detailed_decode",
            "03": "jumpup_metrics",
            "04": "latent_trajectory",
            "05": "residual_stream",
            "06": "cross_evaluation",
            "07": "head_ablation",
            "08": "weight_matrix",
            "09": "mutual_information",
        }.items()
    },
    **{
        analysis_id: REPO_ROOT / "analysis" / "scripts" / "visualize" / f"{analysis_id}_{name}.py"
        for analysis_id, name in {
            "11": "accuracy",
            "12": "jumpup_metrics",
            "13": "residual_stream",
            "14": "latent_trajectory",
            "15": "weight_matrices",
            "16": "mutual_information",
        }.items()
    },
}

COMPUTE_FIELDS = {
    "01": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col", "tgt_col", "seed",
        "batch_size", "max_length", "use_bf16", "n_phases", "checkpoint_steps",
        "teacher_forced_token_accuracy", "teacher_forced_batch_size",
    },
    "02": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col", "tgt_col",
        "seed", "batch_size", "max_length", "use_bf16", "n_phases", "checkpoint_steps",
    },
    "03": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col", "tgt_col",
        "seed", "batch_size", "max_len", "compute_graph_metrics", "checkpoint_steps",
        "phase", "max_phase",
    },
    "04": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col", "tgt_col",
        "seed", "batch_size", "max_len", "checkpoint_steps", "phase", "max_phase",
    },
    "05": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col", "tgt_col",
        "seed", "batch_size", "max_len", "checkpoint_steps", "phase", "max_phase",
    },
    "06": {
        "model_name", "tokenizer_path", "eval_data", "src_col", "tgt_col", "seed",
        "batch_size", "max_length", "use_bf16", "n_phases", "pre_jump_checkpoint", "post_jump_checkpoint",
    },
    "07": {
        "model_name", "tokenizer_path", "eval_data", "src_col", "tgt_col", "seed",
        "batch_size", "max_length", "n_phases", "legacy_all_heads_union", "chiral_heads",
        "final_chiral_heads", "checkpoints",
    },
    "08": {"model_name", "checkpoint_dir", "seed", "checkpoint_steps"},
    "09": {
        "model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "src_col",
        "rs_label_col", "canonical_token_label_col", "seed", "batch_size", "max_len",
        "probe_layers", "chiral_window", "n_splits", "min_samples", "n_probe_shuffle",
        "n_jobs", "enable_nonlinear_probe", "enable_canonical_token_probe", "enable_token_identity_probe", "mlp_epochs",
        "rf_estimators", "checkpoint_steps",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ids(values: list[Any] | None, valid: tuple[str, ...], field: str) -> list[str]:
    if values is None:
        return list(valid)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    normalized: list[str] = []
    for value in values:
        try:
            analysis_id = f"{int(value):02d}"
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid analysis ID in {field}: {value!r}") from error
        if analysis_id not in valid:
            raise ValueError(f"Unsupported analysis ID in {field}: {analysis_id}")
        if analysis_id not in normalized:
            normalized.append(analysis_id)
    return normalized


def _load_yaml(path: Path) -> dict:
    with open(path) as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline manifest must contain a YAML mapping: {path}")
    return value


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def _atomic_write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        yaml.safe_dump(value, file, sort_keys=False)
    temporary.replace(path)


class AnalysisPipeline:
    """Resolve and execute numbered analysis scripts from one manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        selected_models: list[str] | None = None,
        dry_run: bool = False,
        force: bool = False,
    ):
        self.manifest_path = manifest_path.resolve()
        self.manifest_dir = self.manifest_path.parent
        self.manifest = _load_yaml(self.manifest_path)
        if self.manifest.get("version") != 1:
            raise ValueError("Pipeline manifest version must be 1")

        output_root = self.manifest.get("output_root")
        if not isinstance(output_root, str) or not output_root:
            raise ValueError("output_root must be a non-empty path")
        self.output_root = self._absolute_path(output_root)
        self.results_root = self.output_root / "results"
        self.figures_root = self.output_root / "figures"
        self.pipeline_root = self.output_root / ".pipeline"
        self.config_root = self.pipeline_root / "configs"
        self.jump_root = self.pipeline_root / "jumps"
        self.state_path = self.pipeline_root / "pipeline_state.json"
        self.head_selection_path = self.pipeline_root / "head_selection.json"
        self.dry_run = dry_run
        self.force = force

        models = self.manifest.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("models must be a non-empty list")
        self.models: dict[str, dict] = {}
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("Each models entry must be a mapping")
            name = model.get("name")
            checkpoint_dir = model.get("checkpoint_dir")
            if not isinstance(name, str) or not name:
                raise ValueError("Each model must define a non-empty name")
            if name in self.models:
                raise ValueError(f"Duplicate model name: {name}")
            if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
                raise ValueError(f"Model {name} must define checkpoint_dir")
            self.models[name] = model

        if selected_models:
            unknown = sorted(set(selected_models) - self.models.keys())
            if unknown:
                raise ValueError(f"Unknown selected models: {unknown}")
            self.models = {name: self.models[name] for name in selected_models}

        analyses = self.manifest.get("analyses", {})
        if not isinstance(analyses, dict):
            raise ValueError("analyses must be a mapping")
        self.compute_ids = _normalize_ids(analyses.get("enabled"), COMPUTE_IDS, "analyses.enabled")
        self.visualize_ids = _normalize_ids(
            analyses.get("visualize"), VISUALIZE_IDS, "analyses.visualize"
        )
        self.state = self._load_state()

    def _chiral_heads(self, model_name: str, key: str = "chiral_heads") -> list[list[int]] | None:
        """``chiral_heads`` (jump checkpoints) or ``final_chiral_heads`` (final checkpoint) of a model."""
        if key not in HEAD_SELECTION_KEYS:
            raise ValueError(f"Unknown head selection key: {key}")
        value = self._model_settings(model_name, "07").get(key)
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} for {model_name} must be null or a non-empty list")
        for item in value:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(index, int) for index in item)
            ):
                raise ValueError(
                    f"{key} for {model_name} must contain [layer_idx, head_idx] integer pairs"
                )
        return value

    def _missing_head_selections(self) -> list[str]:
        """'model.key' for every head selection analysis 07 still needs."""
        if "07" not in self.compute_ids:
            return []
        return [
            f"{name}.{key}"
            for name in self.models
            for key in HEAD_SELECTION_KEYS
            if self._chiral_heads(name, key) is None
        ]

    def _record_head_selection(self, status: str) -> None:
        if self.dry_run or "07" not in self.compute_ids:
            return
        models = {}
        for name in self.models:
            selections = {key: self._chiral_heads(name, key) for key in HEAD_SELECTION_KEYS}
            models[name] = {
                "status": "awaiting_selection" if None in selections.values() else "selected",
                **selections,
                "heatmap_dir": str(self.figures_root / "12" / name),
            }
        _atomic_write_json(
            self.head_selection_path,
            {
                "version": 1,
                "status": status,
                "models": models,
                "instruction": (
                    "chiral_heads (before_jump / jump_peak / after_jump): top heads by interval_amplitude in "
                    "heatmap_dir/attention_mass_enc_head_ranking.csv. final_chiral_heads: top heads by "
                    "last_step_mass in the same file of the run whose last checkpoint is the final checkpoint "
                    "(this run if final_checkpoint is not set). Set both in the models entry of the pipeline "
                    "manifest and run the same command again."
                ),
                "updated_at": _utc_now(),
            },
        )

    def _absolute_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.manifest_dir / path).resolve()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"version": 1, "tasks": {}}
        with open(self.state_path) as file:
            state = json.load(file)
        if state.get("version") != 1 or not isinstance(state.get("tasks"), dict):
            raise ValueError(f"Invalid pipeline state: {self.state_path}")
        return state

    def _save_state(self) -> None:
        _atomic_write_json(self.state_path, self.state)

    def _path_settings(self, settings: dict) -> dict:
        resolved = dict(settings)
        for key in ("checkpoint_dir", "tokenizer_path", "eval_data"):
            if key in resolved:
                if not isinstance(resolved[key], str) or not resolved[key]:
                    raise ValueError(f"{key} must be a non-empty path")
                resolved[key] = str(self._absolute_path(resolved[key]))
        return resolved

    def _model_settings(self, model_name: str, analysis_id: str) -> dict:
        shared = self.manifest.get("shared", {})
        if not isinstance(shared, dict):
            raise ValueError("shared must be a mapping")
        model = self.models[model_name]
        settings = dict(shared)
        settings.update({
            key: value
            for key, value in model.items()
            if key not in {"name", "display_name", "color", "analysis_options", "final_checkpoint", "trainer_state"}
        })
        settings["model_name"] = model_name

        if analysis_id == "09":
            mutual_information = self.manifest.get("mutual_information", {})
            if not isinstance(mutual_information, dict):
                raise ValueError("mutual_information must be a mapping")
            settings.update(mutual_information)

        global_options = self.manifest.get("analysis_options", {})
        model_options = model.get("analysis_options", {})
        if not isinstance(global_options, dict) or not isinstance(model_options, dict):
            raise ValueError("analysis_options must be a mapping")
        for options in (global_options.get(analysis_id, {}), model_options.get(analysis_id, {})):
            if not isinstance(options, dict):
                raise ValueError(f"analysis_options.{analysis_id} must be a mapping")
            settings.update(options)
        return self._path_settings(settings)

    def compute_config(
        self,
        model_name: str,
        analysis_id: str,
        jump_manifest: dict | None = None,
    ) -> dict:
        if analysis_id not in COMPUTE_IDS:
            raise ValueError(f"Not a compute analysis ID: {analysis_id}")
        settings = self._model_settings(model_name, analysis_id)
        config = {key: settings[key] for key in COMPUTE_FIELDS[analysis_id] if key in settings}

        output_dirs = {
            "01": self.results_root / "accuracy",
            "02": self.results_root / "detailed_decode" / model_name,
            "03": self.results_root / "jumpup_metrics",
            "04": self.results_root / "latent_trajectory",
            "05": self.results_root / "residual_stream",
            "06": self.results_root / "cross_evaluation" / model_name,
            "07": self.results_root / "head_ablation",
            "08": self.results_root / "weight_matrix",
            "09": self.results_root / "mutual_information",
        }
        config["output_dir"] = str(output_dirs[analysis_id])

        if analysis_id in DEPENDENT_IDS:
            if jump_manifest is None:
                raise ValueError(f"Analysis {analysis_id} requires a resolved jump manifest")
            selected = jump_manifest["selected"]
            if analysis_id == "06":
                config["pre_jump_checkpoint"] = selected["before_jump"]["path"]
                config["post_jump_checkpoint"] = selected["after_jump"]["path"]
            else:
                config["checkpoints"] = {
                    label: selected[label]["path"]
                    for label in ("before_jump", "jump_peak", "after_jump", "final")
                }

        missing = sorted(COMPUTE_FIELDS[analysis_id] - config.keys() - self._optional_fields(analysis_id))
        if missing:
            raise ValueError(f"Analysis {analysis_id} for {model_name} is missing config keys: {missing}")
        return config

    @staticmethod
    def _optional_fields(analysis_id: str) -> set[str]:
        common = {
            "src_col", "tgt_col", "seed", "batch_size", "max_length", "max_len",
            "checkpoint_steps", "compute_graph_metrics", "use_bf16", "n_phases",
            "phase", "max_phase", "legacy_all_heads_union",
            "teacher_forced_token_accuracy", "teacher_forced_batch_size",
        }
        if analysis_id == "09":
            common |= {
                "probe_layers", "chiral_window", "n_splits", "min_samples", "n_probe_shuffle",
                "n_jobs", "enable_nonlinear_probe", "enable_canonical_token_probe", "enable_token_identity_probe", "mlp_epochs",
                "rf_estimators", "rs_label_col", "canonical_token_label_col",
            }
        return common

    def visualize_config(self, analysis_id: str) -> dict:
        if analysis_id not in VISUALIZE_IDS:
            raise ValueError(f"Not a visualization analysis ID: {analysis_id}")
        model_names = list(self.models)
        input_dirs = {
            "11": self.results_root / "accuracy",
            "12": self.results_root / "jumpup_metrics",
            "13": self.results_root / "residual_stream",
            "14": self.results_root / "latent_trajectory",
            "15": self.results_root / "weight_matrix",
            "16": self.results_root / "mutual_information",
        }
        config: dict[str, Any] = {
            "input_dir": str(input_dirs[analysis_id]),
            "output_dir": str(self.figures_root / analysis_id),
            "models": model_names,
        }
        if analysis_id == "11":
            for key in ("tokenizer_path", "eval_data", "tgt_col", "n_phases", "teacher_forced_token_accuracy"):
                values = {
                    self._model_settings(name, "01").get(key)
                    for name in model_names
                }
                if len(values) != 1 or (None in values and key in ("tokenizer_path", "eval_data")):
                    raise ValueError(f"Visualization 11 requires all selected models to share one {key}")
                value = values.pop()
                if value is not None:
                    config[key] = value
            trainer_states = {
                name: str(self._absolute_path(self.models[name]["trainer_state"]))
                for name in model_names
                if self.models[name].get("trainer_state")
            }
            if trainer_states:
                config["trainer_states"] = trainer_states
        if analysis_id in {"12", "13", "14", "15", "16"}:
            jump_config = self.manifest.get("jump_detection", {})
            config["jump_threshold_ratio"] = float(jump_config.get("threshold_ratio", 0.15))
            config["jump_ignore_initial_n"] = int(jump_config.get("ignore_initial_n", 10))
        if analysis_id in {"13", "14", "15", "16"}:
            config["jump_logits_dir"] = str(self.results_root / "jumpup_metrics")
        if analysis_id == "16":
            config["display_names"] = {
                name: self.models[name].get("display_name", name) for name in model_names
            }
            colors = {
                name: self.models[name]["color"]
                for name in model_names
                if self.models[name].get("color")
            }
            if colors:
                config["colors"] = colors

        visualization_options = self.manifest.get("visualization_options", {})
        if not isinstance(visualization_options, dict):
            raise ValueError("visualization_options must be a mapping")
        options = visualization_options.get(analysis_id, {})
        if not isinstance(options, dict):
            raise ValueError(f"visualization_options.{analysis_id} must be a mapping")
        config.update(options)
        config["input_dir"] = str(input_dirs[analysis_id])
        config["output_dir"] = str(self.figures_root / analysis_id)
        config["models"] = model_names
        return config

    @staticmethod
    def _fingerprint_path(path: Path, *, recursive: bool) -> list[tuple[str, int, int]]:
        if not path.exists():
            return [(str(path), -1, -1)]
        if path.is_file():
            stat = path.stat()
            return [(str(path), stat.st_size, stat.st_mtime_ns)]
        iterator = path.rglob("*") if recursive else path.iterdir()
        fingerprint: list[tuple[str, int, int]] = []
        for child in sorted(iterator):
            if child.is_file():
                stat = child.stat()
                fingerprint.append((str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns))
            elif not recursive:
                fingerprint.append((str(child.relative_to(path)), 0, child.stat().st_mtime_ns))
        return fingerprint

    def _input_fingerprint(self, config: dict) -> dict:
        fingerprint: dict[str, Any] = {}
        for key in ("tokenizer_path", "eval_data"):
            if config.get(key):
                fingerprint[key] = self._fingerprint_path(Path(config[key]), recursive=False)
        if config.get("checkpoint_dir"):
            checkpoint_dir = Path(config["checkpoint_dir"])
            fingerprint["checkpoint_dir"] = self._fingerprint_path(checkpoint_dir, recursive=False)
            model_files = []
            if checkpoint_dir.is_dir():
                for checkpoint in sorted(checkpoint_dir.glob("checkpoint-*")):
                    for filename in ("model.safetensors", "config.json"):
                        path = checkpoint / filename
                        if path.is_file():
                            stat = path.stat()
                            model_files.append((str(path.relative_to(checkpoint_dir)), stat.st_size, stat.st_mtime_ns))
            fingerprint["checkpoint_files"] = model_files
        for key in ("pre_jump_checkpoint", "post_jump_checkpoint"):
            if config.get(key):
                fingerprint[key] = self._fingerprint_path(Path(config[key]), recursive=False)
        if isinstance(config.get("checkpoints"), dict):
            fingerprint["checkpoints"] = {
                label: self._fingerprint_path(Path(path), recursive=False)
                for label, path in sorted(config["checkpoints"].items())
            }
        if config.get("input_dir"):
            fingerprint["input_dir"] = self._fingerprint_path(Path(config["input_dir"]), recursive=True)
        if config.get("jump_logits_dir"):
            fingerprint["jump_logits_dir"] = self._fingerprint_path(Path(config["jump_logits_dir"]), recursive=True)
        if isinstance(config.get("trainer_states"), dict):
            fingerprint["trainer_states"] = {
                name: self._fingerprint_path(Path(path), recursive=False)
                for name, path in sorted(config["trainer_states"].items())
            }
        return fingerprint

    def _task_hash(self, script_path: Path, config: dict) -> str:
        digest = hashlib.sha256()
        digest.update(script_path.read_bytes())
        # Library code changes must invalidate completed tasks (the scripts' own resume
        # signatures never run for a task the runner skips).
        for library_file in LIBRARY_FILES():
            digest.update(str(library_file.relative_to(REPO_ROOT)).encode())
            digest.update(library_file.read_bytes())
        digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
        digest.update(
            json.dumps(self._input_fingerprint(config), sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()

    def _run_task(self, task_name: str, analysis_id: str, config: dict) -> None:
        script_path = SCRIPT_PATHS[analysis_id]
        if not script_path.is_file():
            raise FileNotFoundError(f"Analysis script not found: {script_path}")
        task_hash = self._task_hash(script_path, config)
        previous = self.state["tasks"].get(task_name, {})
        if not self.force and previous.get("status") == "complete" and previous.get("hash") == task_hash:
            print(f"[skip] {task_name}")
            return

        config_path = self.config_root / f"{task_name.replace(':', '__')}.yml"
        command = [sys.executable, str(script_path), "--config", str(config_path)]
        if self.dry_run:
            print(f"[dry-run] {task_name}: {' '.join(command)}")
            return

        _atomic_write_yaml(config_path, config)
        self.state["tasks"][task_name] = {
            "status": "running",
            "hash": task_hash,
            "config": str(config_path),
            "started_at": _utc_now(),
        }
        self._save_state()
        print(f"[run] {task_name}")
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            self.state["tasks"][task_name].update({
                "status": "failed",
                "returncode": error.returncode,
                "finished_at": _utc_now(),
            })
            self._save_state()
            raise
        self.state["tasks"][task_name].update({
            "status": "complete",
            "finished_at": _utc_now(),
        })
        self._save_state()

    def _checkpoint_paths(self, model_name: str) -> dict[int, Path]:
        checkpoint_dir = Path(self._model_settings(model_name, "03")["checkpoint_dir"])
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
        checkpoints: dict[int, Path] = {}
        for path in checkpoint_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint-"):
                continue
            suffix = path.name.removeprefix("checkpoint-")
            if not suffix.isdigit():
                raise ValueError(f"Invalid checkpoint directory name: {path.name}")
            checkpoints[int(suffix)] = path.resolve()
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directories found in {checkpoint_dir}")
        return checkpoints

    def _final_checkpoint(self, model_name: str, checkpoints: dict[int, Path]) -> dict:
        """Checkpoint used as ``final`` by analysis 07.

        Defaults to the greatest step in ``checkpoint_dir``. A model may set
        ``final_checkpoint`` to a ``checkpoint-N`` directory of another series; the paper
        used step 700,000 of the 5,000-step series while the jump was detected on the
        500-step series.
        """
        value = self.models[model_name].get("final_checkpoint")
        if value is None:
            step = max(checkpoints)
            return {"step": step, "path": str(checkpoints[step])}
        if not isinstance(value, str) or not value:
            raise ValueError(f"final_checkpoint for {model_name} must be a non-empty path")
        path = self._absolute_path(value)
        suffix = path.name.removeprefix("checkpoint-")
        if not path.is_dir() or path.name == suffix or not suffix.isdigit():
            raise FileNotFoundError(f"final_checkpoint must be an existing checkpoint-N directory: {path}")
        return {"step": int(suffix), "path": str(path)}

    def resolve_jump(self, model_name: str) -> dict:
        jump_config = self.manifest.get("jump_detection", {})
        if not isinstance(jump_config, dict):
            raise ValueError("jump_detection must be a mapping")
        source = jump_config.get("source", "perplexity")
        if source != "perplexity":
            raise ValueError(f"Unsupported jump_detection.source: {source!r}")
        threshold_ratio = float(jump_config.get("threshold_ratio", 0.15))
        ignore_initial_n = int(jump_config.get("ignore_initial_n", 10))
        if not 0 < threshold_ratio <= 1:
            raise ValueError("jump_detection.threshold_ratio must be in (0, 1]")
        if ignore_initial_n < 0:
            raise ValueError("jump_detection.ignore_initial_n must be non-negative")

        trajectory_path = (
            self.results_root / "jumpup_metrics" / model_name / "logits_metrics_trajectory.csv"
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Jump trajectory is missing for {model_name}; run analysis 03 first: {trajectory_path}"
            )
        frame = pd.read_csv(trajectory_path, index_col=0)
        if source not in frame:
            raise ValueError(f"Jump trajectory is missing column {source!r}: {trajectory_path}")
        if frame.index.duplicated().any():
            raise ValueError(f"Jump trajectory contains duplicate steps: {trajectory_path}")
        try:
            frame.index = [int(value) for value in frame.index]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Jump trajectory index must contain integer steps: {trajectory_path}") from error

        checkpoints = self._checkpoint_paths(model_name)
        analysis_03 = self.compute_config(model_name, "03")
        requested_steps = analysis_03.get("checkpoint_steps")
        expected_steps = sorted(checkpoints if requested_steps is None else {int(step) for step in requested_steps})
        missing_checkpoints = sorted(set(expected_steps) - checkpoints.keys())
        if missing_checkpoints:
            raise FileNotFoundError(f"Configured analysis 03 checkpoints do not exist: {missing_checkpoints}")
        missing_trajectory = sorted(set(expected_steps) - set(frame.index))
        if missing_trajectory:
            raise ValueError(
                f"Analysis 03 is incomplete for {model_name}; missing trajectory steps: {missing_trajectory}"
            )

        series = frame.loc[expected_steps, source].dropna().sort_index()
        if len(series) != len(expected_steps):
            missing_values = sorted(set(expected_steps) - set(series.index))
            raise ValueError(f"Jump metric contains missing values at steps: {missing_values}")
        start, end, peak = detect_jump_interval(
            series,
            threshold_ratio=threshold_ratio,
            ignore_initial_n=ignore_initial_n,
        )
        start, end, peak = int(start), int(end), int(peak)
        # As in the paper (Tables 3/4, S4/S5): the interval start, the peak and the interval
        # end are themselves analyzed checkpoints and are used directly.
        selected_steps = {
            "before_jump": start,
            "jump_peak": peak,
            "after_jump": end,
        }
        for label, step in selected_steps.items():
            if step not in checkpoints:
                raise ValueError(f"Detected {label} step {step} has no checkpoint for {model_name}")
        selected = {
            label: {"step": step, "path": str(checkpoints[step])}
            for label, step in selected_steps.items()
        }
        selected["final"] = self._final_checkpoint(model_name, checkpoints)
        jump_manifest = {
            "version": 1,
            "model_name": model_name,
            "source": source,
            "trajectory": str(trajectory_path),
            "parameters": {
                "threshold_ratio": threshold_ratio,
                "ignore_initial_n": ignore_initial_n,
            },
            "detected": {"start": start, "peak": peak, "end": end},
            "selected": selected,
            "trajectory_steps": expected_steps,
            "available_checkpoint_steps": sorted(checkpoints),
            "generated_at": _utc_now(),
        }
        if not self.dry_run:
            _atomic_write_json(self.jump_root / f"{model_name}.json", jump_manifest)
        return jump_manifest

    def run_discovery(self) -> None:
        for model_name in self.models:
            for analysis_id in DISCOVERY_IDS:
                if analysis_id in self.compute_ids:
                    self._run_task(
                        f"{model_name}:{analysis_id}",
                        analysis_id,
                        self.compute_config(model_name, analysis_id),
                    )

    def run_compute(self) -> None:
        independent = [
            analysis_id
            for analysis_id in self.compute_ids
            if analysis_id not in DISCOVERY_IDS + DEPENDENT_IDS
        ]
        for model_name in self.models:
            for analysis_id in independent:
                self._run_task(
                    f"{model_name}:{analysis_id}",
                    analysis_id,
                    self.compute_config(model_name, analysis_id),
                )

    def run_dependent(self) -> None:
        dependent = [analysis_id for analysis_id in DEPENDENT_IDS if analysis_id in self.compute_ids]
        if not dependent:
            return
        for model_name in self.models:
            jump_manifest = self.resolve_jump(model_name)
            for analysis_id in dependent:
                self._run_task(
                    f"{model_name}:{analysis_id}",
                    analysis_id,
                    self.compute_config(model_name, analysis_id, jump_manifest),
                )

    def _validate_visual_inputs(self, analysis_id: str) -> None:
        expected: list[Path] = []
        for model_name in self.models:
            expected_path = {
                "11": self.results_root / "accuracy" / f"accuracy_{model_name}.csv",
                "12": (
                    self.results_root / "jumpup_metrics" / model_name / "logits_metrics_trajectory.csv"
                ),
                "13": (
                    self.results_root
                    / "residual_stream"
                    / model_name
                    / "l2_norm_chiral_enc_trajectory.csv"
                ),
                "14": (
                    self.results_root
                    / "latent_trajectory"
                    / model_name
                    / "latent_static_chiral_trajectory.csv"
                ),
                "15": self.results_root / "weight_matrix" / f"weight_matrix_{model_name}.json",
                "16": (
                    self.results_root
                    / "mutual_information"
                    / model_name
                    / f"mutual_information_{model_name}.json"
                ),
            }[analysis_id]
            expected.append(expected_path)
            if analysis_id in {"12", "13", "14", "15", "16"}:
                # Every visualization draws the jump-up interval of analysis 03.
                expected.append(
                    self.results_root / "jumpup_metrics" / model_name / "logits_metrics_trajectory.csv"
                )
            if analysis_id == "13":
                # Fig 6c,d / S6 / S7c,d: checkpoint-to-checkpoint cosine of analysis 03.
                expected.append(
                    self.results_root / "jumpup_metrics" / model_name / "cos_chiral_enc_trajectory.csv"
                )
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Visualization {analysis_id} is missing required compute outputs: {missing}"
            )

    def run_visualize(self, analysis_ids: list[str] | None = None) -> None:
        for analysis_id in analysis_ids if analysis_ids is not None else self.visualize_ids:
            if not self.dry_run:
                self._validate_visual_inputs(analysis_id)
            self._run_task(
                f"visualize:{analysis_id}",
                analysis_id,
                self.visualize_config(analysis_id),
            )

    def run_selection(self) -> None:
        self.run_discovery()
        if "07" in self.compute_ids:
            if "03" not in self.compute_ids:
                raise ValueError("Analysis 07 requires analysis 03 for jump and head selection")
            if "12" not in self.visualize_ids:
                raise ValueError("Analysis 07 requires visualization 12 for manual head selection")
            self.run_visualize(["12"])

    def run_remaining(self) -> None:
        missing = self._missing_head_selections()
        if missing:
            raise ValueError(
                "Head selection is awaiting manual selection: " + ", ".join(missing)
            )
        self.run_compute()
        self.run_dependent()
        remaining_visualizations = self.visualize_ids
        if "07" in self.compute_ids:
            remaining_visualizations = [value for value in self.visualize_ids if value != "12"]
        self.run_visualize(remaining_visualizations)

    def _finish_selection_phase(self) -> bool:
        missing = self._missing_head_selections()
        if not missing:
            self._record_head_selection("selected")
            return True
        self._record_head_selection("awaiting_selection")
        print(
            "[awaiting-selection] Inspect visualization 12 (attention_mass_enc_head_ranking.csv) and set: "
            + ", ".join(missing)
        )
        print(f"[awaiting-selection] State: {self.head_selection_path}")
        return False

    def run(self, stage: str) -> None:
        stages = {
            "selection": self.run_selection,
            "remaining": self.run_remaining,
            "discovery": self.run_discovery,
            "compute": self.run_compute,
            "dependent": self.run_dependent,
            "visualize": self.run_visualize,
        }
        if stage == "all":
            self.run_selection()
            if not self._finish_selection_phase():
                return
            if self.dry_run:
                try:
                    self.run_remaining()
                except FileNotFoundError as error:
                    print(f"[dry-run] remaining analyses deferred until selection completes: {error}")
            else:
                self.run_remaining()
                self._record_head_selection("complete")
            return
        if stage == "selection":
            self.run_selection()
            self._finish_selection_phase()
            return
        if stage == "remaining":
            self.run_remaining()
            self._record_head_selection("complete")
            return
        stages[stage]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end PanCore analysis pipeline.")
    parser.add_argument("--config", required=True, help="Path to the single pipeline manifest.")
    parser.add_argument(
        "--stage",
        choices=(
            "all", "selection", "remaining", "discovery", "compute", "dependent", "visualize"
        ),
        default="all",
    )
    parser.add_argument("--model", action="append", dest="models", help="Run only this model; repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or executing.")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even when their hash is complete.")
    args = parser.parse_args()

    pipeline = AnalysisPipeline(
        Path(args.config),
        selected_models=args.models,
        dry_run=args.dry_run,
        force=args.force,
    )
    pipeline.run(args.stage)


if __name__ == "__main__":
    main()
