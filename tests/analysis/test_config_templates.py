"""Contract tests for individual analysis config templates."""

import yaml

from tests.utils import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "analysis/configs"
INDIVIDUAL_DIR = CONFIG_DIR / "individual"

EXPECTED_FILES = {
    "00_probe_eval_set.yml",
    "01_accuracy_progress.yml",
    "02_detailed_decode.yml",
    "03_jumpup_metrics.yml",
    "04_latent_trajectory.yml",
    "05_residual_stream.yml",
    "06_cross_evaluation.yml",
    "07_head_ablation.yml",
    "08_weight_matrix.yml",
    "09_mutual_information.yml",
    "10_parameter_count.yml",
    "11_accuracy.yml",
    "12_jumpup_metrics.yml",
    "13_residual_stream.yml",
    "14_latent_trajectory.yml",
    "15_weight_matrices.yml",
    "16_mutual_information.yml",
    "17_sampling_schedule.yml",
    "18_seqlen_vs_mw.yml",
}

REQUIRED_KEYS = {
    "00": {"source_csv", "output_dir"},
    "01": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "02": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "03": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "04": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "05": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "06": {
        "model_name", "tokenizer_path", "eval_data", "output_dir",
        "pre_jump_checkpoint", "post_jump_checkpoint",
    },
    "07": {
        "model_name", "chiral_heads", "final_chiral_heads", "tokenizer_path", "eval_data", "output_dir",
        "checkpoints",
    },
    "08": {"model_name", "checkpoint_dir", "output_dir"},
    "09": {"model_name", "checkpoint_dir", "tokenizer_path", "eval_data", "output_dir"},
    "10": {"models", "output_dir"},
    "11": {"input_dir", "output_dir", "tokenizer_path", "eval_data"},
    "12": {"input_dir", "output_dir"},
    "13": {"input_dir", "output_dir", "jump_logits_dir"},
    "14": {"input_dir", "output_dir", "jump_logits_dir"},
    "15": {"input_dir", "output_dir"},
    "16": {"input_dir", "output_dir"},
    "17": {"output_dir"},
    "18": {"zinc_summary", "pubchem_summary", "output_dir"},
}


def test_individual_directory_contains_one_config_per_implemented_script():
    assert {path.name for path in INDIVIDUAL_DIR.glob("*.yml")} == EXPECTED_FILES


def test_pipeline_is_the_only_top_level_config():
    assert {path.name for path in CONFIG_DIR.glob("*.yml")} == {"pipeline.yml"}


def test_individual_configs_are_valid_and_include_required_keys():
    for path in sorted(INDIVIDUAL_DIR.glob("*.yml")):
        config = yaml.safe_load(path.read_text())
        analysis_id = path.name[:2]
        assert isinstance(config, dict), path
        assert REQUIRED_KEYS[analysis_id] <= config.keys(), path
        assert f"analysis/configs/individual/{path.name}" in path.read_text(), path


def test_head_ablation_template_matches_implementation_checkpoint_labels():
    config = yaml.safe_load((INDIVIDUAL_DIR / "07_head_ablation.yml").read_text())
    assert set(config["checkpoints"]) == {
        "before_jump", "jump_peak", "after_jump", "final"
    }


def test_model_names_are_placeholders_in_all_individual_configs():
    for path in sorted(INDIVIDUAL_DIR.glob("0[1-9]_*.yml")):
        config = yaml.safe_load(path.read_text())
        assert config["model_name"] == "your_model", path
    for path in sorted(INDIVIDUAL_DIR.glob("1[1-6]_*.yml")):
        config = yaml.safe_load(path.read_text())
        assert config["models"] == ["your_model"], path
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yml").read_text())
    assert [model["name"] for model in pipeline["models"]] == ["your_model"]
    assert pipeline["models"][0]["chiral_heads"] is None
    assert pipeline["models"][0]["final_chiral_heads"] is None
    assert "model_type" not in pipeline["models"][0]
