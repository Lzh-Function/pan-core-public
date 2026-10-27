# Running PanCORE

Command-line modes, their outputs, and use as a Python library. Back to the [README](../README.md).

Installing the package provides a `pan-core` command. It is the same entry point as
`python -m pan_core.main`; the examples below use the module form, and either works.

```bash
pan-core --config path/to/config.yml [MODE FLAGS] [OPTIONS]
python -m pan_core.main --config path/to/config.yml [MODE FLAGS] [OPTIONS]
```

The CLI sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the variable is
already set. The collator pads each batch to its own longest sequence, so the default
caching allocator sees thousands of block shapes and reserves far more GPU memory than it
allocates; expandable segments keep the two close. No computed value changes.

## Global arguments

| Argument       | Required | Description                                                                       |
| -------------- | -------- | --------------------------------------------------------------------------------- |
| `--config`     | Yes      | Path to the YAML configuration file                                               |
| `--output_dir` | No       | Output directory for checkpoints/logs/results. Defaults to `<config_dir>/output/` |


## Mode: Training (`--do_train`)

Runs the full training pipeline: config → tokenizer → model → data → train → save.

```bash
python -m pan_core.main \
    --config sample/config.yml \
    --output_dir runs/exp01 \
    --do_train
```

- Resumes automatically from the latest checkpoint if one exists in `--output_dir`.
- Final model is saved to `<output_dir>/final_model/`.
- Tokenizer vocabulary is saved alongside the model.


## Mode: Generation / Evaluation (`--do_eval_gen`)

Runs autoregressive greedy decoding (or teacher-forcing) over the evaluation dataset and saves results.

```bash
python -m pan_core.main \
    --config sample/config.yml \
    --output_dir runs/exp01 \
    --do_eval_gen \
    --checkpoint runs/exp01/final_model \
    --csv_output runs/exp01/gen_results.csv
```

| Argument            | Default                              | Description                                                                                 |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `--checkpoint`      | `<output_dir>/final_model`           | Path to the model checkpoint to load                                                        |
| `--encode_only`     | `false`                              | Only run the encoder; saves latent vectors, skips decoding                                  |
| `--latent_decode`   | `false`                              | Decode from pre-computed latent arrays (skips encoding; expects `.npy` at `eval_data.path`) |
| `--teacher_forcing` | `false`                              | Use teacher-forced forward pass instead of autoregressive decoding                          |
| `--csv_output`      | `<output_dir>/gen_results.csv`       | Output CSV path (columns below)                                                             |
| `--encode_output`   | `<output_dir>/encode_latents.npy`    | Output `.npy` path for latent vectors (used when `--encode_only`)                           |
| `--need_logits`     | `false`                              | Also collect and save full output logits                                                    |
| `--logits_output`   | `<output_dir>/gen_output_logits.npy` | Output `.npy` path for logits                                                               |

**Output CSV columns:** an unnamed row index (0 to N-1, in input order) followed by four
columns. Read it back with `pd.read_csv(path, index_col=0)`.

| Column      | Description                                              |
| ----------- | -------------------------------------------------------- |
| (index)     | Row number in the evaluation data                        |
| `judge`     | `True` if `generated == target` (perfect reconstruction) |
| `input`     | Decoded source SMILES                                    |
| `generated` | Model-generated SMILES                                   |
| `target`    | Ground-truth canonical SMILES                            |


## Mode: Accuracy Progress (`--get_acc_progress`)

Evaluates all `checkpoint-N` subdirectories in a directory and produces a training-step accuracy curve.

```bash
python -m pan_core.main \
    --config sample/config.yml \
    --output_dir runs/exp01 \
    --get_acc_progress \
    --checkpoint_dir runs/exp01
```

| Argument           | Required | Description                                        |
| ------------------ | -------- | -------------------------------------------------- |
| `--checkpoint_dir` | Yes      | Directory containing `checkpoint-N` subdirectories |

**Output**: `<output_dir>/acc_progress.csv` with columns `perfect_accuracy`, `partial_accuracy`, and (for phased/dynamic strategies) `phase{i}_data_perfect_accuracy` per phase. Per-checkpoint CSVs are also saved.


## Mode: Hidden State Extraction (`--get_hidden`)

Runs a full forward pass over the evaluation dataset and saves all intermediate hidden states and attention scores.

```bash
python -m pan_core.main \
    --config sample/config.yml \
    --output_dir runs/exp01 \
    --get_hidden \
    --checkpoint runs/exp01/final_model \
    --hidden_output runs/exp01/hidden_output.pkl
```

| Argument          | Default                          | Description                          |
| ----------------- | -------------------------------- | ------------------------------------ |
| `--checkpoint`    | `<output_dir>/final_model`       | Path to the model checkpoint         |
| `--hidden_output` | `<output_dir>/hidden_output.pkl` | Output `.pkl` path for hidden states |

**Output**: A pickle file containing a nested dict:

```python
{
    "outputs": {
        "encoder": [array_layer0, array_layer1, ...],  # each: [N, L, E]
        "decoder": [array_layer0, array_layer1, ...],
    },
    "attention_scores": {
        "encoder": [array_layer0, array_layer1, ...],  # each: [N, H, L, L]
        "decoder": [array_layer0, array_layer1, ...],
    }
}
```

> Note: `torch.compile` is intentionally disabled in this mode because `fullgraph=True` is incompatible with intermediate tensor returns (`need_hidden=True`).


## Using as a library

All four pipeline functions can be imported and called directly without the CLI:

```python
from pan_core.main import (
    run_training,
    run_generation_evaluation,
    run_accuracy_progress,
    get_hidden_outputs,
)

# Example: encode a custom DataFrame and retrieve latents
results, training_args = run_generation_evaluation(
    config="path/to/config.yml",
    output_dir="runs/exp01",
    checkpoint_path="runs/exp01/final_model",
    encode_only=True,
    latent_decode=False,
    encode_output_file="latents.npy",
)
```

`run_generation_evaluation` and `get_hidden_outputs` both accept a pre-loaded `(ModelConfig, TrainArgs)` tuple as the `config` argument, which avoids re-parsing the YAML when called in a loop.

## Repository structure

```
pan-core/
├── .github/workflows/ci.yml  # Lint and test on Python 3.10 – 3.12
├── docs/                     # Data preprocessing, configuration and usage guides
├── sample/
│   ├── config.yml            # Example configuration file (paper settings)
│   └── data/
│       ├── normal_tokens.txt # SMILES vocabulary (185 tokens)
│       └── sample_evaldata.csv # 95 example molecules (input for Data Preprocessing)
├── analysis/                 # Post-hoc checkpoint analysis (see analysis/README.md)
│   ├── pan_core_analysis/    # Shared analysis library used by the scripts
│   │   ├── metrics.py        # Metric functions (CKA, cosine, JSD, attention mass, logits, latent geometry)
│   │   ├── hooks.py          # HiddenStateCaptureHook, LatentCaptureHook, HeadAblationHook
│   │   ├── graph.py          # SMILES → token-level graph distance matrix (RDKit)
│   │   ├── visualization.py  # Jump-up detection, trajectory / heatmap plotting, cmthermal colormap
│   │   └── runner.py         # Greedy decoding evaluation loop, accuracy and resume helpers
│   ├── configs/
│   │   ├── pipeline.yml      # Single-manifest end-to-end analysis config
│   │   └── individual/       # Per-script config templates (00–18)
│   ├── run_pipeline.py       # End-to-end analysis runner
│   └── scripts/
│       ├── compute/          # Compute scripts (00–10)
│       └── visualize/        # Figure generation scripts (11–18)
├── tests/
│   ├── core/                 # Model, tokenizer, config, training and resume tests
│   └── analysis/             # Analysis library, script and pipeline tests
├── src/pan_core/
│   ├── main.py               # CLI entry point and library functions
│   ├── train.py              # HF Trainer subclass (HF_Trainer)
│   ├── config.py             # ModelConfig + TrainArgs (YAML → dataclasses)
│   ├── common/
│   │   ├── tokenizer.py      # SMILESTokenizer (HuggingFace-compatible)
│   │   ├── data_handler.py   # Datasets, samplers, and preprocessing utilities
│   │   └── utils.py          # Shared utilities (KLLoss, move_to_cpu, etc.)
│   └── models/
│       └── transformer/
│           ├── model.py      # PanCoreTransformerModel
│           └── layer.py      # Attention, TransformerBlock, SwiGLU, etc.
```
