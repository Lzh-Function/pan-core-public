# Analysis Scripts

Reproducibility scripts for figures and metrics in:

> Zehao Li, Yasuhiro Yoshikai, Shumpei Nemoto, Hiroyuki Kusuhara, Tadahaya Mizuno.
> **"From Syntax to Semantics: Unveiling the Emergence of Chirality in SMILES Translation Models."**
> arXiv:2605.09949 (2026). https://doi.org/10.48550/arXiv.2605.09949

---

## Setup

Install pan-core with the analysis extra and run every command from the repository root:

```bash
pip install -e ".[analysis]"
python analysis/run_pipeline.py --config /path/to/my_analysis.yml
```

The scripts import the shared library `analysis/pan_core_analysis/` by adding `analysis/` to
`sys.path` themselves; it needs no separate installation. The runner executes every numbered
script with the same Python interpreter it was started with. Analyses `01`–`07` and `09` run
the models and use a GPU when one is visible (they fall back to CPU, which is impractically slow
for full checkpoint series); `00`, `08`, `10` and all visualizations do not need a GPU.

---

## Usage: run the complete analysis

The recommended workflow uses one pipeline manifest and the same command for
both runs. Individual numbered configs are not required.

### 1. Prepare the inputs

Before starting, prepare:

- a checkpoint directory containing `checkpoint-N/` subdirectories;
- a tokenizer vocabulary file;
- an evaluation CSV with the source and target SMILES columns, made of equal
  consecutive length buckets (`n_phases`, default 5); and
- if analysis `09` is enabled, its probe CSV (built by `00_probe_eval_set.py`).

Each checkpoint must contain its model weights and `config.json`. Analyses `07`
and `08` derive the model structure from that file instead of using model-name
profiles.

### 2. Create one pipeline manifest

Copy the template to a project-specific location. Keeping the working copy
outside `analysis/configs/` makes template updates easier to review.

```bash
cp analysis/configs/pipeline.yml /path/to/my_analysis.yml
```

The template holds the paper settings for the 500-step checkpoint series. At
minimum, replace `output_root`, the data paths, and each model's checkpoint
directory. A model `name` is only an arbitrary output label.

```yaml
version: 1
output_root: /path/to/analysis-output

shared:
  tokenizer_path: /path/to/normal_tokens.txt
  eval_data: /path/to/valid_10k.csv
  src_col: random
  tgt_col: canonical
  seed: 42

models:
  - name: your_model
    checkpoint_dir: /path/to/checkpoints
    final_checkpoint: /path/to/other_series/checkpoint-700000   # optional
    chiral_heads: null
    final_chiral_heads: null
```

Optional model keys:

- `final_checkpoint`: a `checkpoint-N` directory used as `final` by analysis `07`
  (paper: step 700,000 of the 5,000-step series). Default: the last checkpoint of
  `checkpoint_dir`.
- `trainer_state`: `trainer_state.json` whose `log_history` gives the training and
  per-bucket evaluation loss of Fig 2 (visualization `11`). Set it only for a
  `checkpoint_dir` that covers the whole training run (paper: manifest B); otherwise the
  loss axis extends far beyond the accuracy axis.
- `display_name`, `color`: labels and colors used by visualization `16`.
- `analysis_options`: per-model overrides with the same structure as the global
  `analysis_options`.

Leave `chiral_heads` and `final_chiral_heads` as `null` for the first run. Head
importance is selected from the attention-mass results of the checkpoints themselves
and cannot be inferred from the checkpoint architecture. Analysis `07` ablates, at
every checkpoint, the chiral heads of that checkpoint:

- `chiral_heads`: used at `before_jump`, `jump_peak` and `after_jump`; the heads with
  the largest change of attention mass on chiral tokens inside the jump-up interval;
- `final_chiral_heads`: used at `final`; the heads with the largest attention mass on
  chiral tokens at the final checkpoint (use the same list when they did not change).

For multiple models, add one `models` entry per model; each entry has its own selections.

If analysis `09` is enabled, also fill in the `mutual_information` section. If
the probe data is not available, remove `9` and `16` from `analyses.enabled` and
`analyses.visualize`, respectively.

### 3. Run phase 1: generate the selection heatmap

```bash
python analysis/run_pipeline.py --config /path/to/my_analysis.yml
```

Phase 1 runs analyses `01` and `03`, then visualization `12`. With a `null` head
selection and analysis `07` enabled, the command exits successfully after writing:

```text
{output_root}/figures/12/{model_name}/
├── attention_mass_enc_heatmap.{png,svg}
├── attention_mass_enc_head_ranking.csv   # interval_amplitude, interval_amplitude_rank, last_step, last_step_mass, last_step_rank
├── attention_mass_dec_heatmap.{png,svg}
└── ... (top-N line plots, entropy / graph-distance heatmaps, perplexity_cdr)

{output_root}/.pipeline/head_selection.json
```

The JSON file records `awaiting_selection`; it is a status record, not another
user-maintained config. Without analysis `07` the pipeline does not pause.

### 4. Select the chiral-sensitive heads

Inspect the attention-mass heatmaps and `attention_mass_enc_head_ranking.csv`, and
enter the selected `[layer_idx, head_idx]` pairs in the same pipeline manifest:

- `chiral_heads`: the top rows by `interval_amplitude_rank`;
- `final_chiral_heads`: the top rows by `last_step_rank` of a run whose last analyzed
  checkpoint is the final checkpoint. Without `final_checkpoint` this is the same run;
  with a `final_checkpoint` from another series, use the ranking of a run over that
  series (paper: manifest B below, step 700,000).

```yaml
models:
  - name: your_model
    checkpoint_dir: /path/to/checkpoints
    chiral_heads: [[6, 5], [7, 1], [7, 4]]
    final_chiral_heads: [[7, 1], [5, 7], [7, 0]]
```

The indices must be integers and must fit the layer/head dimensions stored in
the checkpoints' `config.json`. Duplicate or out-of-range entries fail fast.

### 5. Run phase 2 with the same command

```bash
python analysis/run_pipeline.py --config /path/to/my_analysis.yml
```

The runner skips unchanged phase-1 tasks, derives the jump-up checkpoints from
analysis `03`, and runs the remaining enabled compute analyses plus the remaining
visualizations. Results are written below `{output_root}/results/`, and every
visualization writes both PNG and SVG (300 dpi) below `{output_root}/figures/`.

The jump-up interval is detected on analysis `03`'s perplexity trajectory
(`jump_detection`: rolling mean of 3, `threshold_ratio` 0.15, `ignore_initial_n` 10,
as in the paper). The checkpoints passed to analyses `06` and `07` are:

- `before_jump`: the interval start;
- `jump_peak`: the interval peak;
- `after_jump`: the interval end; and
- `final`: `final_checkpoint` if given, else the greatest available checkpoint step.

Each of these must exist as a checkpoint directory; otherwise the runner raises.
Analysis `06` swaps the encoder/decoder of `before_jump` and `after_jump`.

### Useful controls and recovery

```bash
# Print phase-1 commands without executing them.
python analysis/run_pipeline.py --config /path/to/my_analysis.yml --dry-run

# Run one workflow phase explicitly.
python analysis/run_pipeline.py --config /path/to/my_analysis.yml --stage selection
python analysis/run_pipeline.py --config /path/to/my_analysis.yml --stage remaining

# Restrict execution to one model.
python analysis/run_pipeline.py --config /path/to/my_analysis.yml --model your_model

# Deliberately rerun tasks even when their recorded inputs are unchanged.
python analysis/run_pipeline.py --config /path/to/my_analysis.yml --force
```

Generated per-script configs, jump manifests, and resumable task state live
under `{output_root}/.pipeline/`. A failed task is recorded there; rerunning the
same command resumes from completed tasks. The runner executes the scripts
sequentially in the current process environment; to use a batch scheduler, submit
the runner command itself as one job.

---

## Reproducing the paper

The paper analyzes three models (pancore-baseline, pancore-addonce,
pancore-xattn-adalnzero), each with two checkpoint series:

- **500-step series**: steps 500–74,500 (addonce), 500–64,500 (xattn-adalnzero),
  500–201,500 (baseline);
- **5,000-step series**: steps 5,000–700,000.

Evaluation data (all with columns `canonical`, `random`; five equal length buckets
ZINC (~100), PC (~100), PC (101~200), PC (201~300), PC (301~400)):

| file                              | rows     | used by                                         |
| --------------------------------- | -------- | ----------------------------------------------- |
| `valid_10k.csv`                   | 5×2,000  | `01`, `02`, `06`, `07`; `04` (bucket 1 only)    |
| `sampled_valid_1k_seed_42.csv`    | 5×200    | `03`, `05` (bucket 1 only)                      |
| `sampled_100k.csv` (training)     | 5×20,000 | `02` for Table 2 (train); `00` probe-set source |
| `single_chiral_rs_probe_7484.csv` | 7,484    | `09` (built by `00` from `sampled_100k.csv`)    |

### Manifest A: 500-step series

`analysis/configs/pipeline.yml` as shipped (`checkpoint_dir` = 500-step series,
`final_checkpoint` = `checkpoint-700000` of the 5,000-step series). Run manifest B to
completion first: its `figures/12/{model}/attention_mass_enc_head_ranking.csv` gives
`final_chiral_heads` (`last_step_rank`, last step = 700,000). Take `chiral_heads` only
from manifest A (`interval_amplitude_rank`): on the 5,000-step series the jump-up lies
inside the 10 ignored initial points, so manifest B's interval columns and band are not
the paper interval. The heads of the paper:

| model                   | `chiral_heads` (manifest A, interval amplitude) | `final_chiral_heads` (manifest B, mass at 700,000) |
| ----------------------- | ----------------------------------------------- | -------------------------------------------------- |
| pancore-addonce         | `[[6, 1], [6, 7], [7, 5]]`                      | `[[6, 1], [6, 7], [7, 5]]`                         |
| pancore-xattn-adalnzero | `[[6, 5], [7, 1], [7, 4]]`                      | `[[7, 1], [5, 7], [7, 0]]`                         |
| pancore-baseline        | `[[7, 0], [7, 2], [7, 7]]`                      | `[[7, 7], [7, 0], [1, 0]]`                         |

The random control heads are drawn exactly as in the paper run (one `Random(42)` stream,
pool = final two layers minus `chiral_heads`).

`analysis_options."07".legacy_all_heads_union: true` together with
`final_chiral_heads` = `chiral_heads` reproduces the arXiv v1 run (see "Known
differences" below).

### Manifest B: 5,000-step series

A second manifest with a different `output_root`, `checkpoint_dir` = 5,000-step
series, `trainer_state` = the final checkpoint's `trainer_state.json`, no
`final_checkpoint`, and:

```yaml
analyses:
  enabled: [1, 2, 3, 4, 5]
  visualize: [11, 12, 13, 14]   # 12: final_chiral_heads ranking at step 700,000
analysis_options:
  "02":
    # Table 2 (train): the final checkpoint on the 100k training sample.
    eval_data: /path/to/sampled_100k.csv
    checkpoint_steps: [700000]
  # "01", "03", "04", "05": as in manifest A
visualization_options:
  "13": {show_jump: false, xlim: [0, 80000], cos_ylim: [0.6, 1.0]}
  "14": {show_jump: false, xlim: [0, 80000], ylim: {cosine: [0.95, 1.0], cka: [0.90, 1.0]}}
```

Analysis `05` is only needed because visualization `13` reads its L2 norms together
with `03`'s cosine trajectories (Fig S6 uses only the latter).

### Stand-alone scripts

| script                              | paper                 | config template                       |
| ----------------------------------- | --------------------- | ------------------------------------- |
| `compute/00_probe_eval_set.py`      | probe set (Section C) | `individual/00_probe_eval_set.yml`    |
| `compute/10_parameter_count.py`     | Table S3              | `individual/10_parameter_count.yml`   |
| `visualize/17_sampling_schedule.py` | Fig S1                | `individual/17_sampling_schedule.yml` |
| `visualize/18_seqlen_vs_mw.py`      | Fig S2                | `individual/18_seqlen_vs_mw.yml` (inputs ≈ 2 GB each; needs several GB of memory) |

```bash
python analysis/scripts/compute/00_probe_eval_set.py --config analysis/configs/individual/00_probe_eval_set.yml
```

### Figure and table map

| paper                 | series      | compute → visualize | output                                                                       |
| --------------------- | ----------- | ------------------- | ---------------------------------------------------------------------------- |
| Fig 2                 | 5,000       | `01` → `11`         | `loss_acc_progress_{model}` (needs `trainer_state`), `bucket_accuracy_{model}` |
| Table 2 (eval)        | 5,000       | `01`                | `accuracy_{model}.csv`, row 700000, `phase{p}_accuracy`                      |
| Table 2 (train)       | 5,000       | `02`                | `decode_summary_{model}.csv`, `phase{p}_accuracy`                            |
| Fig 3                 | 500         | `01` → `11`         | `token_accuracy_teacher_forced_{model}_phase{1,2}` (teacher-forced); v1 figure: `token_accuracy_{model}_phase{1,2}` |
| Fig S3 a / b          | 500 / 5,000 | `01` → `11`         | `chiral_accuracy_{model}`                                                    |
| Fig 4                 | 500         | `03` → `12`         | `perplexity_cdr`                                                             |
| Fig 5                 | 500         | `03` → `12`         | `attention_mass_{enc,dec}_heatmap`, `attention_mass_{enc,dec}_top3`          |
| Fig S4                | 500         | `03` → `12`         | `attention_entropy_{enc,dec}_heatmap`, `..._top3`                            |
| Fig S5                | 500         | `03` → `12`         | `graph_dist_from_{enc,dec}_heatmap`, `..._top3`                              |
| Fig 6 a,b / S7 a,b    | 500         | `05` → `13`         | `l2_norm_{background,chiral}_{enc,dec}_lineplot`                             |
| Fig 6 c,d / S7 c,d    | 500         | `03` → `13`         | `cos_{background,chiral}_{enc,dec}_lineplot`                                 |
| Fig S6                | 5,000       | `03` → `13`         | `cos_{background,chiral}_enc_lineplot`                                       |
| Fig 7 a–c             | 500         | `04` → `14`         | `latent_l2_norm`, `latent_cosine`, `latent_cka`                              |
| Fig S8                | 5,000       | `04` → `14`         | `latent_cosine`, `latent_cka`                                                |
| Fig 7 d               | 500         | `09` → `16`         | `figure7d_{model}`                                                           |
| Fig S9 a / b / c      | 500         | `09` → `16`         | `figure7_stereo_probe_comparison` / `figure1_rs_probe` / `figure6_rs_linear_vs_nonlinear` |
| probe numbers in text | 500         | `09` → `16`         | `probe_summary_{model}.json`                                                 |
| Table 3 / S4          | 500         | `06`                | `cross_evaluation_{model}.csv`                                               |
| Table 4 / S5          | 500 + final | `12` (A, B) → `07`  | `head_ablation_{model}.csv`; heads from `attention_mass_enc_head_ranking.csv` |
| Table S3              | —           | `10`                | `parameter_count.csv`                                                        |
| Fig S1                | —           | `17`                | `sampling_schedule`                                                          |
| Fig S2                | —           | `18`                | `zinc_seqlen_mw_hexbin`, `pubchem_seqlen_mw_hexbin_zoomed`                   |

Paper decode settings, kept as defaults: `01`/`02` batch 3584, bf16 decoder loop,
`max_length` 402; `06` fp32, batch 1024; `07` bf16, batch 2048. bf16 greedy outputs
depend slightly on library versions: the paper numbers were computed with torch 2.10, and
re-running `07` with another PyTorch version gave 85–89 % identical generations and
differences of up to 0.015 on the ZINC20 / PubChem (~100) metrics. The "no ablation" rows
of Table 4 come from the fp32 run of Table 3 (pre/post) and from the 5,000-step decode
(final; manifest B `accuracy_{model}.csv`, row 700000), not from `07`, whose bf16
`no_ablation` rows are the precision-matched alternative.

### Known differences from the published paper (arXiv v1)

- **Table 4 / S5, "all chiral heads"**: the original run zeroed heads {1, 5, 6, 7} of
  layers 6 and 7 (8 heads) for pancore-addonce and {1, 4, 5, 6, 7} of layers 6 and 7
  (10 heads) for pancore-xattn-adalnzero instead of the 3 selected heads, and used the
  jump-interval heads also at the final checkpoint. The revised analysis ablates exactly
  the chiral heads of each checkpoint (`chiral_heads` / `final_chiral_heads`).
  `legacy_all_heads_union: true` with `final_chiral_heads` = `chiral_heads` reproduces v1.
- **Fig 3**: the v1 figure was computed from free-running greedy outputs compared position
  by position (`token_accuracy_*`), not teacher-forced as the Methods state. `01` now also
  writes the teacher-forced token accuracy (`token_accuracy_teacher_forced_*`, fp32).
- **Fig 4b**: the v1 band was detected on the CDR trajectory (43,000–52,000 for
  pancore-addonce); `12` draws the perplexity interval on both panels (corrected figure).
- **Fig S4**: heads are ranked by in-interval amplitude (max − min), not by the
  largest entropy decrease.
- **Input @/@@ control (Section C)**: the published "≥ 0.9999 at every checkpoint" was
  computed by an earlier probe version on `valid_10k.csv` (layers 0–7). It holds for
  pancore-addonce and pancore-xattn-adalnzero (minimum 0.99993) but not for
  pancore-baseline (minimum 0.99965 at layer 7). `09` reproduces this probe with
  `enable_token_identity_probe: true`; `16` reports the per-layer minimum in
  `probe_summary_{model}.json`.
- **Fig S1** shows the interpolated bucket probabilities before normalization; the
  training sampler renormalizes them (`normalize: true` in `17`).

---

## Using the library from your own code

| Module             | Contents                                                                                                                                                                                                                                                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metrics.py`       | `compute_linear_cka`, `compute_cosine_similarity`, `compute_jsd`, `compute_entropy`, `compute_attention_mass`, `compute_attention_entropy`, `compute_sar`, `compute_nit`, `compute_logits_metrics` (perplexity, CDR, chiral margin), `compute_l2_norm`, `compute_active_units`, `compute_intrinsic_dimension_twonn`, `compute_isotropy_entropy` |
| `hooks.py`         | `HiddenStateCaptureHook` — context manager that captures per-layer hidden states `[B, L, E]` and attention scores `[B, H, L, L]`; `LatentCaptureHook` — captures the encoder pooling output `[B, E]`; `HeadAblationHook` — pre-hook that zeros specified attention heads (incompatible with `torch.compile(fullgraph=True)`)                    |
| `graph.py`         | `create_token_graph_distance_matrix` — maps a SMILES string to a token-level graph-distance matrix via RDKit; `collate_distance_matrices` — pads and stacks a batch                                                                                                                                                                             |
| `visualization.py` | `detect_jump_interval` / `load_jump_interval` — detect the jump-up interval from a 1-D time series (rolling gradient + threshold walk); `plot_trajectory_with_jumpup` / `plot_heatmap_with_jumpup` — annotated line / heatmap plots; registers the **cmthermal** colormap                                                                        |
| `runner.py`        | `run_generation_evaluation` — greedy decoding loop over any dataset, returns a DataFrame; `compute_accuracy_from_results` / `compute_phase_metrics` — overall, per-bucket and chiral accuracy; `check_resume_signature` — guards resumed runs against config or code changes                                                                     |

The numbered scripts add `analysis/` to `sys.path` themselves, so `pan_core_analysis` needs no
installation. To use it from your own code, do the same:

```python
import sys
sys.path.insert(0, "path/to/pan-core/analysis")

from pan_core.models import PanCoreTransformerModel
from pan_core_analysis.hooks import HiddenStateCaptureHook
from pan_core_analysis.metrics import compute_linear_cka

model = PanCoreTransformerModel.from_pretrained("runs/exp01/checkpoint-10000")
model.eval()

# Capture hidden states from one forward pass
with HiddenStateCaptureHook(model) as hook:
    output = model(input_ids=enc_ids, labels=dec_ids,
                   encoder_attention_mask=enc_mask, need_hidden=True, return_dict=True)

enc_hidden = hook.hidden_states["encoder"]   # list of [B, L, E] numpy arrays
enc_attn   = hook.attn_scores["encoder"]     # list of [B, H, L, L] numpy arrays

# Linear CKA between two layers
cka = compute_linear_cka(enc_hidden[0].reshape(-1, enc_hidden[0].shape[-1]),
                          enc_hidden[1].reshape(-1, enc_hidden[1].shape[-1]))
print(f"Emb → L0 CKA: {cka:.4f}")
```

---

## Directory layout

```
analysis/
├── pan_core_analysis/  # Shared library (metrics, hooks, graph, visualization, runner)
├── configs/
│   ├── pipeline.yml    # Single manifest for the end-to-end workflow (paper settings)
│   └── individual/     # Per-script templates (00–10 and 11–18)
├── run_pipeline.py     # Single-manifest end-to-end runner (analyses 01–09, 11–16)
└── scripts/
    ├── compute/        # 00–10: probe set, checkpoint-wise metrics, parameter count
    └── visualize/      # 11–18: figure generation
```

---

## Pipeline structure

Each compute script reads model checkpoints and writes CSV (or JSON) to an output
directory. Each visualize script reads those outputs and writes PNG + SVG figures.

```
checkpoints  ->  compute/0N_*.py  ->  results/  ->  visualize/1N_*.py  ->  figures/
```

Numbered scripts can also be run directly with their individual templates:

```bash
python analysis/scripts/compute/01_accuracy_progress.py --config analysis/configs/individual/01_accuracy_progress.yml
python analysis/scripts/visualize/11_accuracy.py        --config analysis/configs/individual/11_accuracy.yml
```

---

## Compute scripts

| Script                     | What it computes                                                                                                         | Key outputs                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `00_probe_eval_set.py`     | Single-stereocentre R/S-balanced probe set with R/S and canonical/random @/@@ labels                                    | `single_chiral_rs_probe_{N}.csv`, `..._summary.json`              |
| `01_accuracy_progress.py`  | Exact-match accuracy (pooled and per bucket), chi-/non-chi-SMILES accuracy, free-running position-aligned and teacher-forced token accuracy per bucket | `accuracy_{model}.csv`, `token_accuracy[_teacher_forced]_{model}_phase{p}.csv` |
| `02_detailed_decode.py`    | Full per-sample decode (input / target / generated) for selected checkpoints, with per-bucket summary                   | `decode_{model}_step{N}.csv`, `decode_summary_{model}.csv`        |
| `03_jumpup_metrics.py`     | Attention mass/entropy, SAR, graph/seq distance, NIT, JSD, CKA, checkpoint-to-checkpoint cosine, logits metrics (perplexity, CDR, mean rank) | `*_trajectory.csv` under `{output_dir}/{model}/` |
| `04_latent_trajectory.py`  | Latent geometry: L2 norm, active units, intrinsic dim, isotropy, radial CV, cosine / linear and kernel CKA between checkpoints | 4 CSVs under `{output_dir}/{model}/`                         |
| `05_residual_stream.py`    | Per-layer residual-stream L2 norm; adjacent-layer cosine / CKA (supplementary)                                          | 12 CSVs under `{output_dir}/{model}/`                             |
| `06_cross_evaluation.py`   | Encoder × decoder swap of `before_jump` / `after_jump` (4 combinations), per bucket                                     | `cross_evaluation_{model}.csv`                                    |
| `07_head_ablation.py`      | Ablation of all / single chiral heads and size-matched random heads at 4 checkpoints, per bucket                        | `head_ablation_{model}.csv`                                       |
| `08_weight_matrix.py`      | Weight matrix L2 norm, Frobenius diff, CKA, cosine across checkpoints (supplementary)                                   | `weight_matrix_{model}.json`                                      |
| `09_mutual_information.py` | Linear/nonlinear R/S probes, canonical @/@@ probe, layer-wise mean/window-pool probes, optional input @/@@ control probe | `mutual_information_{model}.json`                                 |
| `10_parameter_count.py`    | Trainable parameters from each checkpoint's `config.json`                                                               | `parameter_count.csv`                                             |

Scripts `03`, `04`, `05`, `09` are the heaviest (`03` and `09` can take many hours
for a full run). Scripts `00`, `08` and `10` do not require a GPU.

### Resume behavior

Scripts `01`–`05` and `09` resume from their existing outputs. Each writes a
`.resume_signature*.json` holding hashes of its config (except `output_dir` and
`checkpoint_steps`), its script and `analysis/pan_core_analysis/*.py`; resuming with a
different config or code, or onto outputs without a signature, raises instead of
mixing results.

| Script           | Resume support                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `01`             | Skips steps already in `accuracy_{model}.csv`                                                          |
| `02`             | Skips steps whose decode CSV exists (reused for the summary)                                           |
| `03`             | Skips computed steps; pending steps must come after the last done step                                 |
| `04`             | Skips computed steps (all 4 CSVs must agree); restores the previous latent for dynamic metrics         |
| `05`             | Skips computed steps (all 12 CSVs must agree)                                                          |
| `09`             | Skips checkpoint steps already present in the output JSON                                              |
| `06`, `07`, `08` | No resume (single pass)                                                                                |

---

## Visualize scripts

| Script                      | Reads from   | Outputs                                                                                                     |
| --------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| `11_accuracy.py`            | `01` output  | Per-bucket accuracy, Fig 2 loss/accuracy, chi-SMILES accuracy, per-token accuracy                           |
| `12_jumpup_metrics.py`      | `03` output  | Perplexity + CDR; heatmaps and top-N line plots of attention mass, entropy, graph distance; SAR/NIT heatmaps |
| `13_residual_stream.py`     | `05` + `03`  | Per-layer L2 norm and checkpoint-to-checkpoint cosine line plots; adjacent-layer heatmaps                    |
| `14_latent_trajectory.py`   | `04` + `03`  | Latent L2 norm, cosine, CKA; intrinsic dim, isotropy entropy, active units                                   |
| `15_weight_matrices.py`     | `08` output  | Weight matrix norm, Frobenius diff, CKA, cosine trajectories                                                 |
| `16_mutual_information.py`  | `09` + `03`  | Probe trajectories, layer heatmaps, nonlinear comparison, @/@@ comparison, Fig 7d, `probe_summary_{model}.json` |
| `17_sampling_schedule.py`   | —            | Delayed-sigmoid bucket sampling probabilities (Fig S1)                                                       |
| `18_seqlen_vs_mw.py`        | summary CSVs | Sequence length vs molecular weight hexbins (Fig S2)                                                         |

Visualizations `12`–`16` draw the jump-up interval detected on `03`'s perplexity
(`jump_threshold_ratio` 0.15, `jump_ignore_initial_n` 10) as a light red band with a
dashed peak line; `13`, `14` and `16` read it from `jump_logits_dir`.
All visualize scripts write both `.png` (300 dpi) and `.svg` per figure.

---

## Config reference

For an end-to-end run, copy `analysis/configs/pipeline.yml` and fill in the
shared paths, models, and optional per-analysis overrides. Leave each model's
`chiral_heads` and `final_chiral_heads` as `null` until the first phase produces
visualization `12`, then fill in only those selections and rerun.

Templates under `analysis/configs/individual/` are for running or debugging one
numbered script directly. The directory contains one template for every script:
compute `00`–`10` and visualize `11`–`18`. Scripts `00`, `10`, `17` and `18` are
not part of the pipeline and are run only this way.

Individual compute configs generally use the following common fields:

```yaml
model_name:      your_model      # arbitrary label used in output filenames
checkpoint_dir:  /path/to/ckpts  # directory containing checkpoint-N subdirs
tokenizer_path:  /path/to/normal_tokens.txt
eval_data:       /path/to/eval.csv
output_dir:      /path/to/results
src_col:         random          # encoder input column in eval_data
tgt_col:         canonical       # decoder target column
seed:            42
n_phases:        5               # equal length buckets of eval_data (01, 02, 06, 07)
phase / max_phase: 1 / 5         # restrict to one bucket (03, 04, 05)
```

Script `06` requires explicit before/after checkpoint paths. Script `07`
requires before-jump, jump-peak, after-jump, and final checkpoint paths and the two
head selections `chiral_heads` / `final_chiral_heads`. The end-to-end runner derives
the checkpoints from analysis `03` (and `final_checkpoint`) automatically.

`model_name` is always an arbitrary output label. Analysis `08` derives its
weight-key subset from each checkpoint's `config.json` and validates those keys
against the state dict. Analysis `07` likewise derives layer/head dimensions
from checkpoint configs. The pipeline pauses for explicit `chiral_heads` /
`final_chiral_heads` selection because head importance is an experimental result rather than an
architecture property.
