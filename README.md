# PanCORE — Pan-Chemical Omniscale Representation Engine

A Transformer-based seq2seq framework that maps chemical structures (SMILES) into a continuous, meaningful latent space.
The name **PanCORE** stands for **Pan-Chemical Omniscale Representation Engine** — reflecting its goal of covering a broad chemical space across multiple molecular scales.

- **Latent representation** — encodes SMILES strings into fixed-length latent vectors for downstream tasks (property prediction, similarity search, conditional generation).
- **Generation** — a decoder for SMILES reconstruction/canonicalization and for generating molecules from the latent space.
- **Broad chemical space** — built to be trained on large datasets of low-to-medium-sized molecules (ZINC, PubChem).
- **Curriculum training** — `normal`, `phased` and length-bucketed `dynamic` strategies, with multi-GPU (DDP) training.

## Status

- **The paper describing PanCORE is in preparation.**
- **Pretrained weights** will be released on Hugging Face soon.
- [`analysis/`](analysis/) contains the code that reproduces the published paper
  *"From Syntax to Semantics: Unveiling the Emergence of Chirality in SMILES Translation Models"*
  ([arXiv:2605.09949](https://doi.org/10.48550/arXiv.2605.09949)). See [analysis/README.md](analysis/README.md).

## Installation

Python 3.10 or newer. Pick the PyTorch build that matches your machine.

With [uv](https://docs.astral.sh/uv/) (the extras select the PyTorch package index for you):

```bash
git clone https://github.com/mizuno-group/pan-core.git
cd pan-core
uv venv && source .venv/bin/activate

uv pip install -e ".[cu130]"   # NVIDIA GPU, CUDA 13.0 build of PyTorch
uv pip install -e ".[cu128]"   # NVIDIA GPU with a driver that cannot run CUDA 13 (older PyTorch)
uv pip install -e ".[cpu]"     # CPU only
```

With pip, install PyTorch from its index first, then pan-core:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or .../cu128, .../cpu
pip install -e .
```

Optional extras: `[analysis]` for the paper-reproduction scripts in `analysis/`, `[dev]` for the
tests and linter (combine them, e.g. `".[cu130,analysis]"`).

## Quick start

**1. Prepare data.** Turn a SMILES column into random → canonical pairs, then tokenize them into
length buckets. This uses the 95 molecules bundled in `sample/data/`:

```python
import os
import pandas as pd
from pan_core.common.data_handler import raw2randomORcanonical, split_and_tokenize

os.makedirs("data", exist_ok=True)
smiles = pd.read_csv("sample/data/sample_evaldata.csv")["smiles"]
raw2randomORcanonical(allsmiles=smiles, randomize=True, canonicalize=True,
                      output_path="data/pairs.parquet", max_workers=4)
split_and_tokenize(csv_path="data/pairs.parquet", token_path="sample/data/normal_tokens.txt",
                   thresholds=[0, 100, 200, 300, 400, 500], max_length=502,
                   randomize=True, canonicalize=True, do_split=True,
                   smiles_format="parquet", max_workers=4)
```

This writes one memmap per length bucket (`data/pairs_0to100.npy`, `data/pairs_100to200.npy`, …)
and logs the number of molecules in each. For large datasets, augmentation and the file
formats, see [docs/data_preprocessing.md](docs/data_preprocessing.md).

**2. Configure.** Copy [`sample/config.yml`](sample/config.yml) (the paper settings) and fill in
`training.data`: one `train_data` entry per bucket with its `path`, `num_data` (the count logged in
step 1) and `dim_data` (the bucket's upper threshold + 2, e.g. 102 for `0to100`), plus `valid_data`.
A bucket that received no molecules gets no `.npy`; leave it out and make
`training.strategy.dynamic.phases` and the sampling-probability lists match the buckets you keep.
See [docs/configuration.md](docs/configuration.md).

**3. Train.**

```bash
pan-core --config my_config.yml --output_dir runs/exp01 --do_train

# several GPUs on one node
torchrun --nproc_per_node=4 -m pan_core.main --config my_config.yml --output_dir runs/exp01 --do_train
```

Training resumes from the latest checkpoint in `--output_dir`, and the final model is written to
`runs/exp01/final_model/`.

**4. Generate and encode.**

```bash
# greedy decoding over training.data.eval_data -> runs/exp01/gen_results.csv
pan-core --config my_config.yml --output_dir runs/exp01 --do_eval_gen

# latent vectors only -> runs/exp01/encode_latents.npy
pan-core --config my_config.yml --output_dir runs/exp01 --do_eval_gen --encode_only
```

The other modes (accuracy over checkpoints, hidden-state extraction) and the Python API are in
[docs/usage.md](docs/usage.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/data_preprocessing.md](docs/data_preprocessing.md) | Input formats, random/canonical pairs, tokenization into length buckets, augmentation, memory notes |
| [docs/configuration.md](docs/configuration.md) | The YAML file: sections, training strategies, model parameters, VAE mode |
| [docs/usage.md](docs/usage.md) | Every CLI mode with its arguments and outputs, use as a library, repository layout |
| [analysis/README.md](analysis/README.md) | Reproducing the figures and tables of arXiv:2605.09949 |

## Tests

```bash
pip install -e ".[cpu,dev,analysis]"
pytest tests
```

The tests run on CPU and need no checkpoints or external data.

## Citation

The paper describing PanCORE is in preparation. Until it is published, please cite the software
(GitHub's "Cite this repository" button gives the same entry, from [CITATION.cff](CITATION.cff)):

```bibtex
@software{li_pancore,
  title   = {PanCORE: Pan-Chemical Omniscale Representation Engine},
  author  = {Li, Zehao and Mizuno, Tadahaya},
  url     = {https://github.com/mizuno-group/pan-core},
  version = {0.1.0},
  year    = {2026}
}
```

If you use the analysis code in `analysis/`, please cite:

```bibtex
@article{li2026syntax,
  title   = {From Syntax to Semantics: Unveiling the Emergence of Chirality in SMILES Translation Models},
  author  = {Li, Zehao and Yoshikai, Yasuhiro and Nemoto, Shumpei and Kusuhara, Hiroyuki and Mizuno, Tadahaya},
  journal = {arXiv preprint arXiv:2605.09949},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.09949}
}
```

The PanCORE entry will be replaced by its paper once published.

## License

[MIT](LICENSE)

## Authors

- [Zehao Li](https://github.com/Lzh-Function) — main contributor
- [Tadahaya Mizuno](https://github.com/tadahayamiz) — correspondence

Built on ideas from [clmpy](https://github.com/mizuno-group/clmpy) (Shumpei Nemoto, Mizuno group).

## Contact

Open a GitHub issue, or contact:

- takuho2002[at]outlook.jp
- tadahaya[at]gmail.com (lead contact)
