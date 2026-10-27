# Data preprocessing

How to turn a SMILES file into the training data PanCORE reads. Back to the [README](../README.md).

PanCORE expects SMILES data in one of two formats at training time:

| Mode         | Format              | Description                                                                                  |
| ------------ | ------------------- | -------------------------------------------------------------------------------------------- |
| `on_the_fly` | CSV or Parquet      | Raw SMILES tokenized at access time; data held as Arrow buffers (CoW-safe across DataLoader workers) |
| `memmap`     | `.npy` memmap       | Pre-tokenized int16 arrays, fastest for large datasets                                       |

> **`on_the_fly` memory note**: SMILES strings are stored as Polars Series (Apache Arrow buffers) rather than Python lists. Arrow buffers are read-only and shared across all fork-based DataLoader workers without Copy-on-Write page duplication — keeping worker memory overhead near zero regardless of dataset size.
>
> If `preprocessing_fn` is provided to `OnTheFlyDataset`, it is applied **per item at access time** (not at load time). For expensive transforms such as RDKit canonicalization, prefer pre-processing the file once with `raw2randomORcanonical` and passing the result directly.

## Step 1 — Prepare a SMILES file

The source file must contain at least one SMILES column.  
Accepted formats: `.csv`, `.csv.gz`, and `.parquet`.  
For seq2seq training (random → canonical), two columns are expected:

```
random,canonical
C1=CC=CC=C1,c1ccccc1
...
```

The column names are configurable via `csv_column_name` in `config.yml`.

[`sample/data/sample_evaldata.csv`](../sample/data/sample_evaldata.csv) (95 molecules, SMILES in the
`smiles` column) can be used to try the steps below, with the vocabulary
[`sample/data/normal_tokens.txt`](../sample/data/normal_tokens.txt).

## Step 2 — Generate Random/Canonical SMILES Pairs

Use `raw2randomORcanonical` from `data_handler.py`.  
The output format is determined by `output_path`: use `.parquet` for Parquet, `.csv.gz` for gzip-compressed CSV, or `.csv` for plain CSV.

```python
from pan_core.common.data_handler import raw2randomORcanonical
import pandas as pd

smiles = pd.read_csv("molecules.csv")["smiles"]

# Parquet output (recommended for large datasets)
raw2randomORcanonical(
    allsmiles=smiles,
    randomize=True,
    canonicalize=True,
    output_path="data/processed.parquet",
    max_workers=8
)

# Gzip CSV output
raw2randomORcanonical(
    allsmiles=smiles,
    randomize=True,
    canonicalize=True,
    output_path="data/processed.csv.gz",
    max_workers=8
)
```

For data augmentation (multiple random SMILES per molecule), the output format is inferred from the input extension (`.parquet` → `_augmented.parquet`, `.csv.gz` → `_augmented.csv.gz`, etc.):

```python
from pan_core.common.data_handler import RandomAugmentation

RandomAugmentation(
    csvpath="data/processed.parquet",
    num_random=10,
    max_workers=8
)
```

## Step 3 — Tokenize and Convert to Memmap (recommended for large datasets)

Accepts `.csv`, `.csv.gz`, or `.parquet` as input.

```python
from pan_core.common.data_handler import split_and_tokenize

split_and_tokenize(
    csv_path="data/processed.parquet",
    token_path="vocab/normal_tokens.txt",
    thresholds=[0, 64, 128, 256, 512],  # pure SMILES token counts (BOS/EOS excluded)
    max_length=512,                      # used only when do_split=False
    randomize=True,
    canonicalize=True,
    do_split=True,          # split into length-bucketed files (for dynamic curriculum)
    smiles_format="parquet", # sidecar SMILES output format: "csv" (default) or "parquet"
    analyze_stride=100,      # record 1-in-N rows for distribution analysis (default 100)
    max_workers=8
)
```

**`thresholds`**: Values represent **pure SMILES token counts excluding BOS/EOS**.  
For example, `thresholds=[0, 64, 128, 256]` creates three buckets; each bucket's NPY stores sequences padded to `threshold_upper + 2` tokens (adding BOS and EOS).

**Output** (`do_split=True`): One `.npy` memmap file per bucket, each with shape `(N, 2, threshold_upper+2)` where `[:, 0, :]` is source and `[:, 1, :]` is target, stored as `int16`.  
Shorter buckets use smaller arrays — the shape varies per bucket.

**Output** (`do_split=False`): A single `.npy` with shape `(N, 2, max_length)`.

A SMILES sidecar file (`.csv.gz` or `.parquet` depending on `smiles_format`) is also written per bucket when `do_split=True`.

**Memory**: Input is read in 1M-row chunks and output is written as it goes, so RAM
is bounded by `max_workers`, not by dataset size. The pool is held to
`chunksize * (max_workers + 2)` rows of in-flight results; at the built-in
`chunksize=10000` and ~1.4 kB per buffered row that is roughly
`14 MB * (max_workers + 2)`.

> Prefer **Parquet** over `.csv.gz` for large inputs. Chunked reads seek within a
> Parquet file, but gzip is not seekable, so each chunk re-inflates from the start
> and the read cost grows with the offset. Plain `.csv` is fine; only the gzipped
> form is affected.

<details><summary>Measurements behind the two notes above</summary>

Benchmark host: AMD Ryzen Threadripper 9960X (24 cores / 48 threads, 5.49 GHz max),
125 GB RAM, ext4 on local SATA (`/dev/sda`); Python 3.12.3, polars 1.42.1,
pyarrow 25.0.0, RDKit 2026.03.3. Input was ZINC20 `random`/`canonical` pairs,
all in the 0–100 bucket.

| What | Result |
| --- | --- |
| `split_and_tokenize`, 8M molecules, `max_workers=24` | 54 kmol/s, parent RSS flat over the run |
| Same, with the result window removed | 52 kmol/s, parent grew 3.7 GB; output byte-identical |
| `max_workers` 4 / 12 / 24 | 22.0 / 54.6 / 55.2 kmol/s — past 12 workers this host gains throughput but not much |
| 1M-row chunk read, Parquet (100.5M-row file) | 0.51–0.57 s at offsets 0 through 20M |
| 1M-row chunk read, `.csv` (8M-row file) | 0.06–0.10 s at offsets 0 through 5M |
| 1M-row chunk read, `.csv.gz` (8M-row file) | 0.13 s at offset 0, 0.23 s at 1M, 0.71 s at 5M |

The parent is single-threaded, so the worker count at which throughput stops
improving is host-specific: a machine with slower cores but many more of them
saturates the parent sooner. Pick `max_workers` by measuring, not by core count.

</details>

## Step 4 — Per-bucket Augmentation (optional)

After `split_and_tokenize`, use `augment_bins` to generate augmented NPY files for selected buckets.  
The original NPY and sidecar files are not modified.

```python
from pan_core.common.data_handler import augment_bins

augment_bins(
    base_dir="data/",
    stem="processed",           # stem used by split_and_tokenize
    token_path="vocab/normal_tokens.txt",
    thresholds=[0, 64, 128, 256, 512],
    bin_num_random={0: 10, 2: 5},   # bin index → random SMILES per molecule
    sidecar_format="parquet",        # must match smiles_format used in split_and_tokenize
    max_workers=8,
)
```

| Argument          | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `base_dir`        | Directory where `split_and_tokenize` wrote its outputs                                   |
| `stem`            | File stem (same as derived from `csv_path` in `split_and_tokenize`)                      |
| `bin_num_random`  | Dict mapping bucket index → number of random SMILES to generate per molecule             |
| `sidecar_format`  | `"parquet"` or `"csv"` — must match `smiles_format` used in `split_and_tokenize`         |

**Output** per *destination* bucket `j`:

| File | Shape / Format | Description |
| ---- | -------------- | ----------- |
| `{stem}_{lo}to{hi}_aug.npy` | `(M, 2, hi+2)` int16 | Augmented memmap; `[:, 0, :]` random, `[:, 1, :]` canonical |
| `{stem}_{lo}to{hi}_aug.parquet` | two columns | `random` + `canonical` sidecar for the augmented pairs |

`augment_bins` returns `{destination bucket index: rows written}` — the `num_data`
to put in `config.yml` for each file.

**Each pair is filed by its own length, not by the bucket it was generated from.**
A randomized SMILES can come out longer than the canonical one it was made from, so
a molecule taken from the 200–300 bucket can produce a variant that only fits the
300–400 one. Writing it under the source bucket would truncate it and cut off the
EOS token, so the destination is chosen per pair by the same rule
`split_and_tokenize` uses. One call therefore writes to every bucket its output
reaches, and two source buckets can contribute to the same destination file.

Pairs longer than `thresholds[-1]` have nowhere to go; they are counted and reported
at the end rather than dropped quietly. Widen `thresholds` to keep them.

`M` may be less than `N × n`: `convertingSMILES` de-duplicates the random variants of
a molecule, and molecules RDKit cannot round-trip are skipped.

Output order follows the input, so the variants of one molecule sit together. The
curriculum samplers shuffle indices every epoch, so this does not correlate batches;
a consumer reading the file sequentially should shuffle it itself.

> **`RandomAugmentation` vs `augment_bins`**: both call the same generator, so the
> augmented pairs are equivalent. `RandomAugmentation` (Step 2) writes SMILES and
> needs `split_and_tokenize` afterwards, and it reads its whole output back to shuffle
> it — about 270 bytes per output row, so a billion augmented rows needs roughly
> 250 GB. `augment_bins` streams throughout and its memory does not grow with the
> dataset. Prefer `augment_bins` for anything large.

## Expected Input Summary

| Field           | Type              | Description                                                                        |
| --------------- | ----------------- | ---------------------------------------------------------------------------------- |
| Source SMILES   | `str`             | Random (non-canonical) SMILES string                                               |
| Target SMILES   | `str`             | Canonical SMILES string (RDKit output)                                             |
| SMILES file     | `.csv` / `.csv.gz` / `.parquet` | Two-column file with `random` and `canonical` columns          |
| Vocabulary file | `.txt`            | One token per line; default vocab size 185                                         |
| Memmap file     | `.npy`            | `int16`, shape `(N, 2, L)` — src at `[:, 0, :]`, tgt at `[:, 1, :]`; `L = threshold_upper + 2` per bucket when `do_split=True` |
| Latent array    | `.npy`            | `float32`, shape `(N, E)` — for `--latent_decode` mode                             |

## Logging

The preprocessing functions print their progress by default, whether they are run
through the CLI or imported into a script. Importing `pan_core` does not configure
logging on its own: a handler is attached to the `pan_core` logger the first time a
preprocessing function runs, and only when nothing else has configured logging.

To control the output yourself, configure logging before calling them — that
setting is then respected and no handler is added:

```python
import logging
logging.basicConfig(level=logging.WARNING)   # quiet
logging.basicConfig(level=logging.INFO)      # the default level
```
