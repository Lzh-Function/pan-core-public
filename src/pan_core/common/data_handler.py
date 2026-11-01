"""
Created by Zehao Li (Takuho Ri)
Created on 2025-02-27 (Thu)  14:30:27 (+09:00)

Datasets, curriculum samplers, the batch collator and the SMILES preprocessing
pipeline of pan-core.
Adapted from clmpy (https://github.com/mizuno-group/clmpy) by Shumpei Nemoto.
"""


import csv
import functools
import gc
import gzip
import logging
import math
import multiprocessing
import os
import random
import threading
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles, rdMolDescriptors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset, Sampler

from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.common.utils import ensure_default_logging

RDLogger.DisableLog('rdApp.*')

# Module logger. Preprocessing progress is emitted at INFO; callers using this
# module as a library enable it with logging.basicConfig(level=logging.INFO).
logger = logging.getLogger(__name__)

# Random-SMILES generation: how many draws a molecule gets before it is dropped.
_MAX_RANDOM_TRIES = 100
_RANDOM_TRIES_PER_VARIANT = 10

# Per-molecule failures are logged in full up to this many times, then demoted to
# DEBUG. Preprocessing runs reach hundreds of millions of molecules, where even a
# 1% failure rate is millions of identical lines. The count is per process, so a
# pool of N workers can emit up to N times this many before the cap bites.
_WARN_LIMIT = 100
_warn_counts: dict[str, int] = {}


def _warn_limited(kind: str, message: str) -> None:
    """Logs a per-molecule failure, capping how many reach WARNING.

    Args:
        kind (str): Groups messages that share a cap, e.g. ``"rdkit_parse"``.
        message (str): The line to log.
    """
    count = _warn_counts.get(kind, 0) + 1
    _warn_counts[kind] = count
    if count < _WARN_LIMIT:
        logger.warning(message)
    elif count == _WARN_LIMIT:
        logger.warning(message)
        logger.warning(
            f"{_WARN_LIMIT} '{kind}' failures reported by this process; further ones "
            "are logged at DEBUG. Enable DEBUG logging to see them all."
        )
    else:
        logger.debug(message)


# =========================================================
# Dataset Classes
# =========================================================

class MemmapDataset(Dataset):
    """
    Dataset for loading pre-tokenized data from .npy memmap files.
    Format: [num_data, 2, dim_data] where [:, 0, :] is src and [:, 1, :] is tgt.
    """
    def __init__(self, path: str, num_data: int, dim_data: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memmap file not found: {path}")

        self.data = np.memmap(path, dtype=np.int16, mode="r", shape=(num_data, 2, dim_data))
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # return metadata for curriculum training sampler
        if idx < 0:
            return {"__metadata__": True, "accumulate_step": abs(idx)}
        # Return source and target sequences
        return self.data[idx, 0], self.data[idx, 1]


def _load_dataframe(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Loads a CSV (.csv, .csv.gz) or Parquet file into a Polars DataFrame."""
    path = str(path)
    if path.endswith(".parquet"):
        return pl.read_parquet(path, columns=columns)
    return pl.read_csv(path, columns=columns, infer_schema_length=0)


class OnTheFlyDataset(Dataset):
    """
    Dataset that loads raw SMILES from CSV and tokenizes on-the-fly.
    Supports 'src' only or 'src' + 'tgt'.
    For CPT (Pretraining), usually src == tgt (autoregressive).
    """
    def __init__(self,
                 file_path_or_df: str | Path | pd.DataFrame | pl.DataFrame,
                 tokenizer: SMILESTokenizer,
                 src_col_name: str = "smiles",
                 tgt_col_name: str | None = None,
                 preprocessing_fn = None,
                 mode: str = "train",
                 maxlen: int|None = None,
                 phase: int|None = None,
                 max_phase: int|None = None):
        """
        Args:
            file_path_or_df (str | Path | pd.DataFrame | pl.DataFrame): Path to a CSV or Parquet file, or a pre-loaded DataFrame.
            tokenizer (SMILESTokenizer): Tokenizer instance with an `.encode()` method.
            src_col_name (str, optional): Column name for source SMILES. Defaults to "smiles".
            tgt_col_name (str, optional): Column name for target SMILES. If None, source is
                copied as target (for CPT/autoregressive pretraining). Defaults to None.
            preprocessing_fn (callable, optional): Optional function applied to SMILES strings
                before tokenization (e.g., canonicalization). Defaults to None.
            mode (str, optional): Dataset mode — "train"/"eval" returns (src, tgt);
                "inference" returns src only. Defaults to "train".
            maxlen (int, optional): If provided, all sequences are padded/truncated to this length.
                Defaults to None (dynamic length, padded by the collator).
            phase (int, optional): 1-based phase index for slicing the dataset into equal parts.
                Requires ``max_phase``. Defaults to None.
            max_phase (int, optional): Total number of phases for dataset slicing. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.maxlen = maxlen
        self.padding = "max_length" if maxlen is not None else False
        self._preprocessing_fn = preprocessing_fn

        # Load Data
        if isinstance(file_path_or_df, (str, Path)):
            if not os.path.exists(file_path_or_df):
                raise FileNotFoundError(f"Data file not found: {file_path_or_df}")
            df = _load_dataframe(file_path_or_df)
        elif isinstance(file_path_or_df, pl.DataFrame):
            df = file_path_or_df
        elif isinstance(file_path_or_df, pd.DataFrame):
            df = pl.from_pandas(file_path_or_df)
        else:
            raise TypeError("file_path_or_df must be a file path (str or Path), a pandas DataFrame, or a polars DataFrame")
        if src_col_name not in df.columns:
            raise ValueError(f"Column '{src_col_name}' not found in input data")

        src_series = df[src_col_name].cast(pl.Utf8)
        if phase is not None and max_phase is not None:
            n = len(src_series)
            num_each_phase = n // max_phase
            start_idx = num_each_phase * (phase - 1)
            end_idx = num_each_phase * phase if phase < max_phase else n
            src_series = src_series[start_idx:end_idx]
        # Keep as Polars Series (Arrow-backed) to avoid CoW page-dirtying in fork-based
        # DataLoader workers. Python list refcount updates dirty shared pages; Arrow buffers
        # are read-only and remain shared across all worker processes.
        self.src_smiles: pl.Series = src_series

        self.has_target = False
        if tgt_col_name is not None and tgt_col_name in df.columns:
            tgt_series = df[tgt_col_name].cast(pl.Utf8)
            if phase is not None and max_phase is not None:
                tgt_series = tgt_series[start_idx:end_idx]
            self.tgt_smiles: pl.Series = tgt_series
            self.has_target = True
        else:
            self.tgt_smiles = self.src_smiles  # For CPT, src is used as target

        del df
        gc.collect()

    def __len__(self):
        return len(self.src_smiles)

    def _encode(self, text):
        """Token IDs for one SMILES string, honouring this dataset's padding setting.

        Without padding -- the training configuration, since ``padding`` is only set
        when ``maxlen`` is given -- the tokenizer's direct path is used; it produces
        the same IDs as ``encode`` and skips its per-call strategy resolution.
        """
        if self.padding is False:
            return self.tokenizer.encode_simple(text)
        return self.tokenizer.encode(text, add_special_tokens=True, padding=self.padding, max_length=self.maxlen)

    def __getitem__(self, idx):
        # return metadata for curriculum training sampler
        if idx < 0:
            return {"__metadata__": True, "accumulate_step": abs(idx)}

        src_text = self.src_smiles[idx]
        if self._preprocessing_fn is not None:
            src_text = self._preprocessing_fn(src_text)

        # Tokenize Source
        src_ids = self._encode(src_text)
        if isinstance(src_ids, list):
            src_ids = torch.tensor(src_ids, dtype=torch.long)

        if self.mode == "inference":
            return src_ids

        # Tokenize Target
        if self.has_target:
            tgt_text = self.tgt_smiles[idx]
            if self._preprocessing_fn is not None:
                tgt_text = self._preprocessing_fn(tgt_text)
            tgt_ids = self._encode(tgt_text)
            if isinstance(tgt_ids, list):
                tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
        else:
            # Copy src as tgt for CPT
            tgt_ids = src_ids.clone()

        return src_ids, tgt_ids


class LatentDataset(Dataset):
    """
    Dataset for loading latent vectors (numpy arrays).
    Used if input is already encoded into latent space.
    """
    def __init__(self, latents: Union[str, np.ndarray]):
        if isinstance(latents, str):
            self.latents = np.load(latents)
        else:
            self.latents = latents

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Returns float tensor [Dimension]
        return torch.from_numpy(self.latents[idx]).float()


class ValidIndexWrapper(Dataset):
    """
    Dataset wrapper that prepends the index to the output.
    Used only during evaluation to track sample IDs across DDP ranks.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < 0:
            return self.dataset[idx]

        output = self.dataset[idx]

        if isinstance(output, tuple) and len(output) == 2:
            return (idx, *output)

        else:
            raise NotImplementedError("EvalIndexWrapper only supports datasets returning tuples (src, tgt).")

# =========================================================
# Data Collator
# =========================================================

@dataclass
class GeneralDataCollator:
    """
    Unified DataCollator for Memmap (int16 numpy) and OnTheFly (long tensor).
    Handles padding and type conversion.
    """
    pad_token_id: int = 0

    def __call__(self, batch):
        # separate normal data and Metadata
        accumulate_step = 1 # Default

        if len(batch) > 0:
            last_item = batch[-1]
            if isinstance(last_item, dict) and last_item.get("__metadata__", False):
                accumulate_step = last_item["accumulate_step"]
                normal_data = batch[:-1]
            else:
                normal_data = batch
        else:
            raise ValueError("Batch is empty.")

        if not normal_data:
             raise ValueError("Batch contains no data.")

        # Detect input type from the first element
        elem = normal_data[0]

        # Case 1: Latent Vector (Single Tensor)
        if isinstance(elem, torch.Tensor) and elem.dtype == torch.float:
            return {
                "latents": torch.stack(normal_data, dim=0),  # [Batch, Dim]
            }

        # Case 2: Sequence Pair (Src, Tgt)
        # elem is tuple (src, tgt)
        if isinstance(elem, tuple):
            indices = None
            if len(elem) == 2:
                src_batch = [item[0] for item in normal_data]
                tgt_batch = [item[1] for item in normal_data]
            elif len(elem) == 3:
                # (idx, src, tgt)
                indices = torch.tensor([item[0] for item in normal_data], dtype=torch.long)
                src_batch = [item[1] for item in normal_data]
                tgt_batch = [item[2] for item in normal_data]
            else:
                raise ValueError(
                    f"A sequence-pair sample must be (src, tgt) or (idx, src, tgt); got a {len(elem)}-tuple."
                )

            # Pad sequences [Batch, Length] -> batch_first=True
            input_ids = self._stack_or_pad(src_batch, self.pad_token_id)
            labels = self._stack_or_pad(tgt_batch, self.pad_token_id)

            # create attention mask
            encoder_attention_mask = (input_ids != self.pad_token_id)
            decoder_attention_mask = (labels != self.pad_token_id)

            # Trim trailing padding to batch-internal max length (src and tgt independently)
            _src_nz = encoder_attention_mask.any(dim=0).nonzero()
            _tgt_nz = decoder_attention_mask.any(dim=0).nonzero()
            src_max_len = (_src_nz[-1].item() + 1) if _src_nz.numel() > 0 else input_ids.size(1)
            tgt_max_len = (_tgt_nz[-1].item() + 1) if _tgt_nz.numel() > 0 else labels.size(1)
            input_ids = input_ids[:, :src_max_len]
            encoder_attention_mask = encoder_attention_mask[:, :src_max_len]
            labels = labels[:, :tgt_max_len]
            decoder_attention_mask = decoder_attention_mask[:, :tgt_max_len]

            batch = {
                "input_ids": input_ids, # [Batch, SeqLen]
                "encoder_attention_mask": encoder_attention_mask, # [Batch, SeqLen]
                "labels": labels, # [Batch, SeqLen]
                "decoder_attention_mask": decoder_attention_mask, # [Batch, SeqLen]
                "accumulate_step": accumulate_step,
            }
            if indices is None:
                return batch
            return {"dataset_indices": indices, **batch}

        # Case 3: Single Sequence (Inference)
        if isinstance(elem, torch.Tensor) and elem.dtype == torch.long:
            input_ids = self._stack_or_pad(list(normal_data), self.pad_token_id)
            encoder_attention_mask = (input_ids != self.pad_token_id)
            return {
                 "input_ids": input_ids, # [Batch, SeqLen]
                 "encoder_attention_mask": encoder_attention_mask, # [Batch, SeqLen]
            }

        raise ValueError(f"Unsupported batch format: {type(elem)}")

    @staticmethod
    def _stack_or_pad(items, pad_token_id: int) -> torch.Tensor:
        """Collates one column of samples into a [Batch, Length] LongTensor.

        MemmapDataset stores fixed-width rows, so in training every sample in the
        column already has the same length and the padding step has nothing to do.
        One np.stack plus one torch.from_numpy then replaces a per-sample tensor
        conversion -- 2 * batch_size of them per batch, 8,192 at batch 4,096 --
        followed by pad_sequence. Ragged input, or anything that is not a 1-D
        numpy array, takes the original path. Both produce the same tensor.

        The dtypes have to match as well. np.stack promotes a mixture, and a
        batch holding both int64 and uint64 lands in float64, which drops integers
        above 2**53 -- 9007199254740993 comes back as ...992. The slow path
        converts each item on its own and keeps them exact. Every memmap this
        reads is int16, so a mixed batch does not arise today; the check is what
        keeps that from mattering.
        """
        first = items[0]
        if isinstance(first, np.ndarray) and first.ndim == 1:
            width = first.shape[0]
            if all(isinstance(x, np.ndarray) and x.shape == (width,)
                   and x.dtype == first.dtype for x in items):
                return torch.from_numpy(np.stack(items)).long()
        return pad_sequence([GeneralDataCollator._to_tensor(x) for x in items],
                            batch_first=True, padding_value=pad_token_id)

    @staticmethod
    def _to_tensor(x):
        """Helper to convert numpy/list to LongTensor"""
        if isinstance(x, torch.Tensor):
            return x.long()
        elif isinstance(x, np.ndarray):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                return torch.from_numpy(x).long()
        elif isinstance(x, list):
            return torch.tensor(x, dtype=torch.long)
        return x


# =========================================================
# Dataset Builders
# ========================================================

class MetadataAwareDatasetWrapper(Dataset):
    """
    Wraps any dataset (including ConcatDataset) to intercept negative indices
    and return them as metadata, preventing ConcatDataset from interpreting them.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < 0:
            return {"__metadata__": True, "accumulate_step": abs(idx)}

        return self.dataset[idx]


def build_train_dataset(args, tokenizer=None):
    """
    Constructs the training dataset based on configuration.
    Handles 'memmap' vs 'on_the_fly' and curriculum (multiple files).

    Raises:
        ValueError: If no training data is configured, or an entry has a blank path.
    """
    for i, cfg in enumerate(args.train_data_config_list):
        if not cfg or not cfg.get("path"):
            raise ValueError(
                f"training.data.train_data[{i}] has no 'path'. Point it at a "
                "training data file, or remove the entry."
            )

    datasets = []
    num_data_list = []

    # Mode: Memmap
    if args.dataset_mode == "memmap":
        if not args.train_data_config_list:
            raise ValueError("dataset_mode is 'memmap' but 'train_data_config_list' is empty.")

        for cfg in args.train_data_config_list:
            # cfg: {path: str, num_data: int, dim_data: int}
            ds = MemmapDataset(
                path=cfg["path"],
                num_data=cfg["num_data"],
                dim_data=cfg["dim_data"]
            )
            datasets.append(ds)
            num_data_list.append(cfg["num_data"])

    # Mode: On-The-Fly
    elif args.dataset_mode == "on_the_fly":
        if not args.train_data_config_list:
            raise ValueError("dataset_mode is 'on_the_fly' but 'train_data_config_list' is empty.")
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for 'on_the_fly' mode.")

        for cfg in args.train_data_config_list:
            ds = OnTheFlyDataset(
                file_path_or_df=cfg['path'],
                tokenizer=tokenizer,
                src_col_name=args.csv_column_name.get("source", "random"),
                tgt_col_name=args.csv_column_name.get("target", "canonical"),
                mode="train"
            )
            datasets.append(ds)
            num_data_list.append(len(ds))
    else:
        raise ValueError(f"Unknown dataset_mode: {args.dataset_mode}")

    # Return ConcatDataset (Expected by Custom Samplers)
    return MetadataAwareDatasetWrapper(ConcatDataset(datasets)), num_data_list


def build_valid_dataset(args, tokenizer=None) -> Dataset:
    """
    Constructs the validation dataset.
    """
    # Mode: Memmap
    if args.dataset_mode == "memmap":
        cfg = args.valid_data_config
        if not cfg or not cfg.get("path"):
            return None
        return ValidIndexWrapper(MemmapDataset(cfg["path"], cfg["num_data"], cfg["dim_data"]))

    # Mode: On-The-Fly
    elif args.dataset_mode == "on_the_fly":
        if not args.valid_data_config:
            return None
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for 'on_the_fly' mode.")

        return ValidIndexWrapper(OnTheFlyDataset(
            file_path_or_df=args.valid_data_config['path'],
            tokenizer=tokenizer,
            src_col_name=args.csv_column_name.get("source", "random"),
            tgt_col_name=args.csv_column_name.get("target", "canonical"),
            mode="eval"
        ))

    return None


def build_eval_dataset(args, tokenizer=None, maxlen: int|None = None, phase: int|None = None, max_phase: int|None = None) -> Dataset:
    """Constructs the evaluation dataset (no index wrapping, used for generation/analysis).

    Args:
        args (TrainArgs): Training arguments containing ``dataset_mode``, ``eval_data_config``,
            and ``csv_column_name``.
        tokenizer (SMILESTokenizer, optional): Required when ``dataset_mode`` is "on_the_fly".
            Defaults to None.
        maxlen (int, optional): Maximum sequence length for tokenization padding/truncation.
            Defaults to None.
        phase (int, optional): 1-based phase index for slicing the dataset. Defaults to None.
        max_phase (int, optional): Total number of phases for dataset slicing. Defaults to None.

    Returns:
        Dataset | None: The evaluation dataset, or None if no eval config is provided.
    """
    # Mode: Memmap
    if args.dataset_mode == "memmap":
        cfg = args.eval_data_config
        if not cfg or not cfg.get("path"):
            return None
        return MemmapDataset(cfg["path"], cfg["num_data"], cfg["dim_data"])

    # Mode: On-The-Fly
    elif args.dataset_mode == "on_the_fly":
        if not args.eval_data_config:
            return None
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for 'on_the_fly' mode.")

        return OnTheFlyDataset(
            file_path_or_df=args.eval_data_config["path"],
            tokenizer=tokenizer,
            src_col_name=args.csv_column_name.get("source", "random"),
            tgt_col_name=args.csv_column_name.get("target", "canonical"),
            maxlen=maxlen,
            phase=phase,
            max_phase=max_phase,
        )

    return None


def build_latent_dataset(latent: np.ndarray):
    """Wraps a pre-computed latent array in a Dataset for ``--latent_decode``.

    Args:
        latent (np.ndarray): Latent vectors, shape ``[N, E]``.

    Returns:
        LatentDataset: Dataset yielding one latent vector per item.
    """
    return LatentDataset(latent)


# =========================================================
# Custom Samplers
# =========================================================

_T_co = TypeVar("_T_co", covariant=True)
class PhasedDistributedSampler(Sampler[_T_co]):
    """
    Distributed Sampler for Phased Curriculum Pretraining.

    This sampler is designed for a continuous pre-training setup where the dataset
    grows over multiple phases. It supports replaying a proportion of data from
    previous phases while training on the current phase's data.

    Args:
        current_phase_idx (int) 0-indexed phase index, e.g., 0 for phase 1)
        epoch (int): current epoch in current phase, default to 0.
        num_epochs (int): total epochs in current phase, default to 1.
        num_data_list (List[int]) list of data sizes for each phase.
        replay_prop (float) proportion of data to replay from previous phases, default to 0.0.
        num_replicas (Optional[int]): Number of processes in distributed training.
        rank (Optional[int]): Rank of the current process.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        seed (int, optional): Random seed. Defaults to 42.
    """

    def __init__(
        self,
        current_phase_idx: int,
        batch_size: int,
        accumulate_steps: int,
        epoch: int = 0,
        num_data_list: Optional[list] = None,
        replay_ratio: float = 0.0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        """Initializes the PhasedDistributedSampler."""
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.current_phase_idx = current_phase_idx
        self.micro_batch_size = batch_size
        self.accumulate_steps = accumulate_steps
        self.macro_batch_size = self.micro_batch_size * self.accumulate_steps
        self.epoch = epoch
        self.num_data_list = [] if num_data_list is None else num_data_list
        self.replay_ratio = replay_ratio
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.batches_yielded = 0
        self.setup()

    def setup(self):
        """
        Sets up the sampler for the current phase.
        """
        self.num_main_data = self.num_data_list[self.current_phase_idx]

        if self.current_phase_idx > 0:
            self.num_replay_data = int(self.num_main_data * self.replay_ratio)  # change replay datanum
        else:
            self.num_replay_data = 0

        # Total samples to consume in this epoch
        self.total_samples_unpadded = self.num_main_data + self.num_replay_data

        # Calculate required padding
        # We need the total samples to be divisible by (num_replicas * macro_batch_size)
        # to ensure every rank gets full macro-batches and stays in sync.
        world_macro_size = self.macro_batch_size * self.num_replicas
        if self.drop_last and self.total_samples_unpadded % self.num_replicas != 0:
            # Round down
           self.num_macro_batches_global = self.total_samples_unpadded // world_macro_size
        else:
            # Round up
            self.num_macro_batches_global = math.ceil(self.total_samples_unpadded / world_macro_size)
        self.total_samples_padded = self.num_macro_batches_global * world_macro_size

        # Number of macro-batches per replica
        self.num_batches_per_replica = self.num_macro_batches_global

    def __iter__(self) -> Iterator[_T_co]:
        """Returns an iterator over the indices of the dataset."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.current_phase_idx + self.epoch)

        indices = []

        # --- 1. Generate Raw Indices ---
        # Current Phase Indices
        current_offset = sum(self.num_data_list[:self.current_phase_idx])
        num_current = self.num_data_list[self.current_phase_idx]

        if self.shuffle:
            current_indices = torch.randperm(num_current, generator=g) + current_offset
        else:
            current_indices = torch.arange(num_current) + current_offset

        indices.extend(current_indices.tolist())

        # Replay Indices (from previous phases)
        if self.num_replay_data > 0 and self.current_phase_idx > 0:
            prev_total_count = sum(self.num_data_list[:self.current_phase_idx])
            # Sample uniformly from all previous data
            if self.shuffle:
                replay_indices = torch.randint(0, prev_total_count, (self.num_replay_data,), generator=g)
            else:
                # If not shuffle, just take start (weird for replay but consistent)
                replay_indices = torch.arange(self.num_replay_data) % prev_total_count

            indices.extend(replay_indices.tolist())

        # --- 2. Global Shuffle ---
        if self.shuffle:
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            perm = torch.randperm(len(indices_tensor), generator=g)
            indices = indices_tensor[perm].tolist()

        # --- 3. Padding ---
        # Pad to match self.total_samples_padded
        padding_size = self.total_samples_padded - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                # Repeat if padding is larger than data
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        elif padding_size < 0:
            # Trim if drop_last caused size reduction
            indices = indices[:self.total_samples_padded]

        assert len(indices) == self.total_samples_padded

        # --- 4. Subsample for Rank ---
        # DDP standard: indices[rank::world_size]
        my_indices = indices[self.rank : self.total_samples_padded : self.num_replicas]

        assert len(my_indices) == self.num_batches_per_replica * self.macro_batch_size

        start_batch_idx = self.batches_yielded

        # --- 5. Yield Macro-Batches ---
        for i in range(self.num_batches_per_replica):
            # skip for resume
            if i < start_batch_idx:
                continue

            start = i * self.macro_batch_size
            end = start + self.macro_batch_size
            batch = my_indices[start:end]

            # Inject Metadata (Negative Index)
            # This allows Trainer to know exactly how to split this batch
            batch.append(-1 * self.accumulate_steps)

            yield batch

            self.batches_yielded = i + 1

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset for the current replica.
        """
        return self.num_batches_per_replica

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.
        """
        if epoch != self.epoch:
            self.batches_yielded = 0
        self.epoch = epoch

    def state_dict(self):
        """Returns the sampler position, so a resumed run continues the same order.

        Returns:
            dict: ``epoch`` and ``batches_yielded``.
        """
        return {
            "epoch": self.epoch,
            "batches_yielded": self.batches_yielded
        }

    def load_state_dict(self, state_dict):
        """Restores the sampler position saved by :meth:`state_dict`.

        Args:
            state_dict (dict): Output of :meth:`state_dict`.
        """
        self.epoch = state_dict["epoch"]
        self.batches_yielded = state_dict["batches_yielded"]


class DynamicBucketDistributedSampler(Sampler[_T_co]):
    """
    A Distributed Sampler for dynamic curriculum learning with bucketing.

    This sampler is designed for training on sequences of varying lengths. It groups
    sequences into buckets based on their length and then samples from these buckets
    with dynamically changing probabilities over epochs. This allows the model to
    first focus on shorter, simpler sequences and gradually move to longer, more
    complex ones.

    It also handles gradient accumulation by yielding consecutive mini-batches from
    the same bucket, ensuring that the model updates are based on data with similar
    characteristics.
    """
    def __init__(
        self,
        batch_sizes: list[int],
        accumulate_steps: list[int],
        num_replicas: int,
        rank: int,
        max_epochs: int,
        total_optimizer_steps: int,
        num_data:list[int],
        start_sampling_probabilities: list[float]|None,
        end_sampling_probabilities: list[float]|None,
        sampling_update_last_epoch: int,
        sampling_interpolation_type: str = "linear",
        # for "sigmoid" type
        sampling_interpolation_steepness: float = 12.0,
        sampling_interpolation_delay_factor: float = 0.8,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Initializes the DynamicBucketDistributedSampler.

        Args:
            batch_sizes (list[int]): A list of batch sizes for each bucket.
            accumulate_steps (list[int]): A list of gradient accumulation steps for each bucket.
            num_replicas (int): The number of processes participating in distributed training.
            rank (int): The rank of the current process.
            max_epochs (int): The total number of training epochs.
            total_optimizer_steps (int): If specified, fixes the total number of
                optimizer steps over all epochs. The steps per epoch will be this value
                divided by max_epochs.
            num_data (list[int]): Number of data of each buckets.
            start_sampling_probabilities (list[float], None): The start sampling probabilities for each bucket.
                if None, then sampling probabilities will be equal to real data distributions.
            end_sampling_probabilities (list[float], None): The final sampling probabilities for each bucket.
                if None, then sampling probabilities will be constant.
            sampling_update_last_epoch (int): The epoch at which the sampling probabilities should reach their final values.
            sampling_interpolation_type (str, optional): The type of interpolation for sampling
                probabilities. Can be "linear" or "sigmoid". Defaults to "linear"
            sampling_interpolation_steepness (float, optional): Steepness (k) for
                'sigmoid' interpolation. Defaults to 12.0.
            sampling_interpolation_delay_factor (float, optional): Delay factor for
                'sigmoid' interpolation (0-1). Defaults to 0.8.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            seed (int, optional): The random seed. Defaults to 0.
        """
        self.batch_sizes = batch_sizes
        self.accumulate_steps = accumulate_steps
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.optimizer_steps_yielded = 0
        self.shuffle = shuffle
        self.seed = seed
        if sampling_interpolation_type not in ["linear", "sigmoid"]:
            raise ValueError("sampling_interpolation_type must be one of 'linear' or 'sigmoid'.")
        self.sampling_interpolation_type = sampling_interpolation_type
        self.sampling_interpolation_steepness = sampling_interpolation_steepness
        self.sampling_interpolation_delay_factor = sampling_interpolation_delay_factor

        # Bucket b holds the ConcatDataset indices [offset_b, offset_b + size_b).
        # Kept as two integers rather than a materialised list(range(...)): at the
        # production scale (some 2e8 molecules) a Python list of ints costs ~36
        # bytes per index, about 8 GB before a single shuffle.
        logger.info("creating buckets...")
        self.bucket_offsets: list[int] = []
        self.bucket_sizes: list[int] = list(num_data)
        num_data_sum = 0
        for num in num_data:
            self.bucket_offsets.append(num_data_sum)
            num_data_sum += num

        # Pre-calculate the sampling probabilities and number of steps for each epoch.
        if start_sampling_probabilities is None:
            start_sampling_probabilities = [num / num_data_sum for num in num_data]
        if end_sampling_probabilities is None:
            end_sampling_probabilities = start_sampling_probabilities
        self._precalculate_probabilities_and_steps(
            start_sampling_probabilities, end_sampling_probabilities,
            sampling_update_last_epoch, max_epochs, total_optimizer_steps
        )

        # 251007-NEW: State variables to track usage across epochs
        num_buckets = len(self.bucket_sizes)
        self.persistent_indices_per_bucket_rank: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.int64) for _ in range(num_buckets)]
        self.current_position_in_bucket: list[int] = [0] * num_buckets
        self.reshuffle_counters: list[int] = [0] * num_buckets
        self._is_setup = False

        # The DataLoader asks for batches ahead of training (worker prefetch), so the
        # live counters run ahead of what has been trained. __iter__ records the
        # counters after every batch it hands out, keyed by optimizer_steps_yielded,
        # and state_dict(consumed=...) saves the entry for the batches actually trained.
        self._iter_start = 0
        self._snapshots: dict[int, tuple[list[int], list[int]]] = {}

    def _precalculate_probabilities_and_steps(self, start_probs, end_probs, update_last_epoch, max_epochs, total_optimizer_steps):
        """
        Pre-calculates the dynamic sampling probabilities and the number of optimizer
        steps for each epoch.
        """
        start_probs = torch.tensor(start_probs, dtype=torch.float32)
        end_probs = torch.tensor(end_probs, dtype=torch.float32)
        self.dynamic_sampling_probabilities = []
        self.optimizer_steps_per_epoch_list = []

        # 251008-NEW: Fixed Optimizer step
        optimizer_steps_per_epoch = total_optimizer_steps // max(1, max_epochs)
        num_buckets = len(start_probs)

        # 251008-NEW: delay-sigmoid calculate
        for epoch in range(max_epochs):
            if epoch >= update_last_epoch:
                current_probs = end_probs
            else:
                t = epoch / max(1, update_last_epoch - 1)

                if self.sampling_interpolation_type == 'linear':
                    current_probs = (1 - t) * start_probs + t * end_probs
                elif self.sampling_interpolation_type == 'sigmoid':
                    t_delayed = torch.zeros(num_buckets)
                    for i in range(num_buckets):
                        offset = self.sampling_interpolation_delay_factor * (i / (num_buckets - 1)) if num_buckets > 1 else 0
                        k = self.sampling_interpolation_steepness

                        y = 1 / (1 + torch.exp(-torch.tensor(k * (t - offset))))
                        y_start = 1 / (1 + torch.exp(torch.tensor(k * offset)))
                        y_end = 1 / (1 + torch.exp(-torch.tensor(k * (1 - offset))))

                        if torch.abs(y_end - y_start) < 1e-9:
                            t_delayed[i] = 1.0 if t > offset else 0.0
                        else:
                            t_delayed[i] = (y - y_start) / (y_end - y_start)

                    t_delayed = torch.clamp(t_delayed, 0, 1)
                    interpolated_probs = (1 - t_delayed) * start_probs + t_delayed * end_probs
                    current_probs = interpolated_probs / torch.sum(interpolated_probs)

            self.dynamic_sampling_probabilities.append(current_probs.tolist())
            self.optimizer_steps_per_epoch_list.append(optimizer_steps_per_epoch)

    # 251007-NEW: Helper method to initialize or reshuffle a bucket's indices for the current rank.
    def _reshuffle_and_reset_bucket(self, bucket_idx: int):
        """
        Shuffles the indices for a given bucket, distributes them among ranks,
        and resets the pointer for the current rank.
        """
        self.persistent_indices_per_bucket_rank[bucket_idx] = self._rank_indices(
            bucket_idx, self.reshuffle_counters[bucket_idx])
        self.current_position_in_bucket[bucket_idx] = 0
        self.reshuffle_counters[bucket_idx] += 1

    def _rank_indices(self, bucket_idx: int, shuffle_round: int) -> torch.Tensor:
        """This rank's share of one bucket's indices for a given shuffle round.

        A pure function of (seed, shuffle_round, bucket, rank, num_replicas), which
        is what lets state_dict store the round number instead of the indices.

        Uses a generator with a seed that changes on each reshuffle to ensure
        different shuffles over time, but is consistent across ranks. The order is
        the one the list-based version produced: there, shuffled_indices[i] was
        all_indices[perm[i]] with all_indices = range(offset, offset + size), that
        is offset + perm[i].

        Returns:
            torch.Tensor: int64 dataset indices, this rank's strided slice.
        """
        size = self.bucket_sizes[bucket_idx]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + shuffle_round)
            order = torch.randperm(size, generator=g)
        else:
            order = torch.arange(size)
        # Distribute indices among replicas
        return order[self.rank::self.num_replicas] + self.bucket_offsets[bucket_idx]

    def _setup_persistent_indices(self):
        """Initializes the persistent indices for all buckets for the first time."""
        for i in range(len(self.bucket_sizes)):
            self._reshuffle_and_reset_bucket(i)
        self._is_setup = True

    def __iter__(self):
        """
        Returns an iterator that yields mini-batches of indices.
        This version uses a persistent state to track consumed indices across epochs.
        """
        if not self._is_setup:
            self._setup_persistent_indices()
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch) # for calculation efficiency, not add self.rank (to ensure all node select same buckets.)

        sampling_probs = torch.tensor(self.dynamic_sampling_probabilities[self.epoch])

        optimizer_steps_target = self.optimizer_steps_per_epoch_list[self.epoch]

        # check iteration step (for loaded states)
        if self.optimizer_steps_yielded == optimizer_steps_target:
            self.optimizer_steps_yielded = 0

        # Mid-epoch resume: replay the bucket draws of the batches already trained in
        # this epoch, so the rest of it draws what an uninterrupted run would have.
        for _ in range(self.optimizer_steps_yielded):
            torch.multinomial(sampling_probs, 1, generator=g)
        self._iter_start = self.optimizer_steps_yielded
        self._snapshots = {self.optimizer_steps_yielded: (list(self.current_position_in_bucket),
                                                          list(self.reshuffle_counters))}

        while self.optimizer_steps_yielded < optimizer_steps_target:
            # Choose a bucket based on the sampling probabilities.
            chosen_bucket_idx = torch.multinomial(sampling_probs, 1, generator=g).item()

            micro_bs = self.batch_sizes[chosen_bucket_idx]
            acc_steps = self.accumulate_steps[chosen_bucket_idx]
            total_samples_needed = micro_bs * acc_steps

            macro_batch_indices = []

            while len(macro_batch_indices) < total_samples_needed:
                current_pos = self.current_position_in_bucket[chosen_bucket_idx]
                indices_for_rank = self.persistent_indices_per_bucket_rank[chosen_bucket_idx]

                # If the bucket's data is exhausted, reshuffle it and reset the pointer.
                if current_pos >= len(indices_for_rank):
                    self._reshuffle_and_reset_bucket(chosen_bucket_idx)
                    current_pos = self.current_position_in_bucket[chosen_bucket_idx]
                    indices_for_rank = self.persistent_indices_per_bucket_rank[chosen_bucket_idx]

                if len(indices_for_rank) == 0:
                    break

                remaining_needed = total_samples_needed -len(macro_batch_indices)
                available = len(indices_for_rank) - current_pos
                take_num = min(remaining_needed, available)
                start = current_pos
                end = current_pos + take_num
                macro_batch_indices.extend(indices_for_rank[start:end].tolist())
                self.current_position_in_bucket[chosen_bucket_idx] += take_num

            if len(macro_batch_indices) == total_samples_needed:
                # Negative sentinel appended as metadata: the collator reads it back
                # as the accumulation-step count for this macro-batch.
                macro_batch_indices.append(-1 * acc_steps)
                self._snapshots[self.optimizer_steps_yielded + 1] = (list(self.current_position_in_bucket),
                                                                     list(self.reshuffle_counters))
                yield macro_batch_indices
                self.optimizer_steps_yielded += 1
            else:
                break

    def __len__(self):
        """
        Returns the number of optimizer steps per epoch for the current replica.
        """
        return self.optimizer_steps_per_epoch_list[self.epoch]

    def set_epoch(self, epoch: int):
        """
        Sets the current epoch. This will affect the sampling probabilities but
        will NOT reset the data consumption state.
        """
        self.epoch = epoch

    def state_dict(self, consumed: int | None = None):
        """
        Returns the state of the sampler as a dictionary.

        The per-bucket indices are not stored: each is a pure function of the
        shuffle round (see _rank_indices), so the round counters and positions are
        enough to rebuild them. Storing them pickled a Python list holding this
        rank's share of every bucket into every checkpoint -- hundreds of MB per
        rank at the production scale.

        Args:
            consumed (int, optional): Batches of the current iteration that have
                been trained. The state is then taken as of that batch rather than
                from the live counters, which include the batches the DataLoader
                prefetched. Without it, a mid-epoch resume skips the prefetched
                batches and the epoch ends that many steps short.
        """
        yielded = self.optimizer_steps_yielded
        positions, counters = self.current_position_in_bucket, self.reshuffle_counters
        if consumed is not None and self._snapshots:
            key = self._iter_start + consumed
            if key in self._snapshots:
                yielded = key
                positions, counters = self._snapshots[key]
            else:
                logger.warning(f"sampler: no state recorded after {key} batches "
                               f"(have {min(self._snapshots)}-{max(self._snapshots)}); saving the live state")
        return {
            "format": 2,
            "optimizer_steps_yielded": yielded,
            "current_position_in_bucket": list(positions),
            "reshuffle_counters": list(counters),
            "_is_setup": self._is_setup,
            # Rebuilding is only exact for the same split; checked on load.
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "bucket_sizes": self.bucket_sizes,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the sampler state from a dictionary.

        Accepts both the current format and the original one, which carried the
        indices themselves as lists.

        Raises:
            ValueError: If the state was saved with a different seed, rank layout or
                dataset sizes, under which the rebuilt indices would differ.
        """
        self.optimizer_steps_yielded = state_dict["optimizer_steps_yielded"]
        self.current_position_in_bucket = list(state_dict["current_position_in_bucket"])
        self.reshuffle_counters = list(state_dict["reshuffle_counters"])
        self._is_setup = state_dict["_is_setup"]

        if "persistent_indices_per_bucket_rank" in state_dict:
            self.persistent_indices_per_bucket_rank = [
                torch.as_tensor(x, dtype=torch.int64)
                for x in state_dict["persistent_indices_per_bucket_rank"]]
            return

        saved = {k: state_dict[k] for k in ("seed", "num_replicas", "rank", "bucket_sizes")}
        current = {"seed": self.seed, "num_replicas": self.num_replicas, "rank": self.rank,
                   "bucket_sizes": self.bucket_sizes}
        if saved != current:
            raise ValueError(f"sampler state was saved for {saved}, but this run is {current}")
        self.persistent_indices_per_bucket_rank = [
            self._rank_indices(b, counter - 1) if counter > 0 else torch.empty(0, dtype=torch.int64)
            for b, counter in enumerate(self.reshuffle_counters)]


"""
DATASET PREPROCESS FUNCTIONS
"""



def _bounded_imap(pool, func, iterable, chunksize, max_workers):
    """``pool.imap`` that stops the workers running far ahead of the consumer.

    ``Pool.imap`` throttles the submitting side only: ``_handle_tasks`` blocks once
    the task pipe is full. Nothing throttles the result side -- ``IMapIterator._items``
    is a deque with no bound -- so when more workers feed results than the
    single-threaded consumer can write out, finished results pile up in the parent.
    Measured on 8M ZINC20 molecules at max_workers=24, on a 24-core / 48-thread
    Ryzen Threadripper 9960X with 125 GB RAM (polars 1.42.1, RDKit 2026.03.3):
    the backlog reached 204 chunks (2.04M rows) and the parent grew by 2.9 GB,
    released only once the backlog drained. The backlog tracks elapsed time, not
    a steady state, so at 200M+ rows it reaches tens of GB. Worker counts above
    what the consumer can absorb buy no throughput here (54.6 kmol/s at 12
    workers, 55.2 at 24) and only widen the gap.

    A semaphore in front of the input generator bounds (submitted - consumed).
    Results, order and counts are unchanged; only how far ahead the pool may work is.

    Args:
        pool: An active ``multiprocessing.Pool``.
        func: Callable applied to each item, as for ``pool.imap``.
        iterable: Input items. Consumed lazily, in step with the consumer.
        chunksize (int): Items per task batch, as for ``pool.imap``.
        max_workers (int): The pool's process count, used to size the window.

    Yields:
        One result per input item, in input order.
    """
    # Must exceed chunksize * max_workers, or a worker cannot be handed a full
    # batch and the pool stalls. The window costs chunksize * (max_workers + 2)
    # rows of buffered results, which is what caps the parent's memory.
    permits = chunksize * (max_workers + 2)
    sem = threading.Semaphore(permits)
    stop = threading.Event()

    def throttled():
        for item in iterable:
            # Timed waits, so the pool's task thread still notices a shutdown.
            # A plain acquire() here would hang pool.terminate() on an error path.
            while not sem.acquire(timeout=0.2):
                if stop.is_set():
                    return
            yield item

    try:
        for result in pool.imap(func, throttled(), chunksize=chunksize):
            sem.release()
            yield result
    finally:
        stop.set()
        for _ in range(permits):          # unpark the task thread if it is waiting
            sem.release()


def csv2SMILESset(path, usecolumn: str):
    """Reads one SMILES column and returns the canonical SMILES as a set.

    Molecules RDKit cannot parse are logged and skipped.

    Args:
        path: CSV, gzipped CSV or Parquet file.
        usecolumn (str): Name of the SMILES column.

    Returns:
        set[str]: Canonical, isomeric SMILES, deduplicated.
    """
    df = _load_dataframe(path, columns=[usecolumn])
    smiles = df[usecolumn].to_list()
    smiles_set = set()
    for smi in smiles:
        try:
            mol = MolFromSmiles(smi, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SetAromaticity(mol)
            smiles_set.add(MolToSmiles(
                mol, canonical=True, isomericSmiles=True))
        except Exception:
            _warn_limited("rdkit_parse", f"RDKit can't handle this molecule -> {smi}")
            continue
    return smiles_set


def ZINCtxt2SMILESset(path, sampling=1.0):
    """Reads a ZINC-style whitespace-separated text file into a set of SMILES.

    The first field of each line is taken as the SMILES. Molecules RDKit cannot
    parse are logged and skipped.

    Args:
        path: Path to the text file.
        sampling (float, optional): Probability of keeping each line. Defaults to 1.0.

    Returns:
        set[str]: Canonical, isomeric SMILES, deduplicated.
    """
    with open(path) as f:
        smiles = f.readlines()
    smiles = [line.split(" ")[0] for line in smiles]
    smiles_set = set()
    for smi in smiles:
        try:
            if random.random() < sampling:
                mol = MolFromSmiles(smi, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                Chem.SetAromaticity(mol)
                smiles_set.add(MolToSmiles(
                    mol, canonical=True, isomericSmiles=True))
        except Exception:
            _warn_limited("rdkit_parse", f"RDKit can't handle this molecule -> {smi}")
            continue
    return smiles_set


def convertingSMILES(smiles, randomize=True, canonicalize=True, num_random: int = 1, sanitize=False):
    """Converts one SMILES into its random and/or canonical forms.

    Args:
        smiles: A SMILES string, or a sequence whose first element is one.
        randomize (bool, optional): Emit a randomized SMILES. Defaults to True.
        canonicalize (bool, optional): Emit the canonical SMILES. Defaults to True.
        num_random (int, optional): Number of random variants per molecule. Values
            above 1 require both ``randomize`` and ``canonicalize``. Defaults to 1.
        sanitize (bool, optional): Passed to RDKit's ``MolFromSmiles``. Defaults to False.

    Returns:
        For ``num_random == 1``, a ``(random, canonical)`` pair with the
        disabled side set to None, or ``(None, None)`` if RDKit failed. For
        ``num_random > 1``, a list of such pairs, or None if RDKit failed.

    Raises:
        AssertionError: If neither conversion is requested, or if
            ``num_random > 1`` without both conversions.
        ValueError: If ``smiles`` is neither a string nor indexable.
    """
    assert randomize or canonicalize, "no randomize, no canonicalize, then what do you want to do ?"
    if num_random > 1:
        assert randomize and canonicalize, "currently, data augumentation is only supported randomize&canonicalize !"
    if not isinstance(smiles, str):
        try:
            smiles = smiles[0]
        except Exception as e:
            raise ValueError(f"invalid input ! > {smiles}") from e

    try:
        mol = MolFromSmiles(smiles, sanitize=sanitize)
        mol.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(mol)
    except Exception:
        _warn_limited("smi2mol", f"invalid smiles (smi2mol failed): {smiles}")
        return (None, None) if num_random == 1 else None
    if mol is None:
        _warn_limited("mol_none", f"invalid smiles (mol object None): {smiles}")
        return (None, None) if num_random == 1 else None
    try:
        rand, cano = None, None
        if canonicalize:
            cano = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if randomize:
            if num_random == 1:
                if canonicalize:
                    # Keep drawing a random SMILES until one canonicalizes back to
                    # `cano`, and drop the molecule if none of _MAX_RANDOM_TRIES does.
                    # The check needs `cano` to exist, hence the branch: without
                    # canonicalize there is nothing to compare against, and comparing
                    # against None would reject every molecule.
                    flag = False
                    cnt = 0
                    while (not flag) and cnt < _MAX_RANDOM_TRIES:
                        rand = MolToSmiles(mol, canonical=False,
                                           doRandom=True, isomericSmiles=True)
                        flag = MolToSmiles(MolFromSmiles(
                            rand, sanitize=sanitize), canonical=True, isomericSmiles=True) == cano
                        cnt += 1
                    if not flag:
                        return (None, None)
                else:
                    rand = MolToSmiles(mol, canonical=False,
                                       doRandom=True, isomericSmiles=True)
                return rand, cano
            elif num_random > 1:
                rand = set()
                cnt = 0
                while len(rand) < num_random and cnt <= num_random * _RANDOM_TRIES_PER_VARIANT:
                    _r = MolToSmiles(mol=mol, canonical=False,
                                     doRandom=True, isomericSmiles=True)
                    if MolToSmiles(MolFromSmiles(_r, sanitize=sanitize), canonical=True, isomericSmiles=True) == cano:
                        rand.add(_r)
                    cnt += 1
                rand = list(rand)
                smiles_set = [[r, cano] for r in rand]
        return (rand, cano) if num_random == 1 else smiles_set
    except Exception:
        _warn_limited("mol2smi", f"invalid smiles (mol2smi failed): {smiles}")
        return (None, None) if num_random == 1 else None


def raw2randomORcanonical(allsmiles: pl.Series | pd.Series | list, randomize: bool, canonicalize: bool, output_path: str, sanitize=False, max_workers: int = 1, log_step=1000000):
    """Converts a SMILES collection into random/canonical pairs and writes them out.

    The output format follows ``output_path``: ``.parquet`` for Parquet, ``.csv.gz``
    for gzipped CSV, anything else for plain CSV. Columns are ``random`` and/or
    ``canonical``, depending on the flags. Molecules RDKit cannot parse are dropped.
    The result is shuffled before it is written.

    Args:
        allsmiles: SMILES strings, as a Polars Series, pandas Series or list.
        randomize (bool): Emit a ``random`` column.
        canonicalize (bool): Emit a ``canonical`` column.
        output_path (str): Destination file; its extension picks the format.
        sanitize (bool, optional): Passed to RDKit. Defaults to False.
        max_workers (int, optional): Worker processes. Defaults to 1.
        log_step (int, optional): Progress is logged every this many molecules.
            Defaults to 1000000.

    Raises:
        ValueError: If neither ``randomize`` nor ``canonicalize`` is set.
    """
    ensure_default_logging()
    if not (randomize or canonicalize):
        raise ValueError(
            "no randomize, no canonicalize, then what do you want to do ?")
    num_smiles = len(allsmiles)
    use_parquet = output_path.endswith(".parquet")
    column = []
    if randomize:
        column.append("random")
    if canonicalize:
        column.append("canonical")
    func = functools.partial(convertingSMILES, randomize=randomize,
                             canonicalize=canonicalize, sanitize=sanitize)

    cnt = 0
    _FLUSH = 100_000

    with multiprocessing.Pool(processes=max_workers) as pool:
        if use_parquet:
            schema = pa.schema([(col, pa.string()) for col in column])
            buffers = {col: [] for col in column}
            with pq.ParquetWriter(output_path, schema) as pq_writer:
                for rand, cano in _bounded_imap(pool, func, allsmiles, 10000, max_workers):
                    if not rand and not cano:
                        continue
                    if randomize:
                        buffers["random"].append(rand)
                    if canonicalize:
                        buffers["canonical"].append(cano)
                    cnt += 1
                    if cnt % _FLUSH == 0:
                        pq_writer.write_table(pa.table(buffers))
                        for k in buffers:
                            buffers[k] = []
                    if cnt % log_step == 0:
                        logger.info(f"{cnt} molecules processed.")
                if any(len(v) > 0 for v in buffers.values()):
                    pq_writer.write_table(pa.table(buffers))
        else:
            use_gz = output_path.endswith(".gz")
            with ExitStack() as stack:
                if use_gz:
                    f = stack.enter_context(gzip.open(output_path, "wt", newline=""))
                else:
                    f = stack.enter_context(open(output_path, 'w', newline=""))
                writer = csv.writer(f)
                writer.writerow(column)
                for rand, cano in _bounded_imap(pool, func, allsmiles, 10000, max_workers):
                    if rand and cano:
                        writer.writerow([rand, cano])
                    elif rand and not cano:
                        writer.writerow([rand])
                    elif not rand and cano:
                        writer.writerow([cano])
                    else:
                        continue
                    cnt += 1
                    if cnt % log_step == 0:
                        logger.info(f"{cnt} molecules processed.")

    logger.info("===== finished =====")
    logger.info(f"Total processed molecules: {cnt}/{num_smiles}")


def RandomAugmentation(csvpath: str, num_random: int, sanitize: bool = False, max_workers: int = 1, log_step=1000000):
    """Writes ``num_random`` random/canonical pairs per molecule to a sibling file.

    The output keeps the input's format and gains an ``_augmented`` suffix, so
    ``data/x.parquet`` becomes ``data/x_augmented.parquet``. The input is left
    untouched.

    Args:
        csvpath (str): Input ``.csv``, ``.csv.gz`` or ``.parquet`` file.
        num_random (int): Random SMILES generated per molecule.
        sanitize (bool, optional): Passed to RDKit. Defaults to False.
        max_workers (int, optional): Worker processes. Defaults to 1.
        log_step (int, optional): Progress is logged every this many molecules.
            Defaults to 1000000.
    """
    ensure_default_logging()
    func = functools.partial(convertingSMILES, randomize=True,
                             canonicalize=True, num_random=num_random, sanitize=sanitize)

    path = str(csvpath)
    if path.endswith(".csv.gz"):
        out_path = path.replace(".csv.gz", "_augmented.csv.gz")
    elif path.endswith(".parquet"):
        out_path = path.replace(".parquet", "_augmented.parquet")
    else:
        out_path = path.replace(".csv", "_augmented.csv")

    use_parquet = out_path.endswith(".parquet")
    use_gz = out_path.endswith(".gz")

    cnt = 0
    i = -1
    _FLUSH = 100_000

    with multiprocessing.Pool(processes=max_workers) as pool:
        if use_parquet:
            schema = pa.schema([("random", pa.string()), ("canonical", pa.string())])
            buf_rand, buf_cano = [], []
            with pq.ParquetWriter(out_path, schema) as pq_writer:
                for i, smiles_set in enumerate(_bounded_imap(pool, func, _read_smiles_file(path, randomize=False, canonicalize=True), 10000, max_workers)):
                    if smiles_set is not None:
                        cnt += 1
                        for pair in smiles_set:
                            buf_rand.append(pair[0])
                            buf_cano.append(pair[1])
                        if len(buf_rand) >= _FLUSH:
                            pq_writer.write_table(pa.table({"random": buf_rand, "canonical": buf_cano}))
                            buf_rand.clear()
                            buf_cano.clear()
                    if (i + 1) % log_step == 0:
                        logger.info(f"{i+1} molecules processed.")
                if buf_rand:
                    pq_writer.write_table(pa.table({"random": buf_rand, "canonical": buf_cano}))
        else:
            with ExitStack() as stack:
                if use_gz:
                    f = stack.enter_context(gzip.open(out_path, "wt", newline=""))
                else:
                    f = stack.enter_context(open(out_path, 'w', newline=""))
                writer = csv.writer(f)
                writer.writerow(["random", "canonical"])
                for i, smiles_set in enumerate(_bounded_imap(pool, func, _read_smiles_file(path, randomize=False, canonicalize=True), 10000, max_workers)):
                    if smiles_set is not None:
                        cnt += 1
                        writer.writerows(smiles_set)
                    if (i + 1) % log_step == 0:
                        logger.info(f"{i+1} molecules processed.")

    logger.info("===== finished =====")
    logger.info(f"Total processed molecules: {cnt}/{i+1}")
    gc.collect()
    logger.info("Loading...")
    df = pl.read_parquet(out_path) if use_parquet else pl.read_csv(out_path)
    logger.info(f"Total datasize: {df.shape}")
    datanum = df.shape[0]
    logger.info("Shuffling...")
    df = df.sample(fraction=1.0, seed=42, shuffle=True)
    logger.info("Exporting...")
    if use_parquet:
        df.write_parquet(out_path)
    else:
        df.write_csv(out_path)
    del df
    gc.collect()
    return datanum


def _read_smiles_file(path: str | Path, randomize: bool, canonicalize: bool, old_style: bool = False):
    """Lazily yields SMILES rows from a CSV (.csv, .csv.gz) or Parquet file using Polars."""
    cols = []
    if randomize:
        cols.append("input" if old_style else "random")
    if canonicalize:
        cols.append("output" if old_style else "canonical")

    path = str(path)
    CHUNK = 1_000_000
    if path.endswith(".parquet"):
        lf = pl.scan_parquet(path).select(cols)
    else:
        lf = pl.scan_csv(path, infer_schema_length=0).select(cols)

    offset = 0
    while True:
        chunk = lf.slice(offset, CHUNK).collect()
        if chunk.is_empty():
            break
        for row in chunk.iter_rows():
            yield np.array(row, dtype=object)
        offset += CHUNK


_worker_tokenizer: SMILESTokenizer | None = None

def _worker_init(token_path: str) -> None:
    """Called once per worker process at pool startup to load the tokenizer into a global."""
    global _worker_tokenizer
    _worker_tokenizer = SMILESTokenizer(token_path)


def _augment_worker(
    canonical_smi: str,
    *,
    num_random: int,
    sanitize: bool = False,
    exclude_wildcard: bool = True,
):
    """Worker for augment_bins: makes ``num_random`` random SMILES and tokenizes them.

    The token id lists come back **unpadded**. Padding needs a target length, and the
    target length is the destination bucket's, which depends on how long the generated
    variant turned out to be -- a randomized SMILES can be longer than the canonical
    one it came from. The caller therefore picks the bucket and pads there, the same
    way split_and_tokenize does.

    Args:
        canonical_smi (str): The molecule to augment.
        num_random (int): Random SMILES to generate.
        sanitize (bool, optional): Passed to RDKit. Defaults to False.
        exclude_wildcard (bool, optional): Reject SMILES containing ``"*"``.
            Defaults to True.

    Returns:
        list[tuple] | None: ``(rand_ids, cano_ids, rand_smi, cano_smi)`` per variant,
        or None if the molecule produced nothing usable.
    """
    if not isinstance(canonical_smi, str) or not canonical_smi:
        return None
    if exclude_wildcard and "*" in canonical_smi:
        return None

    tokenizer = _worker_tokenizer
    if tokenizer is None:
        return None

    smiles_set = convertingSMILES(
        canonical_smi, randomize=True, canonicalize=True,
        num_random=num_random, sanitize=sanitize,
    )
    if smiles_set is None:
        return None

    results = []
    for rand_smi, cano_smi in smiles_set:
        try:
            rand_ids = tokenizer.encode(rand_smi, add_special_tokens=True)
            cano_ids = tokenizer.encode(cano_smi, add_special_tokens=True)
        except Exception:
            continue
        results.append((rand_ids, cano_ids, rand_smi, cano_smi))

    return results if results else None


def process_single_row(row, exclude_wildcard=True):
    """
    Tokenize a single row of SMILES data and compute molecular weight.
    Padding is deferred to the caller so each bucket can use its own max length.
    seqlens contains pure SMILES token counts (BOS/EOS excluded).
    Tokenizer is loaded once per worker process via _worker_init.

    The row is all-or-nothing. One row is one molecule's variants, written to one
    memmap row and one sidecar line, so dropping a single variant would write the
    remaining ones under the wrong column names and leave the memmap row short.
    Any unusable variant therefore rejects the whole row, and None is returned.
    """
    tokenizer = _worker_tokenizer
    smiles_list = row

    if isinstance(smiles_list, np.ndarray):
        smiles_list = smiles_list.tolist()
    elif not isinstance(smiles_list, list):
        smiles_list = [smiles_list]

    valid_smiles = []
    valid_ids = []
    seqlens = []
    mws = []

    for smi in smiles_list:
        # A missing cell arrives as a float (NaN) from both Polars and pandas; the
        # membership test below would raise TypeError and take the worker with it.
        if not isinstance(smi, str) or not smi:
            return None
        if exclude_wildcard and "*" in smi:
            return None

        # --- 1. Tokenization (no padding — caller pads per bucket) ---
        try:
            raw_ids = tokenizer.encode(smi, add_special_tokens=True)
            n_tokens = len(raw_ids) - 2  # exclude BOS + EOS → pure SMILES token length
        except Exception:
            return None

        # --- 2. RDKit Calculation ---
        # A failure to compute the weight is not fatal (mw falls back to 0.0); a
        # failure to parse at all is, because the variant cannot be trusted.
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                return None
            mol.UpdatePropertyCache(strict=False)
            mw = rdMolDescriptors.CalcExactMolWt(mol)
        except Exception:
            mw = 0.0

        valid_smiles.append(smi)
        valid_ids.append(raw_ids)
        seqlens.append(n_tokens)
        mws.append(mw)

    if not valid_smiles:
        return None

    return valid_smiles, valid_ids, seqlens, mws


# ==========================================
# Main Processing Pipeline
# ==========================================

def split_and_tokenize(csv_path, token_path, thresholds: list[int], max_length: int,
                        randomize: bool, canonicalize: bool,
                        do_split: bool = True, output_dir: os.PathLike | None = None,
                        exclude_wildcard: bool = True, analyze: bool = True,
                        compress: bool = True, smiles_format: str = "csv",
                        max_workers: int = 1, old_style: bool = False, log_step: int = 1_000_000,
                        analyze_stride: int = 100):
    """
    Tokenize SMILES from a CSV or Parquet file and write per-bucket NPY memmaps.

    Args:
        csv_path: Input file (.csv, .csv.gz, or .parquet).
        token_path: Path to the SMILESTokenizer vocabulary file.
        thresholds: Bucket boundaries in pure SMILES token counts (BOS/EOS excluded).
                    e.g. [0, 64, 128, 256] creates 3 buckets stored with seqlen 66/130/258.
        max_length: Max sequence length (BOS+tokens+EOS) used when do_split=False.
        randomize: Include randomized SMILES column.
        canonicalize: Include canonical SMILES column.
        do_split: If True, split into per-bucket files. If False, write a single output.
        output_dir: Output directory. Defaults to the input file's directory.
        exclude_wildcard: Drop SMILES containing '*'. Defaults to True.
        analyze: Return a pl.DataFrame of seqlen/mw distributions. Defaults to True.
        compress: Write SMILES sidecar as .csv.gz (ignored when smiles_format="parquet").
        smiles_format: "csv" or "parquet" — format of the SMILES sidecar in split mode.
        max_workers: Number of worker processes.
        old_style: Use legacy "input"/"output" column names instead of "random"/"canonical".
        log_step: Print progress every this many inputs processed.
        analyze_stride: Record analysis data every N rows. Default 100 keeps ~1 M samples
                        for a 100 M-row dataset, avoiding unbounded Python list growth.
    """
    ensure_default_logging()
    tokenizer = SMILESTokenizer(token_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    del tokenizer  # each worker loads its own via _worker_init

    column = []
    if do_split:
        if randomize:
            column.append("random")
        if canonicalize:
            column.append("canonical")

    args_generator = _read_smiles_file(csv_path, randomize, canonicalize, old_style=old_style)
    func = functools.partial(process_single_row, exclude_wildcard=exclude_wildcard)

    base_dir = str(output_dir) if output_dir is not None else os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    # Strip all known extensions to get a clean stem
    stem = base_name
    for ext in ('.csv.gz', '.parquet', '.csv'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break

    use_parquet_smiles = smiles_format == "parquet"
    _PQ_FLUSH = 100_000

    logger.info(f"===== start processing (Split={do_split}, smiles_format={smiles_format}) =====")

    num_processed = 0
    with ExitStack() as stack:
        pool = stack.enter_context(
            multiprocessing.Pool(
                processes=max_workers,
                initializer=_worker_init,
                initargs=(token_path,),
            )
        )

        csv_writers = []
        pq_writers = []
        pq_buffers = []
        bin_handles = []
        temp_bin_paths = []

        # --- Prepare output handles ---
        if do_split:
            num_bins = len(thresholds) - 1
            for i in range(num_bins):
                bin_stem = f"{stem}_{thresholds[i]}to{thresholds[i+1]}"

                if use_parquet_smiles:
                    schema = pa.schema([(col, pa.string()) for col in column])
                    pq_path = os.path.join(base_dir, f"{bin_stem}.parquet")
                    pq_writers.append(stack.enter_context(pq.ParquetWriter(pq_path, schema)))
                    pq_buffers.append({col: [] for col in column})
                    csv_writers.append(None)
                else:
                    if compress:
                        f_csv = stack.enter_context(gzip.open(os.path.join(base_dir, f"{bin_stem}.csv.gz"), "wt", newline=""))
                    else:
                        f_csv = stack.enter_context(open(os.path.join(base_dir, f"{bin_stem}.csv"), 'w', newline=""))
                    w = csv.writer(f_csv)
                    w.writerow(column)
                    csv_writers.append(w)
                    pq_writers.append(None)
                    pq_buffers.append(None)

                bin_path = os.path.join(base_dir, f"{bin_stem}.temp.bin")
                temp_bin_paths.append(bin_path)
                bin_handles.append(stack.enter_context(open(bin_path, "wb")))
        else:
            num_bins = 1
            csv_writers.append(None)
            pq_writers.append(None)
            pq_buffers.append(None)
            bin_path = os.path.join(base_dir, f"{stem}.temp.bin")
            temp_bin_paths.append(bin_path)
            bin_handles.append(stack.enter_context(open(bin_path, "wb")))

        cnts = [0] * (num_bins + 1)
        if analyze:
            res = {"seqlen": [], "mw": []}


        # --- Main loop ---
        for num_processed, result in enumerate(_bounded_imap(pool, func, args_generator, 10000, max_workers), start=1):
            if result is None:
                continue

            smiles_out, ids_out, seqlens, mws = result
            current_max_len = max(seqlens)

            target_idx = -1
            if do_split:
                for idx in range(num_bins):
                    if thresholds[idx] < current_max_len <= thresholds[idx+1]:
                        target_idx = idx
                        break
            else:
                target_idx = 0

            if target_idx != -1:
                bin_max_len = thresholds[target_idx + 1] + 2 if do_split else max_length
                padded_ids = []
                for ids in ids_out:
                    if len(ids) >= bin_max_len:
                        padded_ids.append(ids[:bin_max_len])
                    else:
                        padded_ids.append(ids + [pad_id] * (bin_max_len - len(ids)))

                # SMILES sidecar (split mode only)
                if do_split:
                    if use_parquet_smiles:
                        buf = pq_buffers[target_idx]
                        # strict=False keeps the existing behaviour: process_single_row
                        # can drop an individual variant, making smiles_out shorter
                        # than column, and the shorter one wins.
                        for col, val in zip(column, smiles_out, strict=False):
                            buf[col].append(val)
                        if buf and len(next(iter(buf.values()))) >= _PQ_FLUSH:
                            pq_writers[target_idx].write_table(pa.table(buf))
                            for k in buf:
                                buf[k] = []
                    elif csv_writers[target_idx] is not None:
                        csv_writers[target_idx].writerow(smiles_out)

                arr = np.array(padded_ids, dtype=np.int16)
                bin_handles[target_idx].write(arr.tobytes())
                cnts[target_idx] += 1
            else:
                cnts[-1] += 1

            if analyze and num_processed % analyze_stride == 0:
                res["seqlen"].append(current_max_len)
                res["mw"].append(max(mws))

            if num_processed % log_step == 0:
                logger.info(f"{num_processed} compounds processed.")
                for f in bin_handles:
                    f.flush()

        # Flush remaining parquet buffers before ExitStack closes the writers
        if use_parquet_smiles and do_split:
            for idx in range(num_bins):
                buf = pq_buffers[idx]
                if buf and any(len(v) > 0 for v in buf.values()):
                    pq_writers[idx].write_table(pa.table(buf))

    # --- Post Process: Temp -> NPY (Chunked) ---
    logger.info("===== converting temp binaries to npy (chunked) =====")

    num_variants = 0
    if randomize:
        num_variants += 1
    if canonicalize:
        num_variants += 1
    if num_variants == 0:
        num_variants = 1

    CONVERT_CHUNK_SIZE = 100_000

    for idx in range(num_bins):
        count = cnts[idx]
        temp_bin = temp_bin_paths[idx]

        bin_max_len = thresholds[idx + 1] + 2 if do_split else max_length
        ROW_BYTES = num_variants * bin_max_len * 2  # int16 = 2 bytes

        if do_split:
            npy_path = os.path.join(base_dir, f"{stem}_{thresholds[idx]}to{thresholds[idx+1]}.npy")
        else:
            npy_path = os.path.join(base_dir, f"{stem}.npy")

        if count > 0 and os.path.exists(temp_bin):
            actual_size = os.path.getsize(temp_bin)
            expected_size = count * ROW_BYTES

            if actual_size != expected_size:
                logger.error(f"[FATAL ERROR] File corrupted or incomplete: {temp_bin}")
                logger.info(f"  Expected: {expected_size} bytes ({count} rows)")
                logger.info(f"  Actual:   {actual_size} bytes")
                logger.info(f"  Diff:     {actual_size - expected_size} bytes")
                logger.info("  -> Skipping conversion to avoid misalignment.")
                continue

            logger.info(f"Creating {npy_path} with shape ({count}, {num_variants}, {bin_max_len})")
            shape = (count, num_variants, bin_max_len)
            mm_array = np.memmap(npy_path, dtype=np.int16, mode='w+', shape=shape)

            processed_rows = 0
            with open(temp_bin, "rb") as f_in:
                while processed_rows < count:
                    rows_to_read = min(CONVERT_CHUNK_SIZE, count - processed_rows)
                    bytes_to_read = rows_to_read * ROW_BYTES
                    raw_bytes = f_in.read(bytes_to_read)
                    if len(raw_bytes) != bytes_to_read:
                        logger.error(f"[FATAL ERROR] Read error at row {processed_rows}. Expected {bytes_to_read} bytes, got {len(raw_bytes)}.")
                        break
                    try:
                        chunk_data = np.frombuffer(raw_bytes, dtype=np.int16)
                        chunk_reshaped = chunk_data.reshape(rows_to_read, num_variants, bin_max_len)
                        mm_array[processed_rows : processed_rows + rows_to_read] = chunk_reshaped
                    except ValueError as e:
                        logger.error(f"[FATAL ERROR] Reshape/Assign failed at index {processed_rows}: {e}")
                        break
                    processed_rows += rows_to_read
                    if (processed_rows // CONVERT_CHUNK_SIZE) % 5 == 0:
                        mm_array.flush()

            mm_array.flush()
            del mm_array
            os.remove(temp_bin)
            logger.info(f"-> Finished {os.path.basename(npy_path)}")

        elif os.path.exists(temp_bin):
            os.remove(temp_bin)

    gc.collect()

    logger.info("===== processing finished =====")
    logger.info(f"Total processed inputs: {num_processed}")
    if do_split:
        for i in range(num_bins):
            logger.info(f"Phase {i+1} ({thresholds[i]}to{thresholds[i+1]}): {cnts[i]} compounds")
        logger.info(f"Out of range: {cnts[-1]}")
    else:
        logger.info(f"Total Saved: {cnts[0]} compounds")

    if analyze:
        return pl.DataFrame(res)


def augment_bins(
    base_dir: str | Path,
    stem: str,
    token_path: str,
    thresholds: list[int],
    bin_num_random: dict[int, int],
    sidecar_format: str = "parquet",
    sanitize: bool = False,
    exclude_wildcard: bool = True,
    max_workers: int = 1,
    log_step: int = 1_000_000,
):
    """
    Generate augmented NPY files for specific buckets produced by split_and_tokenize.

    Reads the canonical SMILES sidecar for each requested bucket, generates
    ``bin_num_random[i]`` random SMILES per molecule, tokenizes the (random, canonical)
    pairs, and writes new NPY + sidecar files.  Original files are untouched.

    **Each pair is filed by its own length, not by the bucket it came from.** A
    randomized SMILES can be longer than the canonical one it was generated from, so a
    molecule drawn from the 200-300 bucket can produce a variant that only fits the
    300-400 one. Filing those under the source bucket would truncate them and cut off
    the EOS token, so the destination is chosen per pair with the same rule
    split_and_tokenize uses: the bucket that holds ``max(len(random), len(canonical))``.
    One call therefore writes to every bucket its output reaches, and several source
    buckets can contribute to the same destination file.

    Pairs longer than ``thresholds[-1]`` have no bucket to go to. They are counted and
    reported at the end rather than dropped quietly; widen ``thresholds`` to keep them.

    Output order follows the input, so the variants of one molecule sit together. The
    curriculum samplers shuffle indices every epoch, so this does not correlate the
    batches, but a consumer reading the file sequentially should shuffle it itself.

    Args:
        base_dir: Directory where split_and_tokenize wrote its outputs.
        stem: File stem used in split_and_tokenize (e.g. ``"combined_smiles"``).
        token_path: Vocabulary file path (same as passed to split_and_tokenize).
        thresholds: Same threshold list passed to split_and_tokenize.
        bin_num_random: Dict mapping *source* bucket index → random count per molecule.
                        e.g. ``{0: 10, 2: 5}`` augments the molecules of buckets 0 and 2.
        sidecar_format: ``"parquet"`` or ``"csv"`` — format to read and write.
        sanitize: RDKit sanitize flag passed to convertingSMILES.
        exclude_wildcard: Skip SMILES containing ``"*"``.
        max_workers: Number of worker processes.
        log_step: Log progress every N input molecules.

    Returns:
        dict[int, int]: Rows written per *destination* bucket index. The value is the
        ``num_data`` to put in the config for that file.

    Output files per destination bucket ``j``::

        {base_dir}/{stem}_{lo}to{hi}_aug.npy       shape (M, 2, hi+2)  int16
        {base_dir}/{stem}_{lo}to{hi}_aug.parquet   (or .csv.gz)
    """
    ensure_default_logging()
    base_dir = Path(base_dir)
    tokenizer = SMILESTokenizer(token_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    del tokenizer  # each worker loads its own via _worker_init

    _PQ_FLUSH = 100_000
    CONVERT_CHUNK = 100_000
    use_parquet = sidecar_format == "parquet"
    ext = "parquet" if use_parquet else "csv.gz"
    num_bins = len(thresholds) - 1

    for bin_idx in bin_num_random:
        if not 0 <= bin_idx < num_bins:
            raise ValueError(
                f"bin_num_random refers to bucket {bin_idx}, but thresholds define "
                f"buckets 0..{num_bins - 1}"
            )

    def _sidecar_for(bin_idx: int) -> Path | None:
        lo, hi = thresholds[bin_idx], thresholds[bin_idx + 1]
        if use_parquet:
            p = base_dir / f"{stem}_{lo}to{hi}.parquet"
            return p if p.exists() else None
        for name in (f"{stem}_{lo}to{hi}.csv.gz", f"{stem}_{lo}to{hi}.csv"):
            p = base_dir / name
            if p.exists():
                return p
        return None

    def _canonical_gen(path: Path):
        CHUNK = 1_000_000
        p = str(path)
        lf = (pl.scan_parquet(p).select("canonical") if p.endswith(".parquet")
              else pl.scan_csv(p, infer_schema_length=0).select("canonical"))
        offset = 0
        while True:
            chunk = lf.slice(offset, CHUNK).collect()
            if chunk.is_empty():
                break
            yield from chunk["canonical"]
            offset += CHUNK

    # Destination state, opened once for the whole call because a source bucket can
    # spill into a neighbour and two source buckets can share a destination.
    bin_max_lens = [thresholds[j + 1] + 2 for j in range(num_bins)]
    row_bytes = [2 * L * 2 for L in bin_max_lens]        # 2 variants x L x int16
    temp_paths = [base_dir / f"{stem}_{thresholds[j]}to{thresholds[j + 1]}_aug.temp.bin"
                  for j in range(num_bins)]
    counts = [0] * num_bins
    oversize = 0
    cnt_in = 0

    def _pad(ids: list, target: int) -> list:
        return ids + [pad_id] * (target - len(ids))

    with ExitStack() as stack:
        pool = stack.enter_context(
            multiprocessing.Pool(processes=max_workers, initializer=_worker_init,
                                 initargs=(token_path,))
        )
        temp_handles, writers, buffers = [], [], []
        for j in range(num_bins):
            temp_handles.append(stack.enter_context(open(temp_paths[j], "wb")))
            out_sidecar = base_dir / f"{stem}_{thresholds[j]}to{thresholds[j + 1]}_aug.{ext}"
            if use_parquet:
                schema = pa.schema([("random", pa.string()), ("canonical", pa.string())])
                writers.append(stack.enter_context(pq.ParquetWriter(str(out_sidecar), schema)))
                buffers.append({"random": [], "canonical": []})
            else:
                f_csv = stack.enter_context(gzip.open(str(out_sidecar), "wt", newline=""))
                w = csv.writer(f_csv)
                w.writerow(["random", "canonical"])
                writers.append(w)
                buffers.append(None)

        for bin_idx, num_random in bin_num_random.items():
            lo, hi = thresholds[bin_idx], thresholds[bin_idx + 1]
            sidecar_path = _sidecar_for(bin_idx)
            if sidecar_path is None:
                logger.warning(f"[WARNING] Sidecar not found for bucket {bin_idx} ({lo}to{hi}); skipped")
                continue

            logger.info(f"===== augmenting bucket {bin_idx} ({lo}to{hi}), n={num_random} =====")
            logger.info(f"  sidecar : {sidecar_path}")

            func = functools.partial(_augment_worker, num_random=num_random,
                                     sanitize=sanitize, exclude_wildcard=exclude_wildcard)

            for result in _bounded_imap(pool, func, _canonical_gen(sidecar_path), 1000, max_workers):
                cnt_in += 1
                if result is not None:
                    for rand_ids, cano_ids, rand_smi, cano_smi in result:
                        # Pure SMILES token count, BOS/EOS excluded, as thresholds mean.
                        need = max(len(rand_ids), len(cano_ids)) - 2
                        dest = -1
                        for j in range(num_bins):
                            if thresholds[j] < need <= thresholds[j + 1]:
                                dest = j
                                break
                        if dest == -1:
                            oversize += 1
                            continue

                        L = bin_max_lens[dest]
                        arr = np.array([_pad(rand_ids, L), _pad(cano_ids, L)], dtype=np.int16)
                        temp_handles[dest].write(arr.tobytes())
                        counts[dest] += 1

                        if use_parquet:
                            buf = buffers[dest]
                            buf["random"].append(rand_smi)
                            buf["canonical"].append(cano_smi)
                            if len(buf["random"]) >= _PQ_FLUSH:
                                writers[dest].write_table(pa.table(buf))
                                buf["random"].clear()
                                buf["canonical"].clear()
                        else:
                            writers[dest].writerow([rand_smi, cano_smi])

                if cnt_in % log_step == 0:
                    logger.info(f"  {cnt_in} molecules processed, {sum(counts)} pairs written.")

        if use_parquet:
            for j in range(num_bins):
                if buffers[j]["random"]:
                    writers[j].write_table(pa.table(buffers[j]))

    # Validate and convert each destination temp binary -> NPY
    written = {}
    for j in range(num_bins):
        temp_bin = temp_paths[j]
        count = counts[j]
        lo, hi = thresholds[j], thresholds[j + 1]
        out_npy = base_dir / f"{stem}_{lo}to{hi}_aug.npy"

        if count == 0:
            os.remove(temp_bin)
            base_sidecar = base_dir / f"{stem}_{lo}to{hi}_aug.{ext}"
            if base_sidecar.exists():
                os.remove(base_sidecar)
            continue

        actual_size = os.path.getsize(temp_bin)
        expected_size = count * row_bytes[j]
        if actual_size != expected_size:
            logger.error(f"[FATAL ERROR] Temp binary size mismatch for bucket {j} ({lo}to{hi}).")
            logger.error(f"  Expected {expected_size} bytes ({count} rows), got {actual_size}.")
            os.remove(temp_bin)
            continue

        logger.info(f"  Converting bucket {j} ({lo}to{hi}): {count} rows, "
                    f"shape ({count}, 2, {bin_max_lens[j]})")
        mm = np.memmap(str(out_npy), dtype=np.int16, mode="w+",
                       shape=(count, 2, bin_max_lens[j]))
        processed = 0
        with open(temp_bin, "rb") as f_in:
            while processed < count:
                rows = min(CONVERT_CHUNK, count - processed)
                raw = f_in.read(rows * row_bytes[j])
                if len(raw) != rows * row_bytes[j]:
                    logger.error(f"[FATAL ERROR] Read error at row {processed} of bucket {j}.")
                    break
                mm[processed:processed + rows] = np.frombuffer(raw, dtype=np.int16).reshape(
                    rows, 2, bin_max_lens[j])
                processed += rows
        mm.flush()
        del mm
        os.remove(temp_bin)
        written[j] = count
        logger.info(f"  -> {out_npy.name}: {count} pairs")

    logger.info("===== augment_bins finished =====")
    logger.info(f"Input molecules read: {cnt_in}")
    for j, count in sorted(written.items()):
        logger.info(f"  bucket {j} ({thresholds[j]}to{thresholds[j + 1]}): {count} pairs")
    if oversize:
        logger.warning(
            f"{oversize} pairs exceeded the longest bucket ({thresholds[-1]} tokens) and "
            "were not written. A randomized SMILES can be longer than the canonical one "
            "it came from; widen `thresholds` to keep these."
        )
    return written
