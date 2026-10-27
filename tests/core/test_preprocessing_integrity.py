"""Row-level integrity of the SMILES preprocessing path.

Each test here pins a defect that was found and fixed on 2026-10-23. They are
cheap (no pool, no files) and run on CPU.
"""

import logging
import multiprocessing

import numpy as np
import pytest

from pan_core.common import data_handler as D
from tests.utils import VOCAB_FILE

BENZENE_ETHER = "CCOc1ccccc1"


@pytest.fixture(autouse=True)
def _worker_tokenizer():
    """process_single_row reads a module global that _worker_init populates."""
    D._worker_init(str(VOCAB_FILE))
    yield
    D._worker_tokenizer = None


# --- convertingSMILES -------------------------------------------------------

def test_randomize_only_still_produces_a_smiles():
    """The round-trip check needs a canonical form; without one it must be skipped.

    Comparing against the unset canonical (None) can never succeed, so guarding the
    rejection on `flag` alone would drop every molecule in randomize-only mode.
    """
    rand, cano = D.convertingSMILES(BENZENE_ETHER, randomize=True, canonicalize=False)
    assert isinstance(rand, str) and rand
    assert cano is None


def test_molecule_is_dropped_when_no_random_smiles_round_trips(monkeypatch):
    """Exhausting the attempts must drop the molecule, not emit the last attempt.

    Setting the budget to zero reaches the same branch as a molecule whose random
    forms never canonicalize back.
    """
    monkeypatch.setattr(D, "_MAX_RANDOM_TRIES", 0)
    assert D.convertingSMILES(BENZENE_ETHER, randomize=True, canonicalize=True) == (None, None)


def test_round_tripping_molecule_returns_both_forms():
    rand, cano = D.convertingSMILES(BENZENE_ETHER, randomize=True, canonicalize=True)
    assert isinstance(rand, str) and rand
    assert cano == BENZENE_ETHER


# --- process_single_row -----------------------------------------------------

@pytest.mark.parametrize("row, reason", [
    ([float("nan"), "CCO"], "missing cell"),
    (["C1CC", "CCO"], "unparseable source"),
    (["CCO", "C1CC"], "unparseable target"),
    (["*CC", "CCO"], "wildcard in one variant"),
    (["", "CCO"], "empty cell"),
])
def test_row_is_dropped_when_any_variant_fails(row, reason):
    """One row is one memmap row and one sidecar line, so a partial row would write
    the surviving variant under the wrong column and leave the memmap row short."""
    assert D.process_single_row(np.array(row, dtype=object), exclude_wildcard=True) is None, reason


def test_usable_row_keeps_every_variant():
    out = D.process_single_row(np.array(["CCO", "OCC"], dtype=object), exclude_wildcard=True)
    assert out is not None
    smiles_out, ids_out, seqlens, mws = out
    assert len(smiles_out) == len(ids_out) == len(seqlens) == len(mws) == 2


def test_unknown_element_does_not_drop_the_row():
    """Unknown tokens fall back to <unk> rather than raising, so the row survives."""
    out = D.process_single_row(np.array(["[Og]CC", "CCO"], dtype=object), exclude_wildcard=True)
    assert out is not None and len(out[0]) == 2


# --- warning cap ------------------------------------------------------------

def test_per_molecule_warnings_are_capped(caplog):
    """Hundreds of millions of molecules make an uncapped per-failure warning a
    multi-million-line log."""
    D._warn_counts.clear()
    with caplog.at_level(logging.WARNING, logger="pan_core.common.data_handler"):
        for i in range(D._WARN_LIMIT * 2):
            D._warn_limited("test_kind", f"failure {i}")
    # _WARN_LIMIT failures, plus the one line saying the rest are suppressed
    assert len(caplog.records) == D._WARN_LIMIT + 1
    assert D._warn_counts["test_kind"] == D._WARN_LIMIT * 2
    D._warn_counts.clear()


# --- augment_bins routing ---------------------------------------------------

def _tiny_corpus(tmp_path, n=60):
    """A few molecules whose canonical forms differ enough in length to span buckets."""
    import polars as pl
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C" * 30,
              "CCOc1ccc(cc1)C(=O)NCCc1ccccc1", "OC(=O)CCCCCCCCCCCCCCCCC"]
    rows = [smiles[i % len(smiles)] for i in range(n)]
    p = tmp_path / "corpus.parquet"
    pl.DataFrame({"smiles": rows}).write_parquet(p)
    return p


def test_augment_bins_files_every_pair_by_its_own_length(tmp_path):
    """A randomized SMILES can outgrow the bucket its canonical came from.

    The pair must then be written to the bucket that actually holds it, not truncated
    into the source bucket, and the sidecar must agree with the NPY row for row.
    """
    import polars as pl

    from pan_core.common.data_handler import raw2randomORcanonical, split_and_tokenize

    thresholds = [0, 8, 16, 32, 64]
    src = _tiny_corpus(tmp_path)
    pairs = tmp_path / "pairs.parquet"
    raw2randomORcanonical(allsmiles=pl.read_parquet(src)["smiles"].to_list(),
                          randomize=True, canonicalize=True,
                          output_path=str(pairs), max_workers=1, log_step=10**9)
    split_and_tokenize(csv_path=str(pairs), token_path=str(VOCAB_FILE),
                       thresholds=thresholds, max_length=66,
                       randomize=True, canonicalize=True, do_split=True,
                       output_dir=str(tmp_path), smiles_format="parquet",
                       max_workers=1, analyze=False, log_step=10**9)

    written = D.augment_bins(base_dir=str(tmp_path), stem="pairs",
                             token_path=str(VOCAB_FILE), thresholds=thresholds,
                             bin_num_random={1: 3, 2: 3}, sidecar_format="parquet",
                             max_workers=1, log_step=10**9)
    assert written, "augment_bins wrote nothing"

    tok = D.SMILESTokenizer(str(VOCAB_FILE))
    for j, n_rows in written.items():
        lo, hi = thresholds[j], thresholds[j + 1]
        side = pl.read_parquet(tmp_path / f"pairs_{lo}to{hi}_aug.parquet")
        assert len(side) == n_rows, f"bucket {j}: sidecar and NPY row counts differ"
        mm = np.memmap(tmp_path / f"pairs_{lo}to{hi}_aug.npy",
                       dtype=np.int16, mode="r", shape=(n_rows, 2, hi + 2))
        rows = zip(side["random"], side["canonical"], strict=True)
        for i, (rand_smi, cano_smi) in enumerate(rows):
            for axis, smi in ((0, rand_smi), (1, cano_smi)):
                ids = tok.encode(smi, add_special_tokens=True)
                assert len(ids) <= hi + 2, (
                    f"bucket {j} row {i} axis {axis}: {len(ids)} tokens do not fit {hi + 2}")
                assert list(mm[i, axis, :len(ids)]) == ids, (
                    f"bucket {j} row {i} axis {axis}: NPY does not match the sidecar")
                assert int(mm[i, axis, len(ids) - 1]) == tok.eos_token_id, (
                    f"bucket {j} row {i} axis {axis}: EOS was truncated away")


def test_augment_bins_rejects_an_out_of_range_bucket_index(tmp_path):
    with pytest.raises(ValueError, match="thresholds define"):
        D.augment_bins(base_dir=str(tmp_path), stem="x", token_path=str(VOCAB_FILE),
                       thresholds=[0, 8, 16], bin_num_random={5: 2})


# --- _bounded_imap -------------------------------------------------------------
# Pool.imap throttles only the submitting side; IMapIterator._items has no bound.
# Before the fix, 8M ZINC20 molecules at max_workers=24 on a 24-core Threadripper
# 9960X left 204 chunks (2.04M rows) of finished results in the parent and grew it
# by 2.9 GB. These two tests pin that the window is capped and that capping it
# changes nothing else.

def _double(x):
    return x * 2


def test_bounded_imap_returns_the_same_results_in_the_same_order():
    items = list(range(500))
    with multiprocessing.Pool(processes=2) as pool:
        got = list(D._bounded_imap(pool, _double, iter(items), chunksize=7, max_workers=2))
    assert got == [x * 2 for x in items]


def test_bounded_imap_stops_the_pool_running_ahead_of_the_consumer():
    workers, chunksize, n = 2, 4, 400
    permits = chunksize * (workers + 2)
    produced = []

    def counting_gen():
        for i in range(n):
            produced.append(i)
            yield i

    worst = 0
    with multiprocessing.Pool(processes=workers) as pool:
        for consumed, _ in enumerate(
            D._bounded_imap(pool, _double, counting_gen(), chunksize, workers), start=1
        ):
            worst = max(worst, len(produced) - consumed)
    # One item can be held between the generator's yield and the semaphore.
    assert worst <= permits + 1, (
        f"pool ran {worst} items ahead of the consumer, cap is {permits}")
