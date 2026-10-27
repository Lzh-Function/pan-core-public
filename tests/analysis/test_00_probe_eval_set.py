"""Tests for analysis 00 (probe evaluation set): chirality-token labels and the balanced R/S set."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.utils import load_script, run_script

_PROBE = load_script("analysis/scripts/compute/00_probe_eval_set.py")

# Two buckets of four molecules: R, S, achiral, two stereocentres / R, S, unassigned, invalid
_SOURCE = pd.DataFrame(
    {
        "canonical": [
            "C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", "CCO", "C[C@H](N)[C@@H](C)O",
            "C[C@H](O)CC", "C[C@@H](O)CC", "CC(N)C(=O)O", "C1CC",
        ],
        "random": [
            "N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O", "OCC", "N[C@@H](C)[C@@H](C)O",
            "CC[C@@H](C)O", "CC[C@H](C)O", "NC(C)C(=O)O", "C1CC",
        ],
    },
    index=pd.Index([101, 102, 103, 104, 105, 106, 107, 108], name="id"),
)


@pytest.mark.parametrize(
    "smiles, token, token_id",
    [
        ("C[C@H](N)C(=O)O", "@", 0),
        ("C[C@@H](N)C(=O)O", "@@", 1),
        ("CCO", None, -1),
        ("C[C@H](N)[C@@H](C)O", None, -1),   # '@' and '@@'
        ("C[C@H](N)[C@H](C)O", None, -1),    # two '@'
    ],
)
def test_chiral_token_requires_exactly_one_mark(smiles, token, token_id):
    assert _PROBE.extract_chiral_token(smiles) == token
    assert _PROBE.chiral_token_id(token) == token_id


def test_probe_set_keeps_single_assigned_centres_and_balances_r_and_s():
    final, summary = _PROBE.build_probe_set(_SOURCE, seed=42, max_per_class=10, n_buckets=2)

    assert sorted(final["mol_id"]) == [101, 102, 105, 106]
    assert final["rs_label_id"].value_counts().to_dict() == {0: 2, 1: 2}
    assert (final["rs_label_id"] == final["rs_label"].map({"R": 0, "S": 1})).all()
    assert final.set_index("mol_id")["bucket"].to_dict() == {101: 1, 102: 1, 105: 2, 106: 2}
    assert summary["n_achiral_removed"] == 1
    assert summary["n_invalid_removed"] == 1
    assert summary["n_multi_chiral_removed"] == 1
    assert summary["n_unassigned_removed"] == 1
    assert summary["buckets"]["1"] == {"n_total_in_bucket": 4, "n_valid_R": 1, "n_valid_S": 1, "n_used_R": 1, "n_used_S": 1}


def test_probe_set_caps_each_class_at_max_per_class():
    final, summary = _PROBE.build_probe_set(_SOURCE, seed=42, max_per_class=1, n_buckets=2)

    assert len(final) == 2
    assert summary["n_R"] == summary["n_S"] == 1


def test_probe_set_rejects_unequal_buckets():
    with pytest.raises(ValueError, match="equal buckets"):
        _PROBE.build_probe_set(_SOURCE.iloc[:7], seed=42, max_per_class=10, n_buckets=2)


def test_main_writes_probe_csv_and_summary(tmp_path):
    source = tmp_path / "source.csv"
    _SOURCE.to_csv(source)

    run_script("analysis/scripts/compute/00_probe_eval_set.py",
               {"source_csv": str(source), "output_dir": str(tmp_path / "out"), "n_buckets": 2},
               tmp_path / "config.yml")

    probe = pd.read_csv(tmp_path / "out" / "single_chiral_rs_probe_4.csv")
    summary = json.loads((tmp_path / "out" / "single_chiral_rs_probe_4_summary.json").read_text())
    assert len(probe) == summary["n_total"] == 4
    assert {"rs_label_id", "canonical_chiral_token_id", "random_chiral_token_id", "random"} <= set(probe.columns)
