"""Tests for visualization 18 (sequence length vs molecular weight)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.utils import load_script, run_script

_SCRIPT = "analysis/scripts/visualize/18_seqlen_vs_mw.py"
_SEQLEN_VIS = load_script(_SCRIPT)


@pytest.mark.parametrize("quantile, label", [(0.90, "90%"), (0.99, "99%"), (0.999, "99.9%")])
def test_quantile_labels(quantile, label):
    assert _SEQLEN_VIS._label(quantile) == label


def test_main_writes_both_hexbins(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("zinc", "pubchem"):
        seqlen = rng.integers(10, 400, size=200)
        pd.DataFrame({"seqlen": seqlen, "mw": seqlen * 7.5 + rng.normal(0, 20, size=200), "smiles": "C"}).to_csv(
            tmp_path / f"{name}_summary.csv", index=False)
    config = {
        "zinc_summary": str(tmp_path / "zinc_summary.csv"),
        "pubchem_summary": str(tmp_path / "pubchem_summary.csv"),
        "output_dir": str(tmp_path / "figures"),
    }
    run_script(_SCRIPT, config, tmp_path / "config.yml")

    for stem in ("zinc_seqlen_mw_hexbin", "pubchem_seqlen_mw_hexbin_zoomed"):
        for extension in ("png", "svg"):
            assert (tmp_path / "figures" / f"{stem}.{extension}").is_file(), f"{stem}.{extension}"
