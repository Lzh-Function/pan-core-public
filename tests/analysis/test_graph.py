"""Unit tests for pan_core_analysis.graph."""

from __future__ import annotations

import numpy as np
import pytest

from pan_core_analysis.graph import collate_distance_matrices, create_token_graph_distance_matrix


class _MockTokenizer:
    """Minimal tokenizer mock for graph distance tests."""
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    pad_token = "<PAD>"

    _TOKEN_MAP: dict[str, list[str]] = {
        "CC":  ["C", "C"],
        "CCO": ["C", "C", "O"],
        "CC(=O)O": ["C", "C", "(", "=", "O", ")", "O"],
        "C[C@@H](N)O": ["C", "[", "C", "@", "@", "H", "]", "(", "N", ")", "O"],
    }

    def tokenize(self, smiles: str) -> list[str]:
        return self._TOKEN_MAP.get(smiles, list(smiles))


class TestGraphDistance:
    def test_ethane_shape(self):
        mat = create_token_graph_distance_matrix("CC", _MockTokenizer())
        assert mat is not None
        # tokens: [BOS, C, C, EOS] → seq_len = 4
        assert mat.shape == (4, 4)

    def test_special_tokens_are_minus_one(self):
        mat = create_token_graph_distance_matrix("CC", _MockTokenizer())
        assert mat[0, 0] == -1.0   # BOS diagonal
        assert mat[3, 3] == -1.0   # EOS diagonal

    def test_bonded_atoms_distance_one(self):
        mat = create_token_graph_distance_matrix("CC", _MockTokenizer())
        # tokens [BOS, C, C, EOS]: C at index 1, C at index 2 → bond distance = 1
        assert mat[1, 2] == 1.0

    def test_invalid_smiles_returns_none(self):
        result = create_token_graph_distance_matrix("XXXXINVALID", _MockTokenizer())
        assert result is None

    def test_collate_pads_to_max_size(self):
        a = np.zeros((3, 3))
        b = np.zeros((5, 5))
        out = collate_distance_matrices([a, b])
        assert out.shape == (2, 5, 5)
        assert out[0, 4, 4] == -1.0  # padding for shorter matrix

    def test_collate_none_entry_is_all_pad(self):
        out = collate_distance_matrices([None, np.ones((3, 3))], maxlen=3)
        assert out.shape == (2, 3, 3)
        assert (out[0] == -1.0).all()

    def test_collate_all_none_raises(self):
        with pytest.raises(ValueError):
            collate_distance_matrices([None, None])

    def test_collate_explicit_maxlen(self):
        out = collate_distance_matrices([np.zeros((2, 2))], maxlen=5)
        assert out.shape == (1, 5, 5)
