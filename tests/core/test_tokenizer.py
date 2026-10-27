"""Regression tests for the SMILES tokenizer API."""

import pytest

from pan_core.common.tokenizer import SMILESTokenizer
from tests.utils import VOCAB_FILE


def test_build_inputs_with_special_tokens_for_single_sequence():
    """A single sequence must retain the existing BOS/EOS behavior."""
    tokenizer = SMILESTokenizer(vocab_file=str(VOCAB_FILE))

    result = tokenizer.build_inputs_with_special_tokens([3, 4, 5])

    assert result == [tokenizer.bos_token_id, 3, 4, 5, tokenizer.eos_token_id]


def test_build_inputs_with_special_tokens_rejects_sequence_pair():
    """Unsupported sequence pairs must fail at the tokenizer API boundary."""
    tokenizer = SMILESTokenizer(vocab_file=str(VOCAB_FILE))

    with pytest.raises(NotImplementedError, match="does not support sequence pairs"):
        tokenizer.build_inputs_with_special_tokens([3, 4], [5, 6])
