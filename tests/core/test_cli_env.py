"""The CLI turns on expandable segments for the CUDA caching allocator by default."""

import os
import sys

import pytest

from pan_core.main import main

KEY = "PYTORCH_CUDA_ALLOC_CONF"


def _run_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pan-core", "--help"])
    with pytest.raises(SystemExit):
        main()


def test_default_is_expandable_segments(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    _run_help(monkeypatch)
    assert os.environ[KEY] == "expandable_segments:True"


def test_an_explicit_setting_is_kept(monkeypatch):
    monkeypatch.setenv(KEY, "max_split_size_mb:256")
    _run_help(monkeypatch)
    assert os.environ[KEY] == "max_split_size_mb:256"
