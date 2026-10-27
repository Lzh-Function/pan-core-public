"""Shared paths and helpers for the test suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "sample" / "data" / "normal_tokens.txt"


def load_script(relative_path: str) -> ModuleType:
    """Import a script by its path relative to the repository root.

    Analysis scripts are not importable by name (their filenames start with digits).
    """
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(relative_path: str, config: dict, config_path: Path) -> ModuleType:
    """Write ``config`` to ``config_path`` and run the script's ``main()`` as ``--config config_path``."""
    config_path.write_text(yaml.safe_dump(config))
    module = load_script(relative_path)
    argv = sys.argv
    sys.argv = [str(REPO_ROOT / relative_path), "--config", str(config_path)]
    try:
        module.main()
    finally:
        sys.argv = argv
    return module
