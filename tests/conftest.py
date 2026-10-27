"""
pytest configuration for pan-core tests.

sys.path injection ensures pan_core is always imported from this repository's src/,
regardless of what is installed system-wide, and makes the analysis library
(analysis/pan_core_analysis) importable as the analysis scripts do.

CUDA_VISIBLE_DEVICES=0  — forces single-GPU execution and avoids accidental DDP
                          initialisation inside HF Trainer during tests.
MPLBACKEND=Agg          — figures are rendered without a display.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Always import pan_core from this repo's src/, not any installed copy
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))

# Single GPU to avoid HF Trainer launching DDP in test context
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
