"""Sous Chef lab — docs/LAB.md.

Standard library only at import time. The modules that need torch/unsloth
(train, dpo, export) import them inside their run() functions.
"""

from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = LAB_DIR / "fixtures"
DATA_DIR = LAB_DIR / "data"
RUNS_DIR = LAB_DIR / "runs"

__all__ = ["LAB_DIR", "FIXTURES_DIR", "DATA_DIR", "RUNS_DIR"]
