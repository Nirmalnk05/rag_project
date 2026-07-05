"""Run RAGAS evaluation against the golden set.

Usage:
    python scripts/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_runner import run_eval


if __name__ == "__main__":
    out = run_eval()
    print("\n========== RAGAS scores ==========")
    print(out["scores"])
    print("==================================\n")
