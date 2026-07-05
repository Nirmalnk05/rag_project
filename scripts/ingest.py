"""Run the ingestion pipeline.

Usage:
    python scripts/ingest.py                  # full pipeline (with contextual retrieval)
    python scripts/ingest.py --no-contextual  # skip the LLM contextual aug step
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant + BM25 indexes.")
    parser.add_argument("--no-contextual", action="store_true", help="Skip the contextual augmentation LLM step.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing .md documents to ingest.",
    )
    args = parser.parse_args()

    run_ingestion(docs_dir=args.docs_dir, use_contextual=not args.no_contextual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
