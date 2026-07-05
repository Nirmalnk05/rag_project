"""Run the ingestion pipeline.

Usage:
    python scripts/ingest.py                  # incremental (with contextual retrieval)
    python scripts/ingest.py --rebuild        # drop the collection and re-ingest everything
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
    parser.add_argument("--rebuild", action="store_true", help="Drop the collection and re-ingest everything.")
    parser.add_argument("--force-cloud", action="store_true", help="Force cloud parsing for all PDF/Word docs.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing documents to ingest (pdf/docx/xlsx/md).",
    )
    args = parser.parse_args()

    run_ingestion(docs_dir=args.docs_dir, use_contextual=not args.no_contextual,
                  rebuild=args.rebuild, force_cloud=args.force_cloud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
