"""Re-embed the whole corpus (run after changing EMBED_MODEL).

    python scripts/reembed.py

Forces a full rebuild so every chunk is re-embedded with the current model and
the Qdrant collection is recreated. Equivalent to `scripts/ingest.py --rebuild`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import run_ingestion

if __name__ == "__main__":
    run_ingestion(rebuild=True)
