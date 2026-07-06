"""Pre-download local model weights into the HF cache for air-gapped deploys.

Run this in a CONNECTED environment (or during a connected Docker build) so the
BGE embedding + reranker models and Docling's layout/OCR models are cached on
disk. At runtime in an air-gapped host, set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
so the libraries load from cache and never reach the network.

    python scripts/prefetch_models.py

Note: this does NOT fetch the LLM — pull that separately with `ollama pull <model>`
and transfer the Ollama volume.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings


def main() -> int:
    s = get_settings()

    print(f"[prefetch] embedding model: {s.embed_model}")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(s.embed_model)

    print(f"[prefetch] reranker model: {s.rerank_model}")
    from sentence_transformers import CrossEncoder
    CrossEncoder(s.rerank_model)

    print("[prefetch] docling layout/OCR models…")
    try:
        from docling.document_converter import DocumentConverter
        DocumentConverter()  # triggers model download on first init
    except Exception as e:  # docling optional if only Markdown/Excel are used
        print(f"[prefetch] docling prefetch skipped: {e}")

    print("[prefetch] done. For runtime, set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
