"""Smoke test: verifies the project structure and key imports.

Run with: python -m pytest tests/ -v
Or just: python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_loads():
    """Config should load if ANTHROPIC_API_KEY is set."""
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"  # bypass for import test
    from src.config import get_settings
    s = get_settings()
    assert s.embed_model
    assert s.top_k_rerank > 0


def test_graph_imports():
    """LangGraph wiring should be importable without running."""
    from src.graph import build_state_graph, run_sync, stream_answer  # noqa: F401


def test_retrieval_imports():
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
    from src.retrieval import Retriever, Retrieved  # noqa: F401


def test_ingestion_imports():
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
    from src.ingestion import run_ingestion, Chunk  # noqa: F401


def test_golden_set_parses():
    import json
    with open("eval/golden_set.json") as f:
        golden = json.load(f)
    assert len(golden) >= 5
    for item in golden:
        assert "question" in item and "ground_truth" in item


if __name__ == "__main__":
    test_config_loads()
    test_graph_imports()
    test_retrieval_imports()
    test_ingestion_imports()
    test_golden_set_parses()
    print("OK - all smoke tests passed")
