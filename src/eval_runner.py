"""Offline RAGAS evaluation against a golden Q/A set.

Metrics:
- faithfulness: does the answer match the retrieved context (no hallucination)?
- answer_relevancy: does the answer address the question?
- context_precision: are the top-ranked retrieved chunks the relevant ones?
- context_recall: do the retrieved chunks cover what's needed for the ground truth?
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from src.retrieval import Retriever


GOLDEN_PATH = Path("eval/golden_set.json")


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval() -> dict:
    """Run the system over the golden set, then score with RAGAS."""
    # Lazy import to avoid pulling ragas deps on every API import
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    import asyncio

    from src.orchestrator import Orchestrator

    golden = load_golden()
    retriever = Retriever()
    orchestrator = Orchestrator(retriever)  # honors AGENT_MODE (agent or pipeline)

    rows = []
    for i, item in enumerate(golden, 1):
        print(f"[eval] {i}/{len(golden)}: {item['question'][:60]}...")
        out = asyncio.run(orchestrator.run(item["question"]))
        contexts = [r["text"] for r in out.get("retrieval", [])]
        rows.append(
            {
                "user_input": item["question"],
                "response": out.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )

    ds = Dataset.from_list(rows)
    print("[eval] scoring with RAGAS...")
    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    return {"scores": result, "rows": rows}


if __name__ == "__main__":
    out = run_eval()
    print("\n========== RAGAS scores ==========")
    print(out["scores"])
    print("==================================\n")
