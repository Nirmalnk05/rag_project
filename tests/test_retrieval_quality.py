"""CI retrieval-quality gate.

Runs BM25 (in-process, no Qdrant/models) over a labeled corpus and asserts
recall@k stays above threshold — a lightweight quality regression gate that runs
in CI without Docker, keys, or network. Full RAGAS scoring is a separate,
key-requiring run (scripts/run_eval.py).
"""
import json
from pathlib import Path

from src.retrieval import Retriever

RECALL_THRESHOLD = 0.75
TOP_K = 3

CORPUS = [
    ("c_pump", "pump P-100 alarm code 4F indicates low suction pressure check strainer"),
    ("c_hvac", "chiller CH-200 condenser fin cleaning every 6 weeks in coastal installations"),
    ("c_incident", "earth fault CG-04 seawater ingress degraded deck penetration cable gland winch motor"),
    ("c_bearing", "P-100 bearing alignment tolerance 0.15mm radial dial indicator"),
    ("c_seal", "P-100 mechanical seal cartridge part number MSC-100-A"),
    ("c_e24", "CH-200 fault E-24 refrigerant leak detection shut down evacuate"),
]


def _corpus_chunks():
    return [{"id": cid, "doc_id": "d", "doc_title": "T", "section": "S",
             "text": text, "context": "", "embed_text": text,
             "metadata": {"source_file": f"{cid}.md"}} for cid, text in CORPUS]


def test_retrieval_recall_gate(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(_corpus_chunks()), encoding="utf-8")
    retriever = Retriever(chunks_path=path, load_models=False, qdrant_client=object())

    labels = json.loads(Path("eval/retrieval_labels.json").read_text(encoding="utf-8"))
    hits = 0
    for item in labels:
        results = retriever.sparse_search(item["query"], top_k=TOP_K)
        got = {r.chunk_id for r in results}
        if item["expect_chunk"] in got:
            hits += 1
    recall = hits / len(labels)
    assert recall >= RECALL_THRESHOLD, f"retrieval recall@{TOP_K}={recall:.2f} < {RECALL_THRESHOLD}"
