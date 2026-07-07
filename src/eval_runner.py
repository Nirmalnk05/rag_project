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

from src.config import Settings, get_settings
from src.retrieval import Retriever


GOLDEN_PATH = Path("eval/golden_set.json")


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_ragas_llm_and_embeddings(settings: Settings):
    """Map the app's configured LLM_PROVIDER to a RAGAS judge LLM, so eval
    scoring uses the same backend (Claude, Ollama, or an OpenAI-compatible
    server) as the RAG system itself, instead of RAGAS's OpenAI-only default.

    Embeddings always run locally via the same sentence-transformers model
    used for retrieval — embeddings were never tied to LLM_PROVIDER in this
    app, so there's no reason to change that for eval either.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        chat = ChatAnthropic(model=settings.answer_model,
                             api_key=settings.anthropic_api_key,
                             timeout=settings.llm_timeout_s)
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        chat = ChatOllama(model=settings.answer_model, base_url=settings.ollama_base_url)
    elif settings.llm_provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        # ChatOpenAI requires a non-empty api_key string even against local
        # servers (vLLM/LM Studio) that don't actually check it.
        chat = ChatOpenAI(model=settings.answer_model, base_url=settings.openai_base_url,
                          api_key=settings.openai_api_key or "not-needed")
    else:
        raise ValueError(f"unknown LLM_PROVIDER for RAGAS wiring: {settings.llm_provider}")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


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

    settings = get_settings()
    golden = load_golden()
    retriever = Retriever()
    orchestrator = Orchestrator(retriever)  # honors AGENT_MODE (agent or pipeline)
    ragas_llm, ragas_embeddings = _build_ragas_llm_and_embeddings(settings)

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
    print(f"[eval] scoring with RAGAS (judge: {settings.llm_provider}/{settings.answer_model})...")
    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return {"scores": result, "rows": rows}


if __name__ == "__main__":
    out = run_eval()
    print("\n========== RAGAS scores ==========")
    print(out["scores"])
    print("==================================\n")
