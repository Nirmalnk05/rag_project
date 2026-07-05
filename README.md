# Production-Grade RAG — Reference Implementation

A reference RAG system that goes beyond the "FAISS + LLM" pilot pattern. Built to be run locally, broken, and rebuilt as a learning vehicle.

## What makes it "production-grade"

| Concern | Pilot pattern | This system |
|---|---|---|
| Retrieval | Dense embeddings + FAISS | **Hybrid: BM25 + dense (Qdrant), fused via RRF** |
| Chunking | Fixed-size or semantic | **Contextual retrieval** (LLM-augmented chunk context) |
| Reranking | None or simple | **Cross-encoder reranker** (`bge-reranker-base`) |
| Orchestration | Linear chain | **LangGraph state machine** with query rewrite + grader retry |
| Generation | Single LLM call | Streaming + **strict citation prompt** + faithfulness check |
| Evaluation | ROUGE / manual | **RAGAS** (faithfulness, answer relevancy, context precision/recall) |
| Observability | Print statements | **Langfuse tracing** (optional, env-gated) |
| Serving | Streamlit prototype | **FastAPI + Server-Sent Events** streaming |

## Architecture

```
                 ┌──────────────┐
   User query ──▶│ FastAPI /ask │
                 └──────┬───────┘
                        ▼
        ┌───────── LangGraph orchestrator ──────────┐
        │                                            │
        │   ┌─────────────────────────────┐          │
        │   │ Query rewrite + decompose   │◀──┐      │
        │   └──────────────┬──────────────┘   │      │
        │                  ▼                  │      │
        │   ┌──────────┐  ┌──────────┐       │      │
        │   │ BM25     │  │ Qdrant   │  (parallel)   │
        │   └────┬─────┘  └────┬─────┘       │      │
        │        └────┬────────┘             │      │
        │             ▼                      │      │
        │   ┌─────────────────────────────┐  │      │
        │   │ RRF fusion + cross-encoder  │  │      │
        │   └──────────────┬──────────────┘  │      │
        │                  ▼                  │      │
        │   ┌─────────────────────────────┐  │      │
        │   │ Context grader              │──┘ retry
        │   │ (loops back if insufficient)│         │
        │   └──────────────┬──────────────┘         │
        └──────────────────┼──────────────────────────┘
                           ▼
                 ┌────────────────────┐
                 │ Claude w/ citations│──▶ SSE stream
                 └────────────────────┘
```

Cross-cutting: Langfuse traces, RAGAS eval in CI, PII/safety guardrails.

## Project layout

```
production-rag/
├── src/
│   ├── config.py         # env config
│   ├── ingestion.py      # parse, contextual chunking, index
│   ├── retrieval.py      # hybrid retrieval + reranker
│   ├── graph.py          # LangGraph agentic flow
│   ├── llm.py            # Anthropic client wrapper
│   ├── api.py            # FastAPI + SSE streaming
│   └── eval_runner.py    # RAGAS evaluation
├── data/docs/            # sample maintenance documents
├── eval/golden_set.json  # eval Q/A pairs
├── scripts/
│   ├── ingest.py
│   └── run_eval.py
├── docker-compose.yml    # Qdrant
├── requirements.txt
└── .env.example
```

## Quickstart

### 1. Setup

```bash
git clone <this repo>
cd production-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY
```

### 2. Start Qdrant

```bash
docker compose up -d
# Qdrant UI at http://localhost:6333/dashboard
```

### 3. Ingest sample documents

```bash
python scripts/ingest.py
# Watch: parsing → contextual augmentation → dual indexing
```

### 4. Run the API

```bash
uvicorn src.api:app --reload --port 8000
```

Try a query:

```bash
curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I do if pump P-100 shows alarm code 4F?"}'
```

You'll see streamed tokens + a final citations block.

### 5. Run evaluation

```bash
python scripts/run_eval.py
```

Prints RAGAS scores: faithfulness, answer_relevancy, context_precision, context_recall.

## What to experiment with

Each of these is a 1-2 hour exercise that teaches a production lesson:

1. **Turn off contextual retrieval** in `src/ingestion.py` and re-run eval. Watch context_precision drop.
2. **Disable BM25** in `src/retrieval.py` and query for a part number ("P-100"). Compare to hybrid.
3. **Skip the reranker** and compare top-5 context quality manually.
4. **Force the grader to "insufficient"** once and watch the LangGraph retry loop fire.
5. **Add a new doc** to `data/docs/` and re-ingest. Confirm it's retrievable.
6. **Enable Langfuse** (`LANGFUSE_PUBLIC_KEY` in `.env`) and watch full traces in the UI.

## Honest limits of this codebase

- Uses local embedding/reranker models (`bge-small`, `bge-reranker-base`) — production would use Cohere, OpenAI, or fine-tuned domain models.
- Qdrant runs single-node — production needs replicas + backups.
- No auth on the API — production needs OAuth/JWT + per-user ACLs on retrieval.
- Guardrails are minimal — production would use NeMo Guardrails or LlamaGuard.
- No prompt versioning — production would use Langfuse Prompt Management or similar.

These are the gaps a senior interviewer will probe. Knowing them is the point.

## License

MIT. Built as a learning vehicle.
