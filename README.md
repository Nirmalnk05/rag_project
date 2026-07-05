# Production-Grade RAG — Reference Implementation

A reference agentic RAG system that goes beyond the "FAISS + LLM" pilot pattern.
Runs on **Claude (Anthropic) or a fully local LLM** (Ollama / any OpenAI-compatible
server). Built to be run locally, broken, and rebuilt as a learning vehicle.

## What makes it "production-grade"

| Concern | Pilot pattern | This system |
|---|---|---|
| Retrieval | Dense embeddings + FAISS | **Hybrid: BM25 + dense (Qdrant), fused via RRF, run in parallel** |
| Input formats | Markdown / plain text | **PDF (incl. scanned+OCR), Word, Excel, Markdown** — tables kept intact, figures vision-captioned |
| Chunking | Fixed-size or semantic | **Contextual retrieval** (LLM-augmented chunk context) |
| Reranking | None or simple | **Cross-encoder reranker** (`bge-reranker-base`) |
| Orchestration | Linear chain | **Tool-calling agent** (LLM reasons + calls tools) or corrective-RAG pipeline — switchable |
| LLM backend | One vendor, hard-coded | **Pluggable: Anthropic / Ollama / OpenAI-compatible**, with retry + fallback |
| Generation | Single LLM call | Streaming + **strict citation prompt** + **post-hoc citation verification** |
| Ingestion | Drop & rebuild | **Incremental** (content-hash manifest, resumable augmentation cache) |
| Guardrails | None | **Prompt-injection screening** (input + retrieved chunks) + **PII redaction** |
| Security | Open endpoint | **API-key auth + token-bucket rate limiting** |
| Evaluation | ROUGE / manual | **RAGAS** (faithfulness, answer relevancy, context precision/recall) |
| Observability | Print statements | **JSON logs + request IDs + Langfuse tracing** (wired, env-gated) |
| Serving | Streamlit prototype | **FastAPI + SSE streaming + minimal chat UI** |

## Architecture

```
                 ┌──────────────────────────┐
   User query ──▶│ FastAPI  /  · /ask · /ask/stream
                 │ (auth · rate limit · request-id · guardrail)
                 └──────────────┬───────────┘
                                ▼
        ┌───────── LangGraph orchestrator ──────────┐
        │                                            │
        │   ┌─────────────────────────────┐          │
        │   │ Query rewrite               │◀──┐      │
        │   └──────────────┬──────────────┘   │      │
        │                  ▼                  │      │
        │   ┌──────────┐  ┌──────────┐  (parallel)   │
        │   │ BM25     │  │ Qdrant   │       │      │
        │   └────┬─────┘  └────┬─────┘       │      │
        │        └────┬────────┘             │      │
        │             ▼                      │      │
        │   ┌─────────────────────────────┐  │      │
        │   │ RRF fusion → cross-encoder  │  │      │
        │   │ → injection screen          │  │      │
        │   └──────────────┬──────────────┘  │      │
        │                  ▼                  │      │
        │   ┌─────────────────────────────┐  │      │
        │   │ Context grader              │──┘ retry
        │   │ (loops back if insufficient)│         │
        │   └──────────────┬──────────────┘         │
        └──────────────────┼──────────────────────────┘
                           ▼
                 ┌────────────────────────┐
                 │ LLM w/ citations       │──▶ SSE stream
                 │ → citation verification│    → Chat UI
                 └────────────────────────┘
```

Cross-cutting: JSON logs + request IDs, Langfuse traces (env-gated), RAGAS eval,
guardrails, API-key auth + rate limiting.

## Project layout

```
production-rag/
├── src/
│   ├── config.py           # env config + provider validation
│   ├── providers/          # LLM providers: anthropic, ollama, openai-compat
│   │   ├── base.py         #   interface + typed errors + LLMResult
│   │   ├── retry.py        #   backoff/timeout/fallback wrapper
│   │   └── factory.py      #   provider selection
│   ├── llm.py              # async complete()/stream_complete() shim
│   ├── ingestion.py        # parse, contextual chunking, incremental index
│   ├── retrieval.py        # hybrid retrieval + reranker + filters
│   ├── graph.py            # LangGraph agentic flow (async)
│   ├── guardrails.py       # injection screening + PII redaction
│   ├── citations.py        # post-generation citation verification
│   ├── security.py         # API-key auth + rate limiting
│   ├── observability.py    # JSON logs, request IDs, Langfuse
│   ├── api.py              # FastAPI + SSE streaming
│   ├── static/index.html   # minimal chat UI
│   └── eval_runner.py      # RAGAS evaluation
├── data/docs/              # sample maintenance documents
├── eval/golden_set.json    # eval Q/A pairs
├── scripts/{ingest,run_eval}.py
├── tests/                  # unit tests (mocked LLM/Qdrant/httpx)
├── Dockerfile · docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt · requirements-dev.txt · .env.example
```

## Quickstart

### 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start Qdrant

```bash
docker compose up -d qdrant     # Qdrant UI at http://localhost:6333/dashboard
```

### 3a. Run with Claude (Anthropic)

```bash
# .env: LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY=sk-ant-...
python scripts/ingest.py
uvicorn src.api:app --reload --port 8000
```

### 3b. Run with a local LLM (Ollama) — no API key, fully offline

```bash
ollama pull qwen2.5:7b
# .env:
#   LLM_PROVIDER=ollama
#   ANSWER_MODEL=qwen2.5:7b
#   GRADER_MODEL=qwen2.5:7b
#   CONTEXT_AUG_MODEL=qwen2.5:7b
python scripts/ingest.py
uvicorn src.api:app --reload --port 8000
```

> Other OpenAI-compatible servers (vLLM, LM Studio) work too: set
> `LLM_PROVIDER=openai_compatible` and `OPENAI_BASE_URL`. The embedding and
> reranker models always run locally regardless of provider.

### 4. Open the chat UI

Visit **http://localhost:8000/** — ask a question, watch tokens stream, and see
the rewritten query, grade verdict, source cards, and clickable `[#n]` citations.
Or hit the API directly:

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I do if pump P-100 shows alarm code 4F?"}'
```

### 5. Run evaluation

```bash
python scripts/run_eval.py   # RAGAS: faithfulness, answer_relevancy, context precision/recall
```

## Chat UI

A single static page (`src/static/index.html`, no build step) served at `/`:
streaming answer, a metadata line showing the rewrite + grade, a sources panel
with per-chunk scores, and citation markers that highlight their source card.
A warning badge appears when citation verification finds orphaned references.
It is **demo-grade** — dark theme, laptop-oriented, no mobile/accessibility work.

## Orchestration modes

Set `AGENT_MODE` to choose how a query is answered:

- **`agent` (default) — tool-calling agentic RAG.** The LLM is given tools
  (`search`, `lookup_fault_code`, `get_document_section`, `list_documents`) and
  decides which to call, how many times, and when it has enough to answer. It
  reasons over a multi-step loop, retrieval is an action the model chooses (not a
  fixed step), and `[#n]` citations reference a running list of the sources it
  actually pulled. The `/ask/stream` SSE trace shows each `tool_call`/`tool_result`
  live, and `/ask` returns a `steps` array. **Requires a tool-capable model**:
  any recent Claude, Ollama `llama3.1`+, or an OpenAI-compatible server that
  supports `tools`. Tools live in [src/agent/tools.py](src/agent/tools.py); the
  loop in [src/agent/loop.py](src/agent/loop.py).
- **`pipeline` — corrective RAG (CRAG).** The original fixed LangGraph state
  machine: rewrite → hybrid retrieve → LLM grader → conditional retry → generate.
  No tool-calling; works with any text model. Kept for comparison and as a
  fallback ([src/graph.py](src/graph.py)).

Both modes share retrieval, guardrails, citation verification, observability,
auth, and the chat UI; the [orchestrator](src/orchestrator.py) dispatches by mode.
Agent mode's final answer is streamed as chunked tokens (genuine per-token
streaming of an agent's tool-using final turn is a documented follow-up).

## Ingestion (multi-format, incremental)

Drop `.pdf`, `.docx`, `.xlsx`, or `.md` files into `data/docs/` and run
`python scripts/ingest.py`. A parsing layer (`src/parsing/`) normalizes every
format into a stream of typed elements — **text, tables, figures** — which then
flow through the existing chunking + indexing pipeline:

- **PDF / Word** → Docling (layout-aware; built-in OCR for scanned pages).
- **Excel** → each sheet becomes table chunks; large sheets are split by
  row-group with the **header row repeated** so lookup matrices stay readable.
- **Tables** become **atomic chunks** (never split mid-table).
- **Figures/diagrams** are sent to a vision model (`VISION_MODEL`) for a
  searchable caption; captions cache by image hash. Off by default until you set
  a vision model — then wiring diagrams and HMI screens become retrievable.

Parsing backend is controlled by `PARSE_BACKEND`:
- `local` — Docling only, fully offline.
- `hybrid` (default) — local first; documents that parse poorly (below
  `CLOUD_PARSE_MIN_CHARS`) fall back to a cloud parser if one is configured.
- `cloud` — always use the cloud parser (LlamaParse or Azure Document
  Intelligence). Cloud deps are an optional install; no key → stays local.

Ingestion is incremental: `data/chunks.json` is the single source of truth (BM25
rebuilds from it in memory at startup); contextual blurbs and figure captions are
cached by content hash, so an interrupted run resumes without repaying LLM/vision
calls. Flags: `--rebuild` (drop + re-ingest), `--no-contextual` (skip
augmentation), `--force-cloud` (route all PDF/Word to the cloud parser).

Retrieval can filter by `element_type` (`text`/`table`/`figure`), `page`,
`source_file`, `doc_title`, or `section` via the `filters` field on `/ask`.

## Security & guardrails

- Set `API_KEYS=key1,key2` to require an `X-API-Key` header (empty = open, for
  local dev); `/health` is always open. `RATE_LIMIT_RPM` caps requests per key/IP.
- Queries and retrieved chunks are screened for prompt-injection patterns;
  flagged chunks are dropped from the generation context and logged.
- PII (emails, phone numbers) is redacted in logs always, and in the `/ask`
  answer when `REDACT_PII_IN_ANSWERS=true`. Note: streamed tokens are **not**
  redacted mid-flight.

## Observability

Structured JSON logs carry a `request_id` (propagated from an inbound
`X-Request-ID` or generated) on every line and in every response. Set
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to emit one trace per request with
spans for rewrite/retrieve/grade/generate incl. model + token usage; without
keys it is a zero-overhead no-op.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q      # no API key, no Docker, no network required
```

Unit tests mock the LLM, Qdrant, and HTTP transports, covering RRF, chunking,
grader parsing, retry/backoff/fallback, all three providers, guardrails, citation
verification, auth + rate limiting, ingest manifest diffing, and the API. CI runs
them on every push/PR (`.github/workflows/ci.yml`).

## What to experiment with

1. Turn off contextual retrieval (`--no-contextual`) and re-run eval — watch context_precision drop.
2. Query a part number ("P-100") and compare BM25-heavy vs dense results.
3. Force the grader to "insufficient" once and watch the LangGraph retry loop fire.
4. Swap `LLM_PROVIDER` between anthropic and ollama and compare answers.
5. Add a doc to `data/docs/`, re-ingest — only the new chunks embed.
6. Enable Langfuse and inspect per-node traces.

## Honest limits (the remaining gaps)

This round hardened the pipeline; it deliberately does **not** yet include:

- **Agentic depth**: the agent calls tools and does multi-step retrieval, but
  there's still no conversation memory (single-turn), no query routing/intent
  classification, and no long-horizon planning across turns.
- **Answer self-check** beyond citation verification (no post-hoc faithfulness loop).
- **Data layer**: multi-format ingestion covers PDF/scanned/Word/Excel/Markdown,
  but there are still no per-user retrieval ACLs and no document-revision/freshness
  model; engineering-drawing (CAD/P&ID) and email/ticket sources aren't parsed.
- **Scale**: single-node Qdrant, in-process BM25 (~100k chunks), single-instance
  rate limiter (move to Redis for multi-replica); guardrails are heuristic
  (integration point for NeMo Guardrails / LlamaGuard).
- **MLOps**: no prompt versioning/A-B testing, no online eval / feedback capture,
  no index backup/restore, no semantic cache.

Knowing these — and where in the code they'd plug in — is the point.

## License

MIT. Built as a learning vehicle.
