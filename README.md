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

**Conversation memory (multi-turn).** Pass a `session_id` on `/ask` (the chat UI
generates one per browser session) and prior turns are threaded into the next
answer, so follow-ups like "and how do I fix it?" resolve against earlier context.
History is capped (`MEMORY_MAX_TURNS`) with a TTL and stored in-process
(`MEMORY_BACKEND=memory`) or shared across replicas (`redis`). `POST /session/reset`
or the UI's "New chat" button clears it. Only clean user↔assistant turns are
kept — never the agent's tool-call scratch messages.

**Access control (per-user retrieval ACLs).** With `ENABLE_ACL=true`, retrieval
is filtered by the request principal (`X-User-Id` / `X-Roles` / `X-Sites` headers —
map your real SSO to these) against per-document ACLs. Documents are tagged in
`data/acl.json` (`{"restricted.pdf": {"roles": ["eng"], "sites": ["siteA"]}}`);
unlisted docs are public. A chunk is visible only if the principal satisfies every
ACL dimension it restricts. Enforced in both BM25 and dense retrieval and in the
agent's `search`/`get_document_section`/`list_documents` tools, so a user never
sees — or gets an answer grounded in — documents they aren't cleared for.

**CMMS / system-of-record tool.** With `ENABLE_CMMS=true`, the agent gets a
`query_cmms` tool for live asset status and open work orders — the data that lives
in SAP PM / Maximo / Fiix rather than in manuals. A `MockCMMSConnector`
([src/agent/cmms.py](src/agent/cmms.py)) ships for offline/demo use; it's the seam
for a real connector.

**Semantic answer cache.** With `ENABLE_SEMANTIC_CACHE=true`, a repeated or
near-identical question (cosine ≥ `SEMANTIC_CACHE_THRESHOLD` on the query
embedding) is served from cache without re-running the LLM — the response carries
`cached: true`. Deliberately skipped when ACLs are on (answers differ per
principal), when there's conversation history, or when filters are applied, to
avoid serving a wrong answer ([src/cache.py](src/cache.py)).

**Guardrails backend.** Input screening is pluggable via `GUARDRAILS_BACKEND`:
`heuristic` (default regex injection screen) or `external` — register a real
classifier (LlamaGuard / NeMo Guardrails / a moderation API) with
`guardrails.set_external_screen(fn)` at startup ([src/guardrails.py](src/guardrails.py)).

**Safety escalation.** For high-risk topics (electrical/440V, lockout-tagout,
stored pressure, confined space, refrigerant) the response carries a `safety`
notice recommending the relevant procedure and supervisor sign-off — surfaced as a
banner in the UI ([src/safety.py](src/safety.py), `ENABLE_SAFETY_ESCALATION`).
Heuristic and advisory, not a substitute for site safety systems.

**Groundedness check.** Beyond verifying `[#n]` markers exist, `ENABLE_FAITHFULNESS_CHECK`
adds an LLM-judge pass that checks each claim is actually supported by the retrieved
sources, returning `{grounded, unsupported}` on the response (a warning badge shows
in the UI when a claim isn't grounded). One extra LLM call, non-blocking
([src/faithfulness.py](src/faithfulness.py)).

**Document freshness / versioning.** Revisions are declared in `data/versions.json`
(`{"pump_v1.pdf": {"version": "1", "effective_date": "2024-01-01", "status":
"superseded"}}`). Superseded revisions are **hidden from retrieval by default** so
a technician can't be handed a stale procedure — a safety concern in maintenance;
set `RETRIEVE_SUPERSEDED=true` for historical lookups. Active revisions carry their
version + date into the cited source line (`… > Alarms, rev 3 2026-02-02`) so the
answer reflects which revision it's grounded in.
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

## Deployment

The system runs **fully air-gapped on-prem** or **horizontally-scaled on cloud**;
everything except first-time model-weight downloads works offline.

- **Air-gapped on-prem** — `LLM_PROVIDER=ollama` + `PARSE_BACKEND=local` + baked
  model weights = zero internet egress. Bring up `docker-compose.offline.yml`
  (Qdrant + Ollama + app). Full walkthrough (pre-baking weights, transferring the
  Ollama volume, running disconnected): [docs/deployment/on-prem-airgapped.md](docs/deployment/on-prem-airgapped.md).
- **Cloud (Kubernetes)** — stateless API, scaled to N replicas. Set
  `RATE_LIMIT_BACKEND=redis` so the rate limit is shared across pods; `/ready`
  gates traffic until models load. Manifests in `deploy/k8s/`; guide:
  [docs/deployment/cloud.md](docs/deployment/cloud.md).

**What needs internet (and how to avoid it):** the Anthropic/OpenAI/Azure LLM
APIs (→ use Ollama/local), LlamaParse/Azure cloud parsing (→ `PARSE_BACKEND=local`),
cloud Langfuse (→ disable or self-host), and the one-time download of the BGE and
Docling model weights (→ `scripts/prefetch_models.py` bakes them in during a
connected build, then `HF_HUB_OFFLINE=1` at runtime).

**Agentic RAG offline?** Yes — agent mode works air-gapped with a tool-capable
local model (`llama3.1`+ on Ollama); quality scales with model size.

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
