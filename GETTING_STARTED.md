# Getting Started

Two things live here: **how to run the app** (three ways — pick one) and **what
every file in the repo does** (one or two lines each), so you can orient yourself
without reading the whole codebase.

For feature-level docs (agent tools, ACLs, memory, caching, etc.) see the main
[README.md](README.md). For production/air-gapped/cloud deployment, see
[docs/deployment/](docs/deployment/).

---

## Part 1 — How to run

All three ways need the **embedding + reranker models** (`bge-small`,
`bge-reranker-base`) — these always run locally regardless of which LLM you use,
and download once on first ingest.

### Option A — Local machine, your own API key (Anthropic/Claude)

Fastest way to get a real, high-quality answer if you have an `ANTHROPIC_API_KEY`.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env`:
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```
```bash
docker compose up -d qdrant          # vector DB — needs Docker running
python scripts/ingest.py             # parses data/docs/ into the index
uvicorn src.api:app --reload --port 8000
```
Open **http://localhost:8000/**.

### Option B — Local machine, no API key (Ollama, fully local LLM)

Same as above but swap the provider — nothing ever leaves your machine.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env`:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
ANSWER_MODEL=llama3.1        # must be a tool-capable model for AGENT_MODE=agent
GRADER_MODEL=llama3.1
CONTEXT_AUG_MODEL=llama3.1
```
```bash
# start Ollama (native, or via this repo's docker profile)
ollama serve &                                   # native
# -- or --
docker compose --profile local-llm up -d ollama  # containerized

ollama pull llama3.1                             # or: docker exec rag-ollama ollama pull llama3.1

docker compose up -d qdrant
python scripts/ingest.py
uvicorn src.api:app --reload --port 8000
```
Open **http://localhost:8000/**.

> No tool-capable model available? Set `AGENT_MODE=pipeline` in `.env` to use the
> non-agentic corrective-RAG flow instead — works with any text model.

### Option C — Fully containerized (app + Qdrant, one command)

Nothing installed locally except Docker. Good for a clean demo or a single-box
deployment.

```bash
cp .env.example .env
# edit .env: set your provider (LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY,
# or LLM_PROVIDER=ollama pointing at OLLAMA_BASE_URL=http://ollama:11434)

docker compose up -d --build          # builds+starts qdrant + app
# if using Ollama too:
docker compose --profile local-llm up -d --build

docker exec rag-app python scripts/ingest.py
```
Open **http://localhost:8000/**.

For a **fully air-gapped** version of this (no internet at all, including model
weights) see `docker-compose.offline.yml` + [docs/deployment/on-prem-airgapped.md](docs/deployment/on-prem-airgapped.md).
For **Kubernetes/cloud**, see [docs/deployment/cloud.md](docs/deployment/cloud.md).

### Running the tests (no key, no Docker, no network)

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

### Quick sanity check, any option

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I do if pump P-100 shows alarm code 4F?"}'
```

---

## Part 2 — What each file does

### Root

| File | What it does |
|---|---|
| `README.md` | Main project doc — architecture, feature list, all env-var behaviors explained in depth. |
| `GETTING_STARTED.md` | This file. |
| `requirements.txt` | Runtime Python dependencies. |
| `requirements-dev.txt` | Extra dependencies for running the test suite (pytest, httpx mocks). |
| `pytest.ini` | Pytest config (asyncio mode, test discovery path). |
| `.env.example` | Every configuration variable the app reads, with comments — copy to `.env`. |
| `.env.offline.example` | A pre-filled `.env` profile for the fully air-gapped Ollama+local-parsing setup. |
| `.gitignore` | Excludes venvs, caches, and generated data artifacts from git. |
| `.dockerignore` | Excludes dev-only files from the Docker build context. |
| `Dockerfile` | Multi-stage build for the app image; optional `PREFETCH_MODELS` arg bakes model weights in for offline use. |
| `docker-compose.yml` | Qdrant + app services, plus an optional `local-llm` profile that adds an Ollama container. |
| `docker-compose.offline.yml` | Qdrant + Ollama + app wired for zero-internet-egress deployment. |
| `.github/workflows/ci.yml` | GitHub Actions: installs deps and runs the test suite on every push/PR. |

### `src/` — core application

| File | What it does |
|---|---|
| `config.py` | Central `Settings` object (pydantic) — every env var the app reads, plus `validate_provider()` startup checks. |
| `llm.py` | Thin async `complete()` / `stream_complete()` / `complete_sync()` shim over whichever provider is active. |
| `graph.py` | The **pipeline mode** backend: a LangGraph state machine (rewrite → retrieve → grade → generate) — corrective RAG, no tool-calling. |
| `orchestrator.py` | Routes each query to **agent mode** or **pipeline mode**; also wires in session memory, the semantic cache, and safety/faithfulness checks around either path. |
| `ingestion.py` | The batch pipeline: parse → chunk (text/table/figure aware) → contextually augment → incrementally index. Entry point for `scripts/ingest.py`. |
| `retrieval.py` | `Retriever` — hybrid BM25 + Qdrant dense search, RRF fusion, cross-encoder rerank, plus ACL and document-freshness filtering. |
| `api.py` | The FastAPI app — `/ask`, `/ask/stream` (SSE), `/feedback`, `/session/reset`, `/health`, `/ready`, `/metrics`, and serves the chat UI at `/`. |
| `memory.py` | `SessionStore` — per-session conversation history (in-process or Redis-backed), with a turn cap and TTL. |
| `access.py` | `AccessContext` + `acl_visible()` — per-user/role/site retrieval filtering derived from request headers. |
| `safety.py` | Heuristic detection of high-risk topics (electrical, LOTO, pressure, confined space, refrigerant) and the safety-notice text shown for them. |
| `faithfulness.py` | Optional LLM-judge pass that checks whether an answer's claims are actually supported by its retrieved sources. |
| `cache.py` | `SemanticCache` — serves a cached answer when a new query's embedding is cosine-close to a previous one. |
| `guardrails.py` | Input/retrieved-chunk prompt-injection screening and PII redaction; pluggable backend (heuristic, or wire in an external classifier). |
| `feedback.py` | `FeedbackStore` — appends thumbs-up/down records to a JSONL file and computes summary tallies. |
| `metrics.py` | Dependency-free Prometheus-format counters/histograms, rendered at `/metrics`. |
| `prompts.py` | Versioned prompt registry — every system prompt used by the pipeline, keyed by name and version. |
| `observability.py` | Structured JSON logging with request-ID propagation, plus an env-gated Langfuse tracing wrapper. |
| `citations.py` | `verify_citations()` — checks that `[#n]` markers in an answer point to real retrieved sources. |
| `security.py` | API-key auth check and rate limiting (in-memory token bucket, or Redis-backed fixed-window counter). |
| `eval_runner.py` | Runs the golden Q/A set through the orchestrator and scores the results with RAGAS. |

### `src/providers/` — pluggable LLM backends

| File | What it does |
|---|---|
| `base.py` | `LLMProvider` interface; shared types (`LLMResult`, `ChatResult`, `ToolCall`, `ToolSpec`) and typed errors. |
| `retry.py` | `call_with_retries()` — timeout/backoff/fallback-model wrapper used around every provider call. |
| `factory.py` | `get_provider()` — picks and caches the active provider based on `LLM_PROVIDER`. |
| `anthropic_provider.py` | Claude backend: text completion, streaming, tool-calling, and vision captioning. |
| `ollama_provider.py` | Local Ollama backend (native `/api/chat`) — same capabilities as above. |
| `openai_compat_provider.py` | Backend for any OpenAI-compatible server (vLLM, LM Studio, OpenAI itself). |

### `src/parsing/` — multi-format document ingestion

| File | What it does |
|---|---|
| `model.py` | The normalized `ParsedDocument` / `TextElement` / `TableElement` / `FigureElement` data model every parser emits. |
| `base.py` | `DocumentParser` interface and the `rows_to_markdown()` table-serialization helper. |
| `dispatcher.py` | Routes a file to the right parser by extension; implements the local→cloud fallback for low-confidence parses. |
| `local_parser.py` | Docling-based parser for PDF/Word — layout-aware, OCR for scanned pages, hands figures off for captioning. |
| `excel_parser.py` | Parses `.xlsx` sheets into table chunks, repeating the header row when a sheet is split. |
| `cloud_parser.py` | Optional LlamaParse / Azure Document Intelligence backend, used only when configured. |
| `markdown_parser.py` | Splits Markdown into text/table elements by section — used for `.md` docs and for cloud-parser output. |
| `vision.py` | `FigureCaptioner` — sends extracted images to a vision model and caches the resulting caption by image hash. |

### `src/agent/` — the tool-calling agent

| File | What it does |
|---|---|
| `loop.py` | The agent loop itself: lets the LLM call tools, executes them, feeds results back, repeats until it can answer. |
| `tools.py` | `ToolRegistry` — the actual tools (`search`, `lookup_fault_code`, `get_document_section`, `list_documents`, `query_cmms`). |
| `prompts.py` | Thin wrapper that pulls the agent's system prompt from the versioned registry in `src/prompts.py`. |
| `cmms.py` | `CMMSConnector` interface plus a `MockCMMSConnector` for offline/demo asset & work-order lookups. |

### `src/static/`

| File | What it does |
|---|---|
| `index.html` | The chat UI — a single static page (no build step) with streaming answers, a tool-call trace, and a sources panel. |

### `scripts/` — CLI entry points

| File | What it does |
|---|---|
| `ingest.py` | Run ingestion over `data/docs/` (`--rebuild`, `--no-contextual`, `--force-cloud` flags). |
| `run_eval.py` | Run the RAGAS evaluation against `eval/golden_set.json`. |
| `prefetch_models.py` | Downloads local model weights ahead of time — used to bake them into an offline-capable Docker image. |
| `reembed.py` | Forces a full index rebuild — run this after changing `EMBED_MODEL`. |
| `backup_index.py` | Copies the chunk store/manifests and triggers a Qdrant snapshot. |
| `restore_index.py` | Restores index artifacts from a `backup_index.py` output directory. |

### `tests/` — unit tests (each mirrors one `src` module; mocked LLM/Qdrant/HTTP, no network)

| File | What it tests |
|---|---|
| `test_config.py` | Settings loading and provider validation. |
| `test_retry.py` | The retry/backoff/fallback wrapper. |
| `test_provider_anthropic.py` | Anthropic provider (completion, error mapping). |
| `test_provider_ollama.py` | Ollama provider. |
| `test_provider_openai_compat.py` | OpenAI-compatible provider. |
| `test_provider_chat.py` | Tool-calling `chat()` across all three providers. |
| `test_provider_vision.py` | Vision/image-captioning support across all three providers. |
| `test_llm_shim.py` | `complete()`/`stream_complete()`/`complete_sync()` and provider factory selection. |
| `test_observability.py` | JSON log formatting and the no-op tracer behavior. |
| `test_citations.py` | Citation-marker verification logic. |
| `test_smoke.py` | Basic import/sanity checks and golden-set parsing. |
| `test_parsing_model.py` | The `ParsedDocument`/element data classes. |
| `test_vision_captioner.py` | Figure captioner caching and enable/disable logic. |
| `test_excel_parser.py` | Excel-sheet-to-table-chunk parsing, including header repeat. |
| `test_dispatcher.py` | Parser dispatch by extension and the local/cloud fallback path. |
| `test_markdown_and_cloud.py` | Markdown parser and the cloud-parser Markdown post-processing. |
| `test_security.py` | API-key auth and the in-memory rate limiter. |
| `test_rate_limit_redis.py` | The Redis-backed rate limiter. |
| `test_memory.py` | Session store (in-memory and Redis backends). |
| `test_access.py` | `AccessContext` and ACL visibility rules. |
| `test_agent_loop.py` | The agent's tool-calling loop, including history and step-limit handling. |
| `test_graph_unit.py` | The pipeline-mode LangGraph (routing, grading, guardrail screening). |
| `test_cmms.py` | The CMMS connector and `query_cmms` tool. |
| `test_ingestion_elements.py` | Element-aware chunking and the ACL/version sidecar attach logic. |
| `test_retrieval_unit.py` | Hybrid retrieval, RRF fusion, and ACL/freshness filtering. |
| `test_agent_tools.py` | The tool registry's individual tools. |
| `test_safety.py` | High-risk-topic keyword detection and notice text. |
| `test_faithfulness.py` | The LLM-judge groundedness check. |
| `test_api_agent.py` | The API in agent mode (tool trace, safety, faithfulness fields, streaming). |
| `test_cache.py` | The semantic cache's cosine matching and eviction. |
| `test_orchestrator.py` | Orchestrator dispatch and its session/cache integration. |
| `test_guardrails.py` | Injection screening, PII redaction, and the pluggable guardrails backend. |
| `test_feedback.py` | Feedback recording and summary tallies. |
| `test_retrieval_quality.py` | The CI recall@k regression gate. |
| `test_metrics.py` | Prometheus counter/histogram text rendering. |
| `test_api.py` | The API in pipeline mode (health/ready/feedback/metrics/session endpoints). |
| `test_prompts.py` | The versioned prompt registry. |
| `test_ingestion_unit.py` | Chunk hashing, manifest diffing, point IDs, and the embed-model-change guard. |

### `data/` and `eval/`

| Path | What it is |
|---|---|
| `data/docs/*.md` | Three sample maintenance documents used for local testing/demos. |
| `data/chunks.json`, `data/ingest_manifest.json`, `data/*_cache.json`, `data/acl.json`, `data/versions.json` | **Generated/optional** at ingest time — the index, incremental-ingest manifest, LLM/vision caches, and the ACL/versioning sidecars. Not checked into git. |
| `eval/golden_set.json` | Hand-written Q/A pairs scored by RAGAS in `scripts/run_eval.py`. |
| `eval/retrieval_labels.json` | Query → expected-chunk labels used by the CI retrieval-quality gate. |

### `deploy/k8s/` — example Kubernetes manifests

| File | What it does |
|---|---|
| `app-deployment.yaml` | The app Deployment — replicas, env from ConfigMap/Secret, liveness/readiness probes. |
| `app-service.yaml` | ClusterIP Service exposing the app Deployment. |
| `configmap.example.yaml` | Example non-secret configuration values. |
| `secret.example.yaml` | Example secret values (API keys) — placeholders, not real secrets. |
| `redis.yaml` | Redis Deployment + Service, for the shared rate limiter/cache across replicas. |
| `qdrant-statefulset.yaml` | Self-hosted Qdrant StatefulSet + Service (use managed Qdrant Cloud instead if you have it). |

### `docs/`

| File | What it covers |
|---|---|
| `docs/deployment/on-prem-airgapped.md` | Step-by-step guide to a fully offline, zero-internet deployment. |
| `docs/deployment/cloud.md` | Kubernetes/cloud deployment and multi-replica considerations. |
| `docs/deployment/operations.md` | Metrics, prompt versioning, embedding migration, backup/restore, queued ingestion. |
| `docs/design/hld.drawio` | High-level architecture diagram (editable in [diagrams.net](https://app.diagrams.net)). |
| `docs/design/lld.drawio` | Low-level flowcharts: per-query runtime flow and the ingestion flow. |
| `docs/superpowers/specs/…`, `docs/superpowers/plans/…` | Internal design spec and implementation plan written while building this system — historical reference. |
