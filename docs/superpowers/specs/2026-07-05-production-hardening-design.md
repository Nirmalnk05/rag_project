# Production Hardening + Local LLM Support — Design Spec

**Date:** 2026-07-05
**Status:** Approved scope: "Core hardening" + three cheap adds (PII scrubbing, CI, prompt caching/fallback) + minimal chat UI
**Approach:** A — incremental refactor in place (module layout preserved)

## Goal

Fix all implementation defects found in code review, close the highest-value
production-readiness gaps, and make the system run fully offline on a local
LLM — while keeping the repo recognizable as a learning vehicle.

**Verification standard:** unit tests with mocked LLM/Qdrant/httpx. End-to-end
runs (Docker Qdrant + Ollama/Anthropic) are documented steps for the user, not
part of automated verification.

## Out of scope (deliberately)

Tool-using agent mode, query routing/intent classification, multi-query
decomposition, conversation memory, post-generation faithfulness loop (beyond
citation verification), semantic caching, PDF/Word/OCR parsing, per-user
retrieval ACLs, OpenSearch/native sparse vectors, online eval/user feedback,
eval-set expansion, prompt versioning, A/B testing, index backup/restore,
doc-revision freshness metadata, workflow orchestration, multi-node infra.
The README must not claim any of these.

---

## Section 1 — LLM provider layer (`src/providers/`)

New package replacing the Anthropic-only `src/llm.py`:

- `base.py` — `LLMProvider` abstract interface:
  - `async complete(prompt, *, system, model, max_tokens, temperature) -> LLMResult`
    (`LLMResult` carries `text`, `input_tokens`, `output_tokens`, `model`)
  - `async stream(prompt, *, system, model, max_tokens, temperature) -> AsyncIterator[str]`
  - Typed exceptions: `LLMError` (terminal), `LLMRetryableError` (transient).
- `anthropic_provider.py` — current behavior via `anthropic` SDK.
  - Adds `cache_control: ephemeral` on system prompts (prompt caching).
- `ollama_provider.py` — native Ollama `/api/chat` via `httpx` (stream and
  non-stream). No new SDK dependency.
- `openai_compat_provider.py` — `/v1/chat/completions` via `httpx` with
  configurable `base_url` and optional API key. Covers vLLM, LM Studio,
  OpenAI, Azure-compatible gateways.
- `factory.py` — `get_provider()` selected by `LLM_PROVIDER` env
  (`anthropic` | `ollama` | `openai_compatible`).
- `retry.py` — hand-rolled wrapper (~30 lines): per-call timeout, exponential
  backoff with jitter, max attempts from config. On exhaustion, if
  `fallback_model` is configured, one final attempt with the fallback model;
  otherwise raise `LLMError`.

Config changes (`src/config.py`):
- `llm_provider: str = "anthropic"`
- `anthropic_api_key` becomes optional; validated as required only when
  `llm_provider == "anthropic"` (fail fast at startup with a clear message).
- `ollama_base_url: str = "http://localhost:11434"`
- `openai_base_url: str = "http://localhost:8080/v1"`, `openai_api_key: str = ""`
- `llm_timeout_s: float = 60`, `llm_max_attempts: int = 3`,
  `fallback_model: str = ""` (empty = disabled)
- Per-role models (`answer_model`, `grader_model`, `context_aug_model`)
  unchanged in shape; values are interpreted by the active provider.

`src/llm.py` is kept as a thin compatibility shim delegating to the provider
factory: `complete()` and `stream_complete()` keep their signatures (plus an
added return-type upgrade to `LLMResult` for `complete`), so `ingestion.py`,
`graph.py`, and any user code keep working with minimal diff.

## Section 2 — Graph & API async correctness

- All LangGraph nodes become `async`; graph runs via `ainvoke`/`astream`.
  CPU-bound work (embedding, rerank) runs via `asyncio.to_thread`.
- **Single graph definition**, built **once** at startup and stored on app
  state. The duplicated hand-wired streaming graph in `stream_answer` is
  deleted; streaming uses the same compiled graph with
  `interrupt_before=["generate"]`, then streams generation tokens separately.
- Sparse and dense retrieval run in **parallel** (`asyncio.gather`).
- Grader prompt includes **both** original and rewritten query.
- Generation continues to answer the **original** query (documented decision).
- SSE: mid-stream failures emit `{"type": "error", "message", "request_id"}`
  then `[DONE]`. Sync endpoint gets FastAPI exception handlers returning
  structured JSON errors (`{"error", "request_id"}`) with correct status codes
  (502 for upstream LLM failure, 422 validation, 401/429 from security layer).
- Graceful degradation: grader LLM failure after retries → treated as
  "insufficient, no retry" (proceed to generate with low-confidence note);
  rewrite failure → fall back to original query; generation failure → error
  to client. No silent empty answers.
- `/ask` request model gains optional `filters: dict[str, str]` (see §3).
- README architecture section updated: "decompose" claim removed.

## Section 3 — Retrieval & ingestion

Retrieval (`src/retrieval.py`):
- BM25 pickle **removed**. `data/chunks.json` is the single source of truth;
  `Retriever.__init__` rebuilds `BM25Okapi` in memory from it (documented
  scaling limit: fine to ~100k chunks).
- `qdrant.search()` → `query_points()` (current API).
- Metadata filtering: optional `filters` (exact-match on `source_file`,
  `doc_title`, `section`) applied as Qdrant payload filter and BM25
  post-filter.
- Thread safety: one `threading.Lock` each around embedder and reranker
  `predict/encode` calls; calls always dispatched via `asyncio.to_thread`.

Ingestion (`src/ingestion.py`):
- **Incremental**: `data/ingest_manifest.json` maps chunk-id → content hash.
  - Unchanged chunk: skipped entirely (no LLM augmentation call, no upsert).
  - Changed/new chunk: augmented, upserted.
  - Stale chunk (id in manifest, absent from current build): point deleted
    from Qdrant, removed from manifest and chunk store.
- Context blurbs cached by chunk content hash (`data/context_cache.json`) so
  an interrupted ingest resumes without repaying LLM calls.
- Qdrant point IDs: deterministic `uuid5(NAMESPACE, chunk_id)` so re-ingest
  upserts in place. Collection created if missing, never dropped by default
  (`--rebuild` flag on `scripts/ingest.py` forces the old drop-and-recreate).
- Contextual augmentation goes through the provider layer (works offline).

## Section 4 — Production features

Observability (`src/observability.py`):
- JSON structured logging via stdlib `logging` + custom formatter. Every log
  line carries `request_id` (contextvar), event name, latency where relevant.
- Request-ID middleware: honors inbound `X-Request-ID` else generates one;
  echoed in responses and SSE `final`/`error` events.
- Langfuse wired for real, env-gated exactly as `tracing_enabled` promises:
  one trace per request; spans for rewrite/retrieve/grade/generate with
  model, token usage (from `LLMResult`), and latency. No-op stub when keys
  absent — zero overhead, no import errors.

Guardrails (`src/guardrails.py`):
- Input: length/control-character validation plus heuristic injection
  patterns ("ignore previous instructions", role-play overrides, etc.) →
  reject with 400.
- Retrieved chunks: same injection screen; flagged chunks are dropped from
  the generation context and logged (`chunk_flagged` event). Documented as
  the integration point for NeMo Guardrails / LlamaGuard.
- PII: regex pass (emails, phone numbers, national-ID-like patterns) —
  redacted in logs always; optionally redacted in answers when
  `redact_pii_in_answers=true` (default false).

Citations (`src/citations.py`):
- Parse `[#n]` references from the generated answer.
- Verify: every reference resolves to a retrieved chunk index; flag answers
  with zero citations that contain factual sentences (heuristic: length
  threshold + non-refusal).
- Attach to response: `citations: {valid: bool, used: [n...], orphaned: [n...]}`.
  Non-blocking — clients decide policy.

Security (`src/security.py`):
- `X-API-Key` header checked against comma-separated `API_KEYS` env.
  Unset ⇒ auth disabled (local dev). `/health` always open.
- In-memory token-bucket rate limiter per key (per-IP when auth disabled),
  `RATE_LIMIT_RPM` env, default 30. Documented single-instance limitation.

## Section 5 — Packaging, tests, docs, CI

- `Dockerfile`: multi-stage, `python:3.12-slim`, non-root user, model cache
  layer; app service added to `docker-compose.yml` next to Qdrant.
- `.github/workflows/ci.yml`: pytest on push/PR (mocked tests only — no
  Docker, no API keys, no model downloads; heavyweight imports mocked).
- Unit tests (pytest, `tests/`): RRF math; section splitting + windowed
  chunking; grader JSON fallback parsing; routing (`_route_after_grade` +
  retry decrement); provider request/response mapping for all three providers
  (mocked httpx/SDK); retry/backoff/fallback wrapper; citation verifier;
  guardrail patterns (positive + negative cases); PII redaction; auth +
  rate-limit dependencies; ingest manifest diffing (new/changed/stale).
  Existing smoke tests kept.
- README rewritten to match reality: corrected comparison table (no claimed-
  but-absent features), local-LLM quickstart (Ollama example incl. model
  pull), provider matrix, updated architecture diagram, honest-limits section
  refreshed to the *new* remaining gaps (the out-of-scope list above).
- `.env.example` updated with every new setting.

## Section 6 — Minimal chat UI

A single static page served by the same FastAPI app — demo/debug frontend,
not a product UI. No build step, no framework, no new Python dependencies.

- Files: `src/static/index.html` (+ inline or adjacent `app.js`/`style.css`),
  mounted at `GET /` via FastAPI `StaticFiles`/`FileResponse`.
- Chat transcript with **streaming tokens** rendered as they arrive.
  Because `/ask/stream` is a POST SSE endpoint, the client uses
  `fetch()` + `ReadableStream` line-parsing of `data:` events (not
  `EventSource`, which is GET-only). No API change needed.
- Pipeline transparency panel per answer: rewritten query, grade verdict +
  reason, and the retrieved sources with `doc_title > section` and scores —
  the point is making retrieval quality visible for the learning use case.
- Citations: `[#n]` markers in the answer are clickable and highlight the
  corresponding source card; the `citations` validity object from §4 is
  surfaced (e.g., a warning badge on orphaned citations).
- Optional metadata filters (§3) exposed as simple inputs.
- If auth is enabled: API-key field stored in `localStorage`, sent as
  `X-API-Key`. Errors (400 guardrail rejections, 401, 429, 502, mid-stream
  `error` events) rendered inline in the transcript with `request_id`.
- Styling: plain hand-written CSS, dark theme, usable on a laptop; no
  responsive/mobile work, no accessibility audit (documented as demo-grade).
- Tests: one API test that `GET /` returns the page; a pure-function unit
  test for the SSE line-parser if it's factored into `app.js` testably —
  otherwise UI behavior is verified manually (out of automated scope, same
  standard as end-to-end).

## Error-handling philosophy

Transient LLM/HTTP failures retry with backoff, then fall back (model) if
configured, then degrade explicitly per §2. Guardrail rejections are 400s
with reasons. Nothing swallows an exception into an empty answer.

## Success criteria

1. `pytest` green with no API key, no Docker, no network.
2. `LLM_PROVIDER=ollama` documented path runs end-to-end offline (user-run).
3. `LLM_PROVIDER=anthropic` path unchanged for existing users (user-run).
4. Every README claim corresponds to code that exists.
5. `GET /` serves the chat UI; a browser pointed at it can ask a question,
   watch tokens stream, and inspect sources/citations (user-run check).
