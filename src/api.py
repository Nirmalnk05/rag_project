"""FastAPI service: agentic RAG over JSON + SSE, with auth, rate limiting,
request IDs, structured errors, and a minimal chat UI at /."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings
from src.guardrails import redact_pii, screen_input
from src.orchestrator import Orchestrator
from src.observability import (
    get_tracer, log_event, request_id_var, reset_current_trace, set_current_trace,
    setup_logging,
)
from src.providers.base import LLMError
from src.security import rate_limiter, require_api_key

_logger = logging.getLogger("rag.api")
_STATIC_DIR = Path(__file__).parent / "static"


async def _index_reload_loop(retriever, poll_s: int) -> None:
    """Poll index_meta.json and swap in a newer corpus off the request path.

    chunks.json is a startup snapshot, so until this runs NEW documents are
    invisible to keyword search. Revocations and deletions do not wait for it:
    while the loaded generation is behind, the sparse leg resolves metadata
    from the index itself.
    """
    while True:
        await asyncio.sleep(poll_s)
        try:
            await asyncio.to_thread(retriever.reload_if_stale)
        except Exception:  # noqa: BLE001 — a failed reload must not stop serving
            _logger.warning("index_reload_failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.validate_provider()
    settings.validate_access()
    log_event(_logger, "startup_loading_retriever")
    from src.retrieval import Retriever
    app.state.retriever = await asyncio.to_thread(Retriever)
    app.state.orchestrator = Orchestrator(app.state.retriever, settings=settings)
    app.state.index_reload_task = None
    if settings.index_reload_poll_s > 0:
        app.state.index_reload_task = asyncio.create_task(
            _index_reload_loop(app.state.retriever, settings.index_reload_poll_s))
    log_event(_logger, "startup_ready", provider=settings.llm_provider,
              mode=settings.agent_mode)
    try:
        yield
    finally:
        task = app.state.index_reload_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="Production RAG", version="0.2.0", lifespan=lifespan)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)


app.add_middleware(RequestIDMiddleware)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    log_event(_logger, "llm_error", error=str(exc))
    return JSONResponse(status_code=502, content={
        "error": "upstream LLM failure", "request_id": request_id_var.get(),
    })


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    filters: dict[str, str] | None = None
    session_id: str = Field(default="", max_length=128)


class AskResponse(BaseModel):
    query: str
    answer: str
    rewritten_query: str
    grade: str
    retrieval: list[dict]
    citations: dict
    steps: list[dict]
    safety: str
    faithfulness: dict
    session_id: str
    request_id: str


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {
        "status": "ok",
        "retriever_loaded": getattr(app.state, "retriever", None) is not None,
        "provider": get_settings().llm_provider,
    }


@app.get("/ready")
def ready():
    """Readiness: models + orchestrator are loaded and the app can serve."""
    loaded = (getattr(app.state, "retriever", None) is not None
              and getattr(app.state, "orchestrator", None) is not None)
    if not loaded:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


@app.get("/")
def chat_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    from src.metrics import render
    return PlainTextResponse(render(), media_type="text/plain; version=0.0.4")


class ResetRequest(BaseModel):
    session_id: str = Field(..., max_length=128)


@app.post("/session/reset", dependencies=[Depends(require_api_key)])
def session_reset(req: ResetRequest) -> dict:
    from src.memory import get_store
    get_store().reset(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


class FeedbackRequest(BaseModel):
    rating: str = Field(..., pattern="^(up|down)$")
    query: str = Field(default="", max_length=2000)
    answer: str = Field(default="", max_length=8000)
    request_id: str = Field(default="", max_length=128)
    session_id: str = Field(default="", max_length=128)
    comment: str = Field(default="", max_length=2000)


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest) -> dict:
    from src.feedback import get_feedback_store
    get_feedback_store().record(
        rating=req.rating, query=req.query, answer=req.answer,
        request_id=req.request_id, session_id=req.session_id, comment=req.comment)
    if req.request_id:
        get_tracer().score(trace_id=req.request_id, name="user_feedback",
                          value=1.0 if req.rating == "up" else 0.0, comment=req.comment)
    return {"status": "recorded", "rating": req.rating}


@app.get("/feedback/summary", dependencies=[Depends(require_api_key)])
def feedback_summary() -> dict:
    from src.feedback import get_feedback_store
    return get_feedback_store().summary()


def _record_answer_metrics(endpoint: str, payload: dict, started: float) -> None:
    """The counters every answer path shares.

    Both /ask and /ask/stream come through here. Instrumenting only /ask left
    the chat UI — which posts to /ask/stream — reporting zero requests, zero
    tool calls and zero latency no matter how much it was used.
    """
    from src.metrics import ANSWER_LATENCY, CACHE_HITS, REQUESTS, TOOL_CALLS
    REQUESTS.inc(endpoint=endpoint, status="ok")
    ANSWER_LATENCY.observe(time.perf_counter() - started, endpoint=endpoint)
    if payload.get("cached"):
        CACHE_HITS.inc()
    for step in payload.get("steps", []):
        TOOL_CALLS.inc(tool=step.get("tool", "?"))


def _count_outcome(endpoint: str, status: str) -> None:
    """Record a non-answer outcome. A `status` label that only ever holds "ok"
    hides the one number worth alerting on."""
    from src.metrics import REQUESTS
    REQUESTS.inc(endpoint=endpoint, status=status)


def _guard(query: str, endpoint: str) -> None:
    reason = screen_input(query)
    if reason:
        # a guardrail rejection is a real outcome, not a server failure
        _count_outcome(endpoint, "rejected")
        raise HTTPException(status_code=400, detail=f"query rejected: {reason}")


def _access_for(request: Request):
    """Principal for ACL filtering, or None when ACLs are disabled (dev)."""
    if not get_settings().enable_acl:
        return None
    from src.access import AccessContext
    return AccessContext.from_headers(request.headers)


@app.post("/ask", response_model=AskResponse,
          dependencies=[Depends(require_api_key), Depends(rate_limiter)])
async def ask(req: AskRequest, http_request: Request) -> AskResponse:
    _guard(req.query, "ask")
    settings = get_settings()
    rid = request_id_var.get()
    access = _access_for(http_request)
    trace = get_tracer().trace(name="ask", request_id=rid, input=req.query,
                               session_id=req.session_id)
    tok = set_current_trace(trace)
    started = time.perf_counter()
    try:
        out = await app.state.orchestrator.run(req.query, req.filters, req.session_id, access)
    except Exception:
        _count_outcome("ask", "error")
        raise
    finally:
        reset_current_trace(tok)
    _record_answer_metrics("ask", out, started)
    answer = out.get("answer", "")
    if settings.redact_pii_in_answers:
        answer, _found = redact_pii(answer)
    trace.end(output=answer)
    return AskResponse(
        query=req.query, answer=answer,
        rewritten_query=out.get("rewritten", ""),
        grade=out.get("grade", ""),
        retrieval=out.get("retrieval", []),
        citations=out.get("citations", {}),
        steps=out.get("steps", []),
        safety=out.get("safety", ""),
        faithfulness=out.get("faithfulness", {}),
        session_id=req.session_id,
        request_id=rid,
    )


@app.post("/ask/stream",
          dependencies=[Depends(require_api_key), Depends(rate_limiter)])
async def ask_stream(req: AskRequest, http_request: Request) -> StreamingResponse:
    _guard(req.query, "ask_stream")
    rid = request_id_var.get()
    access = _access_for(http_request)

    async def event_generator() -> AsyncIterator[str]:
        trace = get_tracer().trace(name="ask_stream", request_id=rid, input=req.query,
                                   session_id=req.session_id)
        tok = set_current_trace(trace)
        started = time.perf_counter()
        counted = False
        try:
            async for evt in app.state.orchestrator.stream(req.query, req.filters,
                                                           req.session_id, access):
                if evt.get("type") == "final":
                    evt["request_id"] = rid
                    _record_answer_metrics("ask_stream", evt, started)
                    counted = True
                yield f"data: {json.dumps(evt)}\n\n"
        except Exception as e:  # noqa: BLE001 — stream must report, not vanish
            # the error is delivered as an SSE event with HTTP 200 already sent,
            # so the status cannot be inferred from the response code
            if not counted:
                _count_outcome("ask_stream", "error")
            _logger.exception("stream_failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'request_id': rid})}\n\n"
        finally:
            reset_current_trace(tok)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
