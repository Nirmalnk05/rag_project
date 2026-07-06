"""FastAPI service: agentic RAG over JSON + SSE, with auth, rate limiting,
request IDs, structured errors, and a minimal chat UI at /."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings
from src.guardrails import check_input, redact_pii
from src.orchestrator import Orchestrator
from src.observability import (
    get_tracer, log_event, request_id_var, reset_current_trace, set_current_trace,
    setup_logging,
)
from src.providers.base import LLMError
from src.security import rate_limiter, require_api_key

_logger = logging.getLogger("rag.api")
_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.validate_provider()
    log_event(_logger, "startup_loading_retriever")
    from src.retrieval import Retriever
    app.state.retriever = await asyncio.to_thread(Retriever)
    app.state.orchestrator = Orchestrator(app.state.retriever, settings=settings)
    log_event(_logger, "startup_ready", provider=settings.llm_provider,
              mode=settings.agent_mode)
    yield


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


class ResetRequest(BaseModel):
    session_id: str = Field(..., max_length=128)


@app.post("/session/reset", dependencies=[Depends(require_api_key)])
def session_reset(req: ResetRequest) -> dict:
    from src.memory import get_store
    get_store().reset(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


def _guard(query: str) -> None:
    reason = check_input(query)
    if reason:
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
    _guard(req.query)
    settings = get_settings()
    rid = request_id_var.get()
    access = _access_for(http_request)
    trace = get_tracer().trace(name="ask", request_id=rid, input=req.query)
    tok = set_current_trace(trace)
    try:
        out = await app.state.orchestrator.run(req.query, req.filters, req.session_id, access)
    finally:
        reset_current_trace(tok)
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
    _guard(req.query)
    rid = request_id_var.get()
    access = _access_for(http_request)

    async def event_generator() -> AsyncIterator[str]:
        trace = get_tracer().trace(name="ask_stream", request_id=rid, input=req.query)
        tok = set_current_trace(trace)
        try:
            async for evt in app.state.orchestrator.stream(req.query, req.filters,
                                                           req.session_id, access):
                if evt.get("type") == "final":
                    evt["request_id"] = rid
                yield f"data: {json.dumps(evt)}\n\n"
        except Exception as e:  # noqa: BLE001 — stream must report, not vanish
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
