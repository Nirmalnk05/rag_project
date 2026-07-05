"""Structured JSON logging, request-ID propagation, and Langfuse tracing.

Langfuse is env-gated: without keys, everything is a no-op with zero overhead.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from src.config import get_settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_current_trace: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return  # already configured
    # Drop pre-existing plain StreamHandlers (e.g. a stray basicConfig) so app
    # logs aren't emitted twice, but keep foreign handlers such as pytest's
    # capture handler and uvicorn's.
    root.handlers = [
        h for h in root.handlers
        if not (type(h) is logging.StreamHandler and h.formatter is None)
    ]
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"extra_fields": fields})


# ---------- Langfuse (optional) ----------

class Trace:
    """Wraps a langfuse trace; no-op when _lf_trace is None."""

    def __init__(self, lf_trace=None):
        self._t = lf_trace

    @contextmanager
    def span(self, name: str, **meta):
        start = time.perf_counter()
        lf_span = self._t.span(name=name, input=meta or None) if self._t else None
        try:
            yield
        finally:
            if lf_span:
                lf_span.end(metadata={"latency_s": round(time.perf_counter() - start, 3)})

    def generation(self, name: str, *, model: str, input_tokens: int,
                   output_tokens: int, latency_s: float) -> None:
        if self._t:
            self._t.generation(
                name=name, model=model,
                usage={"input": input_tokens, "output": output_tokens},
                metadata={"latency_s": round(latency_s, 3)},
            )

    def end(self, output: str = "") -> None:
        if self._t:
            self._t.update(output=output[:2000])


class Tracer:
    def __init__(self, client=None):
        self._client = client

    def trace(self, *, name: str, request_id: str, input: str) -> Trace:
        if not self._client:
            return Trace(None)
        return Trace(self._client.trace(name=name, id=request_id, input=input))


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        s = get_settings()
        client = None
        if s.tracing_enabled:
            from langfuse import Langfuse  # lazy: only imported when enabled
            client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
            )
        _tracer = Tracer(client)
    return _tracer


def set_current_trace(trace: Trace | None):
    return _current_trace.set(trace)


def reset_current_trace(token) -> None:
    _current_trace.reset(token)


@contextmanager
def node_span(name: str, **meta):
    """Span helper usable inside graph nodes; no-ops without an active trace."""
    trace = _current_trace.get()
    logger = logging.getLogger("rag.graph")
    start = time.perf_counter()
    if trace is not None:
        with trace.span(name, **meta):
            yield
    else:
        yield
    log_event(logger, f"node_{name}_done", latency_s=round(time.perf_counter() - start, 3))
