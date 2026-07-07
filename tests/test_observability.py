import json
import logging

from src.observability import (
    JsonFormatter, Trace, Tracer, get_tracer, log_event, node_span, record_generation,
    request_id_var, reset_current_trace, set_current_trace, setup_logging,
)


class FakeLangfuseTrace:
    """Duck-typed stand-in for a real langfuse SDK trace object."""

    def __init__(self):
        self.spans: list[tuple] = []
        self.generations: list[dict] = []

    def span(self, *, name, input=None):
        self.spans.append((name, input))

        class _Span:
            def end(self, *, metadata=None):
                pass

        return _Span()

    def generation(self, *, name, model, usage, metadata):
        self.generations.append(
            {"name": name, "model": model, "usage": usage, "metadata": metadata}
        )


class FakeLangfuseClient:
    """Duck-typed stand-in for the top-level langfuse SDK client."""

    def __init__(self):
        self.trace_calls: list[dict] = []
        self.score_calls: list[dict] = []

    def trace(self, **kwargs):
        self.trace_calls.append(kwargs)
        return FakeLangfuseTrace()

    def score(self, **kwargs):
        self.score_calls.append(kwargs)


def test_json_formatter_includes_request_id():
    token = request_id_var.set("req-123")
    try:
        rec = logging.LogRecord("t", logging.INFO, "f", 1, "hello", None, None)
        out = json.loads(JsonFormatter().format(rec))
        assert out["event"] == "hello" and out["request_id"] == "req-123"
        assert out["level"] == "INFO"
    finally:
        request_id_var.reset(token)


def test_log_event_extra_fields(caplog):
    setup_logging()
    logger = logging.getLogger("test.obs")
    with caplog.at_level(logging.INFO, logger="test.obs"):
        log_event(logger, "retrieval_done", chunks=5)
    rec = caplog.records[-1]
    assert rec.extra_fields == {"chunks": 5}


def test_tracer_noop_without_keys():
    tracer = get_tracer()  # no langfuse keys in test env
    trace = tracer.trace(name="ask", request_id="r1", input="q")
    with trace.span("rewrite"):
        pass
    trace.generation("generate", model="m", input_tokens=1, output_tokens=2, latency_s=0.1)
    trace.end(output="ans")  # no exceptions = pass


async def test_node_span_runs_body():
    ran = []
    with node_span("grade"):
        ran.append(1)
    assert ran == [1]


def test_record_generation_calls_trace_when_active():
    fake = FakeLangfuseTrace()
    tok = set_current_trace(Trace(fake))
    try:
        record_generation("agent_step_0", model="m1", input_tokens=10,
                          output_tokens=5, latency_s=0.2)
    finally:
        reset_current_trace(tok)
    assert fake.generations == [{
        "name": "agent_step_0", "model": "m1",
        "usage": {"input": 10, "output": 5}, "metadata": {"latency_s": 0.2},
    }]


def test_record_generation_noop_without_active_trace():
    record_generation("x", model="m", input_tokens=1, output_tokens=1, latency_s=0.01)
    # no exception = pass; there's nothing to assert against without a trace


def test_node_span_creates_tool_span_with_args():
    fake = FakeLangfuseTrace()
    tok = set_current_trace(Trace(fake))
    try:
        with node_span("tool_search", args={"query": "4F"}):
            pass
    finally:
        reset_current_trace(tok)
    assert fake.spans == [("tool_search", {"args": {"query": "4F"}})]


def test_tracer_trace_passes_session_id_when_present():
    client = FakeLangfuseClient()
    tracer = Tracer(client)
    tracer.trace(name="ask", request_id="r1", input="q", session_id="sess-1")
    assert client.trace_calls == [
        {"name": "ask", "id": "r1", "input": "q", "session_id": "sess-1"}
    ]


def test_tracer_trace_omits_session_id_when_absent():
    client = FakeLangfuseClient()
    tracer = Tracer(client)
    tracer.trace(name="ask", request_id="r1", input="q")
    assert client.trace_calls == [{"name": "ask", "id": "r1", "input": "q"}]
    tracer.trace(name="ask", request_id="r2", input="q2", session_id="")
    assert "session_id" not in client.trace_calls[-1]


def test_tracer_trace_noop_without_client():
    tracer = Tracer(None)
    trace = tracer.trace(name="ask", request_id="r1", input="q", session_id="s1")
    assert isinstance(trace, Trace)  # returns a no-op Trace, doesn't raise


def test_tracer_score_calls_client():
    client = FakeLangfuseClient()
    tracer = Tracer(client)
    tracer.score(trace_id="r1", name="user_feedback", value=1.0, comment="great")
    assert client.score_calls == [
        {"trace_id": "r1", "name": "user_feedback", "value": 1.0, "comment": "great"}
    ]


def test_tracer_score_empty_comment_becomes_none():
    client = FakeLangfuseClient()
    tracer = Tracer(client)
    tracer.score(trace_id="r1", name="user_feedback", value=0.0)
    assert client.score_calls[0]["comment"] is None


def test_tracer_score_noop_without_client():
    Tracer(None).score(trace_id="r1", name="user_feedback", value=1.0)  # no exception
