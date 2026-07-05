import json
import logging

from src.observability import (
    JsonFormatter, get_tracer, log_event, node_span, request_id_var, setup_logging,
)


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
