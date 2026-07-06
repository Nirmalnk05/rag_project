from src.metrics import (
    ANSWER_LATENCY, CACHE_HITS, REQUESTS, TOOL_CALLS, render, reset,
)


def setup_function():
    reset()


def test_counter_with_labels_renders():
    REQUESTS.inc(endpoint="ask", status="ok")
    REQUESTS.inc(endpoint="ask", status="ok")
    out = render()
    assert 'rag_requests_total{endpoint="ask",status="ok"} 2.0' in out
    assert "# TYPE rag_requests_total counter" in out


def test_cache_and_tool_counters():
    CACHE_HITS.inc()
    TOOL_CALLS.inc(tool="search")
    TOOL_CALLS.inc(tool="search")
    out = render()
    assert "rag_cache_hits_total 1.0" in out
    assert 'rag_tool_calls_total{tool="search"} 2.0' in out


def test_histogram_count_and_sum():
    ANSWER_LATENCY.observe(0.5)
    ANSWER_LATENCY.observe(1.5)
    out = render()
    assert "rag_answer_latency_seconds_count 2" in out
    assert "rag_answer_latency_seconds_sum 2.0" in out


def test_empty_counter_renders_zero():
    out = render()
    assert "rag_requests_total 0" in out
