"""Minimal in-process metrics with Prometheus text exposition (no dependency).

Enough for counters + latency histograms scraped at /metrics. For richer needs
swap in prometheus_client — the call sites (inc/observe) stay the same shape.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()


class Counter:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with _lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if not self._values:
            out.append(f"{self.name} 0")
        for key, val in self._values.items():
            out.append(f"{self.name}{_fmt_labels(key)} {val}")
        return out


class Histogram:
    """Count + sum only (Prometheus derives averages)."""
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._count: dict[tuple, int] = {}
        self._sum: dict[tuple, float] = {}

    def observe(self, value: float, **labels) -> None:
        key = tuple(sorted(labels.items()))
        with _lock:
            self._count[key] = self._count.get(key, 0) + 1
            self._sum[key] = self._sum.get(key, 0.0) + value

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} summary"]
        for key in self._count:
            lbl = _fmt_labels(key)
            out.append(f"{self.name}_count{lbl} {self._count[key]}")
            out.append(f"{self.name}_sum{lbl} {self._sum[key]}")
        return out


def _fmt_labels(key: tuple) -> str:
    if not key:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in key)
    return "{" + inner + "}"


# ---------- registry ----------

REQUESTS = Counter("rag_requests_total", "Requests by endpoint and status")
CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits")
TOOL_CALLS = Counter("rag_tool_calls_total", "Agent tool calls by tool")
ANSWER_LATENCY = Histogram("rag_answer_latency_seconds", "End-to-end answer latency")

_ALL = [REQUESTS, CACHE_HITS, TOOL_CALLS, ANSWER_LATENCY]


def render() -> str:
    lines: list[str] = []
    for m in _ALL:
        lines.extend(m.render())
    return "\n".join(lines) + "\n"


def reset() -> None:  # test helper
    for m in _ALL:
        if isinstance(m, Counter):
            m._values.clear()
        else:
            m._count.clear()
            m._sum.clear()
