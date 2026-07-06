"""Conversation memory: per-session chat history for multi-turn.

Stores only clean user↔assistant turns (never tool-call scratch messages), capped
at `memory_max_turns` pairs with a TTL. Two backends behind one interface:
`memory` (in-process, single node) and `redis` (shared across replicas).
"""
from __future__ import annotations

import json
import time

from src.config import Settings, get_settings


class SessionStore:
    """In-process session store. Injectable for tests via `now` / `clock`."""

    def __init__(self, settings: Settings | None = None, *, clock=time.time) -> None:
        self._s = settings or get_settings()
        self._clock = clock
        self._data: dict[str, tuple[float, list[dict]]] = {}  # sid -> (ts, turns)

    @property
    def enabled(self) -> bool:
        return self._s.enable_memory

    def _evict_if_stale(self, sid: str) -> None:
        entry = self._data.get(sid)
        if entry and (self._clock() - entry[0]) > self._s.memory_ttl_s:
            del self._data[sid]

    def get_history(self, sid: str) -> list[dict]:
        if not self.enabled or not sid:
            return []
        self._evict_if_stale(sid)
        entry = self._data.get(sid)
        return list(entry[1]) if entry else []

    def append(self, sid: str, user: str, assistant: str) -> None:
        if not self.enabled or not sid:
            return
        turns = self.get_history(sid)
        turns.append({"role": "user", "content": user})
        turns.append({"role": "assistant", "content": assistant})
        # keep the last memory_max_turns pairs
        max_msgs = self._s.memory_max_turns * 2
        turns = turns[-max_msgs:]
        self._data[sid] = (self._clock(), turns)

    def reset(self, sid: str) -> None:
        self._data.pop(sid, None)


class RedisSessionStore:
    """Session store backed by Redis (shared across replicas)."""

    def __init__(self, client, settings: Settings | None = None) -> None:
        self._c = client
        self._s = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self._s.enable_memory

    @staticmethod
    def _key(sid: str) -> str:
        return f"sess:{sid}"

    def get_history(self, sid: str) -> list[dict]:
        if not self.enabled or not sid:
            return []
        raw = self._c.get(self._key(sid))
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)

    def append(self, sid: str, user: str, assistant: str) -> None:
        if not self.enabled or not sid:
            return
        turns = self.get_history(sid)
        turns.append({"role": "user", "content": user})
        turns.append({"role": "assistant", "content": assistant})
        turns = turns[-self._s.memory_max_turns * 2:]
        self._c.set(self._key(sid), json.dumps(turns), ex=self._s.memory_ttl_s)

    def reset(self, sid: str) -> None:
        self._c.delete(self._key(sid))


_store = None


def get_store(settings: Settings | None = None):
    """Module-level singleton session store, chosen by backend."""
    global _store
    if _store is None:
        s = settings or get_settings()
        if s.memory_backend == "redis":
            from src.security import _get_redis
            _store = RedisSessionStore(_get_redis(s.redis_url), s)
        else:
            _store = SessionStore(s)
    return _store


def set_store(store) -> None:  # test hook
    global _store
    _store = store
