"""User feedback capture (thumbs up/down) + summary.

Appends JSONL records for offline analysis and a feedback loop into eval. Simple
file store by default; swap for a DB in production.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings, get_settings


class FeedbackStore:
    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self._s = settings or get_settings()
        self._path = Path(path or self._s.feedback_path)
        self._lock = threading.Lock()

    def record(self, *, rating: str, query: str = "", answer: str = "",
               request_id: str = "", session_id: str = "", comment: str = "") -> dict:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "rating": rating, "query": query, "answer": answer[:2000],
            "request_id": request_id, "session_id": session_id, "comment": comment[:1000],
        }
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def summary(self) -> dict:
        up = down = total = 0
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if r.get("rating") == "up":
                        up += 1
                    elif r.get("rating") == "down":
                        down += 1
        return {"up": up, "down": down, "total": total}


_store: FeedbackStore | None = None


def get_feedback_store(settings: Settings | None = None) -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore(settings)
    return _store


def set_feedback_store(store) -> None:  # test hook
    global _store
    _store = store
