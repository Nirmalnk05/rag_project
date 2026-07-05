"""Figure captioning: turn an image into searchable text via a vision model.

Uses the same provider abstraction as the rest of the system (Claude vision, or
a local vision model like llama3.2-vision via Ollama). Captions are cached by
image content-hash so an interrupted ingest resumes without repaying calls.
Gated by `enable_figure_caption` + a configured `vision_model`; when off, returns
"" so callers degrade to OCR-only text.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from src.config import Settings, get_settings
from src.providers.base import LLMResult
from src.providers.factory import get_provider
from src.providers.retry import call_with_retries

CAPTION_CACHE_PATH = Path("data/caption_cache.json")

CAPTION_PROMPT = (
    "Describe this image from a technical maintenance document in 1-3 sentences. "
    "Focus on any equipment, part numbers, labels, gauge readings, wiring, "
    "connectors, or fault/alarm indicators visible. Output only the description."
)


def _image_key(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()


class FigureCaptioner:
    def __init__(self, *, settings: Settings | None = None, provider=None,
                 cache: dict[str, str] | None = None,
                 cache_path: Path = CAPTION_CACHE_PATH) -> None:
        self._s = settings or get_settings()
        self._provider = provider
        self._cache = cache if cache is not None else {}
        self._cache_path = cache_path

    @property
    def enabled(self) -> bool:
        return self._s.enable_figure_caption and bool(self._s.vision_model)

    async def caption(self, image: bytes, media_type: str) -> str:
        if not self.enabled:
            return ""
        key = _image_key(image)
        if key in self._cache:
            return self._cache[key]

        provider = self._provider or get_provider()

        async def make_call(override: str | None) -> LLMResult:
            text = await provider.caption(
                image=image, media_type=media_type, prompt=CAPTION_PROMPT,
                model=override or self._s.vision_model,
            )
            return LLMResult(text=text, model=self._s.vision_model)

        result = await call_with_retries(make_call, max_attempts=self._s.llm_max_attempts)
        caption = result.text.strip()
        self._cache[key] = caption
        self._save_cache()
        return caption

    def caption_sync(self, image: bytes, media_type: str) -> str:
        return asyncio.run(self.caption(image, media_type))

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
