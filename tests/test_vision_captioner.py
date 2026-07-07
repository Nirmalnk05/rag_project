from src.config import Settings
from src.parsing.vision import FigureCaptioner

IMG = b"fake-image-bytes"


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def caption(self, *, image, media_type, prompt, model, max_tokens=512):
        self.calls += 1
        return "  a centrifugal pump nameplate  "


def _settings(**kw):
    base = dict(_env_file=None, enable_figure_caption=True, vision_model="claude-vision")
    base.update(kw)
    return Settings(**base)


async def test_disabled_returns_empty_no_call():
    fp = FakeProvider()
    cap = FigureCaptioner(settings=_settings(enable_figure_caption=False),
                          provider=fp, cache={}, cache_path=None)
    assert await cap.caption(IMG, "image/png") == ""
    assert fp.calls == 0


async def test_no_vision_model_disabled():
    fp = FakeProvider()
    cap = FigureCaptioner(settings=_settings(vision_model=""), provider=fp,
                          cache={}, cache_path=None)
    assert await cap.caption(IMG, "image/png") == ""
    assert fp.calls == 0


async def test_caption_strips_and_returns():
    fp = FakeProvider()
    cap = FigureCaptioner(settings=_settings(), provider=fp, cache={}, cache_path=None)
    assert await cap.caption(IMG, "image/png") == "a centrifugal pump nameplate"
    assert fp.calls == 1


async def test_cache_hit_avoids_second_call():
    fp = FakeProvider()
    cache = {}
    cap = FigureCaptioner(settings=_settings(), provider=fp, cache=cache, cache_path=None)
    await cap.caption(IMG, "image/png")
    await cap.caption(IMG, "image/png")
    assert fp.calls == 1  # second served from cache
    assert len(cache) == 1


async def test_cache_persisted_to_disk(tmp_path):
    fp = FakeProvider()
    path = tmp_path / "caption_cache.json"
    cap = FigureCaptioner(settings=_settings(), provider=fp, cache={}, cache_path=path)
    await cap.caption(IMG, "image/png")
    assert path.exists()
    import json
    assert list(json.loads(path.read_text()).values()) == ["a centrifugal pump nameplate"]


def test_caption_sync_multiple_calls_reuse_open_loop():
    """Regression test: caption_sync() must not create-and-close a new event
    loop on every call (that breaks a provider client cached across calls —
    see the docstring on caption_sync). Simulates ingestion captioning
    several distinct figures in one run (different images so the cache
    doesn't short-circuit the second/third calls)."""
    fp = FakeProvider()
    cap = FigureCaptioner(settings=_settings(), provider=fp, cache={}, cache_path=None)

    r1 = cap.caption_sync(b"figure-one", "image/png")
    r2 = cap.caption_sync(b"figure-two", "image/png")
    r3 = cap.caption_sync(b"figure-three", "image/png")

    assert [r1, r2, r3] == ["a centrifugal pump nameplate"] * 3
    assert fp.calls == 3  # no cache hits — each was a distinct real call

    import src.llm as llm_mod  # module reference, not `from` -- the global is reassigned
    assert llm_mod._sync_loop is not None and not llm_mod._sync_loop.is_closed()
