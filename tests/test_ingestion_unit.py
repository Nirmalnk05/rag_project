from src.ingestion import (
    _split_into_sections, _windowed_chunks, chunk_content_hash, diff_manifest,
    point_id_for,
)


def test_split_into_sections():
    body = "# Title\nintro\n## A\ncontent a\n## B\ncontent b"
    parts = _split_into_sections(body)
    titles = [t for t, _ in parts]
    assert titles == ["Preamble", "A", "B"]


def test_windowed_chunks_overlap():
    words = " ".join(f"w{i}" for i in range(1500))
    out = _windowed_chunks(words, target=600, overlap=100)
    assert len(out) == 3
    assert out[0].split()[-100:] == out[1].split()[:100]  # overlap preserved


def test_chunk_content_hash_stable_and_sensitive():
    h1 = chunk_content_hash("doc", "sec", "text")
    assert h1 == chunk_content_hash("doc", "sec", "text")
    assert h1 != chunk_content_hash("doc", "sec", "text2")


def test_diff_manifest():
    prev = {"a": "h1", "b": "h2", "c": "h3"}
    cur = {"a": "h1", "b": "h2-changed", "d": "h4"}
    new, changed, stale = diff_manifest(cur, prev)
    assert new == {"d"} and changed == {"b"} and stale == {"c"}


def test_point_id_deterministic_uuid():
    import uuid
    p1, p2 = point_id_for("chunk-1"), point_id_for("chunk-1")
    assert p1 == p2 and p1 != point_id_for("chunk-2")
    uuid.UUID(p1)  # valid uuid string
