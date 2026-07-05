import json

import pytest

from src.retrieval import Retrieved, Retriever, _matches_filters


def _chunk(i, text, section="Ops", title="Pump Manual", src="pump.md"):
    return {
        "id": f"c{i}", "doc_id": "d1", "doc_title": title, "text": text,
        "context": "", "embed_text": text, "section": section,
        "metadata": {"source_file": src},
    }


@pytest.fixture
def retriever(tmp_path):
    chunks = [
        _chunk(1, "pump P-100 alarm code 4F low suction pressure"),
        _chunk(2, "chiller CH-200 condenser fin cleaning", title="Chiller SOP", src="hvac.md"),
        _chunk(3, "bearing alignment tolerance dial indicator"),
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return Retriever(chunks_path=path, load_models=False, qdrant_client=object())


def test_bm25_built_from_chunks_json(retriever):
    out = retriever.sparse_search("P-100 alarm 4F", top_k=3)
    assert out and out[0].chunk_id == "c1" and out[0].source == "bm25"


def test_sparse_search_respects_filters(retriever):
    out = retriever.sparse_search("cleaning condenser chiller CH-200", top_k=3,
                                  filters={"source_file": "pump.md"})
    assert all(o.doc_title == "Pump Manual" for o in out)


def test_matches_filters():
    c = _chunk(1, "x")
    assert _matches_filters(c, {"doc_title": "Pump Manual"})
    assert _matches_filters(c, {"source_file": "pump.md"})
    assert not _matches_filters(c, {"section": "Wrong"})
    assert _matches_filters(c, None)


def test_rrf_orders_by_fused_score():
    a = [Retrieved("c1", "d", "t", "s", "x", "", 9.0, "bm25"),
         Retrieved("c2", "d", "t", "s", "y", "", 8.0, "bm25")]
    b = [Retrieved("c2", "d", "t", "s", "y", "", 0.9, "dense"),
         Retrieved("c3", "d", "t", "s", "z", "", 0.8, "dense")]
    fused = Retriever.reciprocal_rank_fusion([a, b])
    assert fused[0].chunk_id == "c2"          # appears in both lists
    assert {f.source for f in fused} == {"rrf"}


def test_filter_keys_restricted(retriever):
    with pytest.raises(ValueError, match="unsupported filter"):
        retriever.sparse_search("pump", top_k=3, filters={"evil": "x"})
