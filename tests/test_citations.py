from src.citations import verify_citations


def test_valid_citations():
    out = verify_citations("Check the strainer [#1]. Replace seal [#3].", num_sources=5)
    assert out == {"valid": True, "used": [1, 3], "orphaned": []}


def test_orphaned_citation_invalid():
    out = verify_citations("Do the thing [#7].", num_sources=5)
    assert out["valid"] is False and out["orphaned"] == [7] and out["used"] == []


def test_long_factual_answer_without_citations_flagged():
    long_answer = "The pump procedure requires many steps. " * 20
    out = verify_citations(long_answer, num_sources=5)
    assert out["valid"] is False and out["used"] == []


def test_short_refusal_without_citations_ok():
    out = verify_citations("The context does not contain enough information.", num_sources=5)
    assert out["valid"] is True


def test_insufficient_marker_ok_even_if_long():
    ans = "The provided context is insufficient to answer this. " * 10
    assert verify_citations(ans, num_sources=3)["valid"] is True
