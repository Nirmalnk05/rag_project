from src.citations import ensure_citations, verify_citations


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


def test_ensure_citations_appends_trailer_when_uncited_factual():
    long_answer = "The pump procedure requires many steps. " * 20
    fixed, citations = ensure_citations(long_answer, num_sources=2)
    assert fixed == long_answer + "\n\nSources: [#1] [#2]"
    assert citations["valid"] is True and citations["used"] == [1, 2]


def test_ensure_citations_leaves_orphaned_case_untouched():
    """A fabricated citation must stay visibly flagged, not be papered over."""
    answer = "Do the thing [#7]."
    fixed, citations = ensure_citations(answer, num_sources=5)
    assert fixed == answer  # unchanged
    assert citations["valid"] is False and citations["orphaned"] == [7]


def test_ensure_citations_leaves_no_sources_case_untouched():
    long_answer = "The pump procedure requires many steps. " * 20
    fixed, citations = ensure_citations(long_answer, num_sources=0)
    assert fixed == long_answer  # nothing to cite, can't fabricate a source list
    assert citations["valid"] is False


def test_ensure_citations_leaves_refusal_untouched():
    answer = "The context does not contain enough information."
    fixed, citations = ensure_citations(answer, num_sources=3)
    assert fixed == answer and citations["valid"] is True


def test_ensure_citations_leaves_already_valid_untouched():
    answer = "Check the strainer [#1]. Replace seal [#2]."
    fixed, citations = ensure_citations(answer, num_sources=2)
    assert fixed == answer and citations["valid"] is True
