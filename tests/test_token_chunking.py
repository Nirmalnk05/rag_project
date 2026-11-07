"""Chunks are sized in TOKENS, because that is what the embedder truncates on.

bge-small takes 512 tokens and truncates the rest in silence. Measured
tokens-per-word on this corpus: prose 1.23, table rows 1.57, codes + units 2.56,
CJK 5.21 — a 4x spread, so no word target fits all content. The old 600-word
window produced 743 tokens and lost 31% of every full chunk; nothing failed,
because the text was still stored and still shown to the LLM, so only DENSE
recall degraded, on the tail. BM25 indexes the full text, which is why it was
invisible.

Every test here injects a token counter: the suite must not download a model.
"""
import math

import pytest

from src.config import Settings
from src.ingestion import (_element_units, _windowed_chunks, build_chunks,
                           chunk_token_budget, oversize_units)
from src.parsing.dispatcher import Dispatcher
from src.parsing.model import ParsedDocument, TableElement, TextElement

# one token per whitespace-delimited word
WORDS = lambda t: len(t.split())                                    # noqa: E731
# one token per character, ignoring whitespace — stands in for dense content
CHARS = lambda t: sum(len(w) for w in t.split())                    # noqa: E731


def _ratio(r):
    """A counter at a measured tokens-per-word ratio."""
    return lambda t: math.ceil(len(t.split()) * r)


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


# ---------- windows are bounded in tokens, not words ----------

def test_a_text_inside_the_budget_is_one_chunk():
    out = _windowed_chunks("short enough", max_tokens=10, count_tokens=WORDS)
    assert out == ["short enough"]


def test_no_window_exceeds_the_token_budget():
    out = _windowed_chunks(_words(100), max_tokens=10, overlap_tokens=0,
                           count_tokens=WORDS)
    assert out, "expected several windows"
    assert all(WORDS(w) <= 10 for w in out)


@pytest.mark.parametrize("ratio", [1.23, 1.57, 2.56, 5.21])
def test_the_budget_holds_at_every_measured_tokens_per_word_ratio(ratio):
    """A 4x spread in tokens-per-word is exactly why a word target cannot work."""
    counter = _ratio(ratio)
    out = _windowed_chunks(_words(400), max_tokens=512, overlap_tokens=64,
                           count_tokens=counter)
    assert all(counter(w) <= 512 for w in out)


def test_token_dense_content_is_split_more_finely_than_prose():
    """Same words, 4x the tokens per word -> more windows. A word target would
    have produced one chunk for both and silently truncated the dense one."""
    text = _words(200)
    prose = _windowed_chunks(text, max_tokens=100, overlap_tokens=0,
                             count_tokens=_ratio(1.23))
    dense = _windowed_chunks(text, max_tokens=100, overlap_tokens=0,
                             count_tokens=_ratio(5.21))
    assert len(dense) > len(prose)


def test_windows_overlap_by_the_overlap_budget():
    out = _windowed_chunks(_words(50), max_tokens=10, overlap_tokens=3,
                           count_tokens=WORDS)
    assert len(out) > 1
    assert out[0].split()[-3:] == out[1].split()[:3]


def test_windows_cover_every_word():
    out = _windowed_chunks(_words(50), max_tokens=10, overlap_tokens=0,
                           count_tokens=WORDS)
    seen = [w for window in out for w in window.split()]
    assert seen == _words(50).split()


# ---------- splits land on word boundaries, text stays byte-identical ----------

def test_chunk_text_is_byte_identical_to_the_source():
    """Joining on single spaces would rewrite the source: newlines and tabs in
    a table row or a code block carry meaning for the LLM reading the chunk."""
    text = "para one\n\nsecond line\twith a tab  and  double  spaces\nlast bit here"
    out = _windowed_chunks(text, max_tokens=4, overlap_tokens=0, count_tokens=WORDS)
    assert len(out) > 1
    for window in out:
        assert window in text, f"window was rewritten, not sliced: {window!r}"


def test_splits_never_cut_a_word_in_half():
    out = _windowed_chunks(_words(40), max_tokens=7, overlap_tokens=0,
                           count_tokens=WORDS)
    words = set(_words(40).split())
    for window in out:
        assert set(window.split()) <= words


# ---------- what cannot fit is reported, not truncated quietly ----------

def test_a_single_word_larger_than_the_budget_is_kept_whole():
    text = "aaaaaaaaaaaaaaaaaaaa short tail"
    out = _windowed_chunks(text, max_tokens=10, overlap_tokens=0, count_tokens=CHARS)
    assert out[0] == "aaaaaaaaaaaaaaaaaaaa"
    assert "".join(out).replace(" ", "") == text.replace(" ", "")


def test_oversize_units_are_listed_with_their_token_count():
    units = [("fits", "text", 1), ("aaaaaaaaaaaaaaaaaaaa", "table", 2)]
    over = oversize_units(units, max_tokens=10, count_tokens=CHARS)
    assert len(over) == 1
    assert over[0]["element_type"] == "table"
    assert over[0]["tokens"] == 20


# ---------- the budget leaves room for the contextual blurb ----------

def test_contextual_augmentation_reserves_part_of_the_budget():
    """embed_text is the blurb + the chunk, so sizing the chunk to the whole
    window puts the blurb's tokens over the edge and truncates the tail."""
    s = Settings(_env_file=None, chunk_max_tokens=512,
                 chunk_context_reserve_tokens=128)
    assert chunk_token_budget(True, s) < chunk_token_budget(False, s)
    assert chunk_token_budget(False, s) <= 512


def test_the_budget_never_goes_below_one():
    s = Settings(_env_file=None, chunk_max_tokens=8,
                 chunk_context_reserve_tokens=512)
    assert chunk_token_budget(True, s) >= 1


# ---------- element units ----------

def test_element_units_window_text_by_token_budget():
    units = _element_units(TextElement("s", _words(100)),
                           max_tokens=10, overlap_tokens=0, count_tokens=WORDS)
    assert len(units) == 10
    assert all(u[1] == "text" for u in units)


def test_element_units_keep_a_table_atomic_even_when_oversize():
    """A split table row loses its header; truncation is reported instead."""
    units = _element_units(TableElement("Alarms", "| a | b |\n" * 500),
                           max_tokens=10, overlap_tokens=0, count_tokens=WORDS)
    assert len(units) == 1 and units[0][1] == "table"


# ---------- the report reaches the operator ----------

class _StubDispatcher(Dispatcher):
    def __init__(self, doc):
        self._doc = doc

    def parse(self, path, *, force_cloud=False):
        return self._doc


def test_build_chunks_reports_what_cannot_fit(tmp_path, capsys):
    (tmp_path / "m.pdf").write_bytes(b"%PDF-1.4 fake")
    doc = ParsedDocument("Pump Manual", "m.pdf", [
        TableElement("Fault Codes", "| 4F | low pressure |\n" * 40, page=2),
    ])
    build_chunks(tmp_path, use_contextual=False, dispatcher=_StubDispatcher(doc),
                 count_tokens=WORDS, max_tokens=10)
    out = capsys.readouterr().out
    assert "Fault Codes" in out
    assert "token" in out.lower()
    assert "m.pdf" in out


def test_build_chunks_is_quiet_when_everything_fits(tmp_path, capsys):
    (tmp_path / "m.pdf").write_bytes(b"%PDF-1.4 fake")
    doc = ParsedDocument("Pump Manual", "m.pdf",
                         [TextElement("Intro", "the pump operates at 50 bar", page=1)])
    build_chunks(tmp_path, use_contextual=False, dispatcher=_StubDispatcher(doc),
                 count_tokens=WORDS, max_tokens=512)
    assert "exceed" not in capsys.readouterr().out.lower()


def test_short_text_never_reaches_the_tokenizer():
    """Byte length alone can prove a fit, which keeps the common case (and this
    suite, which has no model) off the tokenizer entirely."""
    def _explode(_text):
        raise AssertionError("tokenizer must not be consulted for short text")

    assert _windowed_chunks("the pump operates at 50 bar", max_tokens=512,
                            count_tokens=_explode) == ["the pump operates at 50 bar"]
    assert oversize_units([("short row", "table", 1)], max_tokens=512,
                          count_tokens=_explode) == []
