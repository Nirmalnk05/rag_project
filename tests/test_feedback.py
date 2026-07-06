from src.config import Settings
from src.feedback import FeedbackStore


def _store(tmp_path):
    return FeedbackStore(Settings(_env_file=None), path=tmp_path / "fb.jsonl")


def test_record_appends_and_summary_counts(tmp_path):
    st = _store(tmp_path)
    st.record(rating="up", query="q1", answer="a1", request_id="r1")
    st.record(rating="down", query="q2", answer="a2", comment="wrong code")
    st.record(rating="up", query="q3")
    s = st.summary()
    assert s == {"up": 2, "down": 1, "total": 3}


def test_summary_empty(tmp_path):
    assert _store(tmp_path).summary() == {"up": 0, "down": 0, "total": 0}


def test_record_truncates_long_answer(tmp_path):
    st = _store(tmp_path)
    rec = st.record(rating="up", answer="x" * 5000)
    assert len(rec["answer"]) == 2000


def test_records_are_valid_jsonl(tmp_path):
    import json
    path = tmp_path / "fb.jsonl"
    st = FeedbackStore(Settings(_env_file=None), path=path)
    st.record(rating="up", query="q")
    st.record(rating="down", query="q2")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and all(json.loads(l)["rating"] in ("up", "down") for l in lines)
