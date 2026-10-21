import json
from pathlib import Path

from evaluation.recorded_vectors import RecordedEmbedder
from evaluation.retrieval_eval import recall_at_5
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder, content_tokens
from riya.core.memory import Memory
from riya.core.store import open_db
from tests.bge_cache import live_bge


def _data() -> dict:
    return json.loads(Path("tests/data/retrieval_set.json").read_text(encoding="utf-8"))


def test_retrieval_set_with_hash_embedder():
    data = _data()
    assert len(data["questions"]) == 46
    res = recall_at_5(lambda: Memory(open_db(":memory:"), HashEmbedder(), FakeClock()), data)
    assert res["recall@5"] >= 0.6 and res["never_said_empty"] >= 0.75


def test_never_said_questions_share_no_content_word_with_a_fact():
    data = _data()
    facts = set().union(*(content_tokens(f["text"]) for f in data["facts"]))
    assert sum(1 for q in data["questions"] if q["kind"] == "hard_negative") >= 4
    for q in data["questions"]:
        if q["kind"] in ("never_said", "hard_negative"):
            assert not q["expect"] and not q.get("expect_episodes")
            assert not content_tokens(q["q"]) & facts, q["q"]


def test_retrieval_set_answers_last_time_and_about_me_questions():
    res = recall_at_5(lambda: Memory(open_db(":memory:"), HashEmbedder(), FakeClock()), _data())
    assert res["episode_recall"] == 1.0 and res["about_me_recall"] == 1.0, res


def test_retrieval_set_with_bge(tmp_path, monkeypatch):
    embedder = live_bge(tmp_path, monkeypatch)
    res = recall_at_5(lambda: Memory(open_db(":memory:"), embedder, FakeClock()), _data())
    assert res["recall@5"] >= 0.9 and res["never_said_empty"] >= 0.75, res
    assert res["episode_recall"] == 1.0 and res["about_me_recall"] == 1.0, res


def test_retrieval_set_with_recorded_bge():
    """The same bar on the recorded vectors, so the default suite answers it without loading the model."""
    res = recall_at_5(lambda: Memory(open_db(":memory:"), RecordedEmbedder(), FakeClock()), _data())
    assert res["recall@5"] >= 0.9 and res["never_said_empty"] >= 0.75, res
    assert res["episode_recall"] == 1.0 and res["about_me_recall"] == 1.0, res
