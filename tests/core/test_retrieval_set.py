import json
from pathlib import Path

import pytest

from evaluation.retrieval_eval import recall_at_5
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder, content_tokens
from riya.core.memory import Memory
from riya.core.store import open_db


def _data() -> dict:
    return json.loads(Path("tests/data/retrieval_set.json").read_text(encoding="utf-8"))


def test_retrieval_set_with_hash_embedder():
    data = _data()
    assert len(data["questions"]) == 41
    res = recall_at_5(lambda: Memory(open_db(":memory:"), HashEmbedder(), FakeClock()), data)
    assert res["recall@5"] >= 0.6 and res["never_said_empty"] >= 0.75


def test_never_said_questions_share_no_content_word_with_a_fact():
    data = _data()
    facts = set().union(*(content_tokens(f["text"]) for f in data["facts"]))
    assert sum(1 for q in data["questions"] if q["kind"] == "hard_negative") >= 4
    for q in data["questions"]:
        if not q["expect"]:
            assert not content_tokens(q["q"]) & facts, q["q"]


def test_retrieval_set_with_bge():
    pytest.importorskip("fastembed")
    from riya.core.embed import FastEmbedder
    try:
        embedder = FastEmbedder()
    except Exception as e:  # model not downloaded and no network
        pytest.skip(f"bge model unavailable: {e}")
    res = recall_at_5(lambda: Memory(open_db(":memory:"), embedder, FakeClock()), _data())
    assert res["recall@5"] >= 0.9 and res["never_said_empty"] >= 0.75, res
