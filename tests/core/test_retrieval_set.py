import json
from pathlib import Path

from evaluation.retrieval_eval import recall_at_5
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory
from riya.core.store import open_db


def test_retrieval_set_with_hash_embedder():
    data = json.loads(Path("tests/data/retrieval_set.json").read_text(encoding="utf-8"))
    assert len(data["questions"]) == 30
    res = recall_at_5(lambda: Memory(open_db(":memory:"), HashEmbedder(), FakeClock()), data)
    assert res["recall@5"] >= 0.6 and res["never_said_empty"] >= 0.75
