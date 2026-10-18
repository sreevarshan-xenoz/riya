import numpy as np
import pytest

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory
from riya.core.store import open_db


@pytest.fixture
def mem():
    return Memory(open_db(":memory:"), HashEmbedder(), FakeClock())


def test_add_and_list(mem):
    f = mem.add_fact("The user has an exam on Thursday", source="user_said", key="user.exam_day")
    assert f.status == "active" and f.source == "user_said"
    assert [x.id for x in mem.list_facts()] == [f.id]


def test_duplicate_is_not_added(mem):
    mem.add_fact("The user likes free open source tools", source="user_said")
    assert mem.add_fact("the user likes free open-source tools", source="inferred") is None
    assert len(mem.list_facts()) == 1


def test_correction_supersedes_and_keeps_history(mem):
    old = mem.add_fact("The user's exam is on Thursday", source="user_said", key="user.exam_day")
    new = mem.correct(old.id, "The user's exam is on Friday")
    assert [f.id for f in mem.list_facts()] == [new.id]
    hist = mem.history(new.id)
    assert [f.id for f in hist] == [new.id, old.id]
    assert hist[1].status == "superseded" and hist[1].valid_until is not None


def test_explicit_statement_outranks_inference_on_same_key(mem):
    inferred = mem.add_fact("The user prefers tea", source="inferred", key="user.drink")
    said = mem.add_fact("The user prefers coffee", source="user_said", key="user.drink")
    assert [f.id for f in mem.list_facts()] == [said.id]
    assert mem.history(said.id)[1].id == inferred.id


def test_same_rank_contradiction_stays_visible(mem):
    a = mem.add_fact("The user lives in Chennai", source="inferred", key="user.city")
    b = mem.add_fact("The user lives in Bangalore", source="inferred", key="user.city")
    assert {f.id: f.status for f in mem.list_facts()} == {a.id: "conflict", b.id: "conflict"}


def test_delete_is_real_and_leaves_forgotten_marker(mem):
    f = mem.add_fact("The user's old project is called Nova", source="user_said", key="user.old_project")
    mem.delete_fact(f.id)
    assert mem.list_facts() == []
    assert mem.db.execute("select count(*) from facts").fetchone()[0] == 0
    assert mem.add_fact("The user's old project is called Nova", source="inferred") is None
    assert mem.add_fact("Something else about that project", source="inferred", key="user.old_project") is None


def test_emoji_text_round_trips(mem):
    mem.add_fact("The user's folder is 'cute alya 💞'", source="user_said")
    assert mem.list_facts()[0].text == "The user's folder is 'cute alya 💞'"


def test_hash_embedder_is_normalized_and_tolerates_tokenless_text():
    e = HashEmbedder()
    v = e.embed(["The user likes tea", "💞 !!!", ""])
    assert v.shape == (3, e.dim) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1) < 1e-5
    assert not v[1].any() and not v[2].any()


def test_tokenless_facts_do_not_crash_or_collide(mem):
    a = mem.add_fact("💞", source="user_said")
    b = mem.add_fact("💖", source="user_said")
    assert a is not None and b is not None


def test_old_source_outranking_new_marks_both_conflict(mem):
    said = mem.add_fact("The user lives in Chennai", source="user_said", key="user.city")
    guess = mem.add_fact("The user lives in Bangalore", source="inferred", key="user.city")
    assert {f.id: f.status for f in mem.list_facts()} == {said.id: "conflict", guess.id: "conflict"}


def test_delete_with_episodes_removes_source_episode(mem):
    mem.db.execute("insert into episodes(id, started_at, ended_at, summary) values(1, 0, 1, 'chat')")
    f = mem.add_fact("The user likes tea", source="user_said", episode_id=1)
    mem.delete_fact(f.id, delete_episodes=True)
    assert mem.db.execute("select count(*) from episodes").fetchone()[0] == 0
    assert mem.db.execute("select count(*) from forgotten").fetchone()[0] == 1


def test_identical_tokenless_text_is_a_duplicate(mem):
    assert mem.add_fact("💞", source="user_said") is not None
    assert mem.add_fact("💞", source="user_said") is None
