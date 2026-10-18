import unicodedata

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


def test_cjk_restatement_supersedes_instead_of_colliding(mem):
    a = mem.add_fact("The user's name is 小明", source="user_said", key="user.name")
    b = mem.add_fact("The user's name is 小红", source="user_said", key="user.name")
    assert b is not None and [f.id for f in mem.list_facts()] == [b.id]
    assert mem.history(b.id)[1].id == a.id


def test_tamil_and_hindi_facts_are_distinct_and_embedded(mem):
    e = HashEmbedder()
    assert e.embed(["எனக்கு தேநீர் பிடிக்கும்"]).any() and e.embed(["मुझे चाय पसंद है"]).any()
    assert mem.add_fact("எனக்கு தேநீர் பிடிக்கும்", source="user_said") is not None
    assert mem.add_fact("மேலும் ஒரு வேறு தகவல்", source="user_said") is not None
    assert mem.add_fact("मुझे चाय पसंद है", source="user_said") is not None
    assert mem.add_fact("मेरा घर दिल्ली में है", source="user_said") is not None
    assert len(mem.list_facts()) == 4


def test_accented_text_round_trips_and_matches_decomposed_form(mem):
    composed = unicodedata.normalize("NFC", "The user's favourite place is Café Noir")
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    mem.add_fact(composed, source="user_said")
    assert mem.list_facts()[0].text == composed
    assert mem.add_fact(decomposed, source="inferred") is None


def test_delete_after_bulk_supersession_leaves_nothing_behind(mem):
    mem.add_fact("The user lives in Chennai", source="inferred", key="user.city")
    mem.add_fact("The user lives in Bangalore", source="inferred", key="user.city")
    said = mem.add_fact("The user lives in Madurai", source="user_said", key="user.city")
    mem.delete_fact(said.id)
    assert mem.db.execute("select count(*) from facts").fetchone()[0] == 0
    assert mem.db.execute("select count(*) from forgotten").fetchone()[0] == 3
    assert mem.add_fact("The user lives in Chennai", source="inferred") is None


def test_user_said_overrides_forgotten_marker_but_inference_stays_blocked(mem):
    f = mem.add_fact("The user's old project is called Nova", source="user_said", key="user.old_project")
    mem.delete_fact(f.id)
    assert mem.add_fact("The user's old project is called Nova", source="inferred", key="user.old_project") is None
    assert mem.add_fact("The user's old project is called Nova", source="looked_up") is None
    again = mem.add_fact("The user's old project is called Nova", source="user_said", key="user.old_project")
    assert again is not None and again.status == "active"


def test_check_forgotten_true_blocks_even_user_said(mem):
    f = mem.add_fact("The user's old project is called Nova", source="user_said", key="user.old_project")
    mem.delete_fact(f.id)
    text = "The user's old project is called Nova"
    assert mem.add_fact(text, source="user_said", key="user.old_project", check_forgotten=True) is None
    assert mem.add_fact(text, source="inferred", key="user.old_project", check_forgotten=False) is not None


def test_delete_removes_whole_supersession_chain(mem):
    old = mem.add_fact("The user's exam is on Thursday", source="user_said")
    new = mem.correct(old.id, "The user's exam is on Friday")
    mem.delete_fact(new.id)
    assert mem.db.execute("select count(*) from facts").fetchone()[0] == 0
    assert mem.db.execute("select count(*) from forgotten").fetchone()[0] == 2
    assert mem.add_fact("The user's exam is on Thursday", source="inferred") is None
    assert mem.add_fact("The user's exam is on Friday", source="inferred") is None


def test_delete_from_the_oldest_version_also_removes_the_chain(mem):
    old = mem.add_fact("The user's exam is on Thursday", source="user_said")
    mem.correct(old.id, "The user's exam is on Friday")
    mem.delete_fact(old.id)
    assert mem.db.execute("select count(*) from facts").fetchone()[0] == 0


def test_user_said_resolves_a_conflict_pair(mem):
    mem.add_fact("The user lives in Chennai", source="inferred", key="user.city")
    mem.add_fact("The user lives in Bangalore", source="inferred", key="user.city")
    said = mem.add_fact("The user lives in Madurai", source="user_said", key="user.city")
    assert [f.id for f in mem.list_facts()] == [said.id]


def test_correct_resolves_a_conflict_pair(mem):
    a = mem.add_fact("The user lives in Chennai", source="inferred", key="user.city")
    mem.add_fact("The user lives in Bangalore", source="inferred", key="user.city")
    new = mem.correct(a.id, "The user lives in Madurai")
    assert [f.id for f in mem.list_facts()] == [new.id]


def test_correct_rejects_superseded_and_unknown_ids(mem):
    old = mem.add_fact("The user's exam is on Thursday", source="user_said", key="user.exam_day")
    mem.correct(old.id, "The user's exam is on Friday")
    with pytest.raises(ValueError):
        mem.correct(old.id, "The user's exam is on Saturday")
    with pytest.raises(KeyError):
        mem.correct(9999, "anything")


def test_bogus_source_is_rejected_and_leaves_existing_fact_alone(mem):
    f = mem.add_fact("The user prefers tea", source="inferred", key="user.drink")
    with pytest.raises(ValueError):
        mem.add_fact("The user prefers coffee", source="bogus", key="user.drink")
    assert [(x.id, x.status) for x in mem.list_facts()] == [(f.id, "active")]


def test_failed_insert_rolls_back_the_supersession_update(mem):
    import sqlite3

    f = mem.add_fact("The user prefers tea", source="inferred", key="user.drink")
    with pytest.raises(sqlite3.IntegrityError):
        mem.add_fact("The user prefers coffee", source="user_said", key="user.drink", confidence=None)
    mem.db.commit()
    assert [(x.id, x.status) for x in mem.list_facts(include_superseded=True)] == [(f.id, "active")]


def test_embeddings_from_another_embedder_dim_never_match_or_raise(mem):
    mem.add_fact("The user likes tea", source="user_said")
    other = Memory(mem.db, HashEmbedder(dim=64), mem.clock)
    assert other.add_fact("The user likes coffee", source="inferred") is not None


# --- final review ---------------------------------------------------------------------------------------------------

class _SameVector:
    """Every text embeds to one vector, like bge scoring "exam on Thursday" vs "...Friday" at 0.92."""
    dim, strong_sim, weak_sim = 2, 0.8, 0.6

    def embed(self, texts):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))


def test_a_correction_is_never_a_duplicate():
    hashed = Memory(open_db(":memory:"), HashEmbedder(), FakeClock(), dup_sim=0.3)  # the pair scores 0.33
    for mem in (hashed, Memory(open_db(":memory:"), _SameVector(), FakeClock())):
        assert mem.add_fact("The user's exam is on Thursday", source="user_said") is not None
        assert mem.add_fact("The user's exam is on Friday", source="user_said") is not None
        assert mem.add_fact("the user's exam is on friday!", source="inferred") is None   # same words: a duplicate
        assert len(mem.list_facts()) == 2


def test_a_delete_during_consolidation_is_never_undone(mem):
    import threading
    text = "The user's old project is called Nova"
    f = mem.add_fact(text, source="inferred")
    checked, deleted, real = threading.Event(), threading.Event(), mem._max_sim
    adder = []

    def max_sim(vec, rows):  # the consolidator pauses right after its forgotten check
        out = real(vec, rows)
        if threading.current_thread() is adder[0] and not checked.is_set():
            checked.set()
            deleted.wait(1.0)
        return out
    mem._max_sim = max_sim
    adder.append(threading.Thread(target=lambda: mem.add_fact(text, source="inferred", check_forgotten=True)))
    deleter = threading.Thread(target=lambda: mem.delete_fact(f.id))
    adder[0].start()
    assert checked.wait(2)
    deleter.start()
    deleter.join(0.3)
    deleted.set()
    adder[0].join(2); deleter.join(2)
    assert mem.list_facts() == []
