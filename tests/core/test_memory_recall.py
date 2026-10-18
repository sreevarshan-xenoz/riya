import json

import pytest

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory
from riya.core.store import open_db


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def mem(clock):
    return Memory(open_db(":memory:"), HashEmbedder(), clock)


def test_due_threads_and_marking(mem, clock):
    t = mem.add_thread("Ask how the exam went", due_at=clock.now() + 3600)
    assert mem.due_threads() == []
    clock.advance(3601)
    assert [x.id for x in mem.due_threads()] == [t.id]
    mem.mark_thread(t.id, status="followed_up", raised_at=clock.now())
    assert mem.due_threads() == []


def test_mark_thread_none_leaves_fields_unchanged(mem, clock):
    t = mem.add_thread("Ask about the trip", due_at=clock.now() - 10)
    mem.mark_thread(t.id, raised_at=123.0)
    (got,) = mem.due_threads()
    assert got.status == "open" and got.last_raised_at == 123.0
    mem.mark_thread(t.id, status="resolved")
    assert mem.db.execute("select status, last_raised_at from threads").fetchone()[:] == ("resolved", 123.0)


def test_due_threads_latest_due_first(mem, clock):
    a = mem.add_thread("older", due_at=clock.now() - 100)
    b = mem.add_thread("newer", due_at=clock.now() - 10)
    assert [t.id for t in mem.due_threads()] == [b.id, a.id]


def test_transcripts_purge_after_30_days(mem, clock):
    mem.append_transcript("user", "hi")
    clock.advance(31 * 86400)
    mem.append_transcript("user", "still here")
    assert mem.purge_transcripts() == 1
    assert [t["text"] for t in mem.unconsolidated()] == ["still here"]


def test_unconsolidated_and_mark_consolidated(mem):
    a = mem.append_transcript("user", "one")
    b = mem.append_transcript("riya", "two")
    rows = mem.unconsolidated()
    assert [(r["id"], r["role"], r["text"]) for r in rows] == [(a, "user", "one"), (b, "riya", "two")]
    assert set(rows[0]) == {"id", "at", "role", "text"}
    mem.mark_consolidated([a])
    mem.mark_consolidated([])
    assert [r["id"] for r in mem.unconsolidated()] == [b]


def test_delete_transcript_day(mem, clock):
    mem.append_transcript("user", "day one")
    clock.advance(86400)
    mem.append_transcript("user", "day two")
    first, second = mem.transcript_days()
    assert mem.delete_transcript_day(first) == 1
    assert mem.transcript_days() == [second]
    assert mem.delete_transcript_day("not-a-day") == 0


def test_retrieve_relevant_and_nothing_for_unknown(mem):
    mem.add_fact("The user has a calculus exam on Thursday", source="user_said")
    mem.add_fact("The user's favourite food is biryani", source="user_said")
    r = mem.retrieve("how did your calculus exam go")
    assert r.facts[0].text.startswith("The user has a calculus exam")
    assert mem.retrieve("what is my sister's name").facts == []


def test_retrieve_skips_superseded_and_prefers_recent_and_trusted(mem, clock):
    old = mem.add_fact("The user has a calculus exam on Thursday", source="user_said", key="exam")
    mem.correct(old.id, "The user has a calculus exam on Friday")
    texts = [f.text for f in mem.retrieve("calculus exam").facts]
    assert texts == ["The user has a calculus exam on Friday"]
    mem.add_fact("The user enjoys cycling", source="inferred")
    mem.add_fact("The user enjoys swimming", source="user_said")
    assert [f.source for f in mem.retrieve("enjoys").facts][:2] == ["user_said", "inferred"]
    clock.advance(200 * 86400)  # recency now outweighs the trust gap
    mem.add_fact("The user enjoys hiking", source="inferred")
    assert mem.retrieve("enjoys").facts[0].text == "The user enjoys hiking"


def test_retrieve_includes_episodes_and_due_threads(mem, clock):
    ep = mem.add_episode(clock.now(), clock.now(), "We discussed the calculus exam preparation")
    mem.add_episode(clock.now(), clock.now(), "Played a word game about animals")
    t = mem.add_thread("Ask how the exam went", due_at=clock.now() - 1)
    r = mem.retrieve("calculus exam", k_episodes=2)
    assert ep.id in [e.id for e in r.episodes]
    assert [x.id for x in r.threads] == [t.id]


def test_recent_episodes_newest_first(mem, clock):
    a = mem.add_episode(clock.now(), clock.now() + 60, "Talked about the exam")
    b = mem.add_episode(clock.now() + 100, clock.now() + 200, "Played a word game")
    assert [e.id for e in mem.recent_episodes(2)] == [b.id, a.id]
    assert [e.id for e in mem.recent_episodes(1)] == [b.id]


def test_delete_fact_can_delete_related_episodes(mem, clock):
    ep = mem.add_episode(clock.now(), clock.now(), "The user talked about the calculus exam on Thursday")
    f = mem.add_fact("The user has a calculus exam on Thursday", source="user_said", episode_id=ep.id)
    mem.delete_fact(f.id, delete_episodes=True)
    assert mem.recent_episodes(5) == []


def test_delete_fact_deletes_similar_episodes_but_keeps_unrelated(mem, clock):
    mem.add_episode(clock.now(), clock.now(), "The user has a calculus exam on Thursday soon")
    other = mem.add_episode(clock.now(), clock.now(), "Played a word game about animals")
    f = mem.add_fact("The user has a calculus exam on Thursday", source="user_said")
    mem.delete_fact(f.id, delete_episodes=True)
    assert [e.id for e in mem.recent_episodes(5)] == [other.id]


def test_delete_fact_without_flag_keeps_episodes(mem, clock):
    mem.add_episode(clock.now(), clock.now(), "The user has a calculus exam on Thursday")
    f = mem.add_fact("The user has a calculus exam on Thursday", source="user_said")
    mem.delete_fact(f.id)
    assert len(mem.recent_episodes(5)) == 1


def test_delete_fact_considers_every_version_text(mem, clock):
    mem.add_episode(clock.now(), clock.now(), "The user has a calculus exam on Thursday")
    old = mem.add_fact("The user has a calculus exam on Thursday", source="user_said", key="exam")
    new = mem.correct(old.id, "The user has a chemistry lab on Friday")
    mem.delete_fact(new.id, delete_episodes=True)
    assert mem.recent_episodes(5) == []


def test_import_legacy_json(tmp_path, mem):
    p = tmp_path / "riya_memory.json"
    p.write_text(json.dumps({"summary": "User is building Riya.", "turns": []}), encoding="utf-8")
    ep = mem.import_legacy_json(p)
    assert ep.summary == "User is building Riya." and mem.recent_episodes(1)[0].id == ep.id
    assert ep.outcome == "imported" and ep.started_at == ep.ended_at
    assert mem.import_legacy_json(tmp_path / "nope.json") is None


def test_import_legacy_json_empty_or_bad(tmp_path, mem):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"summary": "  "}), encoding="utf-8")
    assert mem.import_legacy_json(p) is None
    p.write_text("not json", encoding="utf-8")
    assert mem.import_legacy_json(p) is None
    assert mem.recent_episodes(5) == []


class _Dense:
    """Embedder stub: 'sim=0.7 cats' sits at cosine 0.7 from any query ('cats?'), each fact in its own direction."""
    dim, strong_sim, weak_sim = 3, 0.8, 0.6
    WORDS = ("apples", "cats", "pears", "dogs")

    def embed(self, texts):
        import math

        import numpy as np
        out = []
        for t in texts:
            if not t.startswith("sim="):
                out.append([1.0, 0.0, 0.0])
                continue
            sim, phi = float(t.split()[0][4:]), math.pi / 2 * self.WORDS.index(t.split()[-1])
            r = math.sqrt(1 - sim * sim)
            out.append([sim, r * math.cos(phi), r * math.sin(phi)])
        return np.array(out, dtype=np.float32)


def _dense_mem(clock, **kw):
    mem = Memory(open_db(":memory:"), _Dense(), clock, **kw)
    for text in ["sim=0.9 apples", "sim=0.7 cats", "sim=0.7 pears", "sim=0.5 dogs"]:
        mem.add_fact(text, "user_said")
    return mem


def test_hybrid_eligibility_uses_embedder_thresholds(clock):
    # query "cats?" shares a content word with the 0.7 "cats" fact only
    texts = [f.text for f in _dense_mem(clock).retrieve("cats?").facts]
    assert sorted(texts) == ["sim=0.7 cats", "sim=0.9 apples"]  # strong, or weak + shared word; pears, dogs out


def test_explicit_min_sim_overrides_both_thresholds(clock):
    texts = [f.text for f in _dense_mem(clock, min_sim=0.45).retrieve("cats?").facts]
    assert len(texts) == 4
    assert _dense_mem(clock, min_sim=0.95).retrieve("cats?").facts == []


# --- final review: recent episodes and "what do you remember about me?" ---------------------------------------------

DAY = 86400


def test_retrieve_always_includes_the_latest_episodes(mem, clock):
    now = clock.now()
    mem.add_episode(now - 30 * DAY, now - 30 * DAY, "Played a word game about animals")
    calc = mem.add_episode(now - 20 * DAY, now - 20 * DAY, "We discussed the calculus exam preparation")
    prev = mem.add_episode(now - 3 * DAY, now - 3 * DAY, "Planned the weekend trip to Goa")
    last = mem.add_episode(now - DAY, now - DAY, "Chose a new mechanical keyboard")
    assert {e.id for e in mem.retrieve("what did we talk about last time?").episodes} == {last.id, prev.id}
    assert {e.id for e in mem.retrieve("calculus exam").episodes} == {last.id, prev.id, calc.id}
    assert len(mem.retrieve("calculus exam keyboard Goa", k_episodes=3).episodes) == 3


def test_the_second_latest_episode_only_when_it_is_recent(mem, clock):
    now = clock.now()
    mem.add_episode(now - 9 * DAY, now - 9 * DAY, "Planned the weekend trip to Goa")
    last = mem.add_episode(now - 8 * DAY, now - 8 * DAY, "Chose a new mechanical keyboard")
    assert [e.id for e in mem.retrieve("hello").episodes] == [last.id]


@pytest.mark.parametrize("q", ["What do you remember about me?", "what do you know about me", "Tell me everything you "
                               "remember", "what have I told you so far?"])
def test_about_me_returns_the_best_facts_without_a_similarity_gate(mem, clock, q):
    for i in range(10):
        mem.add_fact(f"The user fact number {i} is about topic{i}", source="inferred" if i % 2 else "user_said")
        clock.advance(DAY)
    got = [f.text for f in mem.retrieve(q).facts]
    assert len(got) == 8
    assert got[0] == "The user fact number 8 is about topic8"      # newest of the trusted ones
    assert "The user fact number 9 is about topic9" in got


def test_about_me_with_nothing_stored_is_empty(mem):
    assert mem.retrieve("what do you remember about me?").facts == []


def test_retrieve_includes_at_most_two_threads_relevant_to_the_query(mem, clock):
    due = clock.now() - 60
    exam = mem.add_thread("Ask how the calculus exam went", due_at=due)
    mem.add_thread("Ask whether the calculus exam results are out", due_at=due)
    mem.add_thread("Ask if the calculus exam hall was cold", due_at=due)
    mem.add_thread("Ask about the trip to Goa", due_at=due)
    got = mem.retrieve("how did the calculus exam go").threads
    assert len(got) == 2 and all("calculus exam" in t.text for t in got)
    assert mem.retrieve("what should I cook tonight").threads == []
    assert exam.id in [t.id for t in mem.retrieve("calculus exam went").threads]


def test_stale_open_threads_are_resolved(mem, clock):
    stale = mem.add_thread("Ask about the old trip", due_at=clock.now() - 8 * DAY)
    fresh = mem.add_thread("Ask about the exam", due_at=clock.now() - 6 * DAY)
    assert mem.resolve_stale_threads() == 1
    assert [t.id for t in mem.due_threads()] == [fresh.id]
    assert mem.db.execute("select status from threads where id=?", (stale.id,)).fetchone()[0] == "resolved"
