"""v0.11 Task 5: the personal fill (§4.2). With personal=True (the engine's local reply; the default is off), a personal
request gets up to PERSONAL_EXTRA more live facts at or above the embedder's personal_sim floor, after the eligible ones
and never more than k_facts in all; every other query retrieves exactly what it did before."""
import json
import random
from pathlib import Path

import numpy as np

from evaluation.persona_eval import load_probes
from evaluation.recorded_vectors import RecordedEmbedder
from riya.core import memory
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import PERSONAL_EXTRA, Memory, about_me, personal_request
from riya.core.store import open_db
from tests.embed_stubs import TableEmbedder

DATA = Path(__file__).resolve().parents[1] / "data"
DAY = 86400.0
SNACK = "pick a snack for me, you know what i like"  # a personal request; shares "snack" with SALTY
SALTY = "The user loves salty snack mixes"
DOG_Q = "what's my dog called again"  # a direct question (a dev-set negative)
ROLO = "The user's dog is Rolo"
ABOUT_ME = "what do you remember about me?"


def _dev() -> dict:
    return json.loads((DATA / "personal_set.json").read_text(encoding="utf-8"))


def _mem(sims: dict[str, float], embedder=None) -> tuple[Memory, dict[str, int]]:
    """A memory holding the texts of sims, a minute apart (the first is the oldest); returns it and their ids."""
    mem = Memory(open_db(":memory:"), embedder or TableEmbedder(sims), FakeClock())
    ids = {}
    for text in sims:
        ids[text] = mem.add_fact(text, source="user_said").id
        mem.clock.advance(60)
    return mem, ids


def _others(n: int, lo: float, step: float = 0.01) -> dict[str, float]:
    """n facts at lo, lo + step, ...: no content word in common with SNACK, DOG_Q or each other's query."""
    return {f"The user keeps item number {i} in drawer {i}": round(lo + i * step, 4) for i in range(n)}


def _texts(mem: Memory, ids) -> list[str]:
    return [mem.get_fact(i).text for i in ids]


def test_a_personal_request_fills_up_to_three():
    assert personal_request(SNACK)
    sims = {SALTY: 0.65, **_others(6, 0.50)}
    mem, ids = _mem(sims)
    r = mem.retrieve(SNACK, personal=True)
    assert r.facts[0].id == ids[SALTY]
    top3 = sorted(_others(6, 0.50), key=sims.get, reverse=True)[:3]
    assert [f.id for f in r.facts[1:]] == [ids[t] for t in top3]
    assert len(r.filled) == PERSONAL_EXTRA == 3
    assert r.filled == tuple(f.id for f in r.facts[1:])


def test_the_fill_is_off_by_default():
    """Only the engine's local reply opts in (ledgered ruling): every other caller retrieves as before."""
    mem, ids = _mem({SALTY: 0.65, **_others(6, 0.50)})
    assert len(mem.retrieve(SNACK, personal=True).filled) == 3
    r = mem.retrieve(SNACK)
    assert [f.id for f in r.facts] == [ids[SALTY]] and r.filled == ()
    assert r == mem.retrieve(SNACK, personal=False)


def test_the_total_stays_at_five():
    strong = {f"The user keeps strong thing {i} nearby": 0.85 for i in range(5)}
    four, five = dict(list(strong.items())[:4]), strong
    mem, ids = _mem({**four, **_others(5, 0.50)})
    r = mem.retrieve(SNACK, personal=True)
    assert len(r.facts) == 5 and len(r.filled) == 1
    assert [f.id for f in r.facts[:4]] == [ids[t] for t in reversed(list(four))]  # the eligible four, newest first
    assert r.filled == (r.facts[4].id,)
    mem, _ = _mem({**five, **_others(5, 0.50)})
    r = mem.retrieve(SNACK, personal=True)
    assert len(r.facts) == 5 and r.filled == ()
    assert {f.text for f in r.facts} == set(five)


def test_the_fill_ranks_by_score():
    """M1: by _score, not raw cosine. A fresh user_said fact at 0.50 (score ~0.70) beats a 90-day-old inferred one at
    0.59 (~0.44)."""
    stale, fresh = "The user once mentioned liking marzipan", "The user keeps item number 0 in drawer 0"
    strong = {f"The user keeps strong thing {i} nearby": 0.85 for i in range(4)}
    for eligible in ({}, strong):
        mem = Memory(open_db(":memory:"), TableEmbedder({stale: 0.59, fresh: 0.50, **eligible}), FakeClock())
        old = mem.add_fact(stale, source="inferred")
        mem.clock.advance(90 * DAY)
        new = mem.add_fact(fresh, source="user_said")
        for t in eligible:
            mem.add_fact(t, source="user_said")
        r = mem.retrieve(SNACK, personal=True)
        assert r.filled == ((new.id, old.id) if not eligible else (new.id,))


def test_the_floor_is_inclusive():
    """M2: a cosine equal to the floor is filled (0.5 is exact in float32). A nominal 0.45 under the default floor is
    float32 0.44999998, so it is left out, on the safe side."""
    at = "The user keeps item number 0 in drawer 0"
    mem = Memory(open_db(":memory:"), TableEmbedder({at: 0.5}, personal=0.5), FakeClock())
    kept = mem.add_fact(at, source="user_said")
    assert mem.retrieve(SNACK, personal=True).filled == (kept.id,)
    assert float(np.float32(0.45)) < 0.45
    sims = {"The user keeps item number 1 in drawer 1": 0.46, "The user keeps item number 2 in drawer 2": 0.45,
            "The user keeps item number 3 in drawer 3": 0.44}
    mem, ids = _mem(sims)
    assert _texts(mem, mem.retrieve(SNACK, personal=True).filled) == ["The user keeps item number 1 in drawer 1"]


def test_nothing_above_the_floor_adds_nothing():
    """Review Focus 4: a personal request with nothing at or above the floor, or an empty memory."""
    mem, _ = _mem(_others(6, 0.44, step=0.0))
    r = mem.retrieve(SNACK, personal=True)
    assert r.facts == [] and r.filled == ()
    empty = Memory(open_db(":memory:"), TableEmbedder({}), FakeClock())
    r = empty.retrieve(SNACK, personal=True)
    assert r.facts == [] and r.filled == ()


def test_the_fill_draws_only_live_facts():
    """Review Focus 2: a superseded or deleted fact with a high cosine is never filled."""
    old, new = "The user's go-to treat is caramel popcorn", "The user's go-to treat is now dark chocolate"
    gone, live = "The user hoards sour gummy worms", "The user keeps item number 0 in drawer 0"
    mem = Memory(open_db(":memory:"), TableEmbedder({old: 0.59, new: 0.30, gone: 0.59, live: 0.50}), FakeClock())
    first = mem.add_fact(old, source="user_said", key="treat")
    mem.add_fact(new, source="user_said", key="treat")
    assert mem.get_fact(first.id).status == "superseded"
    doomed = mem.add_fact(gone, source="user_said")
    mem.delete_fact(doomed.id)
    kept = mem.add_fact(live, source="user_said")
    r = mem.retrieve(SNACK, personal=True)
    assert [f.id for f in r.facts] == [kept.id] and r.filled == (kept.id,)
    assert not {old, gone} & {f.text for f in r.facts}  # by text: sqlite reuses a deleted row's id


class _AliasTable(TableEmbedder):
    """A TableEmbedder where each alias embeds exactly as its target text."""

    def __init__(self, sims: dict[str, float], aliases: dict[str, str]):
        super().__init__(sims)
        self.aliases = aliases

    def embed(self, texts: list[str]) -> np.ndarray:
        return super().embed([self.aliases.get(t, t) for t in texts])


def test_a_fill_slot_never_goes_to_a_second_version():
    """M5: a live fact on the key of one already chosen (a conflict pair, or the key of an eligible fact) and a
    near-duplicate by add_fact's rule (correct() can make one) are skipped; the slot goes to the next candidate. In each
    memory the skipped fact outranks the fillers, so without the skip it would take a slot."""
    x1, x2, x3 = _others(3, 0.48)  # at 0.48, 0.49, 0.50: they fill in the order x3, x2, x1
    fudge, toffee = "The user's go-to treat is fudge", "The user's go-to treat is toffee"
    sims = {fudge: 0.56, toffee: 0.55, **_others(3, 0.48)}
    mem = Memory(open_db(":memory:"), TableEmbedder(sims), FakeClock())
    mem.add_fact(fudge, source="looked_up", key="treat")
    mem.add_fact(toffee, source="looked_up", key="treat")
    assert {f.status for f in mem.list_facts()} == {"conflict"}
    for t in (x1, x2, x3):
        mem.add_fact(t, source="looked_up")
    assert _texts(mem, mem.retrieve(SNACK, personal=True).filled) == [fudge, x3, x2]

    sims = {SALTY: 0.65, toffee: 0.55, **_others(3, 0.48)}
    mem = Memory(open_db(":memory:"), TableEmbedder(sims), FakeClock())
    mem.add_fact(SALTY, source="user_said", key="treat")
    mem.add_fact(toffee, source="inferred", key="treat")
    assert {f.status for f in mem.list_facts()} == {"conflict"}
    for t in (x1, x2, x3):
        mem.add_fact(t, source="inferred")
    r = mem.retrieve(SNACK, personal=True)
    assert [f.text for f in r.facts] == [SALTY, x3, x2, x1]

    pretzels, again, note = "The user loves salty pretzels", "The user loves salty pretzels!", "The user owns a kite"
    mem = Memory(open_db(":memory:"), _AliasTable({pretzels: 0.55, note: 0.30, **_others(3, 0.48)},
                                                  {again: pretzels}), FakeClock())
    mem.add_fact(pretzels, source="user_said")
    mem.correct(mem.add_fact(note, source="user_said").id, again)  # correct() never checks the other live facts
    for t in (x1, x2, x3):
        mem.add_fact(t, source="user_said")
    assert {pretzels, again} <= {f.text for f in mem.list_facts()}
    filled = _texts(mem, mem.retrieve(SNACK, personal=True).filled)
    assert len({pretzels, again} & set(filled)) == 1 and filled[1:] == [x3, x2]


def test_episodes_and_threads_are_untouched():
    """M3: a personal turn's episodes and due threads are exactly the non-personal ones."""
    eps = {f"We talked about snack plan number {i}": 0.85 for i in range(4)}
    thread = "Ask how the snack swap went"
    emb = TableEmbedder({SALTY: 0.65, **_others(4, 0.50), **eps, thread: 0.85})
    mem = Memory(open_db(":memory:"), emb, FakeClock())
    for t in [SALTY, *_others(4, 0.50)]:
        mem.add_fact(t, source="user_said")
    mem.clock.advance(30 * DAY)
    for days, summary in zip((20, 8, 3, 1), eps):
        at = mem.clock.now() - days * DAY
        mem.add_episode(at - 1800, at, summary)
    mem.add_thread(thread, due_at=mem.clock.now() - 10)
    on, off = mem.retrieve(SNACK, personal=True), mem.retrieve(SNACK, personal=False)
    assert on.filled and on.episodes and on.threads
    assert on.episodes == off.episodes and on.threads == off.threads


def test_no_fill_off_personal_requests():
    sims = {ROLO: 0.65, SALTY: 0.65, **_others(6, 0.50)}
    mem, ids = _mem(sims)
    assert not personal_request(DOG_Q)
    r = mem.retrieve(DOG_Q, personal=True)
    assert [f.id for f in r.facts] == [ids[ROLO]] and r.filled == ()
    r = mem.retrieve(SNACK, personal=False)
    assert [f.id for f in r.facts] == [ids[SALTY]] and r.filled == ()
    hashed, _ = _mem(sims, embedder=HashEmbedder())
    assert not hasattr(hashed.embedder, "personal_sim")
    r = hashed.retrieve(SNACK, personal=True)
    assert r == hashed.retrieve(SNACK, personal=False) and r.filled == ()


def test_about_me_keeps_the_eight_fact_fill():
    """A9: the explicit about-me path is untouched."""
    assert about_me(ABOUT_ME) and not personal_request(ABOUT_ME)
    mem, _ = _mem({f"The user keeps item number {i} in drawer {i}": 0.5 for i in range(10)})
    r = mem.retrieve(ABOUT_ME, personal=True)
    assert len(r.facts) == 8 and r.filled == ()


def test_about_me_never_fills_even_if_the_detector_fires(monkeypatch):
    """M4, defence in depth: were personal_request ever loosened to fire on an about-me query, the 8-fact fill
    still wins."""
    monkeypatch.setattr(memory, "personal_request", lambda q: True)
    mem, _ = _mem({f"The user keeps item number {i} in drawer {i}": 0.5 for i in range(10)})
    r = mem.retrieve(ABOUT_ME, personal=True)
    assert len(r.facts) == 8 and r.filled == ()


def test_bounds_hold_on_random_memories():
    """A5: whatever the memory, a turn adds at most 3 facts at or above the floor, holds at most 5, and never moves an
    eligible fact."""
    dev = _dev()
    queries = [r["q"] for r in dev["requests"]] + [n["q"] for n in dev["negatives"]]
    rng = random.Random(7)
    fired = 0
    for m in range(200):
        n = rng.randint(0, 12)
        sims = {f"The user keeps thing {m}-{i} somewhere": rng.uniform(0.2, 0.95) for i in range(n)}
        mem = Memory(open_db(":memory:"), TableEmbedder(sims), FakeClock())
        for text in sims:
            mem.add_fact(text, source=rng.choice(["user_said", "looked_up", "inferred"]))
            mem.clock.advance(rng.uniform(0, 40) * DAY)
        q = rng.choice(queries)
        qv = mem.embedder.embed([q])[0]
        r = mem.retrieve(q, personal=True)
        ids = [f.id for f in r.facts]
        assert len(r.filled) <= PERSONAL_EXTRA and len(r.facts) <= 5
        assert set(r.filled) <= set(ids)
        assert tuple(ids[len(ids) - len(r.filled):]) == r.filled  # the fill comes after the eligible facts
        for f in r.facts:
            if f.id in r.filled:
                assert float(np.dot(qv, mem.embedder.embed([f.text])[0])) >= 0.45
        assert r.facts[:len(ids) - len(r.filled)] == mem.retrieve(q, personal=False).facts
        if not personal_request(q):
            assert r.filled == ()
        fired += bool(r.filled)
    assert fired > 0


# --- A8: non-personal retrieval is unchanged, on the recorded bge vectors ---

def _same(mem: Memory, q: str) -> bool:
    """With the fill on, q retrieves exactly what it does with it off, and nothing is filled."""
    on = mem.retrieve(q, personal=True)
    return on == mem.retrieve(q, personal=False) and on.filled == ()


def test_dev_negatives_retrieve_as_before():
    dev = _dev()
    text = {f["id"]: f["text"] for f in dev["facts"]}
    embedder = RecordedEmbedder()
    pooled = Memory(open_db(":memory:"), embedder, FakeClock())
    for t in text.values():
        pooled.add_fact(t, source="user_said")
    for n in dev["negatives"]:
        assert not personal_request(n["q"]), n["id"]
        own = Memory(open_db(":memory:"), embedder, FakeClock())
        for fid in n["memory"]:
            own.add_fact(text[fid], source="user_said")
        assert _same(own, n["q"]), n["id"]
        assert _same(pooled, n["q"]), n["id"]


def test_bank_non_about_me_turns_retrieve_as_before():
    embedder = RecordedEmbedder()
    checked = 0
    for c in load_probes()["conversations"]:
        mem = Memory(open_db(":memory:"), embedder, FakeClock())
        for t in c["facts"]:
            mem.add_fact(t, source="user_said")
        for turn in c["turns"]:
            if turn["kind"] == "about_me" or not turn.get("text") or turn.get("speak_first"):
                continue
            assert _same(mem, turn["text"]), (c["id"], turn["id"])
            checked += 1
    assert checked >= 70


def test_retrieval_set_retrieves_as_before():
    data = json.loads((DATA / "retrieval_set.json").read_text(encoding="utf-8"))
    mem = Memory(open_db(":memory:"), RecordedEmbedder(), FakeClock())
    for fact in data["facts"]:
        mem.add_fact(fact["text"], fact["source"], key=fact.get("key"))
        mem.clock.advance(DAY)
    for ep in data.get("episodes", []):
        at = mem.clock.now() - ep["days_ago"] * DAY
        mem.add_episode(at - 1800, at, ep["summary"])
    questions = [q["q"] for q in data["questions"]]
    assert [q for q in questions if personal_request(q)] == []
    for q in questions:
        assert _same(mem, q), q
