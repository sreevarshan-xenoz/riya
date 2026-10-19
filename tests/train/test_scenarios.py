"""v0.8 Task 1: the seeded fake scenarios, the probe hold-out and the kind-by-kind classifier check."""
import json
import random
from functools import lru_cache

import pytest

from riya.core import games
from riya.core.consolidate import remember_intent
from riya.core.eyes.see import is_screen_question
from riya.core.hands.intent import is_forbidden, match
from riya.core.learn.dials import DIALS, clamp
from riya.core.learn.feedback import detect_feedback
from riya.core.memory import about_me
from riya.core.router import needs_thinking
from riya.core.shaping import is_feelings_question, is_identity_question
from riya.train import scenarios as sc
from riya.train.scenarios import (BASE_T, KINDS, PROBES_PATH, counts, events_of, normalize, overlaps, probe_facts,
                                  probe_texts, scenario, scenarios, thread_of, tool_result)

REPORT_KINDS = {"failed", "lost", "stalled", "waiting", "finished"}


@lru_cache(maxsize=None)
def rows(seed, n=3000):
    return scenarios(seed, n)


def test_counts():
    assert counts() == {"small": 540, "about_me": 300, "unknown": 360, "identity": 180, "remember": 240,
                        "explain": 240, "greeting": 240, "bid": 180, "game": 180, "tool": 180, "feedback": 180,
                        "report": 180}
    assert list(counts()) == list(KINDS)
    assert sum(counts(7, ("small", "bid")).values()) == 7
    assert sum(counts(1001).values()) == 1001
    with pytest.raises(ValueError):
        counts(10, ("nope",))


def test_deterministic_and_seeded():
    assert scenarios(0, 300) == scenarios(0, 300)
    assert scenarios(1, 300) != scenarios(0, 300)
    assert scenario("bid", 5, 0) == next(s for s in rows(0) if s.id == "bid-00005")
    assert scenarios(0, 40, ("bid",))[5] == scenario("bid", 5, 0)
    with pytest.raises(ValueError):
        scenarios(0, 10, ("nope",))


def test_every_kind_is_covered():
    all_rows = rows(0)
    assert len(all_rows) == 3000
    assert {s.kind for s in all_rows} == set(KINDS)
    assert len({s.id for s in all_rows}) == len(all_rows)
    assert [s.kind for s in all_rows] == [k for k in KINDS for _ in range(counts()[k])]


def test_overlap_rule():
    assert overlaps("Hey, Riya!", ["hey riya"])
    assert not overlaps("so what's my sister's name?", ["what's my sister's name?"])
    nervous = next(t["text"] for c in json.loads(PROBES_PATH.read_text(encoding="utf-8"))["conversations"]
                   if c["id"] == "exam_nerves" for t in c["turns"] if t["id"] == "nervous")
    assert overlaps("ok i have my calculus exam tomorrow and stuff", [nervous])
    assert normalize("  What's  UP?! ") == "whats up"


def test_the_probes_load():
    assert len(probe_texts()) >= 20 and len(probe_facts()) >= 5
    assert not any("she speaks first" in t for t in probe_texts())  # speak_first turns carry no user text


def _user_texts(s):
    out = [s.user_text] if s.user_text is not None else []
    return out + [t["content"] for t in s.turns if t["role"] == "user"]


@pytest.mark.parametrize("seed", [0, 1])
def test_no_scenario_text_overlaps_a_probe(seed):
    texts, facts = probe_texts(), {normalize(f) for f in probe_facts()}
    assert texts and facts
    for s in rows(seed):
        for t in _user_texts(s):
            assert not overlaps(t, texts), (s.id, t)
        for f in s.facts:
            assert normalize(f) not in facts, (s.id, f)


def test_a_probe_hit_is_redrawn_and_exhaustion_raises(monkeypatch):
    base = scenario("small", 0, 0)
    real, calls = sc.overlaps, []

    def first_call_hits(text, probes):
        calls.append(text)
        return len(calls) == 1 or real(text, probes)

    monkeypatch.setattr(sc, "overlaps", first_call_hits)
    redrawn = scenario("small", 0, 0)
    assert redrawn != base and redrawn.id == base.id
    monkeypatch.setattr(sc, "overlaps", lambda text, probes: True)
    with pytest.raises(RuntimeError, match="small-3: no probe-free draw"):
        scenario("small", 3, 0)


def test_a_probe_fact_is_redrawn(monkeypatch):
    fact = scenario("about_me", 0, 0).facts[0]
    monkeypatch.setattr(sc, "probe_facts", lambda path=PROBES_PATH: [fact])
    assert fact not in scenario("about_me", 0, 0).facts


def test_every_scenario_takes_its_kinds_branch():
    for s in rows(0):
        t, k = s.user_text, s.kind
        if k in ("greeting", "bid", "report"):
            assert t is None
            continue
        assert isinstance(t, str) and t.strip(), s.id
        assert (remember_intent(t) is not None) == (k == "remember"), (s.id, t)
        assert is_identity_question(t) == (k == "identity"), (s.id, t)
        assert about_me(t) == (k == "about_me"), (s.id, t)
        assert needs_thinking(t) == (k == "explain"), (s.id, t)
        assert (detect_feedback(t) is not None) == (k == "feedback"), (s.id, t)
        assert (match(t, s.now) is not None) == (k == "tool"), (s.id, t)
        assert (games.detect_start(t, None) is not None) == (k == "game" and s.game_start), (s.id, t)
        assert not (is_screen_question(t) or is_forbidden(t) or is_feelings_question(t)), (s.id, t)
        if k == "game" and not s.game_start:
            assert not games.is_stop(t) and not games.about_agents(t), (s.id, t)
        if k == "unknown":
            assert s.asks is not None
            assert s.asks not in {sc.fact_attr(f) for f in s.facts}, s.id


def test_payloads():
    by_kind = {k: [s for s in rows(0) if s.kind == k] for k in KINDS}
    for s in rows(0):
        assert tuple(s.dials) == DIALS, s.id
        assert all(v == clamp(d, v) for d, v in s.dials.items()), s.id
        assert len(s.notes) in (0, 1) and len(s.turns) in (0, 2)
        if s.turns:
            assert [t["role"] for t in s.turns] == ["user", "assistant"]
        assert set(s.mood) == {"valence", "arousal", "warmth", "concern", "mood", "energy", "curiosity"}
        assert all(v == round(v, 3) for v in s.mood.values())
        assert BASE_T <= s.now < BASE_T + 365 * 86400 + 86400
        assert all(f.startswith("The user") for f in s.facts)
    assert all(s.away_s >= 3600 for s in by_kind["greeting"])
    assert {s.away_s < 86400 for s in by_kind["greeting"]} == {True, False}
    assert {bool(s.thread) for s in by_kind["greeting"] + by_kind["bid"]} == {True, False}
    assert all(s.reason for s in by_kind["bid"])
    for s in by_kind["report"]:
        assert 1 <= len(s.events) <= 3 and all(k in REPORT_KINDS for k, _, _ in s.events), s.id
        assert [e.kind for e in events_of(s)] == [k for k, _, _ in s.events]
    assert {len(s.events) for s in by_kind["report"]} == {1, 2, 3}
    assert all(2 <= len(s.facts) <= 4 for s in by_kind["about_me"])
    assert all(1 <= len(s.facts) <= 4 for s in by_kind["unknown"])
    assert all(len(s.facts) <= 3 for s in by_kind["small"])
    assert {len(s.facts) for s in by_kind["small"]} == {0, 1, 2, 3}
    for s in by_kind["tool"]:
        summary, ok, text = s.tool
        assert summary and isinstance(ok, bool) and text
        assert tool_result(s).ok is ok
    assert {s.tool[1] for s in by_kind["tool"]} == {True, False}


def test_game_rows():
    game_rows = [s for s in rows(0) + rows(1) if s.kind == "game"]
    assert {s.game for s in game_rows} == set(games.GAMES)
    assert 0.3 < sum(s.game_start for s in game_rows) / len(game_rows) < 0.5
    for s in game_rows:
        assert s.game in games.GAMES
        if s.game_start:
            assert games.detect_start(s.user_text, None) == s.game, (s.id, s.user_text)
        else:  # CONFLICT 7: every move is judged by the engine, never chat ("off")
            g = games.new_game(s.game, random.Random(s.game_seed))
            assert games.play(g, s.user_text).kind != "off", (s.id, s.game, s.user_text)


def test_helpers_build_the_engines_objects():
    g = next(s for s in rows(0) if s.kind == "greeting" and s.thread)
    t = thread_of(g)
    assert (t.id, t.text, t.due_at, t.status, t.last_raised_at) == (1, g.thread, g.now, "open", None)
    assert thread_of(next(s for s in rows(0) if s.kind == "greeting" and not s.thread)) is None
    s = next(s for s in rows(0) if s.kind == "report")
    assert [(e.kind, e.agent_id, e.name, e.text) for e in events_of(s)] == [
        (k, f"agent-{i}", n, x) for i, (k, n, x) in enumerate(s.events)]


def test_pools_are_big_enough():
    assert len(sc.NAMES) >= 30 and len(sc.PLACES) >= 15 and len(sc.ATTRS) >= 10 and len(sc.LOOKUPS) >= 12
    for a in sc.ATTRS:
        assert "{v}" in a.fact and "{v}" in a.remember and a.unknown
    assert sum(sc.MIX.values()) == 100 and set(sc.MIX) == set(KINDS)
    assert len({normalize(n) for n in sc.NAMES}) == len(sc.NAMES)
