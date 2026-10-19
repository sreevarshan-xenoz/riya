"""v0.8 Task 2: SFT requests built through the engine's own prompt builder, golden-equal to what RiyaCore sends, with
the expect rules, the probe hold-out and the CLI. Synthetic only: nothing reads the user's real data."""
import json
import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from riya.core import games
from riya.core.clock import FakeClock
from riya.core.engine import REPORT_FACE_S, REPORT_FACES
from riya.core.feelings import FeelingState
from riya.core.initiative import Action
from riya.core.learn.dials import clamp
from riya.core.learn.feedback import detect_feedback
from riya.core.prompt import hands_situation
from riya.core.shaping import emoji_limit
from riya.core.store import open_db
from riya.train import build_sft
from riya.train.build_sft import DEFAULT_OUT, build, holdout_violations, main, messages_for, request_for
from riya.train.fmt import EXPECT_KEYS, REQUEST_KEYS, read_jsonl, validate_request, write_jsonl
from riya.train.scenarios import (KINDS, events_of, normalize, probe_facts, probe_texts, scenario, thread_of,
                                  tool_result)
from tests.fakes import make_core

GOLDEN_KINDS = ("small", "about_me", "unknown", "identity", "remember", "explain", "feedback", "greeting", "bid",
                "game", "report")
GOLDEN_SEEDS = (0, 7)
GOLDEN_INDEXES = range(60)  # CONFLICT 7: a wide sweep, so every branch of every kind is exercised
CANARY = "Zorvak-7781"


@lru_cache(maxsize=None)
def built(seed, n):
    return build(seed, n)


def engine_messages(s) -> list[dict]:
    """The messages a real RiyaCore (on fakes) sends its local model for the scenario's inputs."""
    clock = FakeClock(s.now)
    core, _, _, local = make_core(clock)
    core.learn.set_dials(s.dials, "scenario", "explicit", s.now)
    for i, note in enumerate(reversed(s.notes)):
        core.learn.add_note(note, "like", s.now - 10 + i)
    for f in s.facts:
        core.memory.add_fact(f, "user_said")
    core.feelings.state = FeelingState(**s.mood)
    core.turns = [dict(x) for x in s.turns]
    if s.kind == "greeting":
        core._away_s = s.away_s
        core.feelings.on_return(s.away_s)
        list(core.speak_first(Action("greet", True, thread_of(s), "came back")))
    elif s.kind == "bid":
        core._bid(Action("bid", False, thread_of(s), s.reason))
    elif s.kind == "report":
        kinds = {e.kind for e in events_of(s)}
        core.feelings.show(REPORT_FACES[next(k for k in REPORT_FACES if k in kinds)], REPORT_FACE_S)
        list(core._report(events_of(s)))
    elif s.kind == "game":
        core.game_rng = random.Random(s.game_seed)
        if not s.game_start:
            core.game = games.new_game(s.game, random.Random(s.game_seed))
        list(core.reply(s.user_text))
    else:
        list(core.reply(s.user_text))
    return local.calls[-1]


@pytest.mark.parametrize("kind", GOLDEN_KINDS)
def test_request_equals_the_engines_prompt(kind):
    """Review Focus 2: byte for byte, every role and every character, what RiyaCore sends for the same inputs."""
    for seed in GOLDEN_SEEDS:
        for i in GOLDEN_INDEXES:
            s = scenario(kind, i, seed)
            assert messages_for(s) == engine_messages(s), s.id


def test_the_golden_sweep_reaches_every_branch():
    """The sweep above covers each kind's branches (CONFLICT 7), so a branch the builder gets wrong can't hide."""
    rows = [scenario(k, i, seed) for k in GOLDEN_KINDS for seed in GOLDEN_SEEDS for i in GOLDEN_INDEXES]
    by = {k: [s for s in rows if s.kind == k] for k in GOLDEN_KINDS}
    assert {s.game_start for s in by["game"]} == {True, False}
    assert {s.game for s in by["game"] if not s.game_start} == set(games.GAMES)
    assert {bool(s.thread) for s in by["greeting"]} == {True, False}
    assert {bool(s.thread) for s in by["bid"]} == {True, False}
    assert {s.away_s < 86400 for s in by["greeting"]} == {True, False}
    fbs = [detect_feedback(s.user_text) for s in by["feedback"]]
    assert {fb.dial is None for fb in fbs} == {True, False}  # a dial nudge and a style note
    assert any(s.notes for s in rows) and any(s.turns for s in rows)
    assert {len(s.events) for s in by["report"]} == {1, 2, 3}


def test_tool_requests_shape():
    for seed in GOLDEN_SEEDS:
        for i in range(30):
            s = scenario("tool", i, seed)
            messages = messages_for(s)
            system = messages[0]["content"]
            assert messages[0]["role"] == "system"
            assert system.endswith("[Situation] " + hands_situation(s.tool[0], tool_result(s))), s.id
            assert "### Reply rules:" not in system, s.id
            assert messages[-1] == {"role": "user", "content": s.user_text}, s.id


def _dials_after(s) -> dict[str, float]:
    """The dials the engine replies with: a feedback row's nudge applied, as RiyaCore applies it before replying."""
    dials = dict(s.dials)
    if s.kind == "feedback" and (fb := detect_feedback(s.user_text)).dial:
        dials[fb.dial] = clamp(fb.dial, dials[fb.dial] + fb.delta)
    return dials


def test_expect_rules():
    rows = built(0, 600)
    assert {r["kind"] for r in rows} == set(KINDS)
    for r in rows:
        validate_request(r)
        k, e = r["kind"], r["expect"]
        assert tuple(r) == REQUEST_KEYS and tuple(e) == EXPECT_KEYS
        kind, _, index = r["id"].rpartition("-")
        s = scenario(kind, int(index), 0)
        assert e["must_not_invent"] is True
        assert e["must_admit_unknown"] is (k == "unknown")
        assert e["must_say_ai"] is (k == "identity")
        assert e["max_emoji"] == emoji_limit(_dials_after(s)["emoji"]), r["id"]
        if k == "about_me":
            assert e["must_mention"] == r["facts"] and e["must_mention"], r["id"]
            assert sorted(e["must_mention"]) == sorted(s.facts)
        else:
            assert e["must_mention"] == [], r["id"]
        if k in ("remember", "bid", "game"):
            assert e["max_sentences"] == 1
        if k == "explain":
            assert e["max_sentences"] == 8
        if k in ("tool", "report"):
            assert e["max_sentences"] == {"tool": 2, "report": 1}[k]
        if k in ("greeting", "bid") and s.thread:  # CONFLICT 5: the follow-up is part of the memory block
            assert s.thread in r["facts"], r["id"]
    # the facts are the memory block's: every one is listed in the system message
    for r in rows:
        assert all(f in r["messages"][0]["content"] for f in r["facts"]), r["id"]


def test_feedback_rows_reply_with_the_nudged_dials():
    """A feedback row carries the dials from before the nudge; the request applies it where the engine does (before
    the reply), so "no emojis" already allows none."""
    s = next(x for x in (scenario("feedback", i, 0) for i in range(400))
             if (fb := detect_feedback(x.user_text)).dial == "emoji" and fb.delta < 0 and x.dials["emoji"] >= 0.15
             and clamp("emoji", x.dials["emoji"] + fb.delta) < 0.15)
    assert emoji_limit(s.dials["emoji"]) == 1
    assert request_for(s)["expect"]["max_emoji"] == 0


def test_validate_request_rejects_bad_rows():
    good = request_for(scenario("small", 0, 0))
    validate_request(good)
    bad = [
        {k: v for k, v in good.items() if k != "facts"},
        {**good, "extra": 1},
        {**good, "messages": good["messages"][1:]},  # no system message first
        {**good, "messages": good["messages"] + [{"role": "assistant", "content": "hi"}]},  # ends with assistant
        {**good, "messages": [good["messages"][0], {"role": "user", "content": "a"},
                              {"role": "user", "content": "b"}]},  # not alternating
        {**good, "messages": [good["messages"][0], {"role": "user", "content": None}]},
        {**good, "facts": [1]},
        {**good, "expect": {**good["expect"], "max_sentences": 0}},
        {**good, "expect": {**good["expect"], "max_emoji": -1}},
        {**good, "expect": {**good["expect"], "max_emoji": True}},
        {**good, "expect": {**good["expect"], "must_say_ai": 1}},
        {**good, "expect": {**good["expect"], "must_mention": "x"}},
        {**good, "expect": {k: v for k, v in good["expect"].items() if k != "max_emoji"}},
    ]
    for row in bad:
        with pytest.raises(ValueError):
            validate_request(row)


def _row(rid: str, user: str, system: str = "You are Riya.") -> dict:
    return {"id": rid, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def test_no_request_overlaps_a_probe(monkeypatch):
    """Review Focus 1 (CONFLICT 8): the hold-out check flags hand-made violations, the probe sets aren't empty, a full
    build has no violation, and build refuses a violating scenario."""
    texts, facts = probe_texts(), probe_facts()
    assert texts and facts
    window = " ".join(normalize(next(t for t in texts if len(normalize(t).split()) >= 7)).split()[:6])
    rows = [_row("exact", texts[0]), _row("window", f"ok so {window} anyway"),
            _row("fact", "hi", system=f"Stuff.\n- {facts[0].upper()} (user said)"), _row("clean", "hi there")]
    assert holdout_violations(rows, texts, facts) == ["exact", "window", "fact"]
    assert holdout_violations(built(0, 3000), texts, facts) == []

    small, about = scenario("small", 0, 0), scenario("about_me", 0, 0)  # about-me lists every fact in the block
    for bad in (replace(small, user_text=texts[0]), replace(about, facts=(*about.facts, facts[0]))):
        monkeypatch.setattr(build_sft, "scenarios", lambda seed, n, kinds, bad=bad: [bad])
        with pytest.raises(ValueError, match="probe overlap"):
            build(0, 1)


def test_build_is_deterministic_and_the_cli_writes_it(tmp_path, capsys):
    p, q = tmp_path / "a" / "requests.jsonl", tmp_path / "b.jsonl"
    assert main(["--seed", "0", "--n", "48", "--out", str(p)]) == 0
    first = p.read_bytes()
    assert main(["--seed", "0", "--n", "48", "--out", str(p)]) == 0
    assert p.read_bytes() == first
    lines = first.decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines[:-1]) == 48
    for line in lines[:-1]:
        validate_request(json.loads(line))
    assert read_jsonl(p) == build(0, 48)
    out = capsys.readouterr().out
    assert f"Wrote 48 requests to {p}" in out and "small: " in out

    assert main(["--seed", "0", "--n", "20", "--kinds", "bid,report", "--out", str(q)]) == 0
    rows = read_jsonl(q)
    assert len(rows) == 20 and {r["kind"] for r in rows} == {"bid", "report"}
    assert DEFAULT_OUT == Path("scratch_runs/riya_v1/requests.jsonl")


def test_jsonl_round_trip(tmp_path):
    p = tmp_path / "deep" / "x.jsonl"
    write_jsonl([{"a": "é~"}, {"b": 1}], p)
    assert p.read_bytes() == '{"a": "é~"}\n{"b": 1}\n'.encode("utf-8")
    p.write_text('{"a": 1}\n\n  \n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_no_real_user_data_is_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("riya_memory.json").write_text(json.dumps({"summary": f"The user's code is {CANARY}"}), encoding="utf-8")
    real_db = tmp_path / "localappdata" / "Riya" / "riya.db"
    db = open_db(real_db)
    from riya.core.embed import HashEmbedder
    from riya.core.memory import Memory
    Memory(db, HashEmbedder(), FakeClock()).add_fact(f"The user's code is {CANARY}", "user_said")
    db.close()

    paths, real_open = [], build_sft.open_db
    monkeypatch.setattr(build_sft, "open_db", lambda path: paths.append(path) or real_open(path))
    rows = build(0, 60)
    assert rows and paths and all(p == ":memory:" for p in paths)
    assert not any(CANARY in json.dumps(r, ensure_ascii=False) for r in rows)
    for f in Path(build_sft.__file__).parent.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        for word in ("LOCALAPPDATA", "riya.db", "riya_memory", "DEFAULT_DB"):
            assert word not in text, (f.name, word)
