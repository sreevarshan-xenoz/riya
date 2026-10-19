"""Riya v1's SFT requests: each seeded scenario turned into the exact messages RiyaCore sends her local model, built
through the engine's own builders (build_messages, the local persona and style examples, a Memory on an in-memory
database holding the scenario's fake facts, a seeded Feelings, the dials and notes, and the engine's situations), plus
the row's checkable expect rules. Any prompt change in the engine flows into the data, and a golden test holds every
kind but tool rows equal to what RiyaCore sends. Pure, CPU-only and synthetic: the only database is ":memory:".

    python -m riya.train.build_sft [--seed 0] [--n 3000] [--kinds small,bid] [--out PATH]

See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1, §4 and §4.1.2."""
import argparse
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from riya.core import games
from riya.core.clock import FakeClock
from riya.core.consolidate import remember_intent, third_person
from riya.core.embed import HashEmbedder
from riya.core.engine import REPORT_FACE_S, REPORT_FACES, RiyaCore
from riya.core.feelings import Feelings, FeelingState
from riya.core.learn.dials import clamp
from riya.core.learn.feedback import detect_feedback
from riya.core.memory import Memory
from riya.core.persona import persona_prompt, style_examples
from riya.core.prompt import (FEEDBACK_SITUATION, FEELINGS_SITUATION, IDENTITY_SITUATION, bid_situation,
                              build_messages, greeting_situation, hands_situation, likes_block, remember_situation,
                              report_line, report_situation, style_line)
from riya.core.shaping import CAPS, emoji_limit, is_feelings_question, is_identity_question
from riya.core.store import open_db
from riya.train.fmt import validate_request, write_jsonl
from riya.train.scenarios import (DEFAULT_N, DEFAULT_SEED, KINDS, Scenario, events_of, normalize, overlaps,
                                  probe_facts, probe_texts, scenarios, thread_of, tool_result)

DEFAULT_OUT = Path("scratch_runs/riya_v1/requests.jsonl")
REPLY_KINDS = frozenset({"small", "about_me", "unknown", "identity", "remember", "explain", "feedback"})
# the sentence cap the engine enforces where it isn't the turn's own cap: a lead (remember) leaves one, a bid and a game
# turn are one line, and a tool line and a report have their fixed caps
FIXED_SENTENCES = {"remember": 1, "bid": 1, "game": 1, "tool": CAPS["tool"], "report": CAPS["report"]}


def request_for(s: Scenario) -> dict:
    """The scenario as a request: {id, kind, messages, facts, expect}. messages are what RiyaCore would send for the
    same inputs, composed step for step as the engine does; facts are the memory block's (the facts, and a greeting's
    or bid's follow-up thread)."""
    t, kind = s.user_text, s.kind
    # 1. the dials and notes; feedback is read before the reply, as RiyaCore._feedback does
    dials, notes, feedback = dict(s.dials), list(s.notes), False
    if kind == "feedback":
        fb = detect_feedback(t)
        if fb is None:
            raise ValueError(f"{s.id}: a feedback row with no feedback in it")
        feedback = True
        if fb.dial:
            dials[fb.dial] = clamp(fb.dial, dials[fb.dial] + fb.delta)
        else:  # LearnDB.add_note: the newest first (the same text again just moves to the front)
            notes = [fb.note, *(n for n in notes if n != fb.note)]
    # 2. her feelings, live on these dials
    clock = FakeClock(s.now)
    feelings = Feelings(clock, dials)
    feelings.state = FeelingState(**s.mood)
    events = events_of(s)
    if t is not None:
        feelings.on_user_text(t)
    elif kind == "greeting":
        feelings.on_return(s.away_s)
    elif kind == "report":
        kinds = {e.kind for e in events}
        feelings.show(REPORT_FACES[next(k for k in REPORT_FACES if k in kinds)], REPORT_FACE_S)
    # 3. her memory: the scenario's fake facts (and a remember-that turn's fact, saved before she replies)
    memory = Memory(open_db(":memory:"), HashEmbedder(), clock)
    for f in s.facts:
        memory.add_fact(f, "user_said")
    saved = None
    if kind == "remember":
        saved = f"The user said: {third_person(remember_intent(t))}"
        memory.add_fact(saved, "user_said")
    # 4. the turn, its cap and the situation
    verbosity, threads = dials["verbosity"], None
    if kind in REPLY_KINDS:
        turn, _, cap = RiyaCore._shape("reply", "" if kind == "remember" else t, verbosity)
        situation = " ".join(filter(None, [remember_situation(saved) if saved else None,
                                           IDENTITY_SITUATION if is_identity_question(t) else None,
                                           FEELINGS_SITUATION if is_feelings_question(t) else None,
                                           FEEDBACK_SITUATION if feedback else None])) or None
        query = t
    elif kind == "greeting":
        turn, _, cap = RiyaCore._shape("greet", "", verbosity)
        thread = thread_of(s)
        situation = greeting_situation(s.away_s, feelings.state.missing, thread)
        query, threads = s.thread or "came back", [thread] if thread else []
    elif kind == "bid":
        turn = cap = None
        thread = thread_of(s)
        situation = bid_situation(s.reason, thread)
        query, threads = s.thread or s.reason, [thread] if thread else []
    elif kind == "report":
        turn, _, cap = RiyaCore._shape("report", "", verbosity)
        situation, query, threads = report_situation(events), report_line(events, detail=False), []
    elif kind == "game":
        turn, cap = "game", 1
        g = games.new_game(s.game, random.Random(s.game_seed))
        verdict = games.opening(g) if s.game_start else games.play(g, t)
        situation, query = games.game_situation(g, verdict), t
    elif kind == "tool":
        turn, cap = None, CAPS["tool"]
        situation, query = hands_situation(s.tool[0], tool_result(s)), t
    else:
        raise ValueError(f"{s.id}: unknown kind {kind!r}")
    # 5. retrieval: speaking first carries only its chosen thread
    retrieved = memory.retrieve(query)
    if threads is not None:
        retrieved = replace(retrieved, threads=threads)
    # 6. the call, as RiyaCore._speak builds it
    line = style_line(dials)
    messages = build_messages(persona=persona_prompt("local") + (f"\n\n{line}" if line else ""),
                              examples=style_examples(), feelings_tag=feelings.prompt_tag(), retrieved=retrieved,
                              turns=[dict(x) for x in s.turns], user_text=t, situation=situation, now=s.now, kind=turn,
                              cap=cap if turn == "small" else None, likes=likes_block(notes))
    fact_texts = [f.text for f in retrieved.facts]
    expect = {"must_mention": list(fact_texts) if kind == "about_me" else [],
              "must_not_invent": True,
              "must_admit_unknown": kind == "unknown",
              "must_say_ai": kind == "identity",
              "max_sentences": FIXED_SENTENCES.get(kind, cap),
              "max_emoji": emoji_limit(dials["emoji"])}
    # a follow-up thread sits in the memory block too ("- Follow up: ..."), so mentioning it isn't invented
    facts = fact_texts + [th.text for th in retrieved.threads]
    return {"id": s.id, "kind": kind, "messages": messages, "facts": facts, "expect": expect}


def messages_for(s: Scenario) -> list[dict]:
    return request_for(s)["messages"]


def holdout_violations(rows: list[dict], probes: list[str], facts: list[str]) -> list[str]:
    """The ids of rows where a user message overlaps a probe (equal, or a shared 6-word window), or the system message
    holds a probe fact (both normalised)."""
    normed = [n for n in (normalize(f) for f in facts) if n]
    bad = []
    for row in rows:
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        system = normalize(" ".join(m["content"] for m in row["messages"] if m["role"] == "system"))
        if any(overlaps(u, probes) for u in users) or any(f in system for f in normed):
            bad.append(row["id"])
    return bad


def build(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, kinds: Sequence[str] = KINDS) -> list[dict]:
    """Every request for the seed, n and kinds, each validated; ValueError when any overlaps a persona probe."""
    rows = [request_for(s) for s in scenarios(seed, n, kinds)]
    for row in rows:
        validate_request(row)
    if ids := holdout_violations(rows, probe_texts(), probe_facts()):
        raise ValueError(f"probe overlap: {ids[:5]}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m riya.train.build_sft",
                                description="Write Riya v1's SFT requests (synthetic, seeded) as JSON lines.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--kinds", default=",".join(KINDS), help="comma-separated kinds (default: all)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    rows = build(args.seed, args.n, kinds)
    write_jsonl(rows, args.out)
    for kind, count in Counter(r["kind"] for r in rows).items():
        print(f"{kind}: {count}")
    print(f"Wrote {len(rows)} requests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
