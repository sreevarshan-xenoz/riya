"""Riya v1's SFT requests: each seeded scenario turned into the exact messages RiyaCore sends her local model, built
through the engine's own builders (build_messages, the local persona and style examples, a Memory on an in-memory
database holding the scenario's fake facts, a seeded Feelings, the dials and notes, and the engine's situations), plus
the row's checkable expect rules. Any prompt change in the engine flows into the data, and a golden test holds every
kind but tool rows equal to what RiyaCore sends. Every row also carries the prompt fingerprint ("fp") it was built
under, so rows built before a prompt change can be told apart from fresh ones. Pure, CPU-only and synthetic: the only
database is ":memory:".

Engine paths not modelled (each scenario is a plain case): the twin's short greeting (cap 1), a greeting carrying
agent reports or due reminders, and a bid offering a game.

    python -m riya.train.build_sft [--seed 0] [--n 3000] [--kinds small,bid] [--out PATH]

See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1, §4 and §4.1.2."""
import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from riya.agents.hub import AgentEvent
from riya.core import games
from riya.core.clock import FakeClock
from riya.core.consolidate import remember_intent, third_person
from riya.core.embed import HashEmbedder
from riya.core.engine import REPORT_FACE_S, REPORT_FACES, RiyaCore
from riya.core.feelings import Feelings, FeelingState
from riya.core.hands.types import ToolResult
from riya.core.learn.dials import BANDS, DIALS, clamp
from riya.core.learn.feedback import detect_feedback
from riya.core.memory import Fact, Memory, Retrieved, Thread
from riya.core.persona import persona_prompt, style_examples
from riya.core.prompt import (FEEDBACK_SITUATION, FEELINGS_SITUATION, IDENTITY_SITUATION, bid_situation,
                              build_messages, greeting_situation, hands_situation, likes_block, personal_situation,
                              remember_situation, report_line, report_situation, style_line)
from riya.core.shaping import CAPS, emoji_limit, is_feelings_question, is_identity_question
from riya.core.store import open_db
from riya.train.fmt import request_hash, validate_request, write_jsonl
from riya.train.holdout import holdout_violations
from riya.train.scenarios import (DEFAULT_N, DEFAULT_SEED, KINDS, Scenario, ask_record, events_of, fact_record,
                                  probe_facts, probe_texts, scenarios, thread_of, tool_result)

DEFAULT_OUT = Path("scratch_runs/riya_v1/requests.jsonl")
REPLY_KINDS = frozenset({"small", "about_me", "unknown", "identity", "remember", "explain", "feedback"})
# the sentence cap the engine enforces where it isn't the turn's own cap: a lead (remember) leaves one, a bid and a game
# turn are one line, and a tool line and a report have their fixed caps. On a remember row the engine says its
# templated ack (REMEMBER_ACKS) before asking the model, so the one sentence is her reply after it: the filter trains
# only that span (fmt.trained_reply) and rejects an answer that is nothing but an ack (ack_only). A feedback row is the
# same with ack_line(feedback) first, except that the reply after it keeps the turn's full cap
FIXED_SENTENCES = {"remember": 1, "bid": 1, "game": 1, "tool": CAPS["tool"], "report": CAPS["report"]}
FP_NOW = 1_800_000_000.0
# the fingerprint's canary turns: (turn kind, user text), so every reply-rules line (and the AI clause) is rendered
FP_TURNS = ((None, None), ("small", "canary small talk"), ("small", "are you an ai?"), ("explain", "explain canaries"),
            ("about_me", "what do you know about me?"), ("greeting", None), ("game", "canary"), ("report", None))


def prompt_fingerprint() -> str:
    """The sha256 (hex) of build_messages' output for a fixed canary input: the local persona with both extremes of the
    style line, the style examples, a feelings tag, a memory block with a fact and a follow-up, the honesty line, a
    likes block, every situation builder and every reply-rules line. A change to any of them changes it."""
    fact = Fact(1, "The user's canary is called Pim", None, "user_said", 0.9, FP_NOW, None, "active", None)
    thread = Thread(1, "how the canary's recital went", FP_NOW, "open", None)
    retrieved = Retrieved([fact], [], [thread])
    situation = " ".join([
        IDENTITY_SITUATION, FEELINGS_SITUATION, FEEDBACK_SITUATION,
        remember_situation("The user said: their canary is called Pim"), greeting_situation(7200, 0.5, thread),
        bid_situation("worked 3.0 h straight", thread),
        report_situation([AgentEvent("finished", "agent-0", "tests-1", "all 4 tests passed")]),
        hands_situation("work out 2 plus 2", ToolResult(True, "= 4 (2 plus 2)")), personal_situation()])
    persona, examples = persona_prompt("local"), style_examples()
    calls = []
    for i, (turn, text) in enumerate(FP_TURNS):
        dials = {d: BANDS[d][i % 2] for d in DIALS}  # the lowest, then the highest: every style sentence
        line = style_line(dials)
        calls.append(build_messages(
            persona=persona + (f"\n\n{line}" if line else ""), examples=examples,
            feelings_tag=Feelings(FakeClock(FP_NOW), dials).prompt_tag(), retrieved=retrieved,
            turns=[{"role": "user", "content": "canary"}, {"role": "assistant", "content": "Canary~"}],
            user_text=text, situation=situation, now=FP_NOW, kind=turn, cap=2 if turn == "small" else None,
            likes=likes_block(["They like it when I sing like a canary."])))
    blob = json.dumps(calls, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def request_for(s: Scenario, fp: str | None = None) -> dict:
    """The scenario as a request: {id, kind, messages, facts, expect, meta, fp, h}. messages are what RiyaCore would
    send for the same inputs, composed step for step as the engine does; facts are the memory block's (the facts, and a
    greeting's or bid's follow-up thread). fp: the prompt fingerprint to stamp (computed when not given). h: the
    request's content hash (fmt.request_hash)."""
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
    # the structured side of the same facts (for the filter), and what an unknown row asked about
    meta = {"facts": [fact_record(f) for f in facts], "asks": ask_record(s.asks) if s.asks else None}
    row = {"id": s.id, "kind": kind, "messages": messages, "facts": facts, "expect": expect, "meta": meta,
           "fp": prompt_fingerprint() if fp is None else fp}
    return {**row, "h": request_hash(row)}  # the teacher echoes it; the filter pairs an answer only on id and h


def messages_for(s: Scenario) -> list[dict]:
    return request_for(s)["messages"]


def build(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, kinds: Sequence[str] = KINDS) -> list[dict]:
    """Every request for the seed, n and kinds, each validated; ValueError when any overlaps a persona probe."""
    fp = prompt_fingerprint()
    rows = [request_for(s, fp) for s in scenarios(seed, n, kinds)]
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
