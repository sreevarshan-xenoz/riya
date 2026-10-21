"""Retrieval benchmark for Riya's memory: recall@5 over seeded facts and silence on things never said.

  python -m evaluation.retrieval_eval --embedder hash|bge
  python -m evaluation.retrieval_eval --personal [--dev-only] [--personal-sim 0.40|0.45|0.50] [--embedder recorded]

--personal measures v0.11's personal fill (Memory.retrieve(personal=True)) on the personal-request dev set and the
persona probe bank, from the recorded bge vectors (no model is loaded). --dev-only never runs the bank: the floor is
chosen on dev evidence alone.
"""
import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from evaluation.persona_eval import PERSONAL_PROBES, load_probes
from evaluation.recorded_vectors import RecordedEmbedder
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory, personal_request
from riya.core.store import open_db

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "retrieval_set.json"
DEV = ROOT / "tests" / "data" / "personal_set.json"
OUT = Path("scratch_runs") / "retrieval_eval.jsonl"
DAY = 86400.0
HONESTY_KINDS = ("unknown", "ai", "ai_offpath")
NEVER_FIRE_KINDS = HONESTY_KINDS + ("small", "explain", "remember", "greeting")


def recall_at_5(memory_factory: Callable[[], Memory], data: dict) -> dict:
    """recall@5 over questions with an expected fact or episode; never_said_empty over those that expect nothing.

    episode_recall and about_me_recall are the same hit rate over just the "episode" ("what did we talk about last
    time?") and "about_me" ("what do you remember about me?") questions. Also returns the score distribution behind
    those numbers: max_never_top1_sim (the highest cosine any never-said question reaches against a stored fact) and
    min_hit_sim (the lowest cosine between a similarity question and its expected fact); a threshold separates the two
    when max_never_top1_sim < min_hit_sim.

    Facts are seeded a day apart on the memory's clock (it must be a FakeClock); a dataset fact id maps to the stored id,
    so a correction (same key, newer user_said) is expected by its newer id and the superseded one is never retrievable.
    Episodes are then seeded days_ago before the clock's final time.
    """
    memory = memory_factory()
    stored: dict[int, str] = {}
    for fact in data["facts"]:
        added = memory.add_fact(fact["text"], fact["source"], key=fact.get("key"))
        if added is not None:
            stored[added.id] = fact["id"]
        memory.clock.advance(DAY)
    episodes: dict[int, str] = {}
    for ep in data.get("episodes", []):
        at = memory.clock.now() - ep["days_ago"] * DAY
        episodes[memory.add_episode(at - 1800, at, ep["summary"]).id] = ep["id"]
    vecs = {stored[r["id"]]: np.frombuffer(r["embedding"], dtype=np.float32)  # live facts, by dataset id
            for r in memory.db.execute("select id, embedding from facts where status in ('active','conflict')")}
    hits = asked = silent = never = 0
    by_kind: dict[str, list[int]] = {"episode": [0, 0], "about_me": [0, 0]}
    never_top1, hit_sims = [], []
    for q in data["questions"]:
        r = memory.retrieve(q["q"], k_facts=5)
        got, got_eps = {stored.get(f.id) for f in r.facts}, {episodes.get(e.id) for e in r.episodes}
        qv = memory.embedder.embed([q["q"]])[0]
        if q["expect"] or q.get("expect_episodes"):
            hit = all(e in got for e in q["expect"]) and all(e in got_eps for e in q.get("expect_episodes", []))
            asked += 1
            hits += hit
            if q["kind"] in by_kind:
                by_kind[q["kind"]][0] += hit
                by_kind[q["kind"]][1] += 1
            elif q["expect"]:
                hit_sims += [float(np.dot(qv, vecs[e])) for e in q["expect"] if e in vecs]
        else:
            never += 1
            silent += not got
            never_top1.append(max((float(np.dot(qv, v)) for v in vecs.values()), default=0.0))
    rate = {k: n / m if m else 1.0 for k, (n, m) in by_kind.items()}
    return {"recall@5": hits / asked if asked else 0.0, "never_said_empty": silent / never if never else 1.0,
            "episode_recall": rate["episode"], "about_me_recall": rate["about_me"],
            "max_never_top1_sim": round(max(never_top1, default=0.0), 4),
            "min_hit_sim": round(min(hit_sims, default=0.0), 4)}


def seeded(texts: Sequence[str], embedder) -> tuple[Memory, dict[int, str]]:
    """A :memory: Memory on a FakeClock with every text added as user_said at one clock time; also the stored ids,
    mapped to their texts (a near-duplicate text is not stored twice)."""
    memory = Memory(open_db(":memory:"), embedder, FakeClock())
    stored: dict[int, str] = {}
    for text in texts:
        added = memory.add_fact(text, "user_said")
        if added is not None:
            stored[added.id] = text
    return memory, stored


def _ask(memory: Memory, q: str) -> tuple:
    """(the retrieval with the fill on, whether the fill-off retrieval has the same fact ids)."""
    on = memory.retrieve(q, personal=True)
    off = memory.retrieve(q, personal=False)
    return on, [f.id for f in on.facts] == [f.id for f in off.facts]


def _share(n: int, of: int) -> float:
    return n / of if of else 1.0


def personal_report(data: dict, embedder, *, pooled: bool = False) -> dict:
    """The dev set through retrieve(personal=True): each question against its own `memory` facts, or with pooled,
    against every fact. personal_recall is the share of requests with at least one relevant fact in the prompt
    (`misses` names the rest); dragged is the filled facts that aren't relevant, per request (mean and max);
    max_facts and max_filled bound every question; nonpersonal_identical is the share of the questions that
    personal_request does not fire on that retrieve the same facts with the fill on and off."""
    text = {f["id"]: f["text"] for f in data["facts"]}
    pool = seeded([f["text"] for f in data["facts"]], embedder) if pooled else None
    questions = [(r, True) for r in data["requests"]] + [(n, False) for n in data["negatives"]]
    hits, misses, dragged, fired, negatives_fired, identical, plain = 0, [], [], 0, [], 0, 0
    max_facts = max_filled = 0
    for q, is_request in questions:
        memory, _ = pool or seeded([text[i] for i in q["memory"]], embedder)
        on, same = _ask(memory, q["q"])
        fires = personal_request(q["q"])
        max_facts, max_filled = max(max_facts, len(on.facts)), max(max_filled, len(on.filled))
        if not fires:
            plain += 1
            identical += same
        if not is_request:
            if fires:
                negatives_fired.append(q["id"])
            continue
        fired += fires
        relevant = {text[i] for i in q["relevant"]}
        if relevant & {f.text for f in on.facts}:
            hits += 1
        else:
            misses.append(q["id"])
        dragged.append(sum(1 for f in on.facts if f.id in on.filled and f.text not in relevant))
    n = len(data["requests"])
    return {"requests": n, "fired": fired, "fire_rate": _share(fired, n), "negatives_fired": negatives_fired,
            "personal_recall": _share(hits, n), "misses": misses,
            "dragged_mean": round(sum(dragged) / n, 4) if n else 0.0, "dragged_max": max(dragged, default=0),
            "max_facts": max_facts, "max_filled": max_filled, "nonpersonal_identical": _share(identical, plain)}


def bank_report(probes: dict, embedder) -> dict:
    """The persona probe bank, each turn against its own conversation's seeded facts only (remembered facts aren't
    modelled). A speak_first turn is skipped for retrieval but the detector still sees it. fired: the probe ids
    ("conversation.turn") of each kind the detector fires on. personal_hits: the PERSONAL_PROBES conversations whose
    prompt holds a relevant fact (the turn's `relevant` annotation, else every conversation fact). dragged_total:
    filled facts that aren't relevant, over the about-me turns. honesty_facts: the facts any unknown or AI turn would
    carry into its prompt (A10 wants none)."""
    fired: dict[str, list[str]] = {}
    hits, honesty, dragged = [], {}, 0
    max_facts = max_filled = identical = plain = 0
    for conv in probes["conversations"]:
        memory, _ = seeded(conv["facts"], embedder)
        for turn in conv["turns"]:
            pid, kind = f"{conv['id']}.{turn['id']}", turn["kind"]
            fires = personal_request(turn["text"])
            fired.setdefault(kind, [])
            if fires:
                fired[kind].append(pid)
            if turn.get("speak_first"):
                continue
            on, same = _ask(memory, turn["text"])
            max_facts, max_filled = max(max_facts, len(on.facts)), max(max_filled, len(on.filled))
            if not fires:
                plain += 1
                identical += same
            relevant = {conv["facts"][i] for i in turn["relevant"]} if "relevant" in turn else set(conv["facts"])
            if kind == "about_me":
                dragged += sum(1 for f in on.facts if f.id in on.filled and f.text not in relevant)
            if conv["id"] in PERSONAL_PROBES and relevant & {f.text for f in on.facts}:
                hits.append(conv["id"])
            if kind in HONESTY_KINDS and on.facts:
                honesty[pid] = [f.text for f in on.facts]
    return {"fired": fired, "personal_hits": hits, "personal_total": len(PERSONAL_PROBES), "dragged_total": dragged,
            "honesty_facts": honesty, "max_facts": max_facts, "max_filled": max_filled,
            "nonpersonal_identical": _share(identical, plain)}


def format_personal(dev: dict, dev_pooled: dict, bank: dict | None) -> str:
    """One line per number. A bank of None prints the dev lines only."""
    lines = [f"detector fired on {dev['fired']}/{dev['requests']} requests ({dev['fire_rate']:.0%}); "
             f"on negatives: {dev['negatives_fired'] or 'none'}"]
    for label, r in (("own memory", dev), ("pooled", dev_pooled)):
        n = round(r["personal_recall"] * r["requests"])
        lines += [f"personal recall ({label}) {n}/{r['requests']} ({r['personal_recall']:.0%}); "
                  f"misses: {', '.join(r['misses']) or 'none'}",
                  f"dragged ({label}) mean {r['dragged_mean']:.2f}, max {r['dragged_max']}",
                  f"bounds ({label}) max facts {r['max_facts']}, max filled {r['max_filled']}",
                  f"non-personal identical ({label}) {r['nonpersonal_identical']:.0%}"]
    if bank is not None:
        fired = ", ".join(f"{kind} {ids}" for kind, ids in bank["fired"].items() if ids) or "no probe"
        missed = [p for p in PERSONAL_PROBES if p not in bank["personal_hits"]]
        lines += [f"bank personal recall {len(bank['personal_hits'])}/{bank['personal_total']} (prototype 16)",
                  f"bank personal probes without a relevant fact: {', '.join(missed) or 'none'}",
                  f"bank dragged total {bank['dragged_total']}",
                  f"bank honesty probes holding a fact: {bank['honesty_facts'] or 'none'}",
                  f"bank detector fired on: {fired}",
                  f"bank bounds max facts {bank['max_facts']}, max filled {bank['max_filled']}",
                  f"bank non-personal identical {bank['nonpersonal_identical']:.0%}"]
    return "\n".join(lines)


def _personal_main(args) -> int:
    if args.embedder == "bge":
        from riya.core.embed import FastEmbedder
        embedder = FastEmbedder()
    else:
        embedder = {"recorded": RecordedEmbedder, "hash": HashEmbedder}[args.embedder]()
    if args.personal_sim is not None:
        embedder.personal_sim = args.personal_sim
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    own, pooled = personal_report(dev, embedder), personal_report(dev, embedder, pooled=True)
    bank = None if args.dev_only else bank_report(load_probes(), embedder)
    print(format_personal(own, pooled, bank))
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "embedder": args.embedder,
           "personal_sim": getattr(embedder, "personal_sim", None), "dev": own, "dev_pooled": pooled}
    if bank is not None:
        row["bank"] = bank
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return 0


def main(argv: list[str] | None = None) -> dict | int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.retrieval_eval", description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", choices=["hash", "bge", "recorded"], default=None,
                    help="default: recorded with --personal, hash without")
    ap.add_argument("--min-sim", type=float, default=None, help="override the embedder's strong/weak thresholds")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--personal", action="store_true", help="the personal fill on the dev set and the probe bank")
    ap.add_argument("--dev-only", action="store_true",
                    help="with --personal: never run the bank (the floor is chosen on dev evidence only)")
    ap.add_argument("--personal-sim", type=float, default=None, help="override the personal floor (a ruling)")
    args = ap.parse_args(argv)
    if args.dev_only and not args.personal:
        ap.error("--dev-only needs --personal")
    if args.embedder is None:
        args.embedder = "recorded" if args.personal else "hash"
    if args.personal:
        return _personal_main(args)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.embedder == "bge":
        from riya.core.embed import FastEmbedder
        embedder = FastEmbedder()
    elif args.embedder == "recorded":
        embedder = RecordedEmbedder()
    else:
        embedder = HashEmbedder()
    thresholds = ({"min_sim": args.min_sim} if args.min_sim is not None
                  else {"strong_sim": embedder.strong_sim, "weak_sim": embedder.weak_sim})
    res = recall_at_5(lambda: Memory(open_db(":memory:"), embedder, FakeClock(), min_sim=args.min_sim), data)
    print({**thresholds, **res})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "embedder": args.embedder, **thresholds, **res}) + "\n")
    return res


if __name__ == "__main__":
    main()
