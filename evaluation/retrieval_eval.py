"""Retrieval benchmark for Riya's memory: recall@5 over seeded facts and silence on things never said.

  python -m evaluation.retrieval_eval --embedder hash|bge
"""
import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory
from riya.core.store import open_db

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "retrieval_set.json"
OUT = Path("scratch_runs") / "retrieval_eval.jsonl"
DAY = 86400.0


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


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m evaluation.retrieval_eval", description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", choices=["hash", "bge"], default="hash")
    ap.add_argument("--min-sim", type=float, default=None, help="override the embedder's strong/weak thresholds")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.embedder == "bge":
        from riya.core.embed import FastEmbedder
        embedder = FastEmbedder()
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
