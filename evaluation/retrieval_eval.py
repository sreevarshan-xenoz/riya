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
    """recall@5 over questions with an expected fact; never_said_empty over those that expect nothing.

    Also returns the score distribution behind those numbers: max_never_top1_sim (the highest cosine any never-said
    question reaches against a stored fact) and min_hit_sim (the lowest cosine between a question and its expected fact);
    a threshold separates the two when max_never_top1_sim < min_hit_sim.

    Facts are seeded a day apart on the memory's clock (it must be a FakeClock); a dataset fact id maps to the stored id,
    so a correction (same key, newer user_said) is expected by its newer id and the superseded one is never retrievable.
    """
    memory = memory_factory()
    stored: dict[int, str] = {}
    for fact in data["facts"]:
        added = memory.add_fact(fact["text"], fact["source"], key=fact.get("key"))
        if added is not None:
            stored[added.id] = fact["id"]
        memory.clock.advance(DAY)
    vecs = {stored[r["id"]]: np.frombuffer(r["embedding"], dtype=np.float32)  # live facts, by dataset id
            for r in memory.db.execute("select id, embedding from facts where status in ('active','conflict')")}
    hits = asked = silent = never = 0
    never_top1, hit_sims = [], []
    for q in data["questions"]:
        got = {stored.get(f.id) for f in memory.retrieve(q["q"], k_facts=5).facts}
        qv = memory.embedder.embed([q["q"]])[0]
        if q["expect"]:
            asked += 1
            hits += all(e in got for e in q["expect"])
            hit_sims += [float(np.dot(qv, vecs[e])) for e in q["expect"] if e in vecs]
        else:
            never += 1
            silent += not got
            never_top1.append(max((float(np.dot(qv, v)) for v in vecs.values()), default=0.0))
    return {"recall@5": hits / asked if asked else 0.0, "never_said_empty": silent / never if never else 1.0,
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
