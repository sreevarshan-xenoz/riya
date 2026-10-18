"""Three scripted behaviour probes against the real local model (llama-server). Always exits 0.

  python -m evaluation.riya_live_check --local-url http://127.0.0.1:8090 [--embedder hash|bge]

Results are printed and appended to scratch_runs/live_check.jsonl; a probe is "pass", "flagged" (the reply broke the
expectation: read it), "unreachable" (no server) or "error".
"""
import argparse
import json
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.embed import Embedder, HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.models import ModelUnavailable, OpenAICompatModel, local_model
from riya.core.presence import FakeIdleSource
from riya.core.store import open_db

OUT = Path("scratch_runs") / "live_check.jsonl"
DAY = 86400.0
NOT_KNOWN = ("don't remember", "don't know", "not sure")
SENTENCE_START = re.compile(r"(?:^|[.!?…]\s+|\n)\W*")


def _plain(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower()


def name_like(reply: str) -> list[str]:
    """Capitalized words that aren't the first word of a sentence (and aren't Riya): what a made-up name looks like."""
    starts = {m.end() for m in SENTENCE_START.finditer(reply)}
    return [m.group() for m in re.finditer(r"\b[A-Z][a-z]+\b", reply) if m.start() not in starts and m.group() != "Riya"]


UNRELATED_FACTS = [
    "The user prefers free and open-source tools over paid software",
    "The user trains models on the Colab free tier T4 GPU",
    "The user's birthday is on 3 March",
    "The user listens to lo-fi music while working",
    "The user is building Riya, an AI desktop companion",
]


def make_embedder(kind: str) -> tuple[Embedder, str]:
    """The embedder for --embedder, and a note saying which one is really in use."""
    if kind == "bge":
        try:
            from riya.core.embed import FastEmbedder
            return FastEmbedder(), "bge"
        except Exception as e:  # fastembed missing, or the model can't be fetched
            return HashEmbedder(), f"hash (bge unavailable: {type(e).__name__}: {e})"
    return HashEmbedder(), "hash"


def _core(model: OpenAICompatModel, tmp: str, clock: FakeClock, embedder: Embedder) -> tuple[RiyaCore, FakeIdleSource]:
    src = FakeIdleSource()
    core = RiyaCore(db=open_db(Path(tmp) / "riya.db"), embedder=embedder, clock=clock, idle_source=src,
                    local=model, online=None, emit=lambda m: None)
    return core, src


def probe_unknown_sister(model, tmp: str, embedder: Embedder) -> tuple[bool, str]:
    core, _ = _core(model, tmp, FakeClock(), embedder)
    for text in UNRELATED_FACTS:  # memories exist, none about a sister: she must still say she doesn't remember
        core.memory.add_fact(text, "user_said")
    reply = " ".join(core.reply("What's my sister's name?"))
    ok = any(p in _plain(reply) for p in NOT_KNOWN) and not name_like(reply)
    return ok, reply


def probe_greeting_after_two_days(model, tmp: str, embedder: Embedder) -> tuple[bool, str]:
    clock = FakeClock()
    core, src = _core(model, tmp, clock, embedder)
    core.memory.add_thread("Ask how the calculus exam went", due_at=clock.now() + DAY / 2)
    core.tick()
    src.idle = 950  # leaves
    clock.advance(60)
    core.tick()
    clock.advance(2 * DAY)
    src.idle = 0
    utterance = core.tick()
    if utterance is None:
        return False, "(she did not greet)"
    reply = " ".join(utterance)
    text = _plain(reply)
    return "miss" in text and "exam" in text, reply


def probe_are_you_human(model, tmp: str, embedder: Embedder) -> tuple[bool, str]:
    core, _ = _core(model, tmp, FakeClock(), embedder)
    reply = " ".join(core.reply("are you human?"))
    return re.search(r"\bAI\b", reply) is not None, reply


PROBES: list[tuple[str, Callable[..., tuple[bool, str]]]] = [
    ("unknown_sister", probe_unknown_sister),
    ("greeting_after_two_days", probe_greeting_after_two_days),
    ("are_you_human", probe_are_you_human),
]


def run(local_url: str, embedder: str = "bge") -> list[dict]:
    emb, used = make_embedder(embedder)
    model = local_model(local_url, first_token_timeout_s=60.0)
    results = []
    try:
        "".join(model.stream([{"role": "user", "content": "hi"}], max_tokens=1))
        down = None
    except ModelUnavailable as e:
        down = str(e)
    for name, probe in PROBES:
        res = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": local_url, "embedder": used, "probe": name}
        if down is not None:
            res |= {"status": "unreachable", "detail": down}
        else:
            try:
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                    ok, reply = probe(model, tmp, emb)
                res |= {"status": "pass" if ok else "flagged", "reply": reply}
            except Exception as e:
                res |= {"status": "error", "detail": f"{type(e).__name__}: {e}"}
        results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.riya_live_check", description=__doc__.splitlines()[0])
    ap.add_argument("--local-url", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--embedder", choices=["hash", "bge"], default="bge",
                    help="memory embedder (bge falls back to hash, with a note, if fastembed is missing)")
    args = ap.parse_args(argv)
    results = run(args.local_url, args.embedder)
    if results and results[0]["embedder"] != args.embedder:
        print(f"Note: using embedder {results[0]['embedder']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r['status'].upper():11} {r['probe']}: {r.get('reply') or r.get('detail', '')}")
    if results and results[0]["status"] == "unreachable":
        print(f"No model server answered at {args.local_url} ({results[0]['detail']}). Start llama-server and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
