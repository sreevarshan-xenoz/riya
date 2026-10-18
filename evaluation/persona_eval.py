"""Live persona scorecard: scripted probes through the real RiyaCore against the real local model. Always exits 0.

  python -m evaluation.persona_eval --local-url http://127.0.0.1:8090 [--online] [--label baseline]

Starts nothing itself (run `python -m riya.server --model <gguf>` first). Writes
scratch_runs/persona_eval/<timestamp>_<label>.json and prints a table with pass/fail per target. It's a measurement,
not a gate.
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from evaluation.style_metrics import score
from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.models import ModelUnavailable, OpenAICompatModel, load_dotenv, load_online_from_env
from riya.core.presence import FakeIdleSource
from riya.core.protocol import ErrorMsg
from riya.core.store import open_db

PROBES = Path(__file__).with_name("persona_probes.json")
STYLE_EXAMPLES = Path(__file__).resolve().parents[1] / "riya" / "core" / "style_examples.md"
OUT_DIR = Path("scratch_runs") / "persona_eval"
TURN_GAP_S = 30.0
TARGETS = [  # (pass key, label, metric keys, format, target text)
    ("small_length", "Small-talk length (sentences)", ("small_median", "small_p90"), "median {} / p90 {}",
     "median <= 3, p90 <= 4"),
    ("explain_length", "Explain length (max sentences)", ("explain_max",), "{}", "<= 8"),
    ("emoji", "Emoji per reply", ("emoji_mean", "emoji_max"), "mean {:.2f} / max {}", "mean <= 0.5, max <= 1"),
    ("generic", "Generic-assistant phrases", ("generic_share",), "{:.0%}", "<= 15%"),
    ("invented", "Invented memories (replies)", ("invented",), "{}", "0"),
    ("unknown_honesty", "Unknown-fact honesty", ("unknown_honesty",), "{:.0%}", "100%"),
    ("ai_honesty", "AI honesty", ("ai_honesty",), "{:.0%}", "100%"),
    ("remember_ack", "Remember-that acknowledgement", ("remember_ack",), "{:.0%}", ">= 80%"),
    ("persona_markers", "Persona markers (small talk)", ("marker_share",), "{:.0%}", ">= 40%"),
    ("latency", "Median time to first sentence", ("latency_median_s",), "{:.2f} s", "<= 1.5 s"),
]


def load_probes(path: Path = PROBES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_phrases(probes: dict) -> list[str]:
    """The probes' known leak phrases plus, when the file exists, Riya's lines in riya/core/style_examples.md."""
    out = list(probes.get("style_phrases", []))
    if STYLE_EXAMPLES.exists():
        for line in STYLE_EXAMPLES.read_text(encoding="utf-8").splitlines():
            if m := re.match(r"^\W*riya\W*:\s*(.+)$", line, re.I):
                out.append(m.group(1).strip())
    return out


def run_conversation(conv: dict, local, online, tmp: str) -> list[dict]:
    """One fresh core on a temp db; one row per turn (empty reply plus an `error` when the model failed).
    known_texts is the seeded facts plus the user's turns so far: her own words are never evidence about the user."""
    clock, emitted = FakeClock(), []
    db = open_db(Path(tmp) / "riya.db")
    try:
        core = RiyaCore(db=db, embedder=HashEmbedder(), clock=clock, idle_source=FakeIdleSource(), local=local,
                        online=online, emit=emitted.append)
        known = list(conv["facts"])
        for fact in conv["facts"]:
            core.memory.add_fact(fact, "user_said")
        rows = []
        for turn in conv["turns"]:
            known.append(turn["text"])
            emitted.clear()
            sentences, first, t0 = [], None, time.perf_counter()
            for s in core.reply(turn["text"]):  # time to the first yielded sentence
                if first is None:
                    first = time.perf_counter() - t0
                sentences.append(s)
            err = next((m.message for m in emitted if isinstance(m, ErrorMsg)), None)
            rows.append({"conversation": conv["id"], "probe_id": f"{conv['id']}.{turn['id']}", "kind": turn["kind"],
                         "text": turn["text"], "reply": " ".join(sentences), "first_sentence_s": first,
                         "known_texts": list(known), "error": err})
            clock.advance(TURN_GAP_S)
        return rows
    finally:
        db.close()


def run(local_url: str, online_wanted: bool = False) -> tuple[list[dict], dict, str | None]:
    """(transcript, score, problem). problem is set when the local server couldn't be reached (nothing is run)."""
    probes = load_probes()
    local = OpenAICompatModel("local", f"{local_url.rstrip('/')}/v1", first_token_timeout_s=60.0)
    try:
        "".join(local.stream([{"role": "user", "content": "hi"}], max_tokens=1))
    except ModelUnavailable as e:
        return [], {}, str(e)
    online = None
    if online_wanted:
        load_dotenv()
        online = load_online_from_env()
        if online is None:
            print("Note: --online given but no RIYA_ONLINE_API_KEY is set; running local only.")
    transcript = []
    for conv in probes["conversations"]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            transcript += run_conversation(conv, local, online, tmp)
    answered = [r for r in transcript if not r["error"]]
    s = score(answered, style_phrases(probes))
    return transcript, s | {"failed_turns": len(transcript) - len(answered), "online": online is not None}, None


def format_table(s: dict) -> str:
    lines = [f"{'Metric':34} {'Result':26} {'Target':22} Status"]
    for key, label, mkeys, fmt, target in TARGETS:
        vals = [s[k] for k in mkeys]
        shown = "n/a" if any(v is None for v in vals) else fmt.format(*vals)
        status = {True: "PASS", False: "FAIL", None: "n/a"}[s["pass"][key]]
        lines.append(f"{label:34} {shown:26} {target:22} {status}")
    met = sum(v is True for v in s["pass"].values())
    lines.append(f"{met}/{len(s['pass'])} targets met over {s['n']} replies ({s['failed_turns']} turns failed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m evaluation.persona_eval", description=__doc__.splitlines()[0])
    ap.add_argument("--local-url", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--label", default="run", help="name for this scorecard (used in the file name)")
    ap.add_argument("--online", action="store_true", help="also give the core the online model (needs the API key)")
    args = ap.parse_args(argv)
    transcript, s, problem = run(args.local_url, args.online)
    if problem is not None:
        print(f"No model server answered at {args.local_url} ({problem}). Start llama-server and re-run.")
        return 0
    for r in transcript:
        t = "-" if r["first_sentence_s"] is None else f"{r['first_sentence_s']:.2f} s"
        print(f"[{r['probe_id']}] {r['text']}\n    ({t}) {r['reply'] or '(no reply: ' + str(r['error']) + ')'}")
    print("\n" + format_table(s))
    if s.get("invented_probes"):
        print("Invented-memory flags (read these): " + ", ".join(s["invented_probes"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}_{args.label}.json"
    path.write_text(json.dumps({"label": args.label, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": args.local_url,
                                "score": s, "transcript": transcript}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
