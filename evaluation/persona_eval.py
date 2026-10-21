"""Live persona scorecard: scripted probes through the real RiyaCore against the real local model. Exits 0 (see --embedder fast).

  python -m evaluation.persona_eval --local-url http://127.0.0.1:8090 [--online] [--label baseline] [--repeat 2]
                                    [--dials] [--embedder {auto,fast,hash}]

Starts nothing itself (run `python -m riya.server --model <gguf>` first). Writes
scratch_runs/persona_eval/<timestamp>_<label>.json and prints a table with pass/fail per target. It's a measurement,
not a gate. --dials runs the probes at baseline, at low and high teasing/verbosity, and prints a "Dial check": the
dials should change her style, not break her character (the share metrics stay within the noise). --embedder picks
the one embedder the whole invocation measures: auto (default) is the pinned bge model when it loads and verifies, else
the hash embedder with a note; fast is bge or an error (the one case that doesn't exit 0: it raises when the pinned model
can't load or verify); hash is the hash embedder. Its id is recorded as embedder_id.
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from evaluation.style_metrics import irrelevant_share, score
from riya.core.clock import FakeClock
from riya.core import embed
from riya.core.embed import Embedder, HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.initiative import Action
from riya.core.learn.dials import BANDS
from riya.core.models import ModelUnavailable, load_dotenv, load_online_from_env, local_model
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
    ("about_me_recall", "About-me recall", ("about_me_recall",), "{:.0%}", ">= 75%"),
    ("disclaimer", "Disclaimers (non-identity)", ("disclaimer_rate",), "{:.0%}", "<= 5%"),
]
GREET_AWAY_H = 30
DIAL_SETS = {"low": {d: BANDS[d][0] for d in ("teasing", "verbosity")},  # the band edges, by definition
             "high": {d: BANDS[d][1] for d in ("teasing", "verbosity")}}
NOISE_PTS = 0.15  # a share metric moving less than this between two runs is sampling noise
SHARE_KEYS = ("generic_share", "marker_share", "unknown_honesty", "ai_honesty", "remember_ack", "about_me_recall",
              "disclaimer_rate")


EMBEDDERS = ("auto", "fast", "hash")

# The 17 fixed conversations v0.11's A4 counts: the 8 "applied", the 8 "open" and g_about_12 ("what can't i eat")
PERSONAL_PROBES = ("g_about_02", "g_about_03", "g_about_05", "g_about_06", "g_about_09", "g_about_10", "g_about_12",
                   "g_about_13", "g_about_14", "g_about_17", "g_about_19", "g_about_20", "g_about_21", "g_about_24",
                   "g_about_25", "g_about_26", "g_about_27")


def make_embedder(choice: str = "auto") -> tuple[Embedder, str]:
    """(embedder, embedder_id) for one invocation. "fast" is the pinned bge model, verified (errors propagate);
    "auto" tries it and falls back to the hash embedder with a note; "hash" is the hash embedder."""
    if choice == "hash":
        emb = HashEmbedder()
        return emb, emb.identity()
    if choice == "fast":
        emb = embed.FastEmbedder(verify=True)
        return emb, emb.embedder_id
    try:
        return make_embedder("fast")
    except Exception as e:  # noqa: BLE001 - any failure to load or verify means the hash fallback
        print(f"Note: the bge embedder didn't load ({type(e).__name__}: {e}); running on the hash embedder.")
        return make_embedder("hash")


def load_probes(path: Path = PROBES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_phrases(probes: dict) -> list[str]:
    """The probes' known leak phrases plus, when the file exists, Riya's lines in riya/core/style_examples.md that
    are 6+ words (shorter ones like "Hmph." are her voice, not a leak)."""
    out = list(probes.get("style_phrases", []))
    if STYLE_EXAMPLES.exists():
        for line in STYLE_EXAMPLES.read_text(encoding="utf-8").splitlines():
            if (m := re.match(r"^\W*riya\W*:\s*(.+)$", line, re.I)) and len(m.group(1).split()) >= 6:
                out.append(m.group(1).strip())
    return out


def run_conversation(conv: dict, local, online, tmp: str, tag: str = "", dials: dict | None = None,
                     embedder: Embedder | None = None) -> list[dict]:
    """One fresh core on a temp db; one row per turn (empty reply plus an `error` when the model failed). `dials` are
    set on the core's learn store (a temp db, never the user's) right after it is built. `embedder` is shared by every
    conversation of a run (None: a new HashEmbedder).
    known_texts is the seeded facts plus the user's turns so far: her own words are never evidence about the user.
    facts is what her memory holds at that turn. A "speak_first" turn is her greeting after away_hours away."""
    clock, emitted = FakeClock(), []
    db = open_db(Path(tmp) / "riya.db")
    try:
        emb = HashEmbedder() if embedder is None else embedder
        core = RiyaCore(db=db, embedder=emb, clock=clock, idle_source=FakeIdleSource(), local=local,
                        online=online, emit=emitted.append)
        if dials:
            core.learn.set_dials(dials, "persona_eval", "explicit", clock.now())
        known = list(conv["facts"])
        for fact in conv["facts"]:
            core.memory.add_fact(fact, "user_said")
        rows = []
        for turn in conv["turns"]:
            emitted.clear()
            if turn.get("speak_first"):
                away = turn.get("away_hours", GREET_AWAY_H) * 3600.0
                clock.advance(away)
                core.feelings.on_return(away)
                core._away_s = away
                utterance = core.speak_first(Action("greet", True, None, "came back"))
            else:
                known.append(turn["text"])
                utterance = core.reply(turn["text"])
            sentences, first, t0 = [], None, time.perf_counter()
            for s in utterance:  # time to the first yielded sentence
                if first is None:
                    first = time.perf_counter() - t0
                sentences.append(s)
            err = next((m.message for m in emitted if isinstance(m, ErrorMsg)), None)
            row = {"conversation": conv["id"], "probe_id": f"{conv['id']}.{turn['id']}{tag}",
                   "kind": turn["kind"], "text": turn["text"], "reply": " ".join(sentences),
                   "first_sentence_s": first, "known_texts": list(known),
                   "facts": [f.text for f in core.memory.list_facts()], "error": err}
            if "relevant" in turn:  # the fact texts, so a verdict can be re-scored (never scored by the gate)
                row["relevant"] = [conv["facts"][i] for i in turn["relevant"]]
            rows.append(row)
            clock.advance(TURN_GAP_S)
        return rows
    finally:
        db.close()


def run_probes(probes: dict, local, online, repeat: int = 1, dials: dict | None = None,
               embedder: Embedder | None = None) -> list[dict]:
    """Every conversation, fresh each time, `repeat` passes over the probes (sampling is noisy: score them together).
    Rows carry their pass number; probe ids after the first pass get a "#n" suffix."""
    transcript = []
    for n in range(1, repeat + 1):
        for conv in probes["conversations"]:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                rows = run_conversation(conv, local, online, tmp, "" if n == 1 else f"#{n}", dials, embedder)
            transcript += [r | {"pass": n} for r in rows]
    return transcript


def run(local_url: str, online_wanted: bool = False, repeat: int = 1, dials: dict | None = None, *,
        embedder: tuple[Embedder, str] | None = None) -> tuple[list[dict], dict, str | None]:
    """(transcript, score, problem). problem is set when the local server couldn't be reached (nothing is run).
    `embedder` is an (embedder, embedder_id) pair; None builds make_embedder("auto"), after the reachability check.
    The score carries the embedder_id."""
    probes = load_probes()
    local = local_model(local_url, first_token_timeout_s=60.0)
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
    emb, embedder_id = embedder if embedder is not None else make_embedder("auto")
    transcript = run_probes(probes, local, online, repeat, dials, emb)
    answered = [r for r in transcript if not r["error"]]
    s = score(answered, style_phrases(probes))
    return transcript, (s | {"failed_turns": len(transcript) - len(answered), "online": online is not None,
                         "embedder_id": embedder_id}), None


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


def dial_check(base: dict, other: dict) -> list[tuple[str, float | None, bool | None]]:
    """Per share metric: (key, other - base, within the noise). (key, None, None) when either run has no value."""
    out = []
    for k in SHARE_KEYS:
        if base.get(k) is None or other.get(k) is None:
            out.append((k, None, None))
        else:
            d = other[k] - base[k]
            out.append((k, d, abs(round(d, 6)) <= NOISE_PTS))
    return out


def format_dial_check(scores: dict[str, dict]) -> str:
    """The "Dial check": for each dialled set, its small-talk median and emoji mean and the share deltas against
    baseline, then a verdict. `scores` maps "baseline" and each set name to its score."""
    base, moved = scores["baseline"], []
    fmt = lambda v, f: "n/a" if v is None else f.format(v)  # noqa: E731
    lines = ["Dial check (share metrics vs baseline, noise +/-15 points)"]
    for name, s in scores.items():
        lines.append(f"  {name}: small-talk median {fmt(s['small_median'], '{:g}')}, "
                     f"emoji mean {fmt(s['emoji_mean'], '{:.2f}')}")
        if name == "baseline":
            continue
        for key, d, ok in dial_check(base, s):
            lines.append(f"    {key:18} {fmt(d, '{:+.0%}'):>6}  {'n/a' if ok is None else 'ok' if ok else 'OUT OF NOISE'}")
            if ok is False:
                moved.append(f"{name}/{key}")
    lines.append("Verdict: " + ("every share metric stays within the noise at both dial sets." if not moved else
                                "OUT OF NOISE: " + ", ".join(moved) + " (read those transcripts; one run is noisy)."))
    return "\n".join(lines)


def _print_run(transcript: list[dict], s: dict) -> None:
    for r in transcript:
        t = "-" if r["first_sentence_s"] is None else f"{r['first_sentence_s']:.2f} s"
        print(f"[{r['probe_id']}] {r['text']}\n    ({t}) {r['reply'] or '(no reply: ' + str(r['error']) + ')'}")
    print("\n" + format_table(s))
    if s.get("invented_probes"):
        print("Invented-memory flags (read these): " + ", ".join(s["invented_probes"]))
    replies = {r["probe_id"]: r["reply"] for r in transcript}
    print("Tone check (rudeness is an anti-goal; read these): " + ("none" if not s.get("tone_check") else ""))
    for pid in s.get("tone_check", []):
        print(f"    [{pid}] {replies[pid]}")
    share, n = irrelevant_share(transcript)  # informational: not in score(), so never in a verdict
    print("Irrelevant facts used (about-me rows with a distractor): "
          + ("n/a" if share is None else f"{round(share * n)} of {n} ({share:.0%})"))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m evaluation.persona_eval", description=__doc__.splitlines()[0])
    ap.add_argument("--local-url", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--label", default="run", help="name for this scorecard (used in the file name)")
    ap.add_argument("--online", action="store_true", help="also give the core the online model (needs the API key)")
    ap.add_argument("--repeat", type=int, default=1, help="passes over the probes, scored together (default 1)")
    ap.add_argument("--dials", action="store_true",
                    help="run at baseline, low and high teasing/verbosity and print a Dial check")
    ap.add_argument("--embedder", choices=EMBEDDERS, default="auto",
                    help="the one embedder to measure: auto (bge if it loads and verifies, else hash), fast, hash")
    args = ap.parse_args(argv)
    repeat = max(1, args.repeat)
    sets = [("baseline", None), *DIAL_SETS.items()] if args.dials else [("baseline", None)]
    pair = make_embedder(args.embedder)  # one embedder for the whole invocation, every dial set included
    embedder_id = pair[1]
    print(f"Embedder: {embedder_id}")
    results: dict[str, tuple[list[dict], dict]] = {}
    for name, dials in sets:
        # the plain run keeps its three-argument call; with --dials the baseline run's reachability check is the check
        transcript, s, problem = (run(args.local_url, args.online, repeat, dials=dials, embedder=pair) if args.dials
                                  else run(args.local_url, args.online, repeat, embedder=pair))
        if problem is not None:
            print(f"No model server answered at {args.local_url} ({problem}). Start llama-server and re-run.")
            return 0
        results[name] = (transcript, s)
        if args.dials:
            print(f"\n===== {name}" + (f" {dials}" if dials else "") + " =====")
        _print_run(transcript, s)
    if args.dials:
        print("\n" + format_dial_check({n: s for n, (_, s) in results.items()}))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}_{args.label}.json"
    body = {"label": args.label, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": args.local_url, "repeat": repeat,
            "embedder_id": embedder_id}
    if args.dials:
        body |= {"dials": {n: s for n, (_, s) in results.items()}, "transcripts": {n: t for n, (t, _) in results.items()}}
    else:
        ((transcript, s),) = results.values()
        body |= {"score": s, "transcript": transcript}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
