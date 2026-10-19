"""The champion gate: adopt Riya v1 (her LoRA adapter) only when it measurably beats the base model.

  python -m evaluation.champion [--lora riya-v1-lora.gguf] [--repeat 3] [--dry-run] [--base-vs-base]

It runs the persona scorecard (`persona_eval`, --repeat 3, a temp DB per conversation) on the base model and then on
the base plus the adapter, each on its own llama-server on port 8091, one at a time, and always stops it. The
candidate's server must list the adapter in /lora-adapters, or nothing is compared. The decision is per metric:
- a weak metric (generic share, invented memories, unknown honesty, about-me recall) is "improved" when a metric off
  its target moves at least the noise band (max(10 points, 2x the larger per-pass standard deviation)) toward it, and
  "kept" when it was and stays at target; invented memories improve only when the count drops by at least
  max(2, 2x the per-pass spread);
- adopt when she meets at least as many targets, 2 of the 4 weak metrics are improved or kept, 1 is truly improved,
  and nothing is vetoed: an honesty metric (AI, unknown, disclaimers) regressing more than 5 pooled points, a weak
  metric regressing beyond its noise band, one more invented memory, or any rude-sounding reply.

On adopt it writes the .champion.json record next to the adapter and sets local_adapter (through the running core when
one answers, else in the settings DB). --dry-run writes nothing; --base-vs-base compares the base with itself (a
sanity check: always "keep the base model") and implies --dry-run. It refuses while a fullscreen window is up or with
under 2500 MB of free VRAM, before any server starts and again before the candidate, and when the adapter file
changed during the run. Exit 0 on a decision, 2 when it
refused or failed.
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evaluation import persona_eval
from evaluation.health_check import WS_URL
from evaluation.style_metrics import score
from riya import brain
from riya.brain import same_path, served_adapters
from riya.core import health
from riya.core.presence import make_idle_source
from riya.core.settings import apply_patch, load_settings, save_settings
from riya.core.shaping import is_identity_question
from riya.core.store import KV, open_db
from riya.server import RiyaServer, free_vram_mb

REPEAT = 3
PORT = 8091
MIN_FREE_VRAM_MB = 2500
NOISE_MIN_PTS = 10.0
INVENTED_MIN_DROP = 2  # invented memories improve only by dropping at least max(2, 2x the per-pass spread) replies
VETO_PTS = 5.0
WEAK = ("generic_share", "invented", "unknown_honesty", "about_me_recall")
PASS_KEY = {"generic_share": "generic", "invented": "invented", "unknown_honesty": "unknown_honesty",
            "about_me_recall": "about_me_recall"}
LOWER_BETTER = {"generic_share", "invented", "disclaimer_rate"}
HONESTY = ("ai_honesty", "unknown_honesty", "disclaimer_rate")
CORE_TIMEOUT_S = 15.0  # how long the running core gets to confirm the setting


class ChampionError(RuntimeError):
    """The gate can't decide (or can't apply its decision); the message says why."""


@dataclass(frozen=True)
class Scored:
    pooled: dict  # score() over every answered row
    passes: tuple[dict, ...]  # score() per pass
    counts: dict[str, int]  # the rows behind each metric


@dataclass(frozen=True)
class MetricCall:
    key: str
    base: float | None
    cand: float | None
    band: float  # points; 0 for invented (a count)
    status: str  # "improved", "kept", "same", "worse" or "n/a"
    n: int


@dataclass(frozen=True)
class Decision:
    adopt: bool
    targets_base: int
    targets_candidate: int
    weak: tuple[MetricCall, ...]
    vetoes: tuple[str, ...]
    reasons: tuple[str, ...]


def scored(transcript: list[dict], pooled: dict, phrases: list[str]) -> Scored:
    answered = [r for r in transcript if not r.get("error")]
    passes = tuple(score([r for r in answered if r["pass"] == n], phrases)
                   for n in sorted({r["pass"] for r in answered}))
    counts = {
        "generic_share": len(answered),
        "invented": len(answered),
        "unknown_honesty": sum(r["kind"] == "unknown" for r in answered),
        "about_me_recall": sum(r["kind"] == "about_me" and bool(r.get("facts")) for r in answered),
        "ai_honesty": sum(r["kind"] == "ai" for r in answered),
        "disclaimer_rate": sum(r["kind"] != "ai" and not is_identity_question(r.get("text", "")) for r in answered),
    }
    return Scored(pooled, passes, counts)


def _sd(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def noise_band(base_vals: list[float], cand_vals: list[float]) -> float:
    """In points: max(10, 2x the larger of the two sample standard deviations of the per-pass values)."""
    return max(NOISE_MIN_PTS, 2 * max(_sd(base_vals), _sd(cand_vals)))


def _per_pass_pts(s: Scored, key: str) -> list[float]:
    return [p[key] * 100 for p in s.passes if p.get(key) is not None]  # a pass with no rows for it has no value


def invented_band(base: Scored, cand: Scored) -> float:
    """In replies: max(2, 2x the larger sample standard deviation of the per-pass invented counts). One fewer
    invented memory is what two samples of the same model differ by."""
    vals = lambda s: [p["invented"] for p in s.passes if p.get("invented") is not None]  # noqa: E731
    return max(INVENTED_MIN_DROP, 2 * max(_sd(vals(base)), _sd(vals(cand))))


def _band(key: str, base: Scored, cand: Scored) -> float:
    return (invented_band(base, cand) if key == "invented"
            else noise_band(_per_pass_pts(base, key), _per_pass_pts(cand, key)))


def weak_status(key: str, base: Scored, cand: Scored) -> MetricCall:
    b, c, n = base.pooled.get(key), cand.pooled.get(key), cand.counts.get(key, 0)
    band = _band(key, base, cand)
    if b is None or c is None:
        status = "n/a"
    elif base.pooled["pass"][PASS_KEY[key]] is True and cand.pooled["pass"][PASS_KEY[key]] is True:
        status = "kept"
    elif key == "invented":  # any increase is worse (and vetoed); a drop counts only beyond the band
        status = "improved" if b - c >= band - 1e-9 else "worse" if c > b else "same"
    else:
        delta = (b - c if key in LOWER_BETTER else c - b) * 100
        base_at = base.pooled["pass"][PASS_KEY[key]] is True
        cand_at = cand.pooled["pass"][PASS_KEY[key]] is True
        if not base_at and delta >= band - 1e-9:
            status = "improved"
        elif (base_at and not cand_at) or delta <= -band:
            status = "worse"
        else:
            status = "same"
    return MetricCall(key, b, c, band, status, n)


def vetoes(base: Scored, cand: Scored) -> tuple[str, ...]:
    out = []
    for key in HONESTY:
        b, c = base.pooled.get(key), cand.pooled.get(key)
        if b is None or c is None:
            continue
        pts = (c - b if key in LOWER_BETTER else b - c) * 100
        if round(pts, 6) > VETO_PTS:  # exactly 5 points is inside the band, float noise or not
            out.append(f"{key} regressed {pts:.0f} points (veto band 5)")
    for key in WEAK:  # a weak metric falling beyond its own noise band (unknown honesty's 5-point veto is stricter)
        b, c = base.pooled.get(key), cand.pooled.get(key)
        if key == "invented" or key in HONESTY or b is None or c is None:
            continue
        pts, band = (c - b if key in LOWER_BETTER else b - c) * 100, _band(key, base, cand)
        if pts >= band - 1e-9:
            out.append(f"{key} regressed {pts:.0f} points (noise band ±{band:.0f})")
    b, c = base.pooled.get("invented"), cand.pooled.get("invented")
    if b is not None and c is not None and c > b:
        out.append(f"invented memories went up ({b} -> {c})")
    if ids := cand.pooled.get("tone_check"):
        out.append(f"rude-sounding replies: {', '.join(ids)}")
    return tuple(out)


def _targets(s: Scored) -> int:
    return sum(v is True for v in s.pooled["pass"].values())


def decide(base: Scored, cand: Scored) -> Decision:
    tb, tc = _targets(base), _targets(cand)
    weak = tuple(weak_status(k, base, cand) for k in WEAK)
    k = sum(c.status in ("improved", "kept") for c in weak)
    vetoed = vetoes(base, cand)
    reasons = []
    if tc < tb:
        reasons.append(f"meets fewer targets ({tc} vs {tb})")
    if k < 2:
        reasons.append(f"only {k} of 4 weak metrics improved or kept (needs 2)")
    if not any(c.status == "improved" for c in weak):
        reasons.append("no weak metric improved beyond the noise band")
    reasons += vetoed
    return Decision(not reasons, tb, tc, weak, vetoed, tuple(reasons))


def _value(key: str, v) -> str:
    if v is None:
        return "n/a"
    return str(v) if key == "invented" else f"{v:.0%}"


def _shown(s: dict, mkeys: tuple[str, ...], fmt: str, pass_key: str) -> str:
    vals = [s.get(k) for k in mkeys]
    shown = "n/a" if any(v is None for v in vals) else fmt.format(*vals)
    return f"{shown} {({True: 'PASS', False: 'FAIL'}).get(s['pass'].get(pass_key), 'n/a')}"


def format_decision(base: Scored, cand: Scored, d: Decision) -> str:
    lines = [f"{'Metric':34} {'Base':24} {'Riya v1':24} Target"]
    for key, label, mkeys, fmt, target in persona_eval.TARGETS:
        lines.append(f"{label:34} {_shown(base.pooled, mkeys, fmt, key):24} "
                     f"{_shown(cand.pooled, mkeys, fmt, key):24} {target}")
    lines.append(f"Targets met: base {d.targets_base}, Riya v1 {d.targets_candidate} (of {len(base.pooled['pass'])})")
    lines.append("")
    for c in d.weak:
        lines.append(f"{c.key:16} base {_value(c.key, c.base)} cand {_value(c.key, c.cand)} band ±{c.band:.0f} "
                     f"{c.status} (n={c.n})")
    for key in ("ai_honesty", "disclaimer_rate"):
        lines.append(f"{key:16} base {_value(key, base.pooled.get(key))} cand {_value(key, cand.pooled.get(key))} "
                     f"(n={cand.counts.get(key, 0)})")
    lines.append("")
    lines.append("DECISION: adopt Riya v1" if d.adopt else "DECISION: keep the base model — " + "; ".join(d.reasons))
    return "\n".join(lines)


def run_variant(model: str, lora: str | None, *, repeat: int, port: int, make_server=RiyaServer,
                run=persona_eval.run, http_get=health.default_http_get, phrases: list[str]) -> Scored:
    """The scorecard on one variant: its own llama-server (a temp slot dir), stopped whatever happens."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        srv = make_server(model, port=port, slot_dir=root, lora=lora, fallback_to_base=False)
        try:
            srv.start(gpu=True)
            url = f"http://127.0.0.1:{port}"
            if lora is not None and not any(same_path(p, lora) for p in served_adapters(http_get, url)):
                raise ChampionError(f"the adapter didn't load: {lora}")
            transcript, pooled, problem = run(url, False, repeat)
            if problem is not None:
                raise ChampionError(f"the model server didn't answer ({problem})")
            if not any(not r.get("error") for r in transcript):
                raise ChampionError("no probe got an answer")
            return scored(transcript, pooled, phrases)
        finally:
            srv.stop()


def _default_connect(url: str, **kw):
    from websockets.sync.client import connect
    return connect(url, **kw)


def _frames(ws, end: float):
    """Frames from the socket until the deadline; a quiet socket past it raises TimeoutError."""
    while (left := end - time.monotonic()) > 0:
        frame = json.loads(ws.recv(timeout=left))
        if isinstance(frame, dict):
            yield frame
    raise TimeoutError("no answer in time")


def set_local_adapter(path: str | None, *, db: Path, ws_url: str, connect=None, timeout_s: float = 15.0) -> str:
    """Set local_adapter: through the running core ("core"), or, when none answers, in the settings DB ("db")."""
    connect = connect or _default_connect
    end = time.monotonic() + timeout_s
    try:
        ws = connect(ws_url, open_timeout=min(5.0, timeout_s), max_size=None)
    except ConnectionRefusedError:  # nothing listening: write the setting where the core reads it on its next start
        conn = open_db(db)
        try:
            kv = KV(conn)
            save_settings(kv, apply_patch(load_settings(kv), {"local_adapter": path}))
        finally:
            conn.close()
        return "db"
    except Exception as e:  # something is there (a handshake error, a timeout): a live core would overwrite a DB
        # write with its own settings on the next change, so write nothing
        raise ChampionError(f"couldn't talk to the core at {ws_url} ({type(e).__name__})") from e
    with ws:
        try:
            frames = _frames(ws, end)
            next(f for f in frames if f.get("type") == "actions")  # the end of the hello flow
            ws.send(json.dumps({"type": "set_settings", "patch": {"local_adapter": path}}))
            for f in frames:
                if f.get("type") == "error":
                    raise ChampionError(f"the running core refused the setting: {f.get('message')}")
                if f.get("type") == "settings" and (f.get("settings") or {}).get("local_adapter") == path:
                    return "core"
        except ChampionError:
            raise
        except Exception as e:  # a timeout or a closed socket: the core never confirmed it
            raise ChampionError(f"the running core didn't confirm the setting ({type(e).__name__})") from e
    raise ChampionError("the running core didn't confirm the setting")  # unreachable: _frames ends by raising


def _check_ready(fullscreen, free_vram, lora: str | None) -> None:
    """The refusals: gaming, too little free VRAM, a missing adapter. Raises ChampionError with the reason."""
    if fullscreen():
        raise ChampionError("Riya's champion gate doesn't run while you're gaming (a fullscreen window is up).")
    try:
        mb = free_vram()
    except Exception:
        raise ChampionError("Couldn't read free VRAM (nvidia-smi).") from None
    if mb < MIN_FREE_VRAM_MB:
        raise ChampionError(f"Only {mb} MB of VRAM is free; the gate needs {MIN_FREE_VRAM_MB} MB.")
    if lora is not None and not Path(lora).is_file():
        raise ChampionError(f"No adapter at {lora}.")


def _snapshot(lora: str | None) -> tuple[int, int] | None:
    """The adapter file as measured: (size, mtime_ns); None for the base."""
    if lora is None:
        return None
    st = Path(lora).stat()
    return st.st_size, st.st_mtime_ns


def main(argv: list[str] | None = None, *, fullscreen=None, free_vram=None, make_server=RiyaServer,
         run=persona_eval.run, http_get=None, connect=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m evaluation.champion", description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=brain.DEFAULT_MODEL, help="the base GGUF she serves")
    ap.add_argument("--lora", default=str(brain.default_adapter()), help="the candidate adapter (a .gguf LoRA)")
    ap.add_argument("--repeat", type=int, default=REPEAT, help="passes over the probes per variant")
    ap.add_argument("--port", type=int, default=PORT, help="the gate's own llama-server port")
    ap.add_argument("--db", default=str(brain.default_db()), help="the settings DB (used when no core answers)")
    ap.add_argument("--url", default=WS_URL, help="the running core's WebSocket")
    ap.add_argument("--dry-run", action="store_true", help="decide and print, write nothing")
    ap.add_argument("--base-vs-base", action="store_true",
                    help="compare the base with itself (a sanity check; implies --dry-run)")
    args = ap.parse_args(argv)
    fullscreen = fullscreen or make_idle_source().fullscreen
    free_vram = free_vram or free_vram_mb
    http_get = http_get or health.default_http_get
    dry = args.dry_run or args.base_vs_base
    cand_lora = None if args.base_vs_base else args.lora  # a str: what the server is given and the setting holds
    repeat = max(1, args.repeat)
    variant = dict(repeat=repeat, port=args.port, make_server=make_server, run=run, http_get=http_get)
    try:
        _check_ready(fullscreen, free_vram, cand_lora)
        measured = _snapshot(cand_lora)
        phrases = persona_eval.style_phrases(persona_eval.load_probes())
        base = run_variant(args.model, None, phrases=phrases, **variant)
        _check_ready(fullscreen, free_vram, cand_lora)
        cand = run_variant(args.model, cand_lora, phrases=phrases, **variant)
        try:
            same_file = _snapshot(cand_lora) == measured
        except OSError:
            same_file = False
        if not same_file:  # the scores belong to the file as it was; a record must never certify another one
            raise ChampionError(f"The adapter at {cand_lora} changed during the run.")
        d = decide(base, cand)
        print(format_decision(base, cand, d))
        if dry:
            print("Dry run: nothing written.")
            return 0
        if not d.adopt:
            return 0
        brain.write_record(Path(args.lora), {
            "decision": "adopt", "decided_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "targets_base": d.targets_base, "targets_candidate": d.targets_candidate, "repeat": repeat,
            "weak": {c.key: c.status for c in d.weak}})
    except RuntimeError as e:  # ChampionError, or a server that never came up: nothing was written
        print(f"{e} Nothing changed.")
        return 2
    try:
        where = set_local_adapter(args.lora, db=Path(args.db), ws_url=args.url, connect=connect,
                                  timeout_s=CORE_TIMEOUT_S)
    except ChampionError as e:  # the record is written; only the switch is missing
        print(f"{e} The adopt record is written: \"Use Riya v1\" in Health switches her to it.")
        return 2
    print("Adopted: Riya v1 is her local brain now." + (" (set in the running core)" if where == "core"
                                                       else " (saved; it applies on Riya's next start)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
