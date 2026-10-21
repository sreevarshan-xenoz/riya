"""The champion gate: adopt Riya v1 (her LoRA adapter) only when it measurably beats the base model.

  python -m evaluation.champion [--lora riya-v1-candidate-lora.gguf] [--repeat 3] [--dry-run] [--base-vs-base]

It runs the persona scorecard (`persona_eval`, --repeat 3, a temp DB per conversation) on the base model and then on
the base plus the candidate adapter (by default the candidate path next to the adopted adapter), each on its own
llama-server on port 8091, one at a time, and always stops it. That server serves the candidate unconditionally (Riya's
own server serves only an adopted file). Before any probe runs, the base model's server (the user's own llama-server
build) must render her prompts exactly as the trainer's template does (riya.train.fmt.check_template), or it refuses.
The candidate's server must list the adapter in /lora-adapters, or nothing is compared. The decision is
evaluation/gate.py's (v0.10): a paired permutation test per metric, adopt when she meets at least as many targets,
has no more failed turns, keeps or improves 2 of the 4 weak metrics with 1 improved, and nothing is vetoed (a
significant regression in AI or unknown honesty, disclaimers, invented memories, generic share or about-me recall; an
AI probe the base always answers honestly and she never does; any rude-sounding reply). Every mode but --base-vs-base
first hashes the base model and needs a passing gate calibration record for this code and that base
(evaluation/gate_calibration.json), before any server starts.

On adopt it installs the candidate as the adopted adapter (riya-v1-lora.gguf) in one atomic rename, writes the
.champion.json record for the installed file and sets local_adapter to it (through the running core when one answers,
else in the settings DB). On reject it changes nothing: the adopted file and its record stay. --dry-run writes
nothing; --base-vs-base compares the base with itself (a sanity check: always "keep the base model") and implies
--dry-run. --verdict-out PATH (Colab's mode) implies --dry-run too and, after the table, writes the checksummed
verdict (evaluation/verdict.py): both transcripts, the decision, the adapter's and the base's sha256 and size,
llama-server's version and the timings. It combines with --base-vs-base to gather the calibration bank (no calibration
record needed). It refuses while a fullscreen window is up or with under 2500 MB of free VRAM, before any server
starts and again before the candidate, and when the adapter file changed during the run (its size and mtime, or its
sha256). Exit 0 on a decision, 2 when it refused or failed.

The laptop's side of a Colab verdict, on the CPU only (no VRAM check, nothing on the GPU):
  --check-verdict PATH  reads it: the checksum, a candidate verdict, this code's identity and a passing calibration
                        for its repeat and base model; the adapter (--lora) and the base model (--model) are the files
                        it measured (sha256); the transcripts re-scored and decided again here, which must agree with
                        the verdict's own decision. Prints the table; writes nothing, starts nothing.
  --adopt-verdict PATH  the same checks; the decision must be adopt. Then her own llama-server build serves the base
                        and the candidate on the CPU (-ngl 0, an empty CUDA_VISIBLE_DEVICES, a small context): the
                        template check, the adapter listed in /lora-adapters and one 4-token generation, always
                        stopped. Then the atomic install, the installed file hashed again, the record (decided_on
                        "colab", the verdict's checksum and the adapter's sha256) and the setting.
The two exclude each other and --verdict-out, --base-vs-base and --dry-run.
"""
import argparse
import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from evaluation import gate, persona_eval
from evaluation.gate import Decision, MetricCall, Scored, decide, format_decision, scored  # noqa: F401
from evaluation.health_check import WS_URL
from evaluation.verdict import VerdictError, file_identity, load_verdict, make_verdict, write_verdict
from evaluation.verdict import sha256_file as _sha256  # the one streaming hash
from riya import brain, server
from riya.brain import same_path, served_adapters
from riya.core import health
from riya.core.presence import make_idle_source
from riya.core.settings import Settings, apply_patch, load_settings, save_settings
from riya.core.store import KV, open_db
from riya.server import RiyaServer, free_vram_mb
from riya.train import fmt

REPEAT = gate.REPEAT
PORT = 8091
MIN_FREE_VRAM_MB = 2500
CORE_TIMEOUT_S = 15.0  # how long the running core gets to confirm the setting
# the CPU check of her own llama-server before a verdict is adopted: a small context, one 4-token generation
CPU_PORT = PORT
CPU_CTX = 2048
CPU_THREADS = 4
CPU_GEN = {"prompt": "Hi", "n_predict": 4, "temperature": 0}
CPU_TIMEOUT_S = 60.0


class ChampionError(RuntimeError):
    """The gate can't decide (or can't apply its decision); the message says why."""


def check_template(url: str, http_post) -> None:
    """ChampionError unless this llama-server (the user's own build, on the served GGUF) renders her prompts exactly
    as the trainer's template does (fmt.check_template): an adapter trained on another template can't be judged."""
    try:
        problems = fmt.check_template(url, post=http_post)
    except (OSError, ValueError) as e:
        raise ChampionError(f"Couldn't check llama-server's chat template ({e}).") from None
    if problems:
        raise ChampionError("; ".join(problems) + ".")


def _default_http_post(url: str, body: dict, timeout: float) -> dict:
    return fmt._post_json(url, body, timeout)


def run_variant(model: str, lora: str | None, *, repeat: int, port: int, make_server=RiyaServer,
                run=persona_eval.run, http_get=health.default_http_get, phrases: list[str],
                template_post=None) -> Scored:
    """The scorecard on one variant: its own llama-server (a temp slot dir), stopped whatever happens. template_post:
    check the chat template on this server first (check_template) with that poster."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        # its own server serves the candidate as it is (no vetted-only rule): this is how a candidate gets vetted
        srv = make_server(model, port=port, slot_dir=root, lora=lora, fallback_to_base=False, vetted_only=False)
        try:
            srv.start(gpu=True)
            url = f"http://127.0.0.1:{port}"
            if template_post is not None:
                check_template(url, template_post)
            if lora is not None and not any(same_path(p, lora) for p in served_adapters(http_get, url)):
                raise ChampionError(f"the adapter didn't load: {lora}")
            transcript, pooled, problem = run(url, False, repeat)
            if problem is not None:
                raise ChampionError(f"the model server didn't answer ({problem})")
            if not any(not r.get("error") for r in transcript):
                raise ChampionError("no probe got an answer")
            return scored(transcript, phrases)
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


def port_in_use(port: int) -> bool:
    """Something already accepts connections on 127.0.0.1:port (an orphaned llama-server, Riya's own, any program)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _check_free(port: int, port_busy) -> None:
    if port_busy(port):
        raise ChampionError(f"Something is already listening on port {port}: it is already in use (an orphaned "
                            "llama-server? Riya's own?). The gate needs it free for its own server.")


def _check_not_gaming(fullscreen) -> None:
    if fullscreen():
        raise ChampionError("Riya's champion gate doesn't run while you're gaming (a fullscreen window is up).")


def _check_ready(fullscreen, free_vram, lora: str | None, port: int | None = None, port_busy=port_in_use) -> None:
    """The refusals: gaming, too little free VRAM, a missing adapter, a port something already listens on (its
    /health would be taken for the gate's own server). Raises ChampionError with the reason."""
    if port is not None:
        _check_free(port, port_busy)
    _check_not_gaming(fullscreen)
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


def llama_server_version(run=subprocess.run) -> str:
    """The first non-empty line `llama-server --version` prints (stdout, then stderr), else "unknown"."""
    try:
        r = run([server.LLAMA_SERVER, "--version"], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10)
        for text in (getattr(r, "stdout", None), getattr(r, "stderr", None)):
            for line in (text or "").splitlines():
                if line.strip():
                    return line.strip()
    except Exception:  # not installed, hung, not runnable: the verdict just says so
        pass
    return "unknown"


def install(candidate: Path, adopted: Path) -> None:
    """Put the measured candidate at the adopted path in one atomic step: a rename (the runbook keeps both in the
    models folder), or, from another drive, a copy next to the adopted file that is checked byte for byte and then
    renamed over it. ChampionError, with the adopted file and its record untouched, when it can't."""
    if same_path(str(candidate), str(adopted)):
        return  # the gate measured the adopted file itself: nothing to move
    try:
        adopted.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(candidate, adopted)
            return
        except OSError as e:
            if not (getattr(e, "winerror", None) == 17 or e.errno == errno.EXDEV):  # 17: not the same device
                raise
        tmp = adopted.with_name(adopted.name + ".installing")
        try:
            shutil.copy2(candidate, tmp)
            if _sha256(tmp) != _sha256(candidate):
                raise OSError("the copy doesn't match the measured file")
            os.replace(tmp, adopted)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as e:
        raise ChampionError(f"Couldn't install {candidate} as {adopted} ({e}).") from e


# ---- the laptop's side of a Colab verdict (CPU only) -----------------------------------------------------------------

def cpu_env(environ: Mapping[str, str] = os.environ) -> dict:
    """This environment with no GPU visible: llama-server's CUDA build then can't put anything on it."""
    return {**environ, "CUDA_VISIBLE_DEVICES": ""}


def cpu_check(model: str, lora: str, *, make_server=RiyaServer, http_get, http_post, fullscreen,
              port_busy) -> None:
    """Her own llama-server build (Colab's isn't hers) on the CPU only: it renders her template, loads the adapter and
    generates. ChampionError (or the server's RuntimeError) with the reason; the server is stopped whatever happens."""
    _check_free(CPU_PORT, port_busy)
    _check_not_gaming(fullscreen)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        srv = make_server(model, port=CPU_PORT, ctx=CPU_CTX, threads=CPU_THREADS, slot_dir=root, lora=lora,
                          fallback_to_base=False, vetted_only=False, env=cpu_env())
        try:
            srv.start(gpu=False)
            url = f"http://127.0.0.1:{CPU_PORT}"
            check_template(url, http_post)
            if not any(same_path(p, lora) for p in served_adapters(http_get, url)):
                raise ChampionError(f"the adapter didn't load: {lora}")
            try:
                got = http_post(f"{url}/completion", CPU_GEN, CPU_TIMEOUT_S)
            except (OSError, ValueError):  # a timeout, a dropped connection, a garbled answer
                got = None
            if not (isinstance(got, dict) and isinstance(got.get("content"), str)):
                raise ChampionError("her llama-server didn't generate on the CPU")
        finally:
            srv.stop()


def _said(adopt) -> str:
    return "adopt" if adopt is True else "keep" if adopt is False else str(adopt)[:12]


def verdict_steps(path: Path, *, model: str, lora: str) -> tuple[dict, Decision, Scored, Scored]:
    """Steps 1-3 of using a verdict, each a ChampionError with the reason: 1. a sound candidate verdict from this code
    (its identity) under a passing calibration for its repeat and base model; 2. the adapter and the base model here
    are the files it measured (sha256); 3. its transcripts re-scored and decided here, agreeing with its own decision
    in both directions (that field is never trusted on its own). Reads only."""
    try:
        v = load_verdict(Path(path))
    except VerdictError as e:
        raise ChampionError(str(e)) from None
    if v["mode"] != "candidate":
        raise ChampionError("That's a base-vs-base verdict: it can't adopt anything.")
    ours = gate.identity(v["repeat"])
    for field in gate.IDENTITY_FIELDS:
        if v[field] != ours[field]:
            raise ChampionError(f"The verdict is for {field} {str(v[field])[:12]}, "
                                f"this code has {str(ours[field])[:12]}.")
    if v["repeat"] != gate.REPEAT:
        raise ChampionError(f"The verdict is for repeat {str(v['repeat'])[:12]}, "
                            f"this code has {str(gate.REPEAT)[:12]}.")
    if problem := gate.calibration_problem(v["repeat"], base_sha256=v["base_gguf"]["sha256"]):
        raise ChampionError(problem)
    for file, key, wrong in ((lora, "adapter", "This adapter isn't the one the verdict measured."),
                             (model, "base_gguf", "The base model isn't the one the verdict measured.")):
        try:
            sha = file_identity(Path(file))["sha256"]
        except OSError as e:
            raise ChampionError(f"Couldn't read {file}: {e.strerror or e}.") from None
        if sha != v[key]["sha256"]:
            raise ChampionError(wrong)
    phrases = persona_eval.style_phrases(persona_eval.load_probes())
    try:
        base = scored(v["transcripts"]["base"], phrases)
        cand = scored(v["transcripts"]["candidate"], phrases)
        d = decide(base, cand)
    except gate.GateError as e:
        raise ChampionError(str(e)) from None
    except (KeyError, TypeError, ValueError, AttributeError) as e:  # rows without their fields: not a run's transcript
        raise ChampionError(f"The verdict's transcripts can't be scored ({type(e).__name__}: {e}).") from None
    stored = v["decision"].get("adopt") if isinstance(v["decision"], dict) else None
    if stored is not d.adopt:
        raise ChampionError(f"The verdict says {_said(stored)}, but this code decides {_said(d.adopt)}: not adopting "
                            "on a disagreement.")
    return v, d, base, cand


def _unrecorded(adopted: Path, why) -> int:
    print(f"The candidate is installed as {adopted}, but its champion record couldn't be written ({why}). Riya "
          "serves the base model until the gate is run again on it (--lora set to that file).")
    return 2


def _record_and_switch(adopted: Path, record: dict, *, db: Path, ws_url: str, connect, took: str = "") -> int:
    """After the install: the record for the installed file as it is now (its size and mtime), then the setting."""
    try:
        brain.write_record(adopted, record)
    except OSError as e:  # installed but unrecorded: the wrapper serves only a recorded file, so she stays on base
        return _unrecorded(adopted, e)
    try:
        where = set_local_adapter(str(adopted), db=db, ws_url=ws_url, connect=connect, timeout_s=CORE_TIMEOUT_S)
    except Exception as e:  # the record is written; only the switch is missing (never a traceback here)
        why = str(e) if isinstance(e, ChampionError) else f"Couldn't save the setting ({type(e).__name__}: {e})."
        print(f"{why} The adopt record is written: \"Use Riya v1\" in Health switches her to it.")
        return 2
    print("Adopted: Riya v1 is her local brain now. (" + ("set in the running core" if where == "core"
                                                         else "saved; it applies on Riya's next start") + took + ")")
    return 0


def _from_verdict(args, *, adopt: bool, fullscreen, make_cpu_server, http_get, http_post, connect, port_busy,
                  clock: Callable[[], float]) -> int:
    """--check-verdict (steps 1-3 and the table; writes nothing, starts nothing) and --adopt-verdict (then the CPU
    check, the install, the installed file hashed again, the record and the setting). Nothing is written before the
    install, and nothing here reads free VRAM or uses the GPU."""
    t0 = clock()
    lora = str(Path(args.lora).resolve())
    adopted = brain.default_adapter()
    try:
        if adopt:  # the setting it would write must be one the core takes, before anything is written
            try:
                apply_patch(Settings(), {"local_adapter": str(adopted)})
            except ValueError as e:
                raise ChampionError(f"The adopted path can't be saved as local_adapter ({e}).") from None
        v, d, base, cand = verdict_steps(Path(args.verdict), model=args.model, lora=lora)
        print(format_decision(base, cand, d))
        if not adopt:
            print("Check only: nothing written.")
            return 0
        if not d.adopt:
            raise ChampionError("The verdict's decision is to keep the base model.")
        cpu_check(args.model, lora, make_server=make_cpu_server, http_get=http_get, http_post=http_post,
                  fullscreen=fullscreen, port_busy=port_busy)
        install(Path(lora), adopted)  # the adopted file is now the measured candidate, in one rename
    except RuntimeError as e:  # ChampionError, or her server never came up: nothing was written
        print(f"{e} Nothing changed.")
        return 2
    try:
        same = file_identity(adopted)["sha256"] == v["adapter"]["sha256"]
    except OSError as e:
        return _unrecorded(adopted, f"couldn't hash it again: {e.strerror or e}")
    if not same:  # a record certifies only the bytes the verdict measured
        return _unrecorded(adopted, "the installed file isn't the one the verdict measured")
    return _record_and_switch(adopted, {
        "decision": "adopt", "decided_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decided_on": "colab", "verdict_checksum": v["checksum"], "adapter_sha256": v["adapter"]["sha256"],
        "gate_version": v["gate_version"], "probe_sha256": v["probe_sha256"], "repeat": v["repeat"],
        "targets_base": d.targets_base, "targets_candidate": d.targets_candidate,
        "weak": {c.key: c.status for c in d.weak}}, db=Path(args.db), ws_url=args.url, connect=connect,
        took=f"; took {clock() - t0:.0f} s")


def main(argv: list[str] | None = None, *, fullscreen=None, free_vram=None, make_server=RiyaServer,
         run=persona_eval.run, http_get=None, http_post=None, connect=None, port_busy=port_in_use,
         version: Callable[[], str] | None = None, clock: Callable[[], float] = time.monotonic,
         make_cpu_server=RiyaServer) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m evaluation.champion", description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=brain.DEFAULT_MODEL, help="the base GGUF she serves")
    ap.add_argument("--lora", default=str(brain.candidate_adapter()),
                    help="the candidate adapter (a .gguf LoRA); on adopt it is installed as the adopted adapter")
    ap.add_argument("--repeat", type=int, default=REPEAT, help="passes over the probes per variant")
    ap.add_argument("--port", type=int, default=PORT, help="the gate's own llama-server port")
    ap.add_argument("--db", default=str(brain.default_db()), help="the settings DB (used when no core answers)")
    ap.add_argument("--url", default=WS_URL, help="the running core's WebSocket")
    ap.add_argument("--dry-run", action="store_true", help="decide and print, write nothing")
    ap.add_argument("--base-vs-base", action="store_true",
                    help="compare the base with itself (a sanity check; implies --dry-run)")
    ap.add_argument("--verdict-out", metavar="PATH", default=None,
                    help="write a checksummed verdict file (Colab's mode; implies --dry-run; combines with "
                         "--base-vs-base to gather the calibration bank)")
    use = ap.add_mutually_exclusive_group()  # a verdict's two uses (C1): each excludes the run modes below too
    use.add_argument("--check-verdict", metavar="PATH", default=None,
                     help="re-decide a Colab verdict here and print it; writes nothing, starts nothing")
    use.add_argument("--adopt-verdict", metavar="PATH", default=None,
                     help="re-decide a Colab verdict, check her own llama-server on the CPU, then adopt")
    args = ap.parse_args(argv)
    fullscreen = fullscreen or make_idle_source().fullscreen
    free_vram = free_vram or free_vram_mb
    http_get = http_get or health.default_http_get
    if args.verdict_out is not None and not args.verdict_out.strip():
        ap.error("--verdict-out needs a path")  # an empty one must never turn into a live run that can adopt
    for flag, path in (("--check-verdict", args.check_verdict), ("--adopt-verdict", args.adopt_verdict)):
        if path is None:
            continue
        for other, on in (("--verdict-out", args.verdict_out is not None), ("--base-vs-base", args.base_vs_base),
                          ("--dry-run", args.dry_run)):
            if on:
                ap.error(f"{flag} can't be combined with {other}")
        args.verdict = path  # the calibration check is in verdict_steps: it needs the verdict's repeat and base
        return _from_verdict(args, adopt=flag == "--adopt-verdict", fullscreen=fullscreen,
                             make_cpu_server=make_cpu_server, http_get=http_get,
                             http_post=http_post or _default_http_post, connect=connect, port_busy=port_busy,
                             clock=clock)
    dry = args.dry_run or args.base_vs_base or args.verdict_out is not None
    args.lora = str(Path(args.lora).resolve())  # absolute from here on: the server, the snapshot and the install
    cand_lora = None if args.base_vs_base else args.lora  # a str: what the gate's server is given
    adopted = brain.default_adapter()  # where an adopted candidate is installed, and what the setting names
    repeat = max(1, args.repeat)
    variant = dict(repeat=repeat, port=args.port, make_server=make_server, run=run, http_get=http_get)
    try:
        if not dry:  # the setting an adopt would write must be one the core takes, before anything is written
            try:
                apply_patch(Settings(), {"local_adapter": str(adopted)})
            except ValueError as e:
                raise ChampionError(f"The adopted path can't be saved as local_adapter ({e}).") from None
        _check_ready(fullscreen, free_vram, cand_lora, args.port, port_busy)
        base_id = None
        if not args.base_vs_base or args.verdict_out is not None:  # the model is hashed once, before any server starts
            try:
                base_id = file_identity(Path(args.model))
            except OSError as e:
                raise ChampionError(f"Couldn't read the base model {args.model}: {e.strerror or e}.") from None
        if not args.base_vs_base:  # the bank is gathered with --base-vs-base: only a candidate needs a calibration
            if problem := gate.calibration_problem(repeat, base_sha256=base_id["sha256"]):
                print(f"{problem} Nothing changed.")
                return 2
        measured = _snapshot(cand_lora)
        adapter_id = None
        if cand_lora is not None:
            try:
                adapter_id = file_identity(Path(cand_lora))
            except OSError as e:
                raise ChampionError(f"Couldn't read the adapter {cand_lora}: {e.strerror or e}.") from None
        phrases = persona_eval.style_phrases(persona_eval.load_probes())
        t0 = clock()
        base = run_variant(args.model, None, phrases=phrases, template_post=http_post or _default_http_post,
                           **variant)
        t1 = clock()
        _check_ready(fullscreen, free_vram, cand_lora, args.port, port_busy)
        cand = run_variant(args.model, cand_lora, phrases=phrases, **variant)
        t2 = clock()
        try:
            same_file = _snapshot(cand_lora) == measured and (
                cand_lora is None or file_identity(Path(cand_lora)) == adapter_id)
        except OSError:
            same_file = False
        if not same_file:  # the scores belong to the file as it was; a record must never certify another one
            raise ChampionError(f"The adapter at {cand_lora} changed during the run.")
        d = decide(base, cand)
        print(format_decision(base, cand, d))
        if dry:
            print("Dry run: nothing written.")
            if args.verdict_out is not None:
                llama = (version or llama_server_version)()
                total_s = clock() - t0
                v = make_verdict(
                    mode="base-vs-base" if args.base_vs_base else "candidate", identity=gate.identity(repeat),
                    adapter=adapter_id, base_gguf=base_id, llama_server=llama, template_check="ok",
                    base_transcript=list(base.transcript), cand_transcript=list(cand.transcript),
                    decision=gate.decision_dict(d),
                    timings={"base_s": t1 - t0, "candidate_s": t2 - t1, "total_s": total_s},
                    failed={"base": base.failed, "candidate": cand.failed})
                try:
                    write_verdict(v, Path(args.verdict_out))
                except OSError as e:
                    raise ChampionError(f"Couldn't write the verdict {args.verdict_out}: {e.strerror or e}.") from None
                print(f"Verdict written: {args.verdict_out} (total {total_s:.0f} s)")
            return 0
        if not d.adopt:  # the candidate stays where it is; the adopted file and its record are untouched
            return 0
        install(Path(cand_lora), adopted)  # the adopted file is now the measured candidate, in one rename
    except RuntimeError as e:  # ChampionError, or a server that never came up: nothing was written
        print(f"{e} Nothing changed.")
        return 2
    return _record_and_switch(adopted, {
        "decision": "adopt", "decided_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "targets_base": d.targets_base, "targets_candidate": d.targets_candidate, "repeat": repeat,
        "weak": {c.key: c.status for c in d.weak}, "decided_on": "local", "gate_version": gate.GATE_VERSION,
        "probe_sha256": gate.probe_sha256()}, db=Path(args.db), ws_url=args.url, connect=connect)


if __name__ == "__main__":
    raise SystemExit(main())
