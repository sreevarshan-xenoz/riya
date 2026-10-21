"""The gate's verdict file (v0.10): what Colab's `champion --verdict-out` writes and the laptop's adopt re-derives its
decision from. One JSON document holding the identity of the code that scored it, the adapter's and the base GGUF's
sha256, both raw transcripts, the computed decision, the Python and numpy versions (information only) and a checksum
of everything else.

The checksum guards against corruption and stale or mixed-up files (a truncated sync, another run's file). It is not a
signature: the threat is accidents, not a hostile Colab. Stdlib plus evaluation.gate (which brings numpy) and
riya.train.fmt (its atomic writer). Also the repo's one streaming file hash."""
import hashlib
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation import gate
from riya.train import fmt

KIND = "riya-gate-verdict"
MODES = ("candidate", "base-vs-base")
REQUIRED = ("kind", "mode", "gate_version", "probe_sha256", "prompt_fp", "scorer_sha256", "repeat", "alpha", "B",
            "seed", "adapter", "base_gguf", "llama_server", "template_check", "transcripts", "decision", "timings",
            "failed_turns", "created_at", "checksum")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class VerdictError(ValueError):
    """A verdict file that can't be trusted; the message is one line saying why."""


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def checksum_of(body: dict) -> str:
    """The sha256 of the canonical JSON of every key but `checksum`."""
    return hashlib.sha256(canonical({k: v for k, v in body.items() if k != "checksum"})).hexdigest()


def sha256_file(path: Path) -> str:
    """The sha256 of the file, streamed in 1 MiB blocks (a base GGUF is far too big to read whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: Path) -> dict:
    path = Path(path)
    return {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}


def env_info() -> dict:
    """The Python and numpy versions this runs on: information only, never identity and never checked (M4). The gate's
    paired tests draw from numpy's generator, whose stream numpy doesn't promise across versions (NEP 19), so a
    decision near alpha can come out differently on another numpy; the notes below say when the versions differ."""
    return {"python": platform.python_version(), "numpy": np.__version__}


def env_versions(env) -> str:
    env = env if isinstance(env, dict) else {}
    return f"Python {str(env.get('python', '?'))[:20]} with numpy {str(env.get('numpy', '?'))[:20]}"


def env_note(theirs, what: str) -> str | None:
    """A one-line note when `theirs` (a verdict's or a record's env) names other versions than this machine's; None
    when they match or it names none (a file written before the field existed)."""
    ours = env_info()
    if not isinstance(theirs, dict) or all(theirs.get(k) == v for k, v in ours.items()):
        return None
    return (f"Note: {what} ran on {env_versions(theirs)}; this machine has {env_versions(ours)}. Information only: "
            "numpy doesn't promise the same random stream across versions, so a call near the threshold can differ.")


def make_verdict(*, mode: str, identity: dict, adapter: dict | None, base_gguf: dict, llama_server: str,
                 template_check: str, base_transcript: list[dict], cand_transcript: list[dict], decision: dict,
                 timings: dict, failed: dict) -> dict:
    body = {"kind": KIND, "mode": mode, **{k: identity[k] for k in gate.IDENTITY_FIELDS},
            "alpha": gate.ALPHA, "B": gate.B, "seed": gate.GATE_SEED, "adapter": adapter, "base_gguf": base_gguf,
            "llama_server": llama_server, "template_check": template_check,
            "transcripts": {"base": base_transcript, "candidate": cand_transcript}, "decision": decision,
            "timings": timings, "failed_turns": failed, "env": env_info(),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    body = json.loads(canonical(body))  # as a file reads back (tuples are lists): the dict equals its own reload
    body["checksum"] = checksum_of(body)
    return body


def write_verdict(v: dict, path: Path) -> None:
    fmt.write_json(v, path)


def load_verdict(path: Path) -> dict:
    """The verdict at `path`, or a VerdictError with a one-line reason: unreadable or not JSON (a truncated sync), not
    a verdict, a missing key, transcripts that aren't lists, an adapter or base without a sha256 (the adapter only
    in candidate mode), or a checksum that doesn't hold."""
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise VerdictError(f"Couldn't read the verdict {path}: {e.strerror or e}.") from None
    except ValueError:  # JSONDecodeError and UnicodeDecodeError: a cut or garbled file
        raise VerdictError(f"The verdict {path} isn't valid JSON (a truncated copy?).") from None
    if not isinstance(v, dict):
        raise VerdictError(f"The verdict {path} isn't a verdict file.")
    if v.get("kind") != KIND:
        raise VerdictError(f"The file {path} isn't a gate verdict (kind {str(v.get('kind'))[:40]!r}).")
    if v.get("mode") not in MODES:
        raise VerdictError(f"The verdict has an unknown mode ({str(v.get('mode'))[:40]!r}).")
    if missing := [k for k in REQUIRED if k not in v]:
        raise VerdictError(f"The verdict is missing {', '.join(missing)}.")
    t = v["transcripts"]
    if not (isinstance(t, dict) and isinstance(t.get("base"), list) and isinstance(t.get("candidate"), list)):
        raise VerdictError("The verdict's transcripts aren't lists of rows.")
    for key in ("adapter", "base_gguf"):  # a candidate verdict binds an adapter; every verdict names its base
        if key == "adapter" and v["mode"] != "candidate":
            continue
        f = v[key]
        if not (isinstance(f, dict) and isinstance(f.get("sha256"), str) and _HEX64.fullmatch(f["sha256"])):
            raise VerdictError(f"The verdict's {key} has no sha256.")
    if v["checksum"] != checksum_of(v):
        raise VerdictError("The verdict's checksum doesn't match its contents (edited or corrupted).")
    return v
