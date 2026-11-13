"""Where Riya's local brain lives: the served model, her fine-tuned adapter and the champion record next to it.

Stdlib only: riya.server (the llama-server wrapper) imports this, and so do the champion gate and the Health row.
"""
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

from riya.paths import data_dir

ADAPTER_DIR = Path.home() / "models" / "riya"
DEFAULT_MODEL = (ADAPTER_DIR / "Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf").as_posix()
ADAPTER_NAME = "riya-v1-lora.gguf"  # the adopted adapter: only the champion gate installs a file here
CANDIDATE_NAME = "riya-v1-candidate-lora.gguf"  # a new adapter waiting for the gate (the runbook copies it here)
RECORD_SUFFIX = ".champion.json"


def default_adapter() -> Path:
    return ADAPTER_DIR / ADAPTER_NAME


def candidate_adapter() -> Path:
    """Where a retrained adapter waits for the champion gate, next to the adopted one (never over it)."""
    return default_adapter().with_name(CANDIDATE_NAME)


def default_db() -> Path:
    return data_dir() / "riya.db"


def usable_adapter(path: str | None) -> str | None:
    """The absolute path when it names an existing file, else None."""
    return os.path.abspath(path) if path and os.path.isfile(path) else None


def vetted_adapter(path: str | None) -> str | None:
    """The absolute path when it names an existing file that has a matching champion adopt record (read_record: the
    same size and mtime), else None. The only adapters the wrapper serves."""
    usable = usable_adapter(path)
    return usable if usable and read_record(Path(usable)) is not None else None


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def slot_key(lora: str | None) -> str:
    """The slot-cache directory name for one variant: a retrained file at the same path gets a new key."""
    if lora is None:
        return "base"
    st = os.stat(lora)
    ident = f"{os.path.normcase(os.path.abspath(lora))}|{st.st_size}|{st.st_mtime_ns}"
    return "lora-" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]


def read_local_adapter(db: Path) -> str | None:
    """The local_adapter setting, read-only; never creates the file. A DB or JSON error goes to the caller."""
    if not Path(db).exists():
        return None
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        row = conn.execute("select value from kv where key='settings'").fetchone()
    finally:
        conn.close()
    if not row:
        return None
    value = json.loads(row[0]).get("local_adapter")
    return value if isinstance(value, str) else None


def served_adapters(http_get: Callable[[str, float], bytes], base_url: str) -> list[str]:
    """The path of every adapter llama-server has loaded at a non-zero scale; any error or odd shape gives []."""
    try:
        data = json.loads(http_get(f"{base_url}/lora-adapters", 1.5))
        return [e["path"] for e in data if e["scale"] != 0]
    except Exception:
        return []


def record_path(adapter: Path) -> Path:
    return adapter.with_name(adapter.name + RECORD_SUFFIX)


def write_record(adapter: Path, record: dict, stat: os.stat_result | None = None) -> Path:
    """The champion record keyed to the adapter's size and mtime: `stat` when given (the stat a caller hashed the file
    under), else the file as it is now."""
    st = adapter.stat() if stat is None else stat
    out = record_path(adapter)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**record, "adapter": str(adapter), "size": st.st_size, "mtime_ns": st.st_mtime_ns},
                              indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def read_record(adapter: Path) -> dict | None:
    """The champion record when it belongs to this exact file (size and mtime) and adopted it; else None."""
    try:
        rec = json.loads(record_path(adapter).read_text(encoding="utf-8"))
        st = adapter.stat()
        if (isinstance(rec, dict) and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns
                and rec.get("decision") == "adopt"):
            return rec
    except (OSError, ValueError):
        pass
    return None
