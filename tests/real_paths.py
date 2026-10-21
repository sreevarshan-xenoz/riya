"""The user's real folders, as the code names them before any test redirects them (this module is imported by
tests/conftest.py before any fixture runs), and the read-only listing the session guard compares.

Nothing here writes: `listing` only stats."""
import os
from pathlib import Path

from riya import brain
from riya.core import embed

REAL_DEFAULT_MODEL = brain.DEFAULT_MODEL  # the base GGUF she serves, as riya.brain names it
REAL_ADAPTER_DIR = brain.ADAPTER_DIR  # where the adopted adapter and its champion record live
REAL_MODELS = REAL_ADAPTER_DIR.parent  # ~/models: no test may change anything under it
REAL_LOAD_TEXT_EMBEDDING = embed._load_text_embedding  # the one fastembed loader, recorded before any fixture patches it


def listing(root: Path) -> dict[str, tuple[str, int, int]] | None:
    """Every entry under root (recursive, links not followed), by its path relative to root: (kind, size,
    mtime_ns), a folder's size 0. None when root doesn't exist. Read-only: it only stats."""
    root = Path(root)
    if not root.is_dir():
        return None
    out = {}
    for here, dirs, files in os.walk(root):
        for name in dirs + files:
            p = Path(here) / name
            try:
                st = p.lstat()
            except OSError:  # gone between the walk and the stat: a change the next listing shows
                continue
            is_dir = name in dirs  # a folder's size isn't stable on Windows (0, then 4096, mtime unchanged): files only
            out[p.relative_to(root).as_posix()] = ("dir" if is_dir else "file", 0 if is_dir else st.st_size,
                                                   st.st_mtime_ns)
    return out


def changes(before: dict | None, after: dict | None) -> list[str]:
    """What differs between two listings, one line per entry: added, removed or changed (size or mtime)."""
    if before is None and after is None:
        return []
    if before is None or after is None:
        return ["the folder " + ("appeared" if before is None else "disappeared")]
    out = [f"added {k}" for k in sorted(after.keys() - before.keys())]
    out += [f"removed {k}" for k in sorted(before.keys() - after.keys())]
    out += [f"changed {k}: {before[k]} -> {after[k]}" for k in sorted(before.keys() & after.keys())
            if before[k] != after[k]]
    return out


def redirect_models(tmp_path: Path, monkeypatch) -> Path:
    """riya.brain's base model and adapter folder, and so its adopted and candidate adapters, under
    tmp_path/models/riya (tests/conftest.py for every test; tests/gate_helpers.calibrated again, so a gate test module
    imported without the conftest still can't reach the real folder). Returns that folder."""
    models = Path(tmp_path) / "models" / "riya"
    monkeypatch.setattr(brain, "DEFAULT_MODEL", (models / Path(REAL_DEFAULT_MODEL).name).as_posix())
    monkeypatch.setattr(brain, "ADAPTER_DIR", models)  # default_adapter() and candidate_adapter() read it
    return models


def tmp_only(tmp_path: Path, path: Path) -> Path:
    """`path`, after asserting it lies under tmp_path: a fixture about to write a fake adapter calls this first (C1)."""
    assert Path(tmp_path) in Path(path).parents, f"refusing to write a test file outside tmp_path: {path}"
    return Path(path)
