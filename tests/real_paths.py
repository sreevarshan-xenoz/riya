"""The user's real folders, as the code names them before any test redirects them (this module is imported by
tests/conftest.py before any fixture runs), and the read-only listing the session guard compares.

Nothing here writes: `listing` only stats."""
import contextlib
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


# --- no test loads fastembed (v0.11): the loader and the class itself -----------------------------------------------

def no_fastembed(model, cache_dir, local_files_only=False):
    """What embed._load_text_embedding is under pytest, in every scope (tests/conftest.py installs it at configure)."""
    raise RuntimeError(f"tests never load fastembed (model {model}, cache {cache_dir}): "
                       "use tests/data/bge_vectors.npz")


REAL_TEXT_EMBEDDING_INIT = None  # fastembed.TextEmbedding.__init__ as shipped; set by guard_text_embedding


def guard_text_embedding() -> None:
    """Make constructing fastembed.TextEmbedding directly raise too, so a test, fixture or collection-time code that
    skips embed._load_text_embedding still can't download or load a model. Idempotent; a missing fastembed is no
    model to guard. Importing the package loads no model."""
    global REAL_TEXT_EMBEDDING_INIT
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return
    if REAL_TEXT_EMBEDDING_INIT is not None:
        return
    REAL_TEXT_EMBEDDING_INIT = TextEmbedding.__init__

    def guarded(self, model_name="BAAI/bge-small-en-v1.5", cache_dir=None, *args, **kwargs):
        no_fastembed(model_name, cache_dir)
    TextEmbedding.__init__ = guarded


@contextlib.contextmanager
def allow_real_fastembed():
    """For the opt-in live_bge tests only (they load from a tmp copy of the cache): the real constructor, inside the
    block. Use with REAL_LOAD_TEXT_EMBEDDING."""
    if REAL_TEXT_EMBEDDING_INIT is None:
        yield
        return
    from fastembed import TextEmbedding
    guarded = TextEmbedding.__init__
    TextEmbedding.__init__ = REAL_TEXT_EMBEDDING_INIT
    try:
        yield
    finally:
        TextEmbedding.__init__ = guarded
