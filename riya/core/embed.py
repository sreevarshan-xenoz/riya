"""Text embedders: a dependency-free hashing embedder and a lazy fastembed wrapper."""
import hashlib
import importlib.metadata
import json
import logging
import os
import unicodedata
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger("riya.embed")

# The pin (spec 4.7): the one bge model her memory's similarity thresholds were measured on, as fastembed serves it.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"  # what fastembed actually downloads for EMBED_MODEL
EMBED_SNAPSHOT = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
FASTEMBED_VERSION = "0.9.0"
EMBED_FILES: dict[str, tuple[int, str]] = {  # name -> (size, sha256), the five files in the snapshot folder
    "config.json": (706, "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd"),
    "model_optimized.onnx": (66465124, "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"),
    "special_tokens_map.json": (695, "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"),
    "tokenizer.json": (711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    "tokenizer_config.json": (1242, "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53"),
}


def pin_id(snapshot: str, files: Mapping[str, tuple[int, str]], version: str) -> str:
    """The id of an embedder: a digest of the snapshot, every file's (size, sha256) and the fastembed version, then
    the readable parts (the digest comes first: every mismatch message shows the first 12 characters)."""
    canon = json.dumps({"snapshot": snapshot, "files": {n: list(files[n]) for n in sorted(files)},
                        "fastembed": version}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode()).hexdigest()
    return f"{digest[:16]}:bge-small-en-v1.5-onnx-Q@{snapshot[:8]}+fastembed-{version}"


def snapshot_files(snapshot_dir: Path, names: Iterable[str]) -> dict[str, tuple[int, str]]:
    """(size, sha256) of each named file in the folder, read in chunks; a file that isn't there is (-1, "missing")."""
    out = {}
    for name in names:
        path = Path(snapshot_dir) / name
        try:
            h, size = hashlib.sha256(), 0
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
                    size += len(chunk)
            out[name] = (size, h.hexdigest())
        except OSError:
            out[name] = (-1, "missing")
    return out


def installed_fastembed() -> str:
    try:
        return importlib.metadata.version("fastembed")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def pin_problems(snapshot: str, files: Mapping[str, tuple[int, str]], version: str) -> list[str]:
    """One line per difference between what was loaded and the module's pin (read now, so a test can move the pin)."""
    out = []
    if snapshot != EMBED_SNAPSHOT:
        out.append(f"snapshot {snapshot[:12]}… expected {EMBED_SNAPSHOT[:12]}…")
    for name, (size, sha) in EMBED_FILES.items():
        got_size, got_sha = files.get(name, (-1, "missing"))
        if got_sha == "missing":
            out.append(f"{name}: missing")
        elif got_sha != sha:
            out.append(f"{name}: sha256 {got_sha[:12]}… expected {sha[:12]}…")
        elif got_size != size:
            out.append(f"{name}: size {got_size} expected {size}")
    if version != FASTEMBED_VERSION:
        out.append(f"fastembed {version}, pinned {FASTEMBED_VERSION}")
    return out


EMBEDDER_PIN: str = pin_id(EMBED_SNAPSHOT, EMBED_FILES, FASTEMBED_VERSION)


class EmbedderPinError(RuntimeError):
    """The loaded bge model or fastembed isn't the pinned one (FastEmbedder(verify=True): the gate refuses to run)."""


NEGATIONS = frozenset({"no", "not", "never", "nor"})  # content: "is vegetarian" and "is not vegetarian" differ
STOPWORDS = frozenset(
    "a an the and or but if so of to in on at by for with from as is are was were be been am do does did "
    "has have had it its this that these those i me my we our they them their he him his she her "
    "will would can could should about into than then there here what which who "
    "user s riya you".split()
) - NEGATIONS  # never any "n't" word either; tokenize() splits "isn't" into "isn" + "t", both content


def tokenize(text: str) -> list[str]:
    """Unicode word tokens on casefolded NFC text; combining marks (Indic vowel signs) stay inside words."""
    out, cur = [], []
    for c in unicodedata.normalize("NFC", text.casefold()):
        if c.isalnum() or c == "_" or unicodedata.category(c).startswith("M"):
            cur.append(c)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


class Embedder(Protocol):
    """strong_sim: similarity that alone makes a memory relevant; weak_sim: needs a shared content word as well."""
    dim: int
    strong_sim: float
    weak_sim: float

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    strong_sim = weak_sim = 0.2

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def identity(self) -> str:
        return f"hash-{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = [t for t in tokenize(text) if t not in STOPWORDS]
            for g in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]:
                out[i, zlib.crc32(g.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


def model_dir() -> Path:
    """Where Riya keeps downloaded models: RIYA_MODEL_DIR, else %LOCALAPPDATA%/Riya/models (not the temp folder)."""
    if override := os.environ.get("RIYA_MODEL_DIR"):
        return Path(override)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya" / "models"


def _load_text_embedding(model: str, cache_dir: str, local_files_only: bool = False):
    """The only place fastembed is loaded (tests replace it: no test downloads or loads a model)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model, cache_dir=cache_dir, local_files_only=local_files_only)


class FastEmbedder:
    # bge-small cosines are high for anything on the same topic: unrelated text scores 0.5-0.7, real hits 0.7+.
    # personal_sim: the looser floor for a request about the user herself (spec 4.1; measured on tests/data/personal_set.json)
    strong_sim, weak_sim, personal_sim = 0.80, 0.60, 0.45

    def __init__(self, model: str = EMBED_MODEL, cache_dir: str | Path | None = None, *, verify: bool = False,
                 local_files_only: bool = False):
        self._model = _load_text_embedding(model, str(cache_dir or model_dir()), local_files_only)
        snap = getattr(getattr(self._model, "model", None), "_model_dir", None)  # private to fastembed 0.9.0, pinned
        self.snapshot_dir: Path | None = Path(snap) if snap else None
        self.embedder_id = self.identity()
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))
        if self.embedder_id != EMBEDDER_PIN:
            problems = self.problems()
            if verify:
                raise EmbedderPinError("; ".join(problems))
            log.warning("the bge model doesn't match the pin (%s); memory still works", "; ".join(problems))

    def identity(self) -> str:
        if self.snapshot_dir is None:
            return "unverified:no-snapshot"
        return pin_id(self.snapshot_dir.name, snapshot_files(self.snapshot_dir, EMBED_FILES), installed_fastembed())

    def problems(self) -> list[str]:
        if self.snapshot_dir is None:
            return ["no snapshot folder to check: this fastembed doesn't say where it loaded the model from"]
        return pin_problems(self.snapshot_dir.name, snapshot_files(self.snapshot_dir, EMBED_FILES),
                            installed_fastembed())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
