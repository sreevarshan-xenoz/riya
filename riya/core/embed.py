"""Text embedders: a dependency-free hashing embedder and a lazy fastembed wrapper."""
import os
import unicodedata
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

STOPWORDS = frozenset(
    "a an the and or but if so of to in on at by for with from as is are was were be been am do does did "
    "has have had it its this that these those i me my we our they them their he him his she her "
    "will would can could should about into than then there here what which who not no "
    "user s riya you".split()
)


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


class FastEmbedder:
    # bge-small cosines are high for anything on the same topic: unrelated text scores 0.5-0.7, real hits 0.7+
    strong_sim, weak_sim = 0.80, 0.60

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | Path | None = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model, cache_dir=str(cache_dir or model_dir()))
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
