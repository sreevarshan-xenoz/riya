"""Stand-ins for fastembed, so no test loads or downloads a model: a tiny snapshot folder, a fake TextEmbedding that
reports it, a loader that records what it was asked for, and an embedder whose similarities a test writes down."""
import hashlib
from pathlib import Path

import numpy as np

from riya.core import embed
from riya.core.embed import EMBED_FILES, EMBED_SNAPSHOT


def fake_snapshot(root: Path, rev: str = EMBED_SNAPSHOT) -> Path:
    """root/rev holding the five files the pin names, each a few deterministic bytes (the same bytes for any rev)."""
    snap = Path(root) / rev
    snap.mkdir(parents=True, exist_ok=True)
    for name in EMBED_FILES:
        (snap / name).write_bytes(f"fake {name}".encode())
    return snap


class _Inner:
    def __init__(self, snapshot_dir):
        self._model_dir = snapshot_dir


class FakeTextEmbedding:
    """What fastembed's TextEmbedding gives FastEmbedder: .model._model_dir and a deterministic .embed(texts)."""

    def __init__(self, snapshot_dir, dim: int = 4):
        self.model = _Inner(snapshot_dir)
        self.dim = dim

    def embed(self, texts):
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            yield np.random.default_rng(seed).normal(size=self.dim) + 0.5


def fake_loader(snapshot_dir, seen: dict):
    """A stand-in for embed._load_text_embedding: records (model, cache_dir, local_files_only) in seen."""
    def load(model, cache_dir, local_files_only=False):
        seen["model"], seen["cache_dir"], seen["local_files_only"] = model, cache_dir, local_files_only
        return FakeTextEmbedding(snapshot_dir)
    return load


def pin_to(monkeypatch, snapshot_dir):
    """Make the module's pin the given fake snapshot: its name, its five files' size and sha256, the pinned fastembed
    version as the installed one. FastEmbedder(verify=True) on that snapshot then passes."""
    files = embed.snapshot_files(snapshot_dir, EMBED_FILES)
    monkeypatch.setattr(embed, "EMBED_SNAPSHOT", Path(snapshot_dir).name)
    monkeypatch.setattr(embed, "EMBED_FILES", files)
    monkeypatch.setattr(embed, "installed_fastembed", lambda: embed.FASTEMBED_VERSION)
    monkeypatch.setattr(embed, "EMBEDDER_PIN",
                        embed.pin_id(Path(snapshot_dir).name, files, embed.FASTEMBED_VERSION))


class TableEmbedder:
    """Every text outside `sims` is a query at e0, and fact i (the i-th text of `sims`) sits at cosine sims[text]
    from e0, in its own direction. A test states the similarities it wants and gets exactly those."""

    def __init__(self, sims: dict[str, float], *, strong: float = 0.8, weak: float = 0.6,
                 personal: float | None = 0.45):
        self.sims = dict(sims)
        self.index = {text: i + 1 for i, text in enumerate(self.sims)}
        self.dim = len(self.sims) + 1
        self.strong_sim, self.weak_sim = strong, weak
        if personal is not None:
            self.personal_sim = personal

    def identity(self) -> str:
        return "table"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            if text in self.sims:
                s = self.sims[text]
                out[row, 0] = s
                out[row, self.index[text]] = np.sqrt(max(0.0, 1.0 - s * s))
            else:
                out[row, 0] = 1.0
        return out
