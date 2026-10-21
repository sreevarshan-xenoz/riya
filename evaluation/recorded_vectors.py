"""The recorded bge vectors: every text the personal-request dev set, the persona probe bank and the retrieval set
embed, recorded once from the pinned model, so the default test suite (and retrieval_eval) measures bge-sized
thresholds without loading or downloading it.

  python -m evaluation.recorded_vectors --cache DIR [--out PATH]    (DIR: a copy of the fastembed cache; CPU, offline)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from riya.core.embed import FastEmbedder
from riya.train.holdout import probe_facts, probe_texts

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests" / "data" / "bge_vectors.npz"
DEV = ROOT / "tests" / "data" / "personal_set.json"
RETRIEVAL_SET = ROOT / "tests" / "data" / "retrieval_set.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_texts() -> list[str]:
    """Every text to record, sorted and unique: the dev set's facts and requests, the probe bank's facts and
    non-speak-first turns, and the retrieval set's facts, episode summaries and questions."""
    dev, ret = _read(DEV), _read(RETRIEVAL_SET)
    texts = [f["text"] for f in dev["facts"]] + [r["q"] for r in dev["requests"]] + [n["q"] for n in dev["negatives"]]
    texts += probe_facts() + probe_texts()
    texts += [f["text"] for f in ret["facts"]] + [e["summary"] for e in ret["episodes"]] + [q["q"] for q in ret["questions"]]
    return sorted(set(texts))


class UnknownText(KeyError):
    """A text that was never recorded: the corpus changed, so rebuild the file."""

    def __str__(self) -> str:
        return f"no recorded vector for {self.args[0]!r}: rebuild tests/data/bge_vectors.npz " \
               "(python -m evaluation.recorded_vectors --cache DIR)"


class RecordedEmbedder:
    """An Embedder that answers from the recording, with FastEmbedder's thresholds."""
    strong_sim, weak_sim, personal_sim = FastEmbedder.strong_sim, FastEmbedder.weak_sim, FastEmbedder.personal_sim

    def __init__(self, path: Path = VECTORS, *, personal_sim: float | None = None):
        with np.load(path) as z:
            texts, self._vectors = [str(t) for t in z["texts"]], z["vectors"].astype(np.float32)
            self.embedder_id = str(z["embedder_id"])
        self._row = {t: i for i, t in enumerate(texts)}
        self.dim = int(self._vectors.shape[1])
        if personal_sim is not None:
            self.personal_sim = personal_sim

    def identity(self) -> str:
        return self.embedder_id

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t not in self._row:
                raise UnknownText(t)
            rows.append(self._row[t])
        return self._vectors[rows].reshape(len(rows), self.dim).copy()


def build(cache_dir: Path, out: Path = VECTORS) -> int:
    """Embed corpus_texts() with the verified, pinned model loaded offline from cache_dir; write out. Returns the count."""
    embedder = FastEmbedder(cache_dir=cache_dir, verify=True, local_files_only=True)
    texts = corpus_texts()
    vectors = np.concatenate([embedder.embed(texts[i:i + 64]) for i in range(0, len(texts), 64)])
    np.savez_compressed(out, texts=np.array(texts), vectors=vectors.astype(np.float32),
                        embedder_id=np.array(embedder.embedder_id))
    return len(texts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.recorded_vectors", description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="a COPY of the fastembed cache (the original stays read-only)")
    ap.add_argument("--out", type=Path, default=VECTORS)
    args = ap.parse_args(argv)
    n = build(args.cache, args.out)
    with np.load(args.out) as z:  # closed at once: Windows can't rewrite a file numpy still holds
        print(f"{n} texts, embedder {z['embedder_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
