"""The recorded bge vectors (v0.11): every text the dev set, the probe bank and the retrieval set embed, recorded once
from the pinned model, so the default suite measures bge-sized thresholds without loading it."""
import numpy as np
import pytest

from evaluation import recorded_vectors as rv
from riya.core import embed
from riya.core.embed import FastEmbedder
from tests.bge_cache import live_bge


def test_the_recording_covers_the_corpus():
    with np.load(rv.VECTORS) as z:
        texts, vectors, embedder_id = z["texts"], z["vectors"], z["embedder_id"]
    corpus = rv.corpus_texts()
    assert corpus == sorted(set(corpus)) and len(corpus) > 100
    assert [str(t) for t in texts] == corpus
    assert vectors.dtype == np.float32 and vectors.shape == (len(corpus), 384)
    assert np.abs(np.linalg.norm(vectors, axis=1) - 1).max() <= 1e-5
    assert str(embedder_id) == embed.EMBEDDER_PIN


def test_recorded_embedder_follows_fastembedder():
    rec = rv.RecordedEmbedder()
    assert (rec.strong_sim, rec.weak_sim, rec.personal_sim) == (
        FastEmbedder.strong_sim, FastEmbedder.weak_sim, FastEmbedder.personal_sim)
    assert rv.RecordedEmbedder(personal_sim=0.4).personal_sim == 0.4
    assert rec.dim == 384 and rec.identity() == rec.embedder_id == embed.EMBEDDER_PIN
    text = rv.corpus_texts()[0]
    out = rec.embed([text])
    assert out.dtype == np.float32 and out.shape == (1, 384)
    assert rec.embed([]).shape == (0, 384)
    with pytest.raises(rv.UnknownText, match="a sentence nobody recorded"):
        rec.embed([text, "a sentence nobody recorded", "another one"])
    assert issubclass(rv.UnknownText, KeyError)


def test_recorded_vectors_match_live_bge(tmp_path, monkeypatch):
    live = live_bge(tmp_path, monkeypatch)
    rec = rv.RecordedEmbedder()
    assert live.identity() == rec.embedder_id
    texts = rv.corpus_texts()
    worst = 0.0
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        worst = max(worst, float(np.abs(live.embed(batch) - rec.embed(batch)).max()))
    print(f"max abs difference {worst:.3e} over {len(texts)} texts")  # shown with -s
    assert worst <= 1e-4, worst
