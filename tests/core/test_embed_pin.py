"""The bge pin (spec 4.7): FastEmbedder names the model it loaded, `verify` refuses a different one, live mode only
warns, and no test can load fastembed."""
import logging
import re
from pathlib import Path

import pytest

from riya.core import embed
from riya.core.embed import EmbedderPinError, FastEmbedder, HashEmbedder
from tests.embed_stubs import fake_loader, fake_snapshot, pin_to

REPO = Path(__file__).resolve().parents[2]
ONNX_SHA = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
NAMES = {"config.json", "model_optimized.onnx", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}


def test_the_pin_is_the_spec_s():
    assert embed.EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert embed.EMBED_REPO == "Qdrant/bge-small-en-v1.5-onnx-Q"
    assert embed.EMBED_SNAPSHOT == "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
    assert embed.FASTEMBED_VERSION == "0.9.0"
    assert set(embed.EMBED_FILES) == NAMES
    assert embed.EMBED_FILES["model_optimized.onnx"] == (66465124, ONNX_SHA)
    assert {n: s for n, (s, _) in embed.EMBED_FILES.items()} == {
        "config.json": 706, "model_optimized.onnx": 66465124, "special_tokens_map.json": 695,
        "tokenizer.json": 711396, "tokenizer_config.json": 1242}
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for _, sha in embed.EMBED_FILES.values())
    assert embed.EMBEDDER_PIN == embed.pin_id(embed.EMBED_SNAPSHOT, embed.EMBED_FILES, embed.FASTEMBED_VERSION)
    assert re.fullmatch(r"[0-9a-f]{16}", embed.EMBEDDER_PIN[:16])
    assert embed.EMBEDDER_PIN.endswith("bge-small-en-v1.5-onnx-Q@aa8f8b06+fastembed-0.9.0")
    lines = (REPO / "requirements-riya.txt").read_text(encoding="utf-8").splitlines()
    assert "fastembed==0.9.0" in [ln.strip() for ln in lines]


def test_no_test_can_load_fastembed():
    with pytest.raises(RuntimeError, match="tests never load"):
        FastEmbedder()


def build(tmp_path, monkeypatch, **kw):
    snap = fake_snapshot(tmp_path / "bge")
    pin_to(monkeypatch, snap)
    seen = {}
    monkeypatch.setattr(embed, "_load_text_embedding", fake_loader(snap, seen))
    return snap, seen


def test_verify_passes_a_matching_snapshot_and_raises_on_a_tampered_one(tmp_path, monkeypatch):
    snap, seen = build(tmp_path, monkeypatch)
    assert FastEmbedder(verify=True).embedder_id == embed.EMBEDDER_PIN

    tok = snap / "tokenizer.json"
    original = tok.read_bytes()
    tok.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(EmbedderPinError, match="tokenizer.json"):
        FastEmbedder(verify=True)
    tok.write_bytes(original)
    assert FastEmbedder(verify=True).embedder_id == embed.EMBEDDER_PIN

    (snap / "config.json").unlink()
    with pytest.raises(EmbedderPinError, match="missing"):
        FastEmbedder(verify=True)

    other = fake_snapshot(tmp_path / "elsewhere", rev="b" * 40)  # the same bytes under another revision's name
    monkeypatch.setattr(embed, "_load_text_embedding", fake_loader(other, seen))
    with pytest.raises(EmbedderPinError, match="snapshot"):
        FastEmbedder(verify=True)

    monkeypatch.setattr(embed, "_load_text_embedding", fake_loader(snap, seen))
    (snap / "config.json").write_bytes(b"fake config.json")
    monkeypatch.setattr(embed, "installed_fastembed", lambda: "0.9.1")
    with pytest.raises(EmbedderPinError, match="fastembed"):
        FastEmbedder(verify=True)


def test_the_loader_is_asked_for_the_pinned_model_in_the_riya_folder(tmp_path, monkeypatch):
    snap, seen = build(tmp_path, monkeypatch)
    FastEmbedder(cache_dir=tmp_path / "cache", local_files_only=True)
    assert seen == {"model": embed.EMBED_MODEL, "cache_dir": str(tmp_path / "cache"), "local_files_only": True}


def test_live_mode_only_warns_on_a_mismatch(tmp_path, monkeypatch, caplog):
    snap, _ = build(tmp_path, monkeypatch)
    tok = snap / "tokenizer.json"
    tok.write_bytes(tok.read_bytes() + b"!")
    with caplog.at_level(logging.WARNING, logger="riya.embed"):
        emb = FastEmbedder()
    assert emb.embed(["hi"]).shape == (1, 4)
    assert emb.embedder_id != embed.EMBEDDER_PIN
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "riya.embed"]
    assert len(warnings) == 1
    assert "tokenizer.json" in warnings[0].getMessage() and "memory still works" in warnings[0].getMessage()


def test_a_matching_snapshot_logs_nothing(tmp_path, monkeypatch, caplog):
    build(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="riya.embed"):
        FastEmbedder()
    assert not caplog.records


def test_a_model_without_a_snapshot_is_unverified(monkeypatch):
    class Bare:
        def embed(self, texts):
            return iter([[1.0, 0.0]] * len(texts))
    monkeypatch.setattr(embed, "_load_text_embedding", lambda *a, **kw: Bare())
    assert FastEmbedder().embedder_id == "unverified:no-snapshot"
    with pytest.raises(EmbedderPinError):
        FastEmbedder(verify=True)


def test_identity_names_its_parts():
    snap, files, ver = embed.EMBED_SNAPSHOT, dict(embed.EMBED_FILES), embed.FASTEMBED_VERSION
    base = embed.pin_id(snap, files, ver)
    assert embed.pin_id(snap, files, ver) == base
    assert embed.pin_id("c" * 40, files, ver) != base
    assert embed.pin_id(snap, files, "0.9.1") != base
    for name, (size, sha) in files.items():
        assert embed.pin_id(snap, {**files, name: (size + 1, sha)}, ver) != base
        assert embed.pin_id(snap, {**files, name: (size, "0" * 64)}, ver) != base
    assert HashEmbedder().identity() == "hash-2048"
    assert HashEmbedder(dim=64).identity() == "hash-64"


def test_snapshot_files_streams_and_marks_a_missing_file(tmp_path):
    snap = fake_snapshot(tmp_path)
    (snap / "tokenizer.json").unlink()
    files = embed.snapshot_files(snap, sorted(NAMES))
    assert files["tokenizer.json"] == (-1, "missing")
    assert files["config.json"][0] == len(b"fake config.json")


def test_personal_sim_is_fixed():
    assert FastEmbedder.personal_sim == 0.45
    assert not hasattr(HashEmbedder, "personal_sim") and not hasattr(HashEmbedder(), "personal_sim")
