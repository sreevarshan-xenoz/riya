"""persona_eval measures the embedder live Riya uses (v0.11 §4.5): one per invocation, its id on the run. No test loads
fastembed: the loader is a fake, or the conftest guard refuses it."""
import json

import pytest

from evaluation import persona_eval
from riya.core import embed
from riya.core.embed import HashEmbedder
from tests.embed_stubs import fake_loader, fake_snapshot, pin_to
from tests.fakes import FakeModel


def _fast_loads(monkeypatch, tmp_path, calls=None):
    """Make the bge pin a fake snapshot and the loader a counting fake; returns what the loader saw."""
    snap = fake_snapshot(tmp_path / "snap")
    pin_to(monkeypatch, snap)
    seen: dict = {}
    inner = fake_loader(snap, seen)

    def load(model, cache_dir, local_files_only=False):
        seen["calls"] = seen.get("calls", 0) + 1
        return inner(model, cache_dir, local_files_only)

    monkeypatch.setattr(embed, "_load_text_embedding", load)
    return seen


def test_auto_uses_fast_when_it_loads(monkeypatch, tmp_path):
    seen = _fast_loads(monkeypatch, tmp_path)
    emb, embedder_id = persona_eval.make_embedder("auto")
    assert embedder_id == embed.EMBEDDER_PIN and emb.embedder_id == embed.EMBEDDER_PIN
    assert seen["local_files_only"] is False


def test_auto_falls_back_to_hash_with_a_note(capsys):
    emb, embedder_id = persona_eval.make_embedder("auto")  # the conftest guard refuses the real loader
    assert embedder_id == "hash-2048" and isinstance(emb, HashEmbedder)
    assert "Note: the bge embedder didn't load" in capsys.readouterr().out


def test_fast_never_falls_back():
    with pytest.raises(Exception):
        persona_eval.make_embedder("fast")


def _unpinned_bge(monkeypatch, tmp_path):
    """A loadable bge stand-in whose snapshot does NOT match the pin (no pin_to): verify=True must refuse it."""
    snap = fake_snapshot(tmp_path / "snap", rev="not-the-pinned-revision")
    monkeypatch.setattr(embed, "_load_text_embedding", fake_loader(snap, {}))


def test_fast_verifies_the_pin(monkeypatch, tmp_path):
    _unpinned_bge(monkeypatch, tmp_path)
    with pytest.raises(embed.EmbedderPinError):
        persona_eval.make_embedder("fast")


def test_auto_falls_back_to_hash_when_the_model_loads_but_fails_the_pin(monkeypatch, tmp_path, capsys):
    _unpinned_bge(monkeypatch, tmp_path)
    emb, embedder_id = persona_eval.make_embedder("auto")
    assert embedder_id == "hash-2048" and isinstance(emb, HashEmbedder)
    out = capsys.readouterr().out
    assert "Note: the bge embedder didn't load (EmbedderPinError" in out


def test_hash_is_hash():
    emb, embedder_id = persona_eval.make_embedder("hash")
    assert embedder_id == "hash-2048" and isinstance(emb, HashEmbedder)


def test_embedders_are_the_three_choices():
    assert persona_eval.EMBEDDERS == ("auto", "fast", "hash")


def _two_conversations():
    return {"conversations": [
        {"id": f"c{i}", "facts": ["i like tea"], "turns": [{"id": "a", "kind": "small", "text": "hey"}]}
        for i in (1, 2)]}


def test_one_embedder_per_run(monkeypatch, tmp_path):
    seen = _fast_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(persona_eval, "local_model", lambda url, **kw: FakeModel("local", ["Hmph, hi~ I'm an AI."]))
    monkeypatch.setattr(persona_eval, "load_probes", lambda *a, **kw: _two_conversations())
    cores = []
    real_core = persona_eval.RiyaCore

    def spy(*a, **kw):
        cores.append(kw["embedder"])
        return real_core(*a, **kw)

    monkeypatch.setattr(persona_eval, "RiyaCore", spy)

    transcript, score, problem = persona_eval.run("u", repeat=2)
    assert problem is None and len(transcript) == 4 and len(cores) == 4
    assert seen["calls"] == 1 and score["embedder_id"] == embed.EMBEDDER_PIN
    assert all(c is cores[0] for c in cores)

    cores.clear()
    mine = HashEmbedder()
    _, score, problem = persona_eval.run("u", repeat=2, embedder=(mine, "explicit-id"))
    assert problem is None and seen["calls"] == 1  # no further load
    assert len(cores) == 4 and all(c is mine for c in cores) and score["embedder_id"] == "explicit-id"


def test_an_unreachable_server_builds_no_embedder_in_run(monkeypatch):
    from riya.core.models import ModelUnavailable

    class Down:
        def stream(self, *a, **kw):
            raise ModelUnavailable("down")

    monkeypatch.setattr(persona_eval, "local_model", lambda url, **kw: Down())
    monkeypatch.setattr(persona_eval, "make_embedder", lambda *a: pytest.fail("built before the reachability check"))
    assert persona_eval.run("u")[2] == "down"


def test_main_builds_once_and_records_the_id(monkeypatch, tmp_path, capsys):
    pair = (HashEmbedder(), "pair-id")
    built = []
    monkeypatch.setattr(persona_eval, "make_embedder", lambda choice="auto": built.append(choice) or pair)
    got = []

    def fake_run(url, online, repeat, dials=None, *, embedder):
        got.append(embedder)
        return [{"probe_id": "c.a", "text": "hey", "reply": "Hmph.", "first_sentence_s": 0.1, "error": None,
                 "kind": "small"}], _score(), None

    monkeypatch.setattr(persona_eval, "run", fake_run)
    monkeypatch.setattr(persona_eval, "OUT_DIR", tmp_path / "out")
    assert persona_eval.main(["--dials", "--label", "e", "--embedder", "hash"]) == 0
    assert built == ["hash"] and len(got) == 3 and all(g is pair for g in got)
    assert "Embedder: pair-id" in capsys.readouterr().out
    (f,) = (tmp_path / "out").glob("*_e.json")
    assert json.loads(f.read_text(encoding="utf-8"))["embedder_id"] == "pair-id"


def test_main_defaults_to_auto_and_records_the_id_without_dials(monkeypatch, tmp_path):
    pair = (HashEmbedder(), "p")
    built = []
    monkeypatch.setattr(persona_eval, "make_embedder", lambda choice="auto": built.append(choice) or pair)
    monkeypatch.setattr(persona_eval, "run",
                        lambda url, online, repeat, dials=None, *, embedder: ([], _score(), None))
    monkeypatch.setattr(persona_eval, "OUT_DIR", tmp_path / "out")
    assert persona_eval.main(["--label", "p"]) == 0
    assert built == ["auto"]
    (f,) = (tmp_path / "out").glob("*_p.json")
    assert json.loads(f.read_text(encoding="utf-8"))["embedder_id"] == "p"


def _score():
    from tests.core.test_style_metrics import _canned
    return _canned()
