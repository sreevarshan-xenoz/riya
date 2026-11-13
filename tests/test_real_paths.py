"""v0.10 final fix C1: no test reaches the user's real models folder. tests/conftest.py redirects every default path in
riya.brain into tmp_path for every test, and a session guard fails the run if anything under the real models folder
changed. These tests only stat the real folder; they never write to it."""
from pathlib import Path

import pytest

from evaluation import champion
from riya import brain
from tests import gate_helpers, real_paths
from tests.real_paths import changes, listing


def test_every_brain_default_is_redirected_into_tmp(tmp_path):
    for p in (Path(brain.DEFAULT_MODEL), brain.ADAPTER_DIR, brain.default_adapter(), brain.candidate_adapter(),
              brain.default_db()):
        assert tmp_path in p.parents, p
    assert brain.default_adapter() == tmp_path / "models" / "riya" / brain.ADAPTER_NAME
    assert Path(brain.DEFAULT_MODEL).name == Path(real_paths.REAL_DEFAULT_MODEL).name  # the same file name, in tmp
    defaults = champion.parser().parse_args([])
    for p in (defaults.model, defaults.lora, defaults.db):
        assert tmp_path in Path(p).parents, p


def test_the_real_paths_are_the_code_s_own():
    assert real_paths.REAL_ADAPTER_DIR.is_absolute() and real_paths.REAL_ADAPTER_DIR.parts[-2:] == ("models", "riya")
    assert real_paths.REAL_MODELS == real_paths.REAL_ADAPTER_DIR.parent
    assert real_paths.REAL_DEFAULT_MODEL.startswith(real_paths.REAL_ADAPTER_DIR.as_posix() + "/")


def test_listing_and_changes(tmp_path):
    root = tmp_path / "models"
    assert listing(root) is None and changes(None, None) == []
    (root / "riya").mkdir(parents=True)
    (root / "riya" / "base.gguf").write_bytes(b"base")
    before = listing(root)
    assert set(before) == {"riya", "riya/base.gguf"} and before["riya/base.gguf"][:2] == ("file", 4)
    assert changes(before, listing(root)) == []
    assert changes(None, before) == ["the folder appeared"] and changes(before, None) == ["the folder disappeared"]

    (root / "riya" / "riya-v1-lora.gguf").write_bytes(b"GGUF fake adapter")  # what C1 found in the real folder
    (root / "riya" / "base.gguf").write_bytes(b"base, changed")
    found = changes(before, listing(root))
    assert "added riya/riya-v1-lora.gguf" in found
    assert any(line.startswith("changed riya/base.gguf") for line in found)
    (root / "riya" / "base.gguf").unlink()
    assert "removed riya/base.gguf" in changes(before, listing(root))


def test_redirect_models_moves_every_default_even_without_the_conftest(tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(brain, "ADAPTER_DIR", real_paths.REAL_ADAPTER_DIR)  # as a module imported without the conftest
        mp.setattr(brain, "DEFAULT_MODEL", real_paths.REAL_DEFAULT_MODEL)
        assert real_paths.redirect_models(tmp_path / "x", mp) == tmp_path / "x" / "models" / "riya"
        for p in (Path(brain.DEFAULT_MODEL), brain.default_adapter(), brain.candidate_adapter()):
            assert tmp_path / "x" in p.parents, p


def test_the_gate_fixture_redirects_the_models_itself(tmp_path, monkeypatch):
    """C1: tests/gate_helpers.calibrated (autouse in the gate modules) moves the adapter folder too, not only the base."""
    monkeypatch.setattr(brain, "ADAPTER_DIR", real_paths.REAL_ADAPTER_DIR)
    base = gate_helpers.calibrated.__wrapped__(tmp_path, monkeypatch)
    assert brain.DEFAULT_MODEL == str(base) and tmp_path in base.parents
    assert tmp_path in brain.default_adapter().parents and tmp_path in brain.candidate_adapter().parents


def test_tmp_only_refuses_a_path_outside_tmp(tmp_path):
    assert real_paths.tmp_only(tmp_path, tmp_path / "models" / "a.gguf") == tmp_path / "models" / "a.gguf"
    with pytest.raises(AssertionError, match="outside tmp_path"):
        real_paths.tmp_only(tmp_path, real_paths.REAL_ADAPTER_DIR / brain.ADAPTER_NAME)
