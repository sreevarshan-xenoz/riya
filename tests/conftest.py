import pytest

from tests import real_paths  # first: it records the real paths before any fixture redirects them


@pytest.fixture(scope="session", autouse=True)
def real_models_untouched():
    """The session fails if anything under the user's real models folder (%USERPROFILE%\\models) was added, removed
    or changed while it ran (C1: a fake adapter and its champion record once landed there). Read-only: it lists the
    folder at the start and at the end, and never writes. No folder: nothing to guard."""
    before = real_paths.listing(real_paths.REAL_MODELS)
    yield
    if found := real_paths.changes(before, real_paths.listing(real_paths.REAL_MODELS)):
        pytest.fail(f"The test session changed the real models folder {real_paths.REAL_MODELS}: " + "; ".join(found),
                    pytrace=False)


@pytest.fixture(autouse=True)
def no_real_user_files(tmp_path, monkeypatch):
    """No test reaches the real %LOCALAPPDATA%\\Riya, ~/.claude/settings.json, Startup folder or models folder: every
    default path is redirected to tmp_path (riya.brain's base model, adapter folder and so its adopted and candidate
    adapters), and no remembered key carries from one test into the next."""
    from riya import brain
    from riya.agents import hooks, install_hooks
    from riya.core import secrets as keystore
    riya_dir = tmp_path / "localappdata" / "Riya"
    monkeypatch.setattr(keystore, "default_path", lambda: riya_dir / "secrets.env")
    monkeypatch.setattr(install_hooks, "default_settings_path", lambda: tmp_path / "home" / ".claude" / "settings.json")
    monkeypatch.setattr(install_hooks, "default_token_path", lambda: riya_dir / "hook_token")
    monkeypatch.setattr(hooks, "default_token_path", lambda: riya_dir / "hook_token")
    real_paths.redirect_models(tmp_path, monkeypatch)
    from riya import server
    monkeypatch.setattr(server, "LLAMA_SERVER", str(tmp_path / "no-llama-server.exe"))  # nothing runs her real binary
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))  # code that reads these directly (not through
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # the patched defaults) can't reach the real folders
    from riya.core import embed

    def no_fastembed(model, cache_dir, local_files_only=False):
        raise RuntimeError(f"tests never load fastembed (model {model}, cache {cache_dir}): "
                           "use tests/data/bge_vectors.npz")
    monkeypatch.setattr(embed, "_load_text_embedding", no_fastembed)  # the opt-in live_bge tests use REAL_LOAD_TEXT_EMBEDDING
    monkeypatch.delenv("RIYA_MODEL_DIR", raising=False)
    keystore.forget_all()
    yield
    keystore.forget_all()
