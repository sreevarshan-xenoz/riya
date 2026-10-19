import pytest


@pytest.fixture(autouse=True)
def no_real_user_files(tmp_path, monkeypatch):
    """No test reaches the real %LOCALAPPDATA%\\Riya, ~/.claude/settings.json or Startup folder: every default path is
    redirected to tmp_path, and no remembered key carries from one test into the next."""
    from riya.agents import hooks, install_hooks
    from riya.core import secrets as keystore
    riya_dir = tmp_path / "localappdata" / "Riya"
    monkeypatch.setattr(keystore, "default_path", lambda: riya_dir / "secrets.env")
    monkeypatch.setattr(install_hooks, "default_settings_path", lambda: tmp_path / "home" / ".claude" / "settings.json")
    monkeypatch.setattr(install_hooks, "default_token_path", lambda: riya_dir / "hook_token")
    monkeypatch.setattr(hooks, "default_token_path", lambda: riya_dir / "hook_token")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    keystore.forget_all()
    yield
    keystore.forget_all()
