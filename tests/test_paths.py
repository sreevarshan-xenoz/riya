"""Riya's data folder per platform."""

from riya import paths


def test_localappdata_wins_wherever_it_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.data_dir() == tmp_path / "Riya"


def test_windows_without_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)
    assert paths.data_dir() == tmp_path / "AppData" / "Local" / "Riya"


def test_linux_uses_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert paths.data_dir() == tmp_path / "data" / "riya"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)
    assert paths.data_dir() == tmp_path / ".local" / "share" / "riya"
