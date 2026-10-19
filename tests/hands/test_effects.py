"""Effects: key inputs, FakeEffects runs, run_command, WindowsEffects (monkeypatched, never real), output_tail, write_file.

Nothing here opens an app or URL, sends a key, touches the clipboard or runs a command: WindowsEffects methods are only
called with os.startfile, SendInput and subprocess.run replaced.
"""
import subprocess

import pytest

from riya.core.hands import effects as fx
from riya.core.hands import tools
from riya.core.hands.effects import (INPUT, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, VOLUME_PRESSES, FakeEffects,
                                     WindowsEffects, key_inputs, output_tail)
from riya.core.hands.tools import REGISTRY, Call, ToolCtx, validate_call
from riya.core.settings import Settings

ARGV = ["robocopy", "C:/a", "D:/b"]
CMD = {"name": "backup", "argv": ARGV, "cwd": "C:/a", "timeout_s": 30}


@pytest.fixture
def s():
    return Settings(apps=[{"name": "Notepad", "target": "notepad.exe"}], commands=[dict(CMD)])


def run(s, tool, effects=None, riya_dir=None, **args):
    call = validate_call(Call(tool, args), s)
    assert call is not None
    ctx = ToolCtx(settings=s, effects=effects or FakeEffects(), riya_dir=riya_dir)
    return REGISTRY[tool].run(call.args, ctx)


def test_key_inputs_are_extended_down_up_pairs():
    ins = key_inputs(0xAF, 5)
    assert len(ins) == 10 and all(isinstance(i, INPUT) and i.type == 1 and i.ki.wVk == 0xAF for i in ins)
    assert all(i.ki.dwFlags & KEYEVENTF_EXTENDEDKEY for i in ins)
    assert [bool(i.ki.dwFlags & KEYEVENTF_KEYUP) for i in ins] == [False, True] * 5
    assert len(key_inputs(0xAD, 1)) == 2


def test_volume_and_media_presses(s):
    e = FakeEffects()
    assert run(s, "volume", e, action="up").ok and run(s, "volume", e, action="down").ok
    assert run(s, "volume", e, action="mute").ok and run(s, "media", e, action="next").ok
    assert e.calls == [("media_key", "volume_up", 5), ("media_key", "volume_down", 5), ("media_key", "mute", 1),
                       ("media_key", "next", 1)] and VOLUME_PRESSES == 5


def test_open_url_and_app_record_only_validated_targets(s):
    e = FakeEffects()
    assert run(s, "open_url", e, url="example.com/a").ok
    assert run(s, "open_app", e, name="notepad").ok  # the target comes from settings.apps
    assert e.calls == [("open_url", "https://example.com/a"), ("open_app", "notepad.exe")]
    assert validate_call(Call("open_url", {"url": "file:///c:/x"}), s) is None


def test_clipboard_copy_and_failed_effects(s):
    e = FakeEffects()
    r = run(s, "clipboard_copy", e, text="hello")
    assert r.ok and e.calls == [("copy", "hello")]
    bad = FakeEffects(fail=OSError("nope"))
    for tool, args in (("clipboard_copy", {"text": "x"}), ("open_url", {"url": "a.com"}), ("volume", {"action": "up"}),
                       ("open_app", {"name": "Notepad"}), ("media", {"action": "next"})):
        assert run(s, tool, bad, **args).ok is False  # a failing effect is a kind result, not an exception
    assert FakeEffects(fail="boom").fail == "boom"


def test_run_command_ok_and_clamped_timeout(s):
    e = FakeEffects(run_result=(0, "done"))
    r = run(s, "run_command", e, name="backup")
    assert e.calls == [("run", ARGV, "C:/a", 30)] and r.ok and r.text == '"backup" finished: done'
    s.commands = [{**CMD, "timeout_s": 600}]
    e = FakeEffects()
    run(s, "run_command", e, name="backup")
    assert e.calls == [("run", ARGV, "C:/a", 60)]
    s.commands = [{**CMD, "timeout_s": 9999}]
    e = FakeEffects()
    run(s, "run_command", e, name="backup")
    assert e.calls[0][3] == 60


def test_run_command_failures(s):
    r = run(s, "run_command", FakeEffects(run_result=(2, "boom")), name="backup")
    assert not r.ok and r.text == '"backup" failed (exit 2): boom'
    r = run(s, "run_command", FakeEffects(run_result=(None, "timed out after 60 s")), name="backup")
    assert not r.ok and r.text == '"backup" timed out after 60 s'
    assert run(s, "run_command", FakeEffects(fail=OSError()), name="backup").ok is False
    assert run(s, "run_command", FakeEffects(run_result=(0, "")), name="backup").text == '"backup" finished (no output).'


def test_run_command_uses_the_snapshot_not_later_settings(s):
    call = validate_call(Call("run_command", {"name": "backup"}), s)
    s.commands[0]["argv"] = ["evil.exe"]
    e = FakeEffects()
    REGISTRY["run_command"].run(call.args, ToolCtx(settings=s, effects=e))
    assert e.calls == [("run", ARGV, "C:/a", 30)]


def test_run_command_tail_is_redacted_and_fits(s):
    r = run(s, "run_command", FakeEffects(run_result=(1, "noise " * 100 + "api_key=sk-abc123")), name="backup")
    assert len(r.text) <= 200 and r.text.endswith("api_key=***") and "sk-abc123" not in r.text


def test_output_tail():
    out = output_tail("x" * 500 + " api_key=sk-abc123")
    assert len(out) <= 200 and out.endswith("api_key=***") and "sk-abc123" not in out
    assert output_tail("a\r\n\tb\x00\x1b[31mc") == "a b[31mc" and output_tail("") == ""
    assert output_tail("ab" * 300, 10) == "ab" * 5 and len(output_tail("y" * 10**6)) == 200


def test_secret_split_by_the_cut_cannot_survive():
    secret = "sk-" + "A" * 150
    assert output_tail("x" * 400 + " " + secret).endswith("***")
    assert "A" * 20 not in output_tail("x" * 400 + " token=" + secret)


# ---- WindowsEffects, with every OS call replaced ----
def test_windows_run_uses_argv_without_a_shell(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen.update(argv=argv, **kw)
        return subprocess.CompletedProcess(argv, 3, stdout="héllo".encode("utf-8"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert WindowsEffects().run(ARGV, "C:/a", 60) == (3, "héllo")
    assert seen["argv"] == ARGV and seen["shell"] is False and seen["cwd"] == "C:/a" and seen["timeout"] == 60
    assert seen["creationflags"] == fx.CREATE_NO_WINDOW == 0x08000000 and seen["stdin"] == subprocess.DEVNULL


def test_windows_run_timeout_and_start_failure(monkeypatch):
    def timeout(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert WindowsEffects().run(ARGV, None, 60) == (None, "timed out after 60 s")

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", missing)
    assert WindowsEffects().run(ARGV, None, 5) == (None, "couldn't start")


def test_windows_open_and_media(monkeypatch):
    opened, sent = [], []
    monkeypatch.setattr(fx.os, "startfile", lambda t: opened.append(t), raising=False)
    monkeypatch.setattr(fx, "SendInput", lambda ins: sent.append(ins))
    w = WindowsEffects()
    w.open_url("https://example.com/a")
    w.open_app("notepad.exe")
    assert opened == ["https://example.com/a", "notepad.exe"]
    for bad in ("file:///c:/x", "javascript:alert(1)", "C:\\Windows\\system32\\calc.exe", "http://a b"):
        with pytest.raises(ValueError):
            w.open_url(bad)
    assert opened == ["https://example.com/a", "notepad.exe"]
    w.media_key("volume_up", 5)
    w.media_key("mute")
    assert len(sent) == 2 and [(i.ki.wVk, i.ki.dwFlags) for i in sent[0]] == [(i.ki.wVk, i.ki.dwFlags)
                                                                              for i in key_inputs(0xAF, 5)]
    assert len(sent[1]) == 2 and sent[1][0].ki.wVk == 0xAD
    for key, vk in (("play_pause", 0xB3), ("next", 0xB0), ("previous", 0xB1), ("volume_down", 0xAE)):
        w.media_key(key)
        assert sent[-1][0].ki.wVk == vk
    n = len(sent)
    for key, presses in (("explode", 1), ("mute", 0), ("mute", 500)):
        with pytest.raises(ValueError):
            w.media_key(key, presses)
    assert len(sent) == n


def test_input_struct_matches_the_win32_layout():
    import ctypes
    assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


# ---- write_file ----
def test_write_file_into_riya_dir_never_overwrites(s, tmp_path):
    r = run(s, "write_file", riya_dir=tmp_path, filename="groceries", text="milk")
    assert r.ok and (tmp_path / "groceries.txt").read_text(encoding="utf-8") == "milk"
    r = run(s, "write_file", riya_dir=tmp_path, filename="groceries.txt", text="eggs")
    assert r.ok and "groceries (2).txt" in r.text and r.data == {"path": str(tmp_path / "groceries (2).txt")}
    assert (tmp_path / "groceries.txt").read_text(encoding="utf-8") == "milk"
    assert (tmp_path / "groceries (2).txt").read_text(encoding="utf-8") == "eggs"


def test_write_file_creates_the_folder_and_keeps_text_exact(s, tmp_path):
    d = tmp_path / "riya"
    assert run(s, "write_file", riya_dir=d, filename="n.md", text="a\nb é").ok
    assert (d / "n.md").read_bytes() == "a\nb é".encode("utf-8")


def test_write_file_byte_cap_is_enforced_at_run_time(s, tmp_path):
    text = "a" * 65535 + "é"  # 65,536 chars pass validate_call; 65,537 bytes
    assert validate_call(Call("write_file", {"filename": "big", "text": text}), s) is not None
    r = run(s, "write_file", riya_dir=tmp_path, filename="big", text=text)
    assert not r.ok and list(tmp_path.iterdir()) == []
    assert run(s, "write_file", riya_dir=tmp_path, filename="ok", text="a" * 65536).ok


def test_write_file_retries_the_next_suffix_when_a_racer_wins(s, tmp_path, monkeypatch):
    real, n = tools.unique_path, []

    def racy(folder, name):
        p = real(folder, name)
        if not n:  # someone else creates exactly this file between the check and our open
            n.append(1)
            p.write_text("theirs", encoding="utf-8")
        return p

    monkeypatch.setattr("riya.core.hands.tools.unique_path", racy)
    r = run(s, "write_file", riya_dir=tmp_path, filename="race", text="mine")
    assert r.ok and (tmp_path / "race.txt").read_text(encoding="utf-8") == "theirs"
    assert (tmp_path / "race (2).txt").read_text(encoding="utf-8") == "mine"


def test_write_file_gives_up_cleanly(s, tmp_path, monkeypatch):
    monkeypatch.setattr("riya.core.hands.tools.unique_path", lambda folder, name: tmp_path / "same.txt")
    (tmp_path / "same.txt").write_text("x", encoding="utf-8")
    r = run(s, "write_file", riya_dir=tmp_path, filename="same", text="mine")
    assert not r.ok and (tmp_path / "same.txt").read_text(encoding="utf-8") == "x"
    assert not run(s, "write_file", riya_dir=None, filename="a", text="b").ok
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")  # the "folder" is a file: an OSError, reported kindly
    assert not run(s, "write_file", riya_dir=blocker, filename="a", text="b").ok
