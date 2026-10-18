import json

import pytest

from evaluation import eyes_check as ec
from riya.core.eyes.capture import FakeWindowSource, WindowInfo
from riya.core.models import ModelUnavailable
from tests.fakes import FakeModel

JPEG = b"\xff\xd8\xff\xe0fakejpegbytes\xff\xd9"
NOTE = json.dumps({"app": "Notepad", "activity": "reading a text file", "details": "", "notable": None})


def win(hwnd=7, title="riya_eyes_check_1 - Notepad"):
    return WindowInfo(hwnd, title, "notepad.exe", (0, 0, 800, 600), (0, 0, 1920, 1080), False, False)


def test_passed_needs_the_code():
    assert ec.passed("It says: Riya eyes check 4721.") and ec.passed("4721")
    assert not ec.passed("It says Riya eyes check 4712.") and not ec.passed("")


def test_main_skips_without_a_key(capsys):
    assert ec.main([], env={}) == 0
    assert capsys.readouterr().out.strip() == "skipped (no online key)"
    assert ec.main([], env={"RIYA_ONLINE_API_KEY": "  "}) == 0
    assert "skipped (no online key)" in capsys.readouterr().out


def test_main_skip_does_not_touch_windows_or_notepad(monkeypatch):
    monkeypatch.setattr(ec, "live", lambda model: pytest.fail("must not run live"))
    monkeypatch.setattr(ec.subprocess, "Popen", lambda *a, **k: pytest.fail("must not open Notepad"))
    assert ec.main([], env={}) == 0


def test_main_with_a_key_runs_live_with_the_vision_model(monkeypatch):
    seen = []
    monkeypatch.setattr(ec, "live", lambda model: seen.append(model) or True)
    monkeypatch.setattr(ec.sys, "platform", "win32")
    assert ec.main([], env={"RIYA_ONLINE_API_KEY": "k"}) == 0
    assert seen[0].name == "vision" and seen[0].first_token_timeout_s == 20.0


def test_help_works(capsys):
    with pytest.raises(SystemExit) as e:
        ec.main(["--help"], env={})
    assert e.value.code == 0 and "eyes_check" in capsys.readouterr().out


def test_ask_prints_note_and_answer_and_passes(capsys):
    model = FakeModel(replies=[NOTE, "It says Riya eyes check 4721"])
    answer, ok = ec.ask(model, JPEG)
    out = capsys.readouterr().out
    assert ok and "4721" in answer and "Notepad" in out and "answer: It says Riya eyes check 4721" in out
    parts = model.calls[1][-1]["content"]
    assert parts[0]["text"] == "what does this say?" and parts[1]["type"] == "image_url"


def test_ask_fails_on_a_wrong_answer_and_survives_an_unavailable_model(capsys):
    assert ec.ask(FakeModel(replies=[NOTE, "Something unreadable"]), JPEG)[1] is False
    answer, ok = ec.ask(FakeModel(fail=ModelUnavailable("down")), JPEG)
    assert (answer, ok) == ("", False) and "note: none (unavailable)" in capsys.readouterr().out


class FakeUser32:
    def __init__(self):
        self.calls = []

    def ShowWindow(self, hwnd, cmd):
        self.calls.append(("show", hwnd))

    def SetForegroundWindow(self, hwnd):
        self.calls.append(("front", hwnd))

    def keybd_event(self, *a):
        pass


def test_wait_for_window_returns_the_foreground_match():
    src = FakeWindowSource(win())
    assert ec.wait_for_window(src, FakeUser32(), "RIYA_EYES_CHECK_1", timeout_s=1).hwnd == 7


def test_wait_for_window_brings_a_background_window_forward(monkeypatch):
    src = FakeWindowSource(win(1, "Terminal"))
    monkeypatch.setattr(ec, "top_windows", lambda u: [(1, "Terminal"), (7, "riya_eyes_check_1 - Notepad")])
    u, t = FakeUser32(), iter(range(100))

    def front(hwnd):  # the second poll finds it in front
        u.calls.append(("front", hwnd))
        src.info = win(hwnd)

    u.SetForegroundWindow = front
    got = ec.wait_for_window(src, u, "riya_eyes_check_1", timeout_s=10, sleep=lambda s: None, clock=lambda: next(t))
    assert got.hwnd == 7 and ("front", 7) in u.calls


def test_wait_for_window_gives_up(monkeypatch):
    monkeypatch.setattr(ec, "top_windows", lambda u: [])
    t = iter(range(100))
    src = FakeWindowSource(win(1, "Terminal"))
    assert ec.wait_for_window(src, FakeUser32(), "nope", timeout_s=3, sleep=lambda s: None, clock=lambda: next(t)) is None
