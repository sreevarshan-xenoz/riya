import json
import os

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
    monkeypatch.setattr(ec, "WINDOWS", True)
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


class LiveU32(FakeUser32):
    """A user32 stand-in: windows the user had open, and the Notepad window our launch makes."""

    def __init__(self, stays_open=False):
        super().__init__()
        self.stays_open, self.closed = stays_open, []

    def PostMessageW(self, hwnd, msg, w, l):
        assert msg == 0x0010
        self.closed.append(hwnd)

    def IsWindow(self, hwnd):
        return self.stays_open


@pytest.fixture
def live_env(monkeypatch, tmp_path):
    """live() with fake windows: hwnd 5 is a Notepad the user already had open, 9 is the one we open."""
    env = type("Env", (), {})()
    env.u, env.popen_calls, env.subprocess_calls = LiveU32(), [], []
    env.title = f"riya_eyes_check_{os.getpid()} - Notepad"
    env.fg = win(9, env.title)
    env.windows = lambda u: [(5, "notes.txt - Notepad")] + ([(9, env.title)] if env.launched else [])
    env.launched = False
    monkeypatch.setattr(ec, "PAINT_S", 0)
    monkeypatch.setattr(ec, "set_dpi_awareness", lambda: None)
    monkeypatch.setattr(ec, "_user32", lambda: env.u)
    monkeypatch.setattr(ec, "make_window_source", lambda: FakeWindowSource(env.fg))
    monkeypatch.setattr(ec, "top_windows", lambda u: env.windows(u))
    monkeypatch.setattr(ec, "grab_jpeg", lambda info: JPEG)
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(ec.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "t"))
    monkeypatch.setattr(ec.subprocess, "run", lambda *a, **k: env.subprocess_calls.append(a))

    def popen(cmd, **k):
        env.popen_calls.append(cmd)
        env.launched = True

    env.popen = popen
    env.tmp = tmp_path / "t"
    return env


def test_live_closes_only_the_window_it_opened_and_deletes_the_file(live_env):
    model = FakeModel(replies=[NOTE, "Riya eyes check 4721"])
    assert ec.live(model, popen=live_env.popen) is True
    assert live_env.u.closed == [9] and not live_env.tmp.exists()
    assert live_env.popen_calls[0][0] == "notepad.exe" and live_env.subprocess_calls == []


def test_live_never_closes_a_window_that_was_open_before(live_env, capsys):
    live_env.fg = win(5, live_env.title)  # our file opened as a tab in the user's window
    live_env.windows = lambda u: [(5, live_env.title)]
    model = FakeModel(replies=[NOTE, "Riya eyes check 4721"])
    assert ec.live(model, popen=live_env.popen) is True
    assert live_env.u.closed == [] and not live_env.tmp.exists()
    assert "left open" in capsys.readouterr().out


def test_live_failure_branch_never_closes_a_window_that_was_open_before(live_env, monkeypatch):
    live_env.windows = lambda u: [(5, live_env.title)]  # only the pre-existing window matches
    monkeypatch.setattr(ec, "wait_for_window", lambda *a, **k: None)
    assert ec.live(FakeModel(), popen=live_env.popen) is False
    assert live_env.u.closed == [] and not live_env.tmp.exists()


def test_live_failure_branch_closes_a_new_window_that_never_got_focus(live_env, monkeypatch):
    monkeypatch.setattr(ec, "wait_for_window", lambda *a, **k: None)
    assert ec.live(FakeModel(), popen=live_env.popen) is False
    assert live_env.u.closed == [9] and not live_env.tmp.exists()


def test_close_window_leaves_a_window_that_stays_open_and_never_kills(capsys, monkeypatch):
    monkeypatch.setattr(ec.subprocess, "run", lambda *a, **k: pytest.fail("no subprocess call allowed"))
    monkeypatch.setattr(ec.subprocess, "Popen", lambda *a, **k: pytest.fail("no subprocess call allowed"))
    u = LiveU32(stays_open=True)
    assert ec.close_window(u, 9, wait_s=0.2) is False
    assert u.closed == [9] and "still open" in capsys.readouterr().out


def test_live_never_runs_taskkill_or_terminates(live_env):
    ec.live(FakeModel(replies=[NOTE, "no code here"]), popen=live_env.popen)
    calls = live_env.subprocess_calls + [tuple(c[:1]) for c in live_env.popen_calls]  # argv[0] only: tmp paths vary
    assert live_env.subprocess_calls == [] and all("taskkill" not in str(c).lower() for c in calls)


def test_live_setup_failure_is_caught_and_cleaned_up(live_env, monkeypatch):
    def boom():
        raise OSError("no user32")

    monkeypatch.setattr(ec, "_user32", boom)
    assert ec.live(FakeModel(), popen=live_env.popen) is False
    assert not live_env.tmp.exists() and live_env.popen_calls == []


def test_cleanup_survives_a_failing_close(live_env, capsys, tmp_path):
    def bad_close(u, h):
        raise OSError("boom")

    ec.cleanup(LiveU32(), 9, tmp_path / "gone", close=bad_close)
    assert "couldn't close" in capsys.readouterr().out


def test_main_returns_0_when_live_raises(monkeypatch, capsys):
    def boom(model):
        raise RuntimeError("kaput")

    monkeypatch.setattr(ec, "live", boom)
    monkeypatch.setattr(ec, "WINDOWS", True)
    assert ec.main([], env={"RIYA_ONLINE_API_KEY": "k"}) == 0
    assert "FAIL: RuntimeError: kaput" in capsys.readouterr().out


def test_main_returns_0_when_setup_raises(monkeypatch, capsys):
    monkeypatch.setattr(ec, "load_online_from_env", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad env")))
    assert ec.main([], env={}) == 0
    assert "FAIL: ValueError: bad env" in capsys.readouterr().out
