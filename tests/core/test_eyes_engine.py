"""v0.3 eyes in the engine: glances on tick, screen questions, don't-look, notes in memory, prompt and protocol."""
import builtins
import io
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from riya.core.clock import FakeClock
from riya.core.eyes.capture import FakeWindowSource, WindowInfo
from riya.core.eyes.see import NOTE_PROMPT
from riya.core.initiative import Action
from riya.core.memory import DAY
from riya.core.models import ModelUnavailable
from riya.core.prompt import MEMORY_END
from riya.core.protocol import (DontLook, ErrorMsg, EyesMsg, ExpressionMsg, GetMemories, GlanceDelete, Look,
                                MemoriesMsg, SetSettings, SettingsMsg, StateMsg)
from tests.fakes import FakeModel, make_core

NOTE = json.dumps({"app": "VS Code", "activity": "fixing a failing test", "details": "pytest shows one red test",
                   "notable": "TypeError in test_x"})
MON = (0, 0, 1920, 1080)


def jpeg(shade: int = 128) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 36), (shade, shade, shade)).save(out, "JPEG")
    return out.getvalue()


def win(hwnd=1, title="main.py - riya - Visual Studio Code", process="Code.exe", rect=(0, 0, 800, 600), monitor=MON,
        minimized=False, is_riya=False, **kw) -> WindowInfo:
    return WindowInfo(hwnd, title, process, rect, monitor, minimized, is_riya, **kw)


RIYA = WindowInfo(9, "Riya", "electron.exe", (1500, 500, 1800, 1020), MON, False, True)


class Grabber:
    def __init__(self, data: bytes | None = None, on_grab=None):
        self.data, self.on_grab, self.calls = data or jpeg(), on_grab, []

    def __call__(self, info: WindowInfo) -> bytes:
        self.calls.append(info)
        if self.on_grab:
            self.on_grab()
        return self.data


class GateModel(FakeModel):
    """A vision model that waits for `release` before answering, so a test can act while a glance is in flight."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.started, self.release = threading.Event(), threading.Event()

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.started.set()
        assert self.release.wait(5)
        yield from super().stream(messages, max_tokens, temperature)


def eyes_core(info=None, vision=None, local_replies=("Okay.",), grab=None, tmp_path=None, clock=None, online=None):
    clock = clock or FakeClock()
    vision = vision if vision is not None else FakeModel("vision", [NOTE])
    source, grab = FakeWindowSource(info or win()), grab or Grabber()
    core, emitted, src, local = make_core(clock, local_replies=local_replies, online=online, tmp_path=tmp_path,
                                          vision=vision, window_source=source, grab=grab)
    return core, emitted, src, local, clock, vision, source, grab


def wait(core) -> None:
    t = core._glance_thread
    if t is not None:
        t.join(5)
        assert not t.is_alive()


def run(core, clock, seconds: int) -> None:
    for _ in range(seconds):
        clock.advance(1)
        core.tick()
        wait(core)


def of(emitted, kind):
    return [m for m in emitted if isinstance(m, kind)]


def last_system(model) -> str:
    return model.calls[-1][0]["content"]


def situation(model) -> str:
    return next(p for p in last_system(model).split("\n\n") if p.startswith("[Situation]"))


# ---- glances on tick ---------------------------------------------------------------------------------------------

def test_a_settled_window_is_glanced_once_and_the_note_is_kept():
    core, emitted, _, _, clock, vision, _, grab = eyes_core()
    run(core, clock, 3)
    assert grab.calls == [] and vision.calls == []          # still settling (4 s)
    run(core, clock, 3)
    assert len(grab.calls) == 1 and len(vision.calls) == 1
    parts = vision.calls[0][-1]["content"]
    assert parts[0]["text"] == NOTE_PROMPT and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert vision.max_tokens == [400]
    g = core.memory.recent_glances(1)[0]
    assert (g.app, g.activity, g.notable, g.title) == ("VS Code", "fixing a failing test", "TypeError in test_x",
                                                       "main.py - riya - Visual Studio Code")
    assert of(emitted, EyesMsg)[-1] == EyesMsg(enabled=True, dont_look=False, glances_today=1, cap=150)
    run(core, clock, 120)
    assert len(vision.calls) == 1                           # same window, nothing new: no second glance


def test_the_tick_never_waits_for_the_vision_model():
    core, _, _, _, clock, vision, _, _ = eyes_core(vision=GateModel("vision", [NOTE]))
    run(core, clock, 4)
    clock.advance(1)
    started = time.monotonic()
    core.tick()
    assert vision.started.wait(5)
    for _ in range(3):
        clock.advance(1)
        core.tick()
    assert time.monotonic() - started < 1.0 and core.memory.recent_glances(1) == []
    vision.release.set()
    wait(core)
    assert len(core.memory.recent_glances(1)) == 1 and len(vision.calls) == 1


@pytest.mark.parametrize("setup", [
    "dont_look", "eyes_off", "online_off", "no_vision", "denied_title", "denied_process", "unknown_process", "riya",
    "minimized", "gaming", "away", "cap", "extra_title", "extra_process"])
def test_no_capture_unless_every_gate_passes(setup):
    info = {"denied_title": win(title="Sign in to your bank - Chrome"), "denied_process": win(process="KeePassXC.exe"),
            "unknown_process": win(process=""), "riya": win(is_riya=True, process="electron.exe"),
            "minimized": win(minimized=True), "extra_title": win(title="Secret plans - Notepad"),
            "extra_process": win(process="Signal.exe")}.get(setup)
    core, _, src, _, clock, vision, _, grab = eyes_core(info=info)
    if setup == "no_vision":
        core.vision = None
    patch = {"dont_look": {"dont_look": True}, "eyes_off": {"eyes_enabled": False},
             "online_off": {"online_enabled": False}, "cap": {"glance_daily_cap": 0},
             "extra_title": {"eyes_deny_titles": ["secret"]},
             "extra_process": {"eyes_deny_processes": ["signal\\.exe"]}}.get(setup)
    if patch:
        core.handle(SetSettings(patch=patch))
    if setup == "gaming":
        src.is_fullscreen = True
    if setup == "away":
        src.idle = 1000
    run(core, clock, 30)
    assert grab.calls == [] and vision.calls == []


def test_dont_look_switched_on_mid_glance_discards_the_note():
    core, emitted, _, _, clock, vision, _, _ = eyes_core(vision=GateModel("vision", [NOTE]))
    run(core, clock, 4)
    clock.advance(1)
    core.tick()
    assert vision.started.wait(5)
    core.handle(DontLook(on=True))
    vision.release.set()
    wait(core)
    assert core.memory.list_glances() == [] and core.policy.glances_today() == 1   # it was sent: it counts
    assert of(emitted, EyesMsg)[-1].dont_look is True


def test_dont_look_during_the_grab_means_the_image_is_never_sent():
    holder = {}
    grab = Grabber(on_grab=lambda: holder["core"].handle(DontLook(on=True)))
    core, _, _, _, clock, vision, _, _ = eyes_core(grab=grab)
    holder["core"] = core
    run(core, clock, 6)
    assert len(grab.calls) == 1 and vision.calls == [] and core.memory.list_glances() == []


def test_a_failed_glance_backs_off_for_30_minutes():
    core, _, _, local, clock, vision, source, _ = eyes_core(vision=FakeModel("vision", [NOTE],
                                                                             fail=ModelUnavailable("HTTP 429")))
    run(core, clock, 6)
    assert len(vision.calls) == 1 and core.memory.list_glances() == []
    vision.fail = None
    source.info = win(hwnd=2, title="notes.txt - Notepad", process="notepad.exe")
    run(core, clock, 1700)
    assert len(vision.calls) == 1                           # backing off
    run(core, clock, 120)
    assert len(vision.calls) == 2 and core.memory.recent_glances(1)[0].app == "VS Code"
    assert local.calls == []                                # chat model untouched by glances


@pytest.mark.parametrize("reply", ["no idea, sorry", ""])
def test_an_unparseable_or_empty_note_backs_off_for_5_minutes(reply):
    core, _, _, _, clock, vision, source, _ = eyes_core(vision=FakeModel("vision", [reply, NOTE]))
    run(core, clock, 6)
    source.info = win(hwnd=2, title="b")
    run(core, clock, 280)
    assert len(vision.calls) == 1 and core.memory.list_glances() == []
    run(core, clock, 40)
    assert len(vision.calls) == 2 and len(core.memory.list_glances()) == 1


def test_consecutive_unusable_notes_back_off_5_15_30_60_minutes_and_a_note_resets_it():
    vision = FakeModel("vision", ["", "nope", "", "{}", "", NOTE, ""])
    core, _, _, _, clock, _, source, _ = eyes_core(vision=vision)
    sent_at, stream = [], vision.stream
    vision.stream = lambda *a, **k: (sent_at.append(clock.now()), stream(*a, **k))[1]
    run(core, clock, 6 + 300 + 900 + 1800 + 3600 + 3600 + 5)
    gaps = [round(b - a) for a, b in zip(sent_at, sent_at[1:])]
    assert gaps == [300, 900, 1800, 3600, 3600]                  # the 6th reply is a note...
    assert len(core.memory.list_glances()) == 1 and core.policy.glances_today() == 6
    source.info = win(hwnd=2, title="notes.txt - Notepad", process="notepad.exe")
    run(core, clock, 60)
    assert len(sent_at) == 7 and core.policy.glances_today() == 7
    assert core.policy.unusable() == 900                         # ...so the 7th, unusable, backed off 5 min again


@pytest.mark.parametrize("vision", [FakeModel("vision", ["no idea"]), FakeModel("vision", [""]),
                                    FakeModel("vision", [NOTE], fail=ModelUnavailable("HTTP 429"))])
def test_every_sent_glance_counts_whatever_came_back(vision, tmp_path):
    core, emitted, _, _, clock, _, _, _ = eyes_core(vision=vision, tmp_path=tmp_path)
    run(core, clock, 6)
    assert len(vision.calls) == 1 and core.memory.list_glances() == []
    assert core.policy.glances_today() == 1 and of(emitted, EyesMsg)[-1].glances_today == 1
    again, *_ = eyes_core(tmp_path=tmp_path, clock=clock)
    assert again.policy.glances_today() == 1                     # saved like any other glance


def test_unusable_replies_stop_at_the_daily_cap():
    core, _, _, _, clock, vision, _, _ = eyes_core(vision=FakeModel("vision", [""]))
    core.handle(SetSettings(patch={"glance_daily_cap": 2}))
    run(core, clock, 6 + 300 + 900 + 1800)
    assert len(vision.calls) == 2 and core.policy.glances_today() == 2


@pytest.mark.parametrize("vision", [FakeModel("vision", [""]), FakeModel("vision", [NOTE], fail=ModelUnavailable("x"))])
def test_a_failed_screen_question_still_counts(vision):
    core, emitted, _, _, clock, _, _, _ = eyes_core(vision=vision, local_replies=["Can't see it~"])
    list(core.reply("what's wrong with this?"))
    assert len(vision.calls) == 1 and core.policy.glances_today() == 1
    assert of(emitted, EyesMsg)[-1].glances_today == 1


def test_empty_screen_answers_escalate_the_backoff_too():
    core, _, _, _, clock, vision, _, _ = eyes_core(vision=FakeModel("vision", [""]), local_replies=["Hmm~"])
    list(core.reply("what's wrong with this?"))
    list(core.reply("what's wrong with this?"))
    assert core.policy.unusable() == 1800                        # 300 and 900 were used by the two questions


@pytest.mark.parametrize("fail", [ModelUnavailable("online: <urlopen error [WinError 10061] refused>"),
                                  ModelUnavailable("online: HTTP 503"), RuntimeError("socket blew up")])
def test_an_outage_backs_off_for_30_minutes(fail):
    core, _, _, _, clock, vision, source, _ = eyes_core(vision=FakeModel("vision", [NOTE], fail=fail))
    run(core, clock, 6)
    vision.fail = None
    source.info = win(hwnd=2, title="b")
    run(core, clock, 1700)
    assert len(vision.calls) == 1
    run(core, clock, 120)
    assert len(vision.calls) == 2


def test_same_window_is_glanced_again_after_5_minutes_only_when_it_changed():
    grab = Grabber(jpeg(40))
    core, _, _, _, clock, vision, _, _ = eyes_core(grab=grab, vision=FakeModel("vision", [NOTE]))
    run(core, clock, 6)
    assert len(vision.calls) == 1
    run(core, clock, 400)
    assert len(vision.calls) == 1                           # peeked, but the screen didn't change
    assert 1 < len(grab.calls) < 20                         # peeks are throttled, never every tick
    grab.data = jpeg(220)
    run(core, clock, 40)
    assert len(vision.calls) == 2


def test_unicode_titles_and_notes_round_trip():
    note = json.dumps({"app": "Chrome", "activity": "படிக்கிறார் 📚 notes", "details": "日本語のページ", "notable": None},
                      ensure_ascii=False)
    core, emitted, _, local, clock, *_ = eyes_core(info=win(title="தமிழ் notes 📚 – Chrome", process="chrome.exe"),
                                                   vision=FakeModel("vision", [note]))
    run(core, clock, 6)
    g = core.memory.list_glances()[0]
    assert (g.title, g.activity, g.details) == ("தமிழ் notes 📚 – Chrome", "படிக்கிறார் 📚 notes", "日本語のページ")
    core.handle(GetMemories())
    assert of(emitted, MemoriesMsg)[-1].glances[0].title == g.title
    list(core.reply("hey"))
    assert "படிக்கிறார் 📚 notes (Chrome)" in last_system(local)


def test_a_private_note_keeps_no_title():
    private = json.dumps({"app": "Chrome", "activity": "something private", "details": "", "notable": None})
    core, _, _, _, clock, _, _, _ = eyes_core(info=win(title="Chat with Priya - WhatsApp", process="WhatsApp.exe"),
                                              vision=FakeModel("vision", [private]))
    run(core, clock, 6)
    g = core.memory.list_glances()[0]
    assert g.activity == "something private" and g.title == "" and g.details == ""


def test_no_file_is_ever_written_while_she_looks(tmp_path, monkeypatch):
    core, _, _, _, clock, vision, _, _ = eyes_core(tmp_path=tmp_path, local_replies=["Hm."])
    vision.replies = [NOTE, "That's a TypeError, dummy."]

    def no_files(*a, **kw):
        raise AssertionError(f"a file was opened: {a!r}")

    monkeypatch.setattr(builtins, "open", no_files)
    monkeypatch.setattr(io, "open", no_files)
    monkeypatch.setattr(Path, "write_bytes", no_files)
    monkeypatch.setattr(Image.Image, "save", no_files)
    run(core, clock, 6)
    assert list(core.reply("what's wrong with this error?"))
    assert len(vision.calls) == 2 and len(core.memory.list_glances()) == 1


# ---- look ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rect,monitor,buddy,side", [
    ((0, 0, 800, 600), MON, None, "left"),
    ((760, 0, 1160, 600), MON, None, "center"),
    ((1200, 0, 1920, 1080), MON, None, "right"),
    ((2000, 0, 2600, 1080), (1920, 0, 3840, 1080), None, "left"),   # its own monitor's centre, not the primary's
    ((560, 0, 1360, 1080), MON, [1600, 500], "left"),                # her centre is 1600 + 150
    ((1450, 0, 2050, 1080), MON, [1600, 500], "center"),
    ((200, 0, 600, 1080), MON, [10, 500], "right"),
])
def test_look_side_is_relative_to_her(rect, monitor, buddy, side):
    core, emitted, _, _, clock, _, _, _ = eyes_core(info=win(rect=rect, monitor=monitor))
    if buddy:
        core.handle(SetSettings(patch={"buddy_position": buddy}))
    run(core, clock, 1)
    assert of(emitted, Look)[-1] == Look(side=side, watching=True)


def test_look_is_sent_only_when_it_changes():
    core, emitted, src, _, clock, _, source, _ = eyes_core()
    run(core, clock, 10)
    assert of(emitted, Look) == [Look(side="left", watching=True)]
    source.info = win(hwnd=2, rect=(1200, 0, 1920, 1080))
    run(core, clock, 3)
    src.is_fullscreen = True                                 # a game: she doesn't watch
    run(core, clock, 3)
    source.info = None
    run(core, clock, 3)
    assert of(emitted, Look) == [Look(side="left", watching=True), Look(side="right", watching=True),
                                 Look(side="right", watching=False), Look(side=None, watching=False)]


def test_she_does_not_watch_while_covering_her_eyes_or_a_denylisted_window():
    core, emitted, _, _, clock, _, source, _ = eyes_core()
    core.handle(DontLook(on=True))
    run(core, clock, 2)
    assert of(emitted, Look)[-1].watching is False
    core.handle(DontLook(on=False))
    source.info = win(title="PayPal - Chrome")
    run(core, clock, 2)
    assert of(emitted, Look)[-1].watching is False


DESKTOP = WindowInfo(5, "", "explorer.exe", MON, MON, False, False, "Progman")
TASKBAR = WindowInfo(6, "", "explorer.exe", (0, 1032, 1920, 1080), MON, False, False, "Shell_TrayWnd")


@pytest.mark.parametrize("shell", [DESKTOP, TASKBAR, WindowInfo(7, "", "explorer.exe", MON, MON, False, False, "WorkerW"),
                                   WindowInfo(8, "Search", "SearchHost.exe", MON, MON, False, False, "x")])
def test_the_desktop_and_taskbar_are_never_glanced_or_grabbed(shell):
    core, emitted, _, local, clock, vision, _, grab = eyes_core(info=shell)
    run(core, clock, 400)
    assert grab.calls == [] and vision.calls == []
    assert of(emitted, Look)[-1] == Look(side=None, watching=False)
    assert list(core.reply("what's on my screen?"))
    assert grab.calls == [] and vision.calls == []
    assert "no window in front of you" in situation(local) and "explorer" not in situation(local)


def test_a_click_through_overlay_is_never_grabbed():
    core, emitted, _, local, clock, vision, _, grab = eyes_core(info=win(ex_style=0x80020))
    run(core, clock, 10)
    assert grab.calls == [] and vision.calls == [] and of(emitted, Look)[-1].watching is False
    list(core.reply("what's on my screen?"))
    assert grab.calls == [] and "see-through overlay" in situation(local)


# ---- don't look ----------------------------------------------------------------------------------------------------

def test_dont_look_covers_her_eyes_and_persists(tmp_path):
    core, emitted, _, _, clock, _, _, _ = eyes_core(tmp_path=tmp_path)
    core.handle(DontLook(on=True))
    assert core.settings.dont_look is True
    assert of(emitted, EyesMsg)[-1] == EyesMsg(enabled=True, dont_look=True, glances_today=0, cap=150)
    assert of(emitted, ExpressionMsg)[-1].name == "covering_eyes"
    assert of(emitted, SettingsMsg)[-1].settings["dont_look"] is True
    run(core, clock, 31)
    assert of(emitted, ExpressionMsg)[-1].name == "covering_eyes" and of(emitted, StateMsg)[-1].expression == \
        "covering_eyes"
    again, *_ = eyes_core(tmp_path=tmp_path, clock=clock)
    hello = again.hello_messages()
    assert [m for m in hello if isinstance(m, StateMsg)][0].expression == "covering_eyes"
    assert [m for m in hello if isinstance(m, EyesMsg)][0].dont_look is True
    emitted.clear()
    core.handle(DontLook(on=False))
    assert of(emitted, EyesMsg)[-1].dont_look is False
    assert of(emitted, ExpressionMsg)[-1].name != "covering_eyes"


def test_hello_carries_eyes():
    core, *_ = eyes_core()
    assert [m for m in core.hello_messages() if isinstance(m, EyesMsg)] == [
        EyesMsg(enabled=True, dont_look=False, glances_today=0, cap=150)]
    assert [m.type for m in core.hello_messages()][:3] == ["hello", "settings", "state"]


def test_eyes_message_follows_settings_changes():
    core, emitted, *_ = eyes_core()
    core.handle(SetSettings(patch={"glance_daily_cap": 20}))
    assert of(emitted, EyesMsg)[-1].cap == 20 and core.policy.cap == 20
    core.handle(SetSettings(patch={"eyes_enabled": False}))
    assert of(emitted, EyesMsg)[-1].enabled is False
    n = len(of(emitted, EyesMsg))
    core.handle(SetSettings(patch={"volume": 0.3}))
    assert len(of(emitted, EyesMsg)) == n                    # only on change


# ---- screen questions ----------------------------------------------------------------------------------------------

def test_a_screen_question_sends_the_image_with_her_full_prompt_and_is_shaped():
    vision = FakeModel("vision", ["Oh, that's the editor. A test is red. Line four looks off. Also five. 😀😀"])
    core, emitted, _, local, clock, _, _, grab = eyes_core(vision=vision)
    said = list(core.reply("what am I looking at?"))
    assert said == ["Oh, that's the editor.", "A test is red.", "Line four looks off."]   # small: 3 sentences
    msgs = vision.calls[0]
    system = msgs[0]["content"]
    assert "Companion layer" in system and "### What you know about them" in system and "[Riya's state" in system
    parts = msgs[-1]["content"]
    assert parts[0] == {"type": "text", "text": "what am I looking at?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert vision.max_tokens[0] >= 400 and local.calls == [] and len(grab.calls) == 1
    assert core.policy.glances_today() == 1
    assert core.turns[-2] == {"role": "user", "content": "what am I looking at?"}       # no image kept in turns


def test_a_screen_question_skips_the_min_gap():
    vision = FakeModel("vision", [NOTE, "It's a TypeError."])
    core, _, _, _, clock, _, _, grab = eyes_core(vision=vision)
    run(core, clock, 6)
    assert len(vision.calls) == 1
    assert list(core.reply("what's wrong with this error?")) == ["It's a TypeError."]
    assert len(vision.calls) == 2


def test_typing_to_her_looks_at_the_window_behind_her():
    vision = FakeModel("vision", ["Looks fine to me."])
    core, _, _, _, clock, _, source, grab = eyes_core(vision=vision)
    core.vision = None                                        # no glances, just remember the window
    run(core, clock, 2)
    core.vision = vision
    source.info = RIYA
    run(core, clock, 30)
    source.windows[1] = win(rect=(100, 100, 900, 700))       # moved meanwhile: the fresh rect is the one grabbed
    list(core.reply("can you see this error?"))
    assert [(i.hwnd, i.rect) for i in grab.calls] == [(1, (100, 100, 900, 700))]


@pytest.mark.parametrize("change", ["minimized", "retitled", "denylisted", "closed", "other_process"])
def test_the_window_behind_her_is_reread_before_the_grab(change):
    core, _, _, local, clock, vision, source, grab = eyes_core()
    core.vision = None
    run(core, clock, 2)
    core.vision = vision
    source.info = RIYA
    run(core, clock, 5)
    if change == "closed":
        del source.windows[1]
    else:
        source.windows[1] = {"minimized": win(minimized=True), "retitled": win(title="notes.txt - Notepad"),
                             "denylisted": win(title="PayPal - Log in"),
                             "other_process": win(process="KeePassXC.exe")}[change]
    list(core.reply("can you see this error?"))
    assert grab.calls == [] and vision.calls == []
    system = last_system(local)
    assert "you can't see right now: the window changed" in system
    assert "PayPal" not in system and "main.py" not in system


def test_a_background_glance_rereads_its_window_before_grabbing():
    core, _, _, _, clock, vision, source, grab = eyes_core()
    run(core, clock, 4)
    source.window = lambda hwnd: win(title="Sign in to PayPal")   # retitled between the tick and the grab
    run(core, clock, 3)
    assert grab.calls == [] and vision.calls == []


# ---- covered windows (z-order) ---------------------------------------------------------------------------------

VAULT = WindowInfo(20, "Bitwarden", "Bitwarden.exe", (600, 400, 1000, 800), MON, False, False)  # overlaps (0,0,800,600)


@pytest.mark.parametrize("above,grabbed", [
    ([VAULT], False),                                                        # a password manager over part of it
    ([win(hwnd=21, title="Toast", process="x.exe", rect=(700, 500, 1100, 900), ex_style=0x80020)], False),
    ([win(hwnd=22, title="Spotify", process="Spotify.exe", rect=(900, 0, 1900, 1000))], True),  # beside it
    ([win(hwnd=23, title="Spotify", process="Spotify.exe", rect=(800, 0, 1900, 600))], True),   # touching only
    ([replace(VAULT, cloaked=True)], True),                                  # on another virtual desktop
    ([replace(VAULT, minimized=True)], True),
    ([replace(RIYA, rect=(500, 300, 800, 600))], True),                      # her own window
])
def test_a_background_glance_never_grabs_a_covered_window(above, grabbed):
    core, _, _, _, clock, vision, source, grab = eyes_core()
    source.stacks[1] = above
    run(core, clock, 10)
    assert (len(grab.calls) == 1 and len(vision.calls) == 1) if grabbed else (grab.calls == [] and vision.calls == [])


def test_a_covered_window_is_only_part_of_its_monitor():
    w = win(rect=(-300, 0, 500, 600))                          # partly off the left of its monitor
    core, _, _, _, clock, vision, source, grab = eyes_core(info=w)
    source.stacks[1] = [win(hwnd=24, title="x", process="x.exe", rect=(-300, 0, -10, 600))]  # covers only the off part
    run(core, clock, 10)
    assert len(grab.calls) == 1


def test_a_screen_question_never_grabs_a_covered_window():
    core, _, _, local, clock, vision, source, grab = eyes_core(vision=FakeModel("vision", ["Looks fine."]))
    source.stacks[1] = [VAULT]
    list(core.reply("what's on my screen?"))
    assert grab.calls == [] and vision.calls == []
    assert "on top of" in situation(local) and "Bitwarden" not in last_system(local)


def test_a_window_above_that_cant_be_read_means_no_look():
    core, _, _, local, clock, vision, source, grab = eyes_core()
    source.above = lambda hwnd: (_ for _ in ()).throw(RuntimeError("the window is gone"))
    run(core, clock, 10)
    list(core.reply("what's on my screen?"))
    assert grab.calls == [] and vision.calls == []


def test_her_own_window_over_the_window_behind_her_is_fine_but_others_are_not():
    vision = FakeModel("vision", ["Looks fine to me.", "Fine."])
    core, _, _, local, clock, _, source, grab = eyes_core(vision=vision)
    core.vision = None
    run(core, clock, 2)
    core.vision = vision
    source.info = RIYA
    source.stacks[1] = [replace(RIYA, rect=(500, 300, 900, 700))]   # she sits over part of it
    run(core, clock, 5)
    list(core.reply("can you see this error?"))
    assert len(grab.calls) == 1
    source.stacks[1] = [replace(RIYA, rect=(500, 300, 900, 700)), VAULT]
    list(core.reply("can you see this error?"))
    assert len(grab.calls) == 1 and "on top of" in situation(core.local)


def test_the_foreground_changing_between_tick_and_grab_means_no_grab():
    core, _, _, _, clock, vision, source, grab = eyes_core()
    run(core, clock, 4)
    read = source.window

    def window(hwnd):                                          # the user switches windows right as the worker starts
        source.info = win(hwnd=2, title="notes.txt - Notepad", process="notepad.exe")
        return read(hwnd)

    source.window = window
    clock.advance(1)
    core.tick()
    wait(core)
    assert grab.calls == [] and vision.calls == []


@pytest.mark.parametrize("setup,why", [
    ("dont_look", "covering your eyes"),
    ("denied", "private"),
    ("eyes_off", "switched off"),
    ("cap", "enough times today"),
    ("no_vision", "online model"),
])
def test_when_she_cannot_look_the_situation_says_why(setup, why):
    info = win(title="Net banking - HDFC") if setup == "denied" else None
    core, _, _, local, _, vision, _, grab = eyes_core(info=info)
    if setup == "no_vision":
        core.vision = None
    patch = {"dont_look": {"dont_look": True}, "eyes_off": {"eyes_enabled": False},
             "cap": {"glance_daily_cap": 0}}.get(setup)
    if patch:
        core.handle(SetSettings(patch=patch))
    list(core.reply("what's wrong with this?"))
    system = last_system(local)
    assert "you can't see right now:" in system and why in system
    assert grab.calls == [] and vision.calls == []
    if setup == "denied":
        assert "HDFC" not in system


def test_a_failed_screen_question_falls_back_to_local_and_rearms_the_backoff():
    vision = FakeModel("vision", [NOTE], fail=ModelUnavailable("HTTP 429"))
    core, _, _, local, clock, _, source, _ = eyes_core(vision=vision, local_replies=["I can't see it right now~"])
    assert list(core.reply("what's wrong with this?")) == ["I can't see it right now~"]
    system = last_system(local)
    assert "couldn't see" in system and "main.py - riya - Visual Studio Code" in system
    assert all(isinstance(m["content"], str) for m in local.calls[-1])     # the local model never gets the image
    vision.fail = None
    run(core, clock, 1700)                                   # a settled window, but the backoff holds
    assert len(vision.calls) == 1
    assert list(core.reply("what's wrong with this?"))       # a forced question may still try during backoff
    assert len(vision.calls) == 2
    vision.fail = ModelUnavailable("HTTP 429")
    list(core.reply("what's wrong with this?"))              # ...and failing re-arms it
    vision.fail = None
    source.info = win(hwnd=3, title="c")
    run(core, clock, 1700)
    assert len(vision.calls) == 3
    run(core, clock, 200)
    assert len(vision.calls) == 4


def test_dont_look_after_the_grab_but_before_the_send_never_sends_the_image():
    core, _, _, local, _, vision, _, grab = eyes_core(local_replies=["Eyes covered~"])
    core._eyes_context = lambda: core.handle(DontLook(on=True))   # runs between the grab and the send
    assert list(core.reply("what's wrong with this?")) == ["Eyes covered~"]
    assert len(grab.calls) == 1 and vision.calls == []
    system = last_system(local)
    assert "covering your eyes" in system and core.policy.glances_today() == 0
    assert "main.py" not in system and "Code.exe" not in system      # no guessing from the window either


def test_eyes_closed_before_the_question_grabs_nothing():
    core, _, _, local, _, vision, _, grab = eyes_core()
    core._dont_look = True             # closed by another thread after the settings were read
    list(core.reply("what's wrong with this?"))
    assert grab.calls == [] and vision.calls == []


def test_a_screen_question_waits_for_a_glance_in_flight_and_the_cap_holds():
    vision = GateModel("vision", [NOTE, "A red test."])
    core, _, _, local, clock, _, _, grab = eyes_core(vision=vision)
    core.handle(SetSettings(patch={"glance_daily_cap": 1}))
    run(core, clock, 4)
    clock.advance(1)
    core.tick()
    assert vision.started.wait(5)
    asker = threading.Thread(target=lambda: list(core.reply("what's wrong with this?")))
    asker.start()
    asker.join(0.3)
    assert asker.is_alive() and len(grab.calls) == 1        # waiting: no second capture while the glance runs
    vision.release.set()
    asker.join(5)
    assert len(vision.calls) == 1 and len(grab.calls) == 1   # the glance used the cap: the question can't look
    assert core.policy.glances_today() == 1 and "enough times today" in last_system(local)


def test_no_background_glance_starts_while_a_screen_question_runs():
    vision = GateModel("vision", ["A red test."])
    core, _, _, _, clock, _, _, grab = eyes_core(vision=vision)
    clock.advance(10)
    asker = threading.Thread(target=lambda: list(core.reply("what's wrong with this?")))
    asker.start()
    assert vision.started.wait(5)
    for _ in range(10):
        clock.advance(1)
        assert core.tick() is None
    assert len(grab.calls) == 1
    vision.release.set()
    asker.join(5)
    assert len(vision.calls) == 1


def test_a_stale_expression_is_never_sent_after_dont_look():
    core, emitted, *_ = eyes_core()
    real, other = core.feelings.expression, []

    def slow_expression():
        if not other:                  # don't-look arrives from the socket thread mid-computation
            other.append(threading.Thread(target=core.handle, args=(DontLook(on=True),)))
            other[0].start()
            other[0].join(0.3)
            return "happy"
        return real()

    core.feelings.expression = slow_expression
    core._sync_expression()
    other[0].join(5)
    assert of(emitted, ExpressionMsg)[-1].name == "covering_eyes"


def test_glances_today_survives_a_restart_the_same_day(tmp_path):
    core, _, _, _, clock, *_ = eyes_core(tmp_path=tmp_path)
    run(core, clock, 6)
    assert core.policy.glances_today() == 1
    again, *_ = eyes_core(tmp_path=tmp_path, clock=clock)
    assert again.policy.glances_today() == 1
    assert [m for m in again.hello_messages() if isinstance(m, EyesMsg)][0].glances_today == 1
    clock.advance(DAY)
    later, *_ = eyes_core(tmp_path=tmp_path, clock=clock)
    assert later.policy.glances_today() == 0


def test_a_forced_look_counts_toward_the_saved_total(tmp_path):
    core, _, _, _, clock, vision, _, _ = eyes_core(tmp_path=tmp_path, vision=FakeModel("vision", ["Looks fine."]))
    list(core.reply("what's wrong with this?"))
    again, *_ = eyes_core(tmp_path=tmp_path, clock=clock)
    assert again.policy.glances_today() == 1


def test_a_failed_capture_is_said_honestly():
    def broken(info):
        raise OSError("mss failed")

    core, _, _, local, _, vision, _, _ = eyes_core(grab=broken)
    list(core.reply("what's wrong with this?"))
    assert "you can't see right now: the screen capture didn't work" in last_system(local) and vision.calls == []


def test_an_empty_screen_answer_backs_off_only_5_minutes():
    vision = FakeModel("vision", ["", NOTE])
    core, _, _, local, clock, _, source, _ = eyes_core(vision=vision, local_replies=["Hmm, can't see it~"])
    assert list(core.reply("what's wrong with this?")) == ["Hmm, can't see it~"]
    source.info = win(hwnd=2, title="b")
    run(core, clock, 280)
    assert len(vision.calls) == 1
    run(core, clock, 40)
    assert len(vision.calls) == 2


def test_screen_question_while_gaming_is_refused_but_a_break_still_looks():
    vision = FakeModel("vision", ["A red test."])
    core, _, src, local, clock, _, _, grab = eyes_core(vision=vision)
    src.idle = 150
    run(core, clock, 1)
    assert core.presence.state == "break"
    list(core.reply("what's wrong with this?"))
    assert len(grab.calls) == 1
    src.idle, src.is_fullscreen = 0, True
    run(core, clock, 1)
    list(core.reply("what's wrong with this?"))
    assert len(grab.calls) == 1 and "in a game" in last_system(local)


# ---- notes in memory, prompt, bids and consolidation ---------------------------------------------------------------

def test_memories_list_and_delete_glances():
    core, emitted, *_ = eyes_core()
    g = core.memory.add_glance("VS Code", "main.py", "fixing a test", "pytest", None)
    core.handle(GetMemories())
    assert of(emitted, MemoriesMsg)[-1].glances[0].model_dump() == {
        "id": g.id, "at": g.at, "app": "VS Code", "title": "main.py", "activity": "fixing a test",
        "details": "pytest", "notable": None}
    core.handle(GlanceDelete(id=g.id))
    assert of(emitted, MemoriesMsg)[-1].glances == [] and core.memory.list_glances() == []


def test_glances_are_purged_after_7_days():
    core, _, _, _, clock, *_ = eyes_core()
    core.memory.add_glance("A", "t", "old", "", None)
    clock.advance(6 * DAY)
    core.memory.add_glance("B", "t", "newer", "", None)
    clock.advance(1.5 * DAY)
    assert core.memory.purge_glances() == 1
    assert [g.activity for g in core.memory.list_glances()] == ["newer"]


def test_the_daily_purge_drops_old_glances():
    core, _, _, _, clock, *_ = eyes_core()
    core.vision = None
    core.memory.add_glance("A", "t", "old", "", None)
    clock.advance(8 * DAY)
    run(core, clock, 1)
    assert core.memory.list_glances() == []


def test_the_prompt_says_what_they_are_doing_outside_the_memory_block():
    core, _, _, local, clock, *_ = eyes_core()
    core.vision = None
    for activity in ("reading docs", "writing tests", "fixing a failing test"):
        core.memory.add_glance("VS Code", "main.py", activity, "pytest shows one red test", None)
        clock.advance(60)
    list(core.reply("hey"))
    system = last_system(local)
    now = system.index("### What they're doing now: fixing a failing test")
    assert now > system.index(MEMORY_END)
    assert "writing tests" in system[now:] and "reading docs" in system[now:]
    clock.advance(11 * 60)
    list(core.reply("hey"))
    system = last_system(local)
    assert "### What they're doing now" not in system and "fixing a failing test" in system


@pytest.mark.parametrize("patch,now_line,block", [({"dont_look": True}, False, True), ({"eyes_enabled": False}, False, False),
                                                   ({}, True, True)])
def test_dont_look_drops_the_now_line_and_eyes_off_drops_the_eyes_block(patch, now_line, block):
    core, _, _, local, clock, *_ = eyes_core(local_replies=["Hey~"])
    core.vision = None
    core.memory.add_glance("VS Code", "main.py", "fixing a failing test", "pytest shows one red test", "TypeError")
    core.handle(SetSettings(patch=patch))
    list(core.reply("hey"))
    system = last_system(local)
    assert ("### What they're doing now" in system) is now_line
    assert ("What you glimpsed on their screen" in system) is block and ("fixing a failing test" in system) is block
    core._bid(Action("bid", False, None, "worked 2.1 h straight"))
    assert ("fixing a failing test" in situation(local)) is now_line     # nor is it a bid's reason


def test_a_bid_can_use_the_latest_note():
    core, emitted, _, local, clock, *_ = eyes_core(local_replies=["Still on that bug?"])
    core.vision = None
    core.memory.add_glance("VS Code", "main.py", "fixing a failing test", "", "TypeError in test_x")
    core._bid(Action("bid", False, None, "worked 2.1 h straight"))
    assert "fixing a failing test" in situation(local) and "TypeError in test_x" in situation(local)
    clock.advance(31 * 60)
    core._bid(Action("bid", False, None, "worked 2.6 h straight"))
    assert "fixing a failing test" not in situation(local)


def test_a_private_note_is_never_a_bid_reason():
    core, _, _, local, *_ = eyes_core(local_replies=["Hey~"])
    core.vision = None
    core.memory.add_glance("Chrome", "", "something private", "", None)
    core._bid(Action("bid", False, None, "worked 2.1 h straight"))
    assert "private" not in situation(local)


def test_consolidation_sees_the_chunks_notes_but_not_private_ones():
    core, _, _, local, clock, *_ = eyes_core(local_replies=["Okay.", json.dumps(
        {"facts": [], "episode_summary": "", "user_mood": None, "threads": []})])
    core.vision = None
    core.memory.add_glance("Spotify", "x", "playing music long before", "", None)
    clock.advance(7200)
    core.memory.add_glance("VS Code", "main.py", "building Riya in Python", "", None)
    core.memory.add_glance("Chrome", "", "something private", "", None)
    clock.advance(60)
    list(core.reply("ugh this bug"))
    clock.advance(700)
    core.consolidator.run()
    sent = local.calls[-1][1]["content"]
    assert "building Riya in Python" in sent and "something private" not in sent and "Spotify" not in sent


# ---- settings ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("patch", [{"eyes_deny_titles": ["("]}, {"eyes_deny_processes": "x"},
                                   {"eyes_deny_titles": [3]}, {"glance_daily_cap": -1}, {"glance_daily_cap": True},
                                   {"glance_min_gap_s": 1}, {"glance_min_gap_s": "45"}, {"dont_look": "yes"},
                                   {"eyes_enabled": None}])
def test_eyes_settings_are_validated(patch):
    core, emitted, *_ = eyes_core()
    core.handle(SetSettings(patch=patch))
    assert isinstance(emitted[-1], ErrorMsg)


def test_eyes_settings_defaults():
    core, *_ = eyes_core()
    s = core.settings
    assert (s.eyes_enabled, s.glance_daily_cap, s.glance_min_gap_s, s.eyes_deny_titles, s.eyes_deny_processes,
            s.dont_look) == (True, 150, 45, [], [], False)


def test_min_gap_setting_reaches_the_policy():
    core, _, _, _, clock, vision, source, _ = eyes_core()
    core.handle(SetSettings(patch={"glance_min_gap_s": 120}))
    run(core, clock, 6)
    source.info = win(hwnd=2, title="b")
    run(core, clock, 100)
    assert len(vision.calls) == 1
    run(core, clock, 30)
    assert len(vision.calls) == 2


# ---- feelings question (carried from v0.2) -------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["do you have feelings?", "Do you actually have feelings", "can you feel sad?",
                                  "do you really have feelings riya"])
def test_a_feelings_question_gets_its_situation(text):
    core, _, _, local, *_ = eyes_core()
    core.vision = None
    list(core.reply(text))
    assert "They're asking whether you have feelings." in last_system(local)


def test_companion_layer_scopes_not_knowing_to_their_life():
    text = (Path(__file__).resolve().parents[2] / "riya" / "core" / "companion.md").read_text(encoding="utf-8")
    text = " ".join(text.split())
    assert "If they ask about something in their life that isn't in your memory, say you don't know." in text
    assert "If you don't remember, say so." not in text


def test_build_core_sets_dpi_awareness_first_and_gives_eyes_their_own_model(tmp_path, monkeypatch):
    import argparse

    from riya.core import __main__ as runner
    from riya.core.presence import FakeIdleSource
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "set_dpi_awareness", lambda: calls.append("dpi"))
    monkeypatch.setattr(runner, "make_idle_source", lambda: calls.append("idle") or FakeIdleSource())
    monkeypatch.setattr(runner, "make_window_source", lambda: calls.append("windows") or FakeWindowSource())
    monkeypatch.setenv("RIYA_ONLINE_API_KEY", "k")
    args = argparse.Namespace(db=str(tmp_path / "riya.db"), local_url="http://127.0.0.1:1", hash_embed=True,
                              legacy_memory=str(tmp_path / "none.json"))
    core = runner.build_core(args, lambda m: None)
    assert calls[0] == "dpi" and "windows" in calls
    assert core.vision is not core.online and core.vision.first_token_timeout_s == 20
    assert core.online.first_token_timeout_s == 8


# ---- no body, quitting ---------------------------------------------------------------------------------------------

def test_no_glance_or_screen_look_without_a_connected_body():
    present = [False]
    vision = FakeModel("vision", [NOTE, "Looks fine."])
    clock, source, grab = FakeClock(), FakeWindowSource(win()), Grabber()
    core, emitted, _, local = make_core(clock, vision=vision, window_source=source, grab=grab,
                                        body_present=lambda: present[0])
    run(core, clock, 400)
    assert grab.calls == [] and vision.calls == []
    list(core.reply("what's on my screen?"))
    assert grab.calls == [] and vision.calls == [] and "window isn't open" in situation(local)
    present[0] = True                                          # the body connects again
    run(core, clock, 10)
    assert len(grab.calls) == 1 and len(vision.calls) == 1


def test_a_body_is_assumed_in_tests_and_evaluation():
    core, *_ = eyes_core()
    assert core.body_present() is True


def test_quit_core_asks_the_runner_to_stop():
    from riya.core.protocol import QuitCore
    core, emitted, *_ = eyes_core()
    assert not core.quit_requested.is_set()
    n = len(emitted)
    core.handle(QuitCore())
    assert core.quit_requested.is_set() and emitted[n:] == []
