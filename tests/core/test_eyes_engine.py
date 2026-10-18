"""v0.3 eyes in the engine: glances on tick, screen questions, don't-look, notes in memory, prompt and protocol."""
import builtins
import io
import json
import threading
import time
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
        minimized=False, is_riya=False) -> WindowInfo:
    return WindowInfo(hwnd, title, process, rect, monitor, minimized, is_riya)


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
    assert core.memory.list_glances() == [] and core.policy.glances_today() == 0
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


def test_an_unparseable_note_also_backs_off():
    core, _, _, _, clock, vision, source, _ = eyes_core(vision=FakeModel("vision", ["no idea, sorry"]))
    run(core, clock, 6)
    source.info = win(hwnd=2, title="b")
    run(core, clock, 600)
    assert len(vision.calls) == 1 and core.memory.list_glances() == []


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
    source.info = win(hwnd=9, title="Riya", process="electron.exe", is_riya=True)
    run(core, clock, 30)
    list(core.reply("can you see this error?"))
    assert [i.hwnd for i in grab.calls] == [1]


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
    assert "covering your eyes" in last_system(local) and core.policy.glances_today() == 0


def test_a_failed_capture_is_said_honestly():
    def broken(info):
        raise OSError("mss failed")

    core, _, _, local, _, vision, _, _ = eyes_core(grab=broken)
    list(core.reply("what's wrong with this?"))
    assert "you can't see right now: the screen capture didn't work" in last_system(local) and vision.calls == []


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
