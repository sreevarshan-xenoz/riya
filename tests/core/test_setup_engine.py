"""v0.7 Task 4: Health and setup through the engine: the confirm round trip, the key staying in the core, Health off the
core lock, the onboarded flag, voices and autostart.

Nothing here reaches the real ~/.claude/settings.json, %LOCALAPPDATA%\\Riya, the Startup folder, the mic, audio, setx
or the network: every path is under tmp_path, the installer's runner is a recorder and every model is a FakeModel."""
import base64
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from riya.agents import install_hooks as ih
from riya.core import engine, health
from riya.core import secrets as keystore
from riya.core.clock import FakeClock
from riya.core.eyes.capture import FakeWindowSource, WindowInfo
from riya.core.models import ModelUnavailable, ToolReply
from riya.core.protocol import (ErrorMsg, GetMemories, HealthGet, HealthMsg, OnboardingDone, ProtocolError,
                                SetSettings, SettingsMsg, SetupAction, SetupConfirm, SetupRequest, SetupResult,
                                VoicesMsg, dump, parse_body_message)
from riya.core.secrets import KEY_VAR, InvalidKey, OnlineKey
from riya.core.setup import (CANCELLED, CONFIRM_GATED, EYES_IMAGE, NONCE_REFUSED, REFUSED_SUMMARY, SAMPLE_TEXT,
                             TESTS_KEY, Setup, SetupDeps)
from tests.fakes import FakeModel, make_core, wait_for

CANARY = "CANARY-k3y-0123456789abcdef"
DOTENV = "DOTENV-1234567890"


def keyed(tmp_path) -> OnlineKey:
    return OnlineKey(tmp_path / "secrets.env", DOTENV, ".env")


def deps(tmp_path, key=None, replies=None, fail=None, **over) -> SetupDeps:
    """SetupDeps on tmp paths: a recording runner (`runs`), voices alba and eve, and a recording make_model (`made`,
    `models`) that, like the real one, builds nothing without a key."""
    runs, made, models = [], [], []

    def runner(cmd, **kw):
        runs.append((cmd, kw))
        return SimpleNamespace(returncode=0)

    def make_model(env, name, timeout_s):
        made.append((name, timeout_s))
        if not env.get(KEY_VAR):
            return None
        m = FakeModel(name, (replies or {}).get(name, ("ok",)), fail=fail)
        m.api_key = env[KEY_VAR]
        models.append(m)
        return m

    kw = dict(key=key or OnlineKey(tmp_path / "secrets.env", None, None), environ={},
              claude_settings=tmp_path / "claude" / "settings.json", token_path=tmp_path / "tok" / "hook_token",
              runner=runner, voices=lambda: ["alba", "eve"], make_model=make_model)
    kw.update(over)
    d = SetupDeps(**kw)
    d.runs, d.made, d.models = runs, made, models
    return d


def confirms(emitted, after=0):
    return [m for m in emitted[after:] if isinstance(m, SetupConfirm)]


def results(emitted, after=0):
    return [m for m in emitted[after:] if isinstance(m, SetupResult)]


def ask(core, emitted, action, **args) -> SetupConfirm:
    mark = len(emitted)
    core.handle(SetupRequest(action=action, args=args))
    return wait_for(emitted, SetupConfirm, after=mark)


def run(core, emitted, action, **args) -> SetupResult:
    """A plain action from a click, waited for."""
    mark = len(emitted)
    core.handle(SetupRequest(action=action, args=args))
    return wait_for(emitted, SetupResult, after=mark)


def confirmed(core, emitted, action) -> SetupResult:
    """A gated action: the confirm, then Yes, waited for."""
    c = ask(core, emitted, action)
    mark = len(emitted)
    core.handle(SetupAction(nonce=c.nonce, yes=True))
    return wait_for(emitted, SetupResult, after=mark)


def pos(emitted, m) -> int:
    """m's index in emitted by identity (equal messages compare equal, so list.index can find an earlier one)."""
    return next(i for i, x in enumerate(emitted) if x is m)


def health_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "riya-health")


def wait_health_idle(core, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while True:
        with core._health_lock:
            if not core._health_running:
                return
        assert time.monotonic() < deadline, "the health worker never finished"
        time.sleep(0.01)


# ---- Review Focus 2: an outward action only after a fresh confirm ---------------------------------------------------

def test_hooks_connect_needs_a_confirm_and_runs_once_on_temp_paths(tmp_path):
    clock = FakeClock()
    d = deps(tmp_path)
    core, emitted, *_ = make_core(clock, setup_deps=d)
    core.handle(SetupRequest(action="hooks_connect"))
    (c,) = confirms(emitted)
    assert c.action == "hooks_connect" and c.text.endswith(ih.RESTART_LINE)
    assert not d.claude_settings.exists() and d.runs == [] and results(emitted) == []

    mark = len(emitted)
    core.handle(SetupAction(nonce=c.nonce, yes=True))
    r = wait_for(emitted, SetupResult, after=mark)
    assert r.action == "hooks_connect" and r.ok
    assert ih.riya_events(ih.read_settings(d.claude_settings)) and len(d.runs) == 1

    mark = len(emitted)                                   # the same nonce again
    core.handle(SetupAction(nonce=c.nonce, yes=True))
    assert [m.message for m in emitted[mark:] if isinstance(m, ErrorMsg)] == [NONCE_REFUSED]
    time.sleep(0.1)
    assert len(d.runs) == 1 and results(emitted, mark) == []
    newest = core.hands.actions()[0]
    assert (newest.tool, newest.summary, newest.outcome) == ("setup", REFUSED_SUMMARY, "refused")

    stale = ask(core, emitted, "hooks_connect")           # 60 s old: refused
    clock.advance(60)
    mark = len(emitted)
    core.handle(SetupAction(nonce=stale.nonce, yes=True))
    assert [m.message for m in emitted[mark:] if isinstance(m, ErrorMsg)] == [NONCE_REFUSED]
    time.sleep(0.1)
    assert len(d.runs) == 1

    fresh = ask(core, emitted, "hooks_connect")           # 59.9 s old: still good
    clock.advance(59.9)
    mark = len(emitted)
    core.handle(SetupAction(nonce=fresh.nonce, yes=True))
    assert wait_for(emitted, SetupResult, after=mark).ok and len(d.runs) == 2

    before = d.claude_settings.read_bytes()               # No: nothing runs
    no = ask(core, emitted, "hooks_connect")
    mark = len(emitted)
    core.handle(SetupAction(nonce=no.nonce, yes=False))
    r = wait_for(emitted, SetupResult, after=mark)
    assert (r.action, r.ok, r.detail) == ("hooks_connect", False, CANCELLED)
    time.sleep(0.1)
    assert d.claude_settings.read_bytes() == before and len(d.runs) == 2


def test_nothing_outward_runs_without_a_confirm(tmp_path, monkeypatch):
    performed = []
    monkeypatch.setattr(Setup, "perform", lambda self, action, args: performed.append(action))
    d = deps(tmp_path, key=keyed(tmp_path))
    online, vision = FakeModel("online"), FakeModel("vision")
    core, emitted, *_ = make_core(online=online, vision=vision, setup_deps=d)
    for action in CONFIRM_GATED:
        mark = len(emitted)
        core.handle(SetupRequest(action=action))
        new = emitted[mark:]
        assert len(new) == 1 and isinstance(new[0], SetupConfirm) and new[0].action == action, action
    time.sleep(0.2)
    assert results(emitted) == []
    assert performed == [] and d.made == [] and d.runs == []
    assert online.calls == online.tool_calls == vision.calls == vision.tool_calls == []


def test_model_tools_and_hooks_never_reach_setup(tmp_path, monkeypatch):
    called = []
    for name in ("request", "answer", "perform"):
        monkeypatch.setattr(Setup, name, lambda self, *a, _name=name, **kw: called.append(_name))
    as_json = '{"type": "setup_request", "action": "hooks_connect"}'
    online = FakeModel("online", (as_json,), tool_replies=(ToolReply([("hooks_connect", "{}")], None),))
    clock = FakeClock()
    core, emitted, _, local = make_core(clock, local_replies=(as_json,), online=online, setup_deps=deps(tmp_path))
    list(core.reply("connect claude code"))
    list(core.reply("test your eyes"))
    list(core.reply("run hooks_connect"))                 # a command: the online model is offered her tools
    assert online.tool_calls and local.calls
    core.agent_inbox.put({"hook_event_name": "Stop", "session_id": "abc123def", "cwd": "E:/riya",
                          "last_assistant_message": as_json})
    for _ in range(30):
        clock.advance(1)
        if (said := core.tick()) is not None:
            list(said)
    assert called == []
    assert not any(isinstance(m, (SetupConfirm, SetupResult)) for m in emitted)


# ---- Review Focus 1: the key never leaves the core ------------------------------------------------------------------

def test_the_key_never_leaves_the_core(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    keystore.install_redaction()
    try:
        d = deps(tmp_path, fail=ModelUnavailable(f"HTTP 401 {CANARY}"))
        core, emitted, *_ = make_core(setup_deps=d)
        assert run(core, emitted, "online_key_set", key=CANARY).ok

        mark = len(emitted)
        core.handle(HealthGet())
        h = wait_for(emitted, HealthMsg, after=mark)
        (row,) = [i for i in h.items if i.key == "online"]
        assert "••••" + CANARY[-4:] in row.detail and h.key_source == "app"

        for action in ("online_test", "eyes_test"):       # the failing fake's error holds the key
            r = confirmed(core, emitted, action)
            assert r.action == action and not r.ok
        assert {name for name, _ in d.made} >= {"online-test", "eyes-test"}

        hello = core.hello_messages()
        list(core.reply("hi"))
        list(core.reply("explain how my keys work"))       # routed online: the failing fake's error is logged
        assert any("online model failed" in r.getMessage() for r in caplog.records)
        core.handle(GetMemories())
        sinks = {
            "emitted": "\n".join(dump(m) for m in emitted),
            "hello": "\n".join(dump(m) for m in hello),
            "logs": caplog.text,
            "db": "\n".join(core.kv.conn.iterdump()),   # facts, transcripts, KV and the action log
            "nonces": str(core.setup.nonces.__dict__),
        }
        for name, text in sinks.items():
            assert CANARY not in text, name
        assert (tmp_path / "secrets.env").read_text(encoding="utf-8") == f"{KEY_VAR}={CANARY}\n"
    finally:
        keystore.uninstall_redaction()


def test_a_bad_setup_request_never_echoes_its_args(tmp_path):
    raw = '{"type":"setup_request","action":"online_key_set","args":{"key":["' + CANARY + '"]}}'
    with pytest.raises(ProtocolError) as e:
        parse_body_message(raw)
    assert str(e.value) == "invalid setup_request"
    core, emitted, *_ = make_core(setup_deps=deps(tmp_path))
    r = run(core, emitted, "online_key_set", key=CANARY + " x")
    assert not r.ok and r.detail == InvalidKey.BAD_CHARS
    assert all(CANARY not in dump(m) for m in emitted)


# ---- Review Focus 5: set and clear rebuild both models --------------------------------------------------------------

def test_key_set_and_clear_rebuild_both_models(tmp_path):
    d = deps(tmp_path, key=keyed(tmp_path))
    core, emitted, *_ = make_core(online=FakeModel("old-online"), vision=FakeModel("old-vision"), setup_deps=d)
    start = (core.online, core.vision)
    assert start[0] is not None and start[1] is not None
    assert run(core, emitted, "online_key_set", key=CANARY).ok
    assert core.online.api_key == CANARY and core.vision.api_key == CANARY
    assert core.online is not core.vision and core.online is not start[0] and core.vision is not start[1]
    assert core.router.online is core.online
    assert (core.online.name, core.vision.name) == ("online", "vision")
    mark = len(emitted)
    core.handle(HealthGet())
    assert wait_for(emitted, HealthMsg, after=mark).key_source == "app"

    saved = (core.online, core.vision)
    r = run(core, emitted, "online_key_clear")
    assert r.ok
    assert core.online.api_key == DOTENV and core.vision.api_key == DOTENV
    assert core.online is not saved[0] and core.vision is not saved[1] and core.router.online is core.online
    wait_health_idle(core)
    mark = len(emitted)
    core.handle(HealthGet())
    assert wait_for(emitted, HealthMsg, after=mark).key_source == ".env"

    after_clear = (core.online, core.vision)
    r = run(core, emitted, "online_key_clear")            # a .env key can't be cleared from Riya
    assert not r.ok and ".env" in r.detail
    assert (core.online, core.vision) == after_clear and core.online is after_clear[0]


# ---- Review Focus 3: Health never takes the core lock ---------------------------------------------------------------

def test_health_runs_off_the_core_lock(tmp_path):
    asked = []

    def voices():
        asked.append(1)
        return ["alba", "eve"]

    core, emitted, *_ = make_core(setup_deps=deps(tmp_path, voices=voices))
    core.handle(HealthGet())
    h = wait_for(emitted, HealthMsg)
    v = wait_for(emitted, VoicesMsg)
    assert v == VoicesMsg(items=[], current="alba") and pos(emitted, v) < pos(emitted, h)
    assert asked == []                                    # the health thread never lists (imports) the TTS voices
    wait_health_idle(core)

    core.setup.load_voices()
    gathered, real_ctx = [], core._health_ctx

    def ctx():
        gathered.append((threading.current_thread().name, core._lock.locked()))
        return real_ctx()

    core._health_ctx = ctx
    with core._lock:
        mark = len(emitted)
        t0 = time.monotonic()
        core.handle(HealthGet())
        assert time.monotonic() - t0 < 0.5
        h = wait_for(emitted, HealthMsg, after=mark)      # it arrived while the test still held the lock
    assert gathered == [("riya-health", True)]            # gathered on its own thread, the core lock held elsewhere
    core._health_ctx = real_ctx
    assert [i.key for i in h.items] == list(health.ROWS)
    v = wait_for(emitted, VoicesMsg, after=mark)
    assert v == VoicesMsg(items=["alba", "eve"], current="alba") and pos(emitted, v) < pos(emitted, h)
    wait_health_idle(core)

    release = threading.Event()
    core.local_url = "http://127.0.0.1:1"
    core.health_http = lambda url, timeout_s: release.wait(10) and b""
    core.health_timeout_s = 0.3
    most, sampling = [0], threading.Event()

    def sample():
        while not sampling.is_set():
            most[0] = max(most[0], health_threads())
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        mark = len(emitted)
        t0 = time.monotonic()
        core.handle(HealthGet())
        for _ in range(3):                                # mid-pass: no second thread, one coalesced rerun
            core.handle(HealthGet())
        first = wait_for(emitted, HealthMsg, after=mark, timeout_s=2)
        assert time.monotonic() - t0 < 2
        (local,) = [i for i in first.items if i.key == "local"]
        assert local.detail == "No answer within 0.3 s."
        second = wait_for(emitted, HealthMsg, after=pos(emitted, first) + 1, timeout_s=2)
        time.sleep(1.0)
        assert [m for m in emitted[mark:] if isinstance(m, HealthMsg)] == [first, second]
        assert pos(emitted, second) > pos(emitted, first)
        wait_health_idle(core)
    finally:
        sampling.set()
        sampler.join(2)
        release.set()
    assert most[0] == 1


# ---- voices, the eye test, autostart -----------------------------------------------------------------------------

def test_voices_and_voice_choose(tmp_path):
    core, emitted, *_ = make_core(setup_deps=deps(tmp_path))
    core.setup.load_voices()
    r = run(core, emitted, "voice_choose", voice="eve")
    assert r.ok and core.settings.voice == "eve"
    v = wait_for(emitted, VoicesMsg, after=pos(emitted, r))
    assert v == VoicesMsg(items=["alba", "eve"], current="eve")

    spoken = []
    core.voice_sampler = lambda voice, text: spoken.append((voice, text))
    assert run(core, emitted, "voice_sample", voice="eve").ok
    assert spoken == [("eve", SAMPLE_TEXT)]
    core.voice_sampler = None
    r = run(core, emitted, "voice_sample", voice="eve")
    assert not r.ok and r.detail == "Her voice isn't running (text only)."


def test_eyes_test_never_grabs_the_screen(tmp_path):
    seen = []

    class Windows(FakeWindowSource):
        def foreground(self):
            seen.append("foreground")
            return super().foreground()

    win = WindowInfo(hwnd=1, title="secret.txt", process="notepad.exe", rect=(0, 0, 800, 600),
                     monitor=(0, 0, 1920, 1080), minimized=False, is_riya=False)
    d = deps(tmp_path, key=keyed(tmp_path), replies={"eyes-test": ("42",)})
    core, emitted, *_ = make_core(window_source=Windows(win), grab=lambda w: seen.append("grab") or b"",
                                  setup_deps=d)
    r = confirmed(core, emitted, "eyes_test")
    assert r.ok and r.action == "eyes_test"
    assert seen == []
    (model,) = [m for m in d.models if m.name == "eyes-test"]
    (msgs,) = model.calls
    url = msgs[0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == Path(EYES_IMAGE).read_bytes()
    assert core.kv.get(TESTS_KEY)["eyes_test"]["ok"] is True


def test_autostart_action_uses_the_guarded_startup_dir(tmp_path, monkeypatch):
    core, emitted, *_ = make_core(setup_deps=deps(tmp_path))
    core.startup_dir = tmp_path / "startup"
    shortcut = core.startup_dir / "riya.cmd"
    assert run(core, emitted, "autostart", on=True).ok and shortcut.exists()
    assert run(core, emitted, "autostart", on=False).ok and not shortcut.exists()

    assert run(core, emitted, "autostart", on=True).ok    # the Health repair: on, but the shortcut went missing
    shortcut.unlink()
    r = run(core, emitted, "autostart", on=True)
    assert (r.ok, r.detail) == (True, "Starts with Windows now.") and shortcut.exists()

    def no_appdata(*a, **kw):
        raise KeyError("APPDATA")

    monkeypatch.setattr(engine, "set_autostart", no_appdata)
    r = run(core, emitted, "autostart", on=True)
    assert (r.ok, r.detail) == (False, "Couldn't change the Startup shortcut.")
    mark = len(emitted)
    core.handle(SetSettings(patch={"autostart": True}))
    new = emitted[mark:]
    (err,) = [m for m in new if isinstance(m, ErrorMsg)]
    assert err.message.startswith("couldn't change autostart:")
    settings = [i for i, m in enumerate(new) if isinstance(m, SettingsMsg)]
    assert settings and settings[0] < new.index(err)


# ---- fix round 1: racing patches, one read of the online model, an unreadable Health field --------------------------

def test_racing_patches_never_leave_the_older_state_behind(tmp_path):
    """A (voice eve) is held just as it sends its SettingsMsg, while B (online off) is patched from another thread.
    However they interleave, the runtime and the last SettingsMsg carry both."""
    core, emitted, *_ = make_core(setup_deps=deps(tmp_path))
    b_sent, done = threading.Event(), []
    real_emit = core.emit

    def emit(m):
        if isinstance(m, SettingsMsg) and m.settings["voice"] == "eve" and m.settings["online_enabled"]:
            threading.Thread(target=lambda: done.append(core.patch_settings({"online_enabled": False}))).start()
            b_sent.wait(0.5)  # with the fix B waits for A's lock, so this times out; without it B runs to the end
        real_emit(m)
        if isinstance(m, SettingsMsg) and not m.settings["online_enabled"]:
            b_sent.set()

    core.emit = emit
    a = threading.Thread(target=lambda: done.append(core.patch_settings({"voice": "eve"})))
    a.start()
    a.join(5)
    deadline = time.monotonic() + 5
    while len(done) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(done) == 2
    last = [m for m in emitted if isinstance(m, SettingsMsg)][-1].settings
    assert (last["voice"], last["online_enabled"]) == ("eve", False)
    assert core.router.online_enabled is False
    assert (core.settings.voice, core.settings.online_enabled) == ("eve", False)


def test_consolidation_reads_the_online_model_once(tmp_path):
    from dataclasses import replace

    from riya.core.settings import save_settings
    core, *_ = make_core(setup_deps=deps(tmp_path))

    class Cleared(FakeModel):
        def __bool__(self):  # the key is cleared right as the consolidator asks
            core.swap_online(None, None)
            return True

    model = Cleared("online")
    core.online = core.router.online = model
    save_settings(core.kv, replace(core.settings, online_consolidation=True))
    assert core._consolidation_model() is model
    assert core.online is None and core._consolidation_model() is core.local


def test_an_unreadable_health_field_warns_only_its_row(tmp_path, monkeypatch):
    from dataclasses import replace

    from riya.core.settings import save_settings
    core, emitted, *_ = make_core(setup_deps=deps(tmp_path))
    save_settings(core.kv, replace(core.settings, autostart=True))
    monkeypatch.delenv("APPDATA")                         # core.startup_dir is None: the real default can't resolve
    core.handle(HealthGet())
    h = wait_for(emitted, HealthMsg)
    rows = {i.key: i for i in h.items}
    assert [i.key for i in h.items] == list(health.ROWS)
    assert (rows["autostart"].status, rows["autostart"].detail) == ("warn", "This check failed (KeyError).")
    assert all("This check failed" not in i.detail for k, i in rows.items() if k != "autostart")
    assert rows["memory"].status == "ok" and rows["local"].detail == "No local server is configured."
