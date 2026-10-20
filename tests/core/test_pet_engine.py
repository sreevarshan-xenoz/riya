"""v0.9 Task 3: Little Riya in the engine: agent_activity outside the core lock, presence on change, focus_agent ->
agent_window (looked up at click time, never focused by the core) and the pet settings. Offline: fake clock, fake
PeerOS, no real windows, tables or processes."""
import json
import pathlib
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from riya.agents.hooks import create_hook_app
from riya.agents.peer import FakePeerOS, Proc, TcpRow, TopWindow
from riya.core import engine, settings
from riya.core import protocol as proto
from riya.core.clock import FakeClock
from riya.core.protocol import AgentActivity, AgentsMsg, AgentWindow, FocusAgent, PresenceMsg, SetSettings, SettingsMsg
from riya.core.settings import Settings, apply_patch
from tests.core.test_agents_engine import ask, failure, hook, stop
from tests.fakes import make_core

TOKEN = "t" * 64
SID = "abc123def"


@pytest.fixture(autouse=True)
def noon(monkeypatch):
    monkeypatch.setattr(engine, "local_time", lambda t, tz=None: datetime(2027, 1, 15, 12, 0))


def activities(emitted, after=0):
    return [m for m in emitted[after:] if isinstance(m, AgentActivity)]


def pre(tool, sid=SID, **tool_input):
    return hook("PreToolUse", sid, tool_name=tool, tool_input=tool_input)


def started(**kw):
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, **kw)
    core.tick()
    emitted.clear()
    return core, emitted, src, clock


# --- agent_activity ------------------------------------------------------------------------------------------------

def test_pre_tool_use_becomes_agent_activity():
    core, emitted, src, clock = started()
    core.take_hook(pre("Edit", file_path="E:/riya/riya/core/engine.py"))
    core.tick()
    assert activities(emitted) == [AgentActivity(session=SID, label="editing engine.py", kind="edit", at=clock.now())]
    assert core.agents.snapshot()[0]["status"] == "busy"  # the hub saw it too
    assert not [m for m in emitted if m.type in ("say", "say_end")]


def test_activity_is_rate_limited_per_session_latest_wins():
    core, emitted, src, clock = started()
    for name in ("a.py", "b.py", "c.py"):
        core.take_hook(pre("Read", file_path=name))
    core.take_hook(pre("Read", sid="s2", file_path="z.py"))
    core.tick()
    got = [(a.session, a.label) for a in activities(emitted)]
    assert got == [(SID, "reading a.py"), ("s2", "reading z.py")]
    emitted.clear()
    clock.advance(1)
    core.tick()
    assert activities(emitted) == []
    clock.advance(0.5)
    core.tick()
    assert [a.label for a in activities(emitted)] == ["reading c.py"]
    clock.advance(10)
    core.tick()
    labels = [m.label for m in emitted if isinstance(m, AgentActivity)]
    assert "reading b.py" not in labels


def test_activity_flows_while_she_is_speaking():
    core, emitted, src, clock = started()
    core._lock.acquire()
    try:
        core.take_hook(stop())
        assert core.tick() is None
        assert activities(emitted) == [AgentActivity(session=SID, label="all done!", kind="done", at=clock.now())]
        assert not [m for m in emitted if m.type in ("say", "say_end")]  # the report waits for the lock
    finally:
        core._lock.release()


def test_every_reaction_kind_reaches_the_body():
    core, emitted, src, clock = started()
    payloads = [hook("UserPromptSubmit", "s1", prompt="do it"), ask("s2"), stop("s3"),
                hook("PostToolUseFailure", "s4", tool_name="Bash", error="x"),
                hook("StopFailure", "s5", error="rate_limit"), hook("SubagentStop", "s6")]
    for p in payloads:
        core.take_hook(p)
    core.tick()
    got = {a.session: (a.kind, a.label) for a in activities(emitted)}
    assert got == {"s1": ("think", "thinking…"), "s2": ("wave", "needs you! running tests"), "s3": ("done", "all done!"),
                   "s4": ("oops", "oops!"), "s5": ("fail", "that didn't work"), "s6": ("sparkle", "helper done")}


def test_a_pre_tool_use_right_after_a_permission_request_cannot_hide_needs_you():
    """Review (Task 1, Minor 5): "needs you" never rides only on the gated stream. The wave is sent at once even right
    after a tool bubble, a PreToolUse behind it can't replace it, and the hub's agents message carries needs_attention
    whatever the gate did."""
    core, emitted, src, clock = started()
    core.take_hook(pre("Bash", command="ls"))
    core.take_hook(ask())  # inside the 1.5 s gap of the PreToolUse before it
    core.take_hook(pre("Edit", file_path="late.py"))  # and a PreToolUse right behind it
    core.tick()
    got = activities(emitted)
    assert [(a.kind, a.label) for a in got] == [("run", "running ls"), ("wave", "needs you! running tests")]
    agents = [m for m in emitted if isinstance(m, AgentsMsg)]
    assert agents and agents[-1].items[0].needs_attention is True and agents[-1].items[0].status == "waiting"
    # the held PreToolUse is only a bubble: it can neither unsend the wave nor clear the hub's waiting state
    emitted.clear()
    clock.advance(2)
    core.tick()
    assert [a.kind for a in activities(emitted)] == ["edit"]
    assert core.agents.snapshot()[0]["needs_attention"] is True


def test_needs_you_reaches_the_body_in_agents_even_when_the_activity_step_is_busy():
    core, emitted, src, clock = started()
    assert core._activity_lock.acquire(blocking=False)  # another thread is in the activity step: it is skipped
    try:
        core.take_hook(ask())
        core.tick()
    finally:
        core._activity_lock.release()
    assert activities(emitted) == []
    agents = [m for m in emitted if isinstance(m, AgentsMsg)]
    assert agents and agents[-1].items[0].needs_attention is True
    core.tick()
    assert [a.kind for a in activities(emitted)] == ["wave"]  # and the pose follows on the next tick


def test_activity_step_never_raises_and_one_bad_payload_does_not_stop_the_rest():
    core, emitted, src, clock = started()
    for bad in (None, "x", 7, [], {"hook_event_name": "PreToolUse"}, {"session_id": "s", "hook_event_name": 5}):
        core.take_hook(bad)
    core.take_hook(stop())
    core.tick()
    assert [a.kind for a in activities(emitted)] == ["done"]


# --- presence ------------------------------------------------------------------------------------------------------

def test_presence_message_only_on_change():
    core, emitted, src, clock = started()
    types_before = [m.type for m in core.hello_messages()]
    for _ in range(3):
        clock.advance(1)
        core.tick()
    assert [m for m in emitted if isinstance(m, PresenceMsg)] == []
    src.idle = 200
    clock.advance(1)
    core.tick()
    assert [m for m in emitted if isinstance(m, PresenceMsg)] == [PresenceMsg(state="break")]
    for _ in range(3):
        clock.advance(1)
        src.idle = 201
        core.tick()
    assert len([m for m in emitted if isinstance(m, PresenceMsg)]) == 1
    src.is_fullscreen, src.idle = True, 0
    clock.advance(1)
    core.tick()
    assert [m.state for m in emitted if isinstance(m, PresenceMsg)] == ["break", "gaming"]
    assert [m.type for m in core.hello_messages()] == types_before  # hello's order is untouched
    assert "presence" not in types_before


# --- focus_agent -> agent_window -----------------------------------------------------------------------------------

TERMINAL = dict(rows=[TcpRow(50000, 8766, 4242)],
                procs=[Proc(4242, 4000, "node.exe"), Proc(4000, 10, "windowsterminal.exe"),
                       Proc(10, 0, "explorer.exe")],
                wins=[TopWindow(77, 4000, "riya \u2014 Windows Terminal")])


def hooked(core, event="UserPromptSubmit"):
    c = TestClient(create_hook_app(TOKEN, core.take_hook, peers=core.peers))
    body = {"hook_event_name": event, "session_id": SID, "cwd": "E:/riya", "prompt": "hi"}
    assert c.post("/hook", content=json.dumps(body), headers={"X-Riya-Token": TOKEN}).status_code == 200


def test_focus_agent_replies_with_the_window_to_its_connection():
    core, emitted, src, local = make_core(peer_os=FakePeerOS(**TERMINAL))
    hooked(core)
    got = []
    core.handle(FocusAgent(id=SID), reply=got.append)
    assert got == [AgentWindow(id=SID, hwnd=77)]
    assert emitted == []


def test_focus_agent_unknown_or_unresolved_is_null():
    core = make_core(peer_os=FakePeerOS(**TERMINAL))[0]
    hooked(core)
    got = []
    core.handle(FocusAgent(id="nobody"), reply=got.append)
    assert got == [AgentWindow(id="nobody", hwnd=None)]
    nowin = make_core(peer_os=FakePeerOS(rows=TERMINAL["rows"], procs=TERMINAL["procs"]))[0]
    hooked(nowin)
    got.clear()
    nowin.handle(FocusAgent(id=SID), reply=got.append)
    assert got == [AgentWindow(id=SID, hwnd=None)]
    broken = FakePeerOS(**TERMINAL)
    core2 = make_core(peer_os=broken)[0]
    hooked(core2)
    broken.raises = {"windows"}
    got.clear()
    core2.handle(FocusAgent(id=SID), reply=got.append)
    assert got == [AgentWindow(id=SID, hwnd=None)]
    broken.raises = {"created"}
    core2.handle(FocusAgent(id=SID), reply=got.append)
    assert got[-1] == AgentWindow(id=SID, hwnd=None)
    assert got[-1].model_dump()["hwnd"] is None


def test_focus_agent_after_the_agent_exited_or_its_pid_was_reused_is_null():
    """Review (Task 2, Minor 3): the pid cached at hook time is checked again at click time."""
    os_ = FakePeerOS(**TERMINAL, created={4242: 100.0, 4000: 50.0, 10: 1.0})
    core = make_core(peer_os=os_)[0]
    hooked(core)
    got = []
    core.handle(FocusAgent(id=SID), reply=got.append)
    assert got == [AgentWindow(id=SID, hwnd=77)]
    os_.times[4242] = 900.0  # the pid now belongs to a younger process: a reuse, not our agent
    core.handle(FocusAgent(id=SID), reply=got.append)
    assert got[-1] == AgentWindow(id=SID, hwnd=None)
    os_.times[4242] = 100.0
    os_.procs = [p for p in os_.procs if p.pid != 4242]  # and when it exited
    os_.times.pop(4242)
    core.handle(FocusAgent(id=SID), reply=got.append)
    assert got[-1] == AgentWindow(id=SID, hwnd=None)


def test_focus_agent_does_not_walk_into_a_reused_parent():
    """A parent born after its child is a reused pid: the chain stops there, so its window is never picked."""
    os_ = FakePeerOS(rows=TERMINAL["rows"], procs=TERMINAL["procs"], wins=TERMINAL["wins"],
                     created={4242: 100.0, 4000: 5000.0, 10: 1.0})
    core = make_core(peer_os=os_)[0]
    hooked(core)
    got = []
    core.handle(FocusAgent(id=SID), reply=got.append)
    assert got == [AgentWindow(id=SID, hwnd=None)]


def test_nothing_focuses_without_a_click(monkeypatch):
    from riya.agents import peer
    os_ = FakePeerOS(**TERMINAL)
    core, emitted, src, local = make_core(peer_os=os_)

    def boom(*a, **k):
        raise AssertionError("the core focused a window")

    monkeypatch.setattr(peer, "focus", boom)
    events = ["SessionStart", "UserPromptSubmit", "PreToolUse", "PermissionRequest", "PostToolUse",
              "PostToolUseFailure", "Stop", "StopFailure", "SubagentStop", "Notification", "SessionEnd"]
    for i in range(200):
        sid = f"s{i % 3}"
        core.take_hook({"hook_event_name": events[i % len(events)], "session_id": sid, "cwd": "E:/riya",
                        "tool_name": "Bash", "tool_input": {"command": "ls"}, "notification_type": "permission_prompt"})
    for _ in range(30):
        core.clock.advance(1)
        core.tick()
    assert not [m for m in emitted if isinstance(m, AgentWindow)]
    assert set(os_.calls) <= {"tcp_rows"}, os_.calls  # never windows, processes or created
    src = pathlib.Path(engine.__file__).read_text(encoding="utf-8")
    assert "SetForegroundWindow" not in src and "peer.focus" not in src and "import focus" not in src


# --- settings -----------------------------------------------------------------------------------------------------

def test_pet_settings_defaults_and_validation():
    s = Settings()
    assert (s.pet_enabled, s.pet_scale, s.reduced_motion, s.pet_walks_when) == (True, 0.6, "system", "idle")
    assert settings.PET_SCALE_RANGE == (0.4, 1.0) and settings.REDUCED_MOTION == ("system", "on", "off")
    assert settings.PET_WALKS_WHEN == ("idle", "always_but_focus")
    for scale in (0.4, 1, 1.0):
        assert apply_patch(s, {"pet_scale": scale}).pet_scale == scale
    assert apply_patch(s, {"pet_walks_when": "always_but_focus"}).pet_walks_when == "always_but_focus"
    assert apply_patch(s, {"reduced_motion": "on", "pet_enabled": False}).pet_enabled is False
    for bad in ({"pet_scale": 0.39}, {"pet_scale": 1.01}, {"pet_scale": True}, {"pet_scale": "0.6"},
                {"reduced_motion": "maybe"}, {"pet_walks_when": "always"}, {"pet_walks_when": "breaks_only"},
                {"pet_enabled": 1}):
        with pytest.raises(ValueError):
            apply_patch(s, bad)


def test_set_pet_scale_reaches_the_body():
    core, emitted, src, clock = started()
    core.handle(SetSettings(patch={"pet_scale": 0.8}))
    got = [m for m in emitted if isinstance(m, SettingsMsg)]
    assert got and got[-1].settings["pet_scale"] == 0.8


def test_the_hook_receiver_hands_hooks_to_take_hook():
    from riya.core import __main__ as runner
    core = make_core()[0]
    seen = {}

    class Made:
        def __init__(self, enqueue, peers=None):
            seen.update(enqueue=enqueue, peers=peers)

        def start(self):
            return self

    runner.start_receiver(core, make=Made)
    assert seen["enqueue"] == core.take_hook and seen["peers"] is core.peers

