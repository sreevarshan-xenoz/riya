"""Run times in the agents hub (v0.12 §4.2, Resolution 12, A16, Review Focus 1)."""
import pytest

from riya.agents.hub import AgentHub
from riya.core.clock import FakeClock
from riya.core.protocol import AgentInfo

SID = "abcdef123456"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def hub(clock):
    return AgentHub(clock)


def ev(name, sid=SID, cwd=r"E:\work\riya", **kw):
    return {"session_id": sid, "hook_event_name": name, "cwd": cwd, **kw}


def at(clock, t0, t):
    clock.t = t0 + t


def test_a_claude_turn_is_timed_without_its_waiting(hub, clock):
    t0 = clock.t
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    at(clock, t0, 60)
    hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "ls"}))
    at(clock, t0, 160)
    hub.on_claude_event(ev("PostToolUse"))
    at(clock, t0, 400)
    (done,) = hub.on_claude_event(ev("Stop", last_assistant_message="ok"))
    assert done.kind == "finished" and done.run_s == 300


def test_a_turn_whose_start_she_never_saw_has_no_run_time(hub, clock):
    hub.on_claude_event(ev("PreToolUse", tool_name="Read"))
    clock.advance(200)
    (done,) = hub.on_claude_event(ev("Stop"))
    assert done.kind == "finished" and done.run_s is None


def test_a_stop_failure_carries_its_run_time(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    clock.advance(90)
    (failed,) = hub.on_claude_event(ev("StopFailure", error="rate_limit"))
    assert failed.kind == "failed" and failed.run_s == 90 and failed.error == "rate_limit"


def test_the_next_turn_starts_its_own_clock(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="one"))
    clock.advance(50)
    hub.on_claude_event(ev("Stop"))
    clock.advance(1000)
    hub.on_claude_event(ev("UserPromptSubmit", prompt="two"))
    clock.advance(70)
    (done,) = hub.on_claude_event(ev("Stop"))
    assert done.run_s == 70


def test_a_colab_run_is_timed_from_its_first_running(hub, clock):
    t0 = clock.t
    hub.on_colab("w3", "running", "epoch 1")
    at(clock, t0, 600)
    hub.on_colab("w3", "stalled", "no output")
    at(clock, t0, 900)
    hub.on_colab("w3", "running", "epoch 2")
    at(clock, t0, 2000)
    (done,) = hub.on_colab("w3", "done", "saved")
    assert done.kind == "finished" and done.run_s == 2000

    fresh = AgentHub(clock)  # a restart: the start was never seen
    (done,) = fresh.on_colab("w3", "done", "saved")
    assert done.run_s is None


def test_a_lost_run_has_no_time(hub, clock):
    hub.on_colab("w3", "running", "epoch 1")
    clock.advance(500)
    (lost,) = hub.on_colab("w3", "lost", "gone")
    assert lost.kind == "lost" and lost.run_s is None
    clock.advance(100)
    (done,) = hub.on_colab("w3", "done", "saved")
    assert done.run_s is None  # the lost run's start was cleared


def test_a_clock_jump_back_gives_no_run_time(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    clock.advance(-3600)
    (done,) = hub.on_claude_event(ev("Stop"))
    assert done.run_s is None
    hub.on_colab("w3", "running", "x")
    clock.advance(-10)
    (done,) = hub.on_colab("w3", "done", "x")
    assert done.run_s is None


def test_run_bucket_and_running(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    hub.on_claude_event(ev("UserPromptSubmit", sid="zzzzzz999", cwd="", prompt="go"))
    hub.on_colab("w3", "running", "x")
    assert hub.run_bucket(SID) == "claude:riya"
    assert hub.run_bucket("zzzzzz999") == "claude:*"
    assert hub.run_bucket("colab:w3") == "colab:w3"
    assert hub.run_bucket("nope") is None
    clock.advance(30)
    runs = {r.agent_id: r for r in hub.running()}
    assert set(runs) == {SID, "zzzzzz999", "colab:w3"}
    assert runs[SID].bucket == "claude:riya" and runs[SID].started_at == clock.t - 30
    hub.on_claude_event(ev("PreToolUse", sid="never-saw-start", tool_name="Read"))
    hub.on_claude_event(ev("Stop"))
    assert {r.agent_id for r in hub.running()} == {"zzzzzz999", "colab:w3"}


def test_the_snapshot_keys_are_unchanged(hub):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    assert set(hub.snapshot()[0]) == {"id", "kind", "name", "status", "since", "detail", "needs_attention"}
    assert set(hub.snapshot()[0]) == set(AgentInfo.model_fields)


def test_a_new_prompt_after_an_unanswered_permission_starts_clean(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit", prompt="one"))
    hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "rm"}))
    clock.advance(600)  # denied with Esc: no hook fires, she still looks waiting
    hub.on_claude_event(ev("UserPromptSubmit", prompt="two"))
    clock.advance(200)
    (done,) = hub.on_claude_event(ev("Stop"))
    assert done.run_s == 200


def test_a_stale_turn_is_not_timed(hub, clock):
    from riya.agents.hub import STALE_S
    hub.on_claude_event(ev("UserPromptSubmit", prompt="go"))
    clock.advance(STALE_S + 1)
    hub.tick()
    hub.on_claude_event(ev("PostToolUse"))
    clock.advance(10)
    (done,) = hub.on_claude_event(ev("Stop"))
    assert done.run_s is None
