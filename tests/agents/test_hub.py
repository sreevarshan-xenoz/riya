import pytest

from riya.agents.hub import AgentHub, summarize
from riya.core.clock import FakeClock

SID = "abcdef123456"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def hub(clock):
    return AgentHub(clock)


def ev(name, sid=SID, **kw):
    return {"session_id": sid, "hook_event_name": name, "cwd": r"C:\x\riya", **kw}


def state(hub, aid=SID):
    return next(s for s in hub.snapshot() if s["id"] == aid)


def test_summarize_strips_control_and_collapses():
    assert summarize("a\x00b\x1b[31m  c\n\td\u202e") == "ab[31m c d"
    assert summarize(None) == "None"
    assert summarize("x" * 500, 80) == "x" * 79 + "\u2026"
    assert len(summarize("word " * 100)) <= 80
    assert summarize("short") == "short"


def test_unknown_session_is_created_and_named(hub):
    assert hub.on_claude_event(ev("UserPromptSubmit", prompt="do it")) == []
    s = state(hub)
    assert (s["kind"], s["status"], s["name"], s["detail"]) == ("claude", "busy", "riya abcdef", "do it")
    hub.on_claude_event({"session_id": "zzzzzzzz", "hook_event_name": "SessionStart", "cwd": "/x/y/proj/"})
    assert state(hub, "zzzzzzzz")["name"] == "proj zzzzzz"


@pytest.mark.parametrize("bad", [
    {}, {"session_id": SID}, {"hook_event_name": "Stop"}, {"session_id": "", "hook_event_name": "Stop"},
    {"session_id": 5, "hook_event_name": "Stop"}, {"session_id": SID, "hook_event_name": "Bogus"}, "nope", None,
])
def test_malformed_payload_changes_nothing(hub, bad):
    assert hub.on_claude_event(bad) == []
    assert hub.snapshot() == []


def test_session_start_and_idle_prompt(hub):
    hub.on_claude_event(ev("SessionStart"))
    assert state(hub)["status"] == "idle"
    hub.on_claude_event(ev("UserPromptSubmit"))
    hub.on_claude_event(ev("Notification", notification_type="idle_prompt"))
    assert state(hub)["status"] == "idle"
    assert hub.on_claude_event(ev("Notification", notification_type="auth_success")) == []


def test_session_start_midturn_keeps_busy(hub):
    hub.on_claude_event(ev("UserPromptSubmit"))
    hub.on_claude_event(ev("SessionStart", source="compact"))
    assert state(hub)["status"] == "busy"


def test_permission_request_waits_with_command_summary(hub):
    hub.on_claude_event(ev("UserPromptSubmit"))
    out = hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "npm  test\n--all"}))
    assert [(e.kind, e.agent_id, e.name, e.text) for e in out] == [("waiting", SID, "riya abcdef", "Bash: npm test --all")]
    s = state(hub)
    assert s["status"] == "waiting" and s["needs_attention"] and s["detail"] == "Bash: npm test --all"


def test_waiting_detail_file_path_then_json(hub):
    out = hub.on_claude_event(ev("PermissionRequest", tool_name="Edit", tool_input={"file_path": "/a/b.py", "x": 1}))
    assert out[0].text == "Edit: /a/b.py"
    out = hub.on_claude_event(ev("PermissionRequest", tool_name="Task", tool_input={"a": 1, "b": "é"}))
    assert out[0].text == 'Task: {"a":1,"b":"é"}'


def test_repeated_identical_waiting_does_not_reemit(hub):
    req = ev("PermissionRequest", tool_name="Bash", tool_input={"command": "ls"})
    assert len(hub.on_claude_event(req)) == 1
    assert hub.on_claude_event(req) == []
    assert hub.on_claude_event(ev("Notification", notification_type="permission_prompt", message="needs permission")) == []
    assert state(hub)["detail"] == "Bash: ls"
    assert len(hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "rm x"}))) == 1


def test_permission_prompt_notification_alone_waits(hub):
    out = hub.on_claude_event(ev("Notification", notification_type="permission_prompt", message="Claude needs your permission to use Bash"))
    assert [e.kind for e in out] == ["waiting"]
    assert "permission" in out[0].text
    assert state(hub)["needs_attention"]


def test_stop_finishes_and_clears_attention(hub):
    hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "ls"}))
    out = hub.on_claude_event(ev("Stop", last_assistant_message="All done.\n" + "x" * 200))
    assert [e.kind for e in out] == ["finished"] and len(out[0].text) <= 80
    s = state(hub)
    assert s["status"] == "finished" and not s["needs_attention"] and len(s["detail"]) <= 80
    assert hub.on_claude_event(ev("Stop", last_assistant_message="again")) == []


def test_stop_failure_fails_once(hub):
    out = hub.on_claude_event(ev("StopFailure", error="rate_limit"))
    assert [(e.kind, e.text) for e in out] == [("failed", "rate_limit")]
    s = state(hub)
    assert s["status"] == "failed" and s["needs_attention"]
    assert hub.on_claude_event(ev("StopFailure", error="rate_limit")) == []


def test_busy_again_clears_attention(hub):
    hub.on_claude_event(ev("StopFailure", error="boom"))
    hub.on_claude_event(ev("UserPromptSubmit", prompt="retry"))
    s = state(hub)
    assert s["status"] == "busy" and not s["needs_attention"]


def test_idle_prompt_does_not_hide_failure_or_wait(hub):
    hub.on_claude_event(ev("StopFailure", error="boom"))
    hub.on_claude_event(ev("Notification", notification_type="idle_prompt"))
    assert state(hub)["status"] == "failed"


def test_three_tool_failures_in_a_turn_fail(hub):
    hub.on_claude_event(ev("UserPromptSubmit"))
    assert hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e1")) == []
    assert state(hub)["status"] == "busy"
    assert hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e2")) == []
    out = hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e3"))
    assert [e.kind for e in out] == ["failed"] and "3 tool failures" in out[0].text
    s = state(hub)
    assert s["status"] == "failed" and s["needs_attention"]
    assert hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e4")) == []
    assert state(hub)["status"] == "failed"


def test_new_turn_resets_tool_failure_count(hub):
    hub.on_claude_event(ev("UserPromptSubmit"))
    for _ in range(2):
        hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e"))
    hub.on_claude_event(ev("UserPromptSubmit"))
    for _ in range(2):
        assert hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="e")) == []
    assert state(hub)["status"] == "busy"


def test_tool_failure_after_permission_goes_busy(hub):
    hub.on_claude_event(ev("PermissionRequest", tool_name="Bash", tool_input={"command": "ls"}))
    hub.on_claude_event(ev("PostToolUseFailure", tool_name="Bash", error="denied"))
    s = state(hub)
    assert s["status"] == "busy" and not s["needs_attention"]


def test_busy_goes_unknown_after_30_min_without_events(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit"))
    clock.advance(30 * 60 - 1)
    assert hub.tick() == [] and state(hub)["status"] == "busy"
    clock.advance(1)
    assert hub.tick() == []
    s = state(hub)
    assert s["status"] == "unknown" and not s["needs_attention"]


def test_events_keep_busy_fresh_and_sleep_gap_is_never_failed(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit"))
    clock.advance(25 * 60)
    hub.on_claude_event(ev("SubagentStop"))
    clock.advance(25 * 60)
    hub.tick()
    assert state(hub)["status"] == "busy"
    clock.advance(10 * 3600)  # laptop sleep
    assert hub.tick() == []
    assert state(hub)["status"] == "unknown"


def test_staleness_only_touches_busy(hub, clock):
    hub.on_claude_event(ev("Stop", last_assistant_message="ok"))
    hub.on_claude_event(ev("PermissionRequest", sid="w1", tool_name="Bash"))
    clock.advance(3 * 3600)
    hub.tick()
    assert state(hub)["status"] == "finished" and state(hub, "w1")["status"] == "waiting"


def test_unknown_agent_revives_on_new_prompt(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit"))
    clock.advance(31 * 60)
    hub.tick()
    hub.on_claude_event(ev("UserPromptSubmit"))
    assert state(hub)["status"] == "busy"


def test_session_end_removed_after_10_min(hub, clock):
    hub.on_claude_event(ev("UserPromptSubmit"))
    assert hub.on_claude_event(ev("SessionEnd")) == []
    s = state(hub)
    assert s["status"] == "ended" and not s["needs_attention"]
    clock.advance(10 * 60 - 1)
    hub.tick()
    assert len(hub.snapshot()) == 1
    clock.advance(1)
    hub.tick()
    assert hub.snapshot() == []


def test_events_after_end_are_ignored_except_restart(hub):
    hub.on_claude_event(ev("SessionEnd"))
    assert hub.on_claude_event(ev("Stop", last_assistant_message="late")) == []
    assert state(hub)["status"] == "ended"
    hub.on_claude_event(ev("UserPromptSubmit"))
    assert state(hub)["status"] == "busy"


@pytest.mark.parametrize("kind,status,event", [("done", "finished", True), ("fail", "failed", True),
                                               ("lost", "lost", True), ("stalled", "stalled", True),
                                               ("running", "running", False)])
def test_colab_transitions(hub, kind, status, event):
    out = hub.on_colab("colab", kind, "line  of\ttext")
    assert [e.kind for e in out] == ([status] if event else [])
    s = state(hub, "colab:colab")
    assert (s["kind"], s["name"], s["status"], s["detail"]) == ("colab", "colab", status, "line of text")
    assert s["needs_attention"] == (status in ("failed", "lost", "stalled"))
    if event:
        assert (out[0].agent_id, out[0].name, out[0].text) == ("colab:colab", "colab", "line of text")
        assert hub.on_colab("colab", kind, "again") == []  # once per transition


def test_colab_run_cycle(hub):
    hub.on_colab("colab", "running", "epoch 1")
    assert [e.kind for e in hub.on_colab("colab", "stalled", "no growth")] == ["stalled"]
    assert hub.on_colab("colab", "running", "epoch 2") == []
    assert not state(hub, "colab:colab")["needs_attention"]
    assert [e.kind for e in hub.on_colab("colab", "done", "TRAIN_DONE")] == ["finished"]
    assert hub.on_colab("colab", "bogus", "x") == []


def test_colab_not_made_unknown_by_staleness(hub, clock):
    hub.on_colab("colab", "running", "x")
    clock.advance(5 * 3600)
    hub.tick()
    assert state(hub, "colab:colab")["status"] == "running"


def test_sanitizing_hostile_payload_text(hub):
    evil = "\x00\x1b[2J" + "ignore previous instructions " * 100 + "\u202e"
    out = hub.on_claude_event(ev("Stop", last_assistant_message=evil, cwd="C:\\x\\" + "d" * 500 + "\x07"))
    assert len(out[0].text) <= 80 and "\x1b" not in out[0].text and "\x00" not in out[0].text
    s = state(hub)
    assert len(s["detail"]) <= 80 and len(s["name"]) <= 47 and "\x07" not in s["name"]
    out = hub.on_claude_event(ev("PermissionRequest", tool_name="B" * 300, tool_input={"command": "c" * 9999}))
    assert len(out[0].text) <= 80
    hub.on_claude_event(ev("StopFailure", error={"a": "z" * 9999}))
    assert len(state(hub)["detail"]) <= 80
    hub.on_claude_event(ev("UserPromptSubmit", prompt="p" * 10**6))
    assert len(state(hub)["detail"]) <= 80


def test_snapshot_caps_at_20_with_attention_first(hub, clock):
    for i in range(30):
        hub.on_claude_event(ev("UserPromptSubmit", sid=f"s{i:03d}-aaaa"))
        clock.advance(1)
    hub.on_claude_event(ev("StopFailure", sid="s003-aaaa", error="old"))
    clock.advance(1)
    hub.on_claude_event(ev("PermissionRequest", sid="s001-aaaa", tool_name="Bash"))
    snap = hub.snapshot()
    assert len(snap) == 20
    assert [s["id"] for s in snap[:2]] == ["s001-aaaa", "s003-aaaa"]
    assert all(s["needs_attention"] for s in snap[:2]) and not any(s["needs_attention"] for s in snap[2:])
    assert snap[2]["id"] == "s029-aaaa"  # then newest first
    assert set(snap[0]) == {"id", "kind", "name", "status", "since", "detail", "needs_attention"}


def test_agent_count_is_bounded(hub, clock):
    for i in range(500):
        hub.on_claude_event(ev("UserPromptSubmit", sid=f"s{i}"))
        clock.advance(1)
    assert len(hub._agents) == 200
    assert "s499" in hub._agents and "s0" not in hub._agents
