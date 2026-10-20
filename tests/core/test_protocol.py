import json
from pathlib import Path

import pytest

from riya.core.protocol import (EXPRESSIONS, Mouth, ProtocolError, Say, UserText, dump, parse_body_message,
                                schema)


def test_parse_user_text():
    m = parse_body_message('{"type":"user_text","text":"hi 💞"}')
    assert isinstance(m, UserText) and m.text == "hi 💞"


@pytest.mark.parametrize("raw", ['{"type":"nope"}', "not json", '{"type":"user_text","text":""}'])
def test_parse_rejects(raw):
    with pytest.raises(ProtocolError):
        parse_body_message(raw)


@pytest.mark.parametrize("raw", ["[]", "3", "null", '{"text":"hi"}', '{"type":"user_text"}'])
def test_parse_rejects_non_object_missing_type_and_fields(raw):
    with pytest.raises(ProtocolError):
        parse_body_message(raw)


def test_parse_no_arg_messages():
    assert parse_body_message('{"type":"talk_hotkey"}').type == "talk_hotkey"
    assert parse_body_message('{"type":"get_memories"}').type == "get_memories"


def test_core_messages_carry_type():
    assert json.loads(dump(Say(utterance_id="u1", text="hi", spoken=True, kind="reply")))["type"] == "say"


def test_mouth_level_bounded():
    with pytest.raises(ValueError):
        Mouth(level=1.5)


def test_expressions():
    assert len(EXPRESSIONS) == 14 and "missed_you" in EXPRESSIONS and "covering_eyes" in EXPRESSIONS


def test_feelings_reexports_expressions():
    from riya.core import feelings
    assert feelings.EXPRESSIONS is EXPRESSIONS


def test_committed_schema_matches():
    assert json.loads(Path("docs/protocol.schema.json").read_text(encoding="utf-8")) == schema()


def test_episode_and_thread_messages():
    from riya.core.protocol import EpisodeDelete, MemoriesMsg, ThreadResolve
    assert parse_body_message('{"type":"episode_delete","id":3}') == EpisodeDelete(id=3)
    assert parse_body_message('{"type":"thread_resolve","id":4}') == ThreadResolve(id=4)
    with pytest.raises(ProtocolError):
        parse_body_message('{"type":"episode_delete"}')
    old_style = MemoriesMsg(facts=[], transcript_days=[])      # the body's existing fixture has neither list
    assert old_style.episodes == [] and old_style.threads == []


def test_eyes_messages():
    from riya.core.protocol import DontLook, EyesMsg, GlanceDelete, GlanceInfo, Look, MemoriesMsg
    assert parse_body_message('{"type":"dont_look","on":true}') == DontLook(on=True)
    assert parse_body_message('{"type":"glance_delete","id":7}') == GlanceDelete(id=7)
    for raw in ('{"type":"dont_look"}', '{"type":"glance_delete"}', '{"type":"dont_look","on":"yes"}'):
        with pytest.raises(ProtocolError):
            parse_body_message(raw)
    assert json.loads(dump(Look(side="left", watching=True))) == {"type": "look", "side": "left", "watching": True}
    assert Look(side=None, watching=False).side is None
    with pytest.raises(ValueError):
        Look(side="up", watching=True)
    assert json.loads(dump(EyesMsg(enabled=True, dont_look=False, glances_today=3, cap=150)))["type"] == "eyes"
    m = MemoriesMsg(facts=[], transcript_days=[], glances=[GlanceInfo(id=1, at=2.0, app="a", title="t",
                                                                       activity="x", details="", notable=None)])
    assert json.loads(dump(m))["glances"][0]["activity"] == "x"
    assert MemoriesMsg(facts=[], transcript_days=[]).glances == []


def test_quit_core_message():
    from riya.core.protocol import QuitCore
    assert parse_body_message('{"type":"quit_core"}') == QuitCore()
    fixture = Path("body/test/fixtures/body_quit_core.json")
    assert parse_body_message(fixture.read_text(encoding="utf-8")) == QuitCore()


def test_hands_messages():
    from riya.core.protocol import (ActionInfo, ActionsClear, ActionsMsg, Confirm, ConfirmDone, ConfirmRequest,
                                    MemoriesMsg, NoteDelete, NoteInfo, ReminderCancel, ReminderInfo, RemindersMsg)
    assert parse_body_message('{"type":"confirm","id":"abc","yes":true}') == Confirm(id="abc", yes=True)
    assert parse_body_message('{"type":"reminder_cancel","id":3}') == ReminderCancel(id=3)
    assert parse_body_message('{"type":"note_delete","id":4}') == NoteDelete(id=4)
    assert parse_body_message('{"type":"actions_clear"}') == ActionsClear()
    for raw in ('{"type":"confirm","id":"abc","yes":"yes"}', '{"type":"confirm","id":"abc","yes":1}',
                '{"type":"confirm","id":"abc"}', '{"type":"confirm","yes":true}', '{"type":"reminder_cancel"}',
                '{"type":"note_delete"}'):
        with pytest.raises(ProtocolError):
            parse_body_message(raw)
    assert json.loads(dump(ConfirmRequest(id="a", summary="run x", expires_at=5.0))) == {
        "type": "confirm_request", "id": "a", "summary": "run x", "expires_at": 5.0}
    for outcome in ("ran", "started", "failed", "declined", "expired", "replaced", "cancelled"):
        assert json.loads(dump(ConfirmDone(id="a", outcome=outcome)))["outcome"] == outcome
    with pytest.raises(ValueError):
        ConfirmDone(id="a", outcome="refused")
    r = RemindersMsg(items=[ReminderInfo(id=1, kind="timer", due_at=2.0, text="timer")])
    assert json.loads(dump(r)) == {"type": "reminders", "items": [{"id": 1, "kind": "timer", "due_at": 2.0,
                                                                   "text": "timer"}]}
    a = ActionsMsg(items=[ActionInfo(id=1, at=2.0, tool="open_url", tier=1, summary="open x", outcome="ran")])
    assert json.loads(dump(a))["items"][0]["outcome"] == "ran"
    assert MemoriesMsg(facts=[], transcript_days=[]).notes == []
    m = MemoriesMsg(facts=[], transcript_days=[], notes=[NoteInfo(id=1, at=2.0, text="milk")])
    assert json.loads(dump(m))["notes"] == [{"id": 1, "at": 2.0, "text": "milk"}]


def test_learning_messages():
    from typing import get_args

    from riya.core.learn import twin
    from riya.core.protocol import (DialInfo, LearningMsg, LearningReset, StyleNoteDelete, StyleNoteInfo,
                                    TraitChangeInfo, TwinInfo, TwinReset)
    assert parse_body_message('{"type":"note_delete_style","id":2}') == StyleNoteDelete(id=2)
    assert parse_body_message('{"type":"twin_reset","kind":"bid"}') == TwinReset(kind="bid")
    assert parse_body_message('{"type":"learning_reset"}') == LearningReset()
    for raw in ('{"type":"twin_reset","kind":"nope"}', '{"type":"twin_reset"}', '{"type":"note_delete_style"}'):
        with pytest.raises(ProtocolError):
            parse_body_message(raw)
    assert get_args(TwinReset.model_fields["kind"].annotation) == twin.KINDS
    m = LearningMsg(dials={"teasing": DialInfo(value=0.35, lo=0.1, hi=0.8)},
                    history=[TraitChangeInfo(at=1.0, dial="teasing", old=0.5, new=0.35, reason="r", source="explicit")],
                    notes=[StyleNoteInfo(id=1, at=1.0, text="t", polarity="like")],
                    twin=[TwinInfo(kind="bid", bucket="night/break/long", p=0.5, n=2.0)], enabled=True)
    out = json.loads(dump(m))
    assert out["type"] == "learning" and out["dials"]["teasing"] == {"value": 0.35, "lo": 0.1, "hi": 0.8}
    assert out["twin"] == [{"kind": "bid", "bucket": "night/break/long", "p": 0.5, "n": 2.0}]
    with pytest.raises(ValueError):
        TraitChangeInfo(at=1.0, dial="teasing", old=0.5, new=0.35, reason="r", source="nope")
    with pytest.raises(ValueError):
        StyleNoteInfo(id=1, at=1.0, text="t", polarity="nope")


def test_setup_messages():
    from typing import get_args

    from riya.core import setup
    from riya.core.protocol import (HealthGet, HealthItemInfo, HealthMsg, OnboardingDone, SetupAction,
                                    SetupActionName, SetupConfirm, SetupRequest, SetupResult, VoicesMsg)
    assert parse_body_message('{"type":"health_get"}') == HealthGet()
    assert parse_body_message('{"type":"onboarding_done"}') == OnboardingDone()
    assert parse_body_message('{"type":"setup_request","action":"autostart","args":{"on":true}}') == SetupRequest(
        action="autostart", args={"on": True})
    assert parse_body_message('{"type":"setup_request","action":"online_test"}') == SetupRequest(action="online_test")
    assert parse_body_message('{"type":"setup_action","nonce":"n1","yes":true}') == SetupAction(nonce="n1", yes=True)
    for raw in ('{"type":"setup_request","action":"rm_rf"}', '{"type":"setup_action","nonce":"n1","yes":"yes"}',
                '{"type":"setup_action","nonce":"","yes":true}', '{"type":"setup_action","nonce":"' + "n" * 65 +
                '","yes":true}', '{"type":"setup_request"}'):
        with pytest.raises(ProtocolError):
            parse_body_message(raw)
    with pytest.raises(ProtocolError, match="^invalid setup_request$"):
        parse_body_message('{"type":"setup_request","action":"rm_rf"}')
    assert get_args(SetupActionName) == setup.ACTIONS
    assert get_args(SetupConfirm.model_fields["action"].annotation) == setup.CONFIRM_GATED
    SetupResult(action="autostart", ok=True, detail="x" * 120)
    with pytest.raises(ValueError):
        SetupResult(action="autostart", ok=True, detail="x" * 121)
    with pytest.raises(ValueError):
        HealthItemInfo(key="local", status="ok", detail="x" * 121)
    with pytest.raises(ValueError):
        HealthItemInfo(key="local", status="bad", detail="x")
    h = HealthMsg(items=[HealthItemInfo(key="online", status="warn", detail="d", fix="online_test")], key_source="app")
    assert json.loads(dump(h)) == {"type": "health", "items": [{"key": "online", "status": "warn", "detail": "d",
                                                                "fix": "online_test"}], "key_source": "app"}
    assert HealthMsg(items=[]).key_source is None
    assert json.loads(dump(VoicesMsg(items=["alba"], current="alba"))) == {"type": "voices", "items": ["alba"],
                                                                         "current": "alba"}
    c = SetupConfirm(nonce="n", action="hooks_connect", text="t")
    assert json.loads(dump(c))["type"] == "setup_confirm"
    with pytest.raises(ValueError):
        SetupConfirm(nonce="n", action="autostart", text="t")
    test_committed_schema_matches()


CANARY = "CANARY-k3y-0123456789abcdef"
KEYED = [  # (raw, the fixed error): the canary under an unknown type, wrong-shaped known types, and not JSON
    ('{"type":"setup_requesx","args":{"key":"' + CANARY + '"}}', "invalid message"),
    ('{"type":"nope","k":"' + CANARY + '"}', "invalid message"),
    ('{"type":["' + CANARY + '"]}', "invalid message"),
    ('["' + CANARY + '"]', "invalid message"),
    ('{"type":"set_settings","patch":"' + CANARY + '"}', "invalid set_settings"),
    ('{"type":"user_text","text":["' + CANARY + '"]}', "invalid user_text"),
    ('{"type":"memory_correct","id":"' + CANARY + '","text":"x"}', "invalid memory_correct"),
    ('{"type":"setup_request","action":"online_key_set","args":{"key":["' + CANARY + '"]}}', "invalid setup_request"),
    ('{"type":"user_text","text":"' + CANARY + '"', "invalid message: not JSON"),
]


def leaks(text: str) -> bool:
    """Any of the canary, or its head or tail (pydantic cuts a long input out of the middle)."""
    return any(part in text for part in (CANARY, CANARY[:10], CANARY[-10:]))


@pytest.mark.parametrize("raw,expected", KEYED)
def test_parse_errors_never_echo_input(raw, expected):
    with pytest.raises(ProtocolError) as e:
        parse_body_message(raw)
    assert str(e.value) == expected and not leaks(str(e.value))
    assert e.value.__cause__ is None and e.value.__suppress_context__  # no chained pydantic text for a traceback


def test_pet_messages():
    from typing import get_args

    from riya.agents.hub import ACTIVITY_KINDS
    from riya.core import presence
    from riya.core.protocol import ActivityKind, AgentWindow, FocusAgent, PresenceMsg
    assert parse_body_message('{"type":"focus_agent","id":"abc123def"}') == FocusAgent(id="abc123def")
    for raw in ('{"type":"focus_agent","id":""}', '{"type":"focus_agent","id":"' + "x" * 129 + '"}',
                '{"type":"focus_agent"}', '{"type":"focus_agent","id":5}'):
        with pytest.raises(ProtocolError):
            parse_body_message(raw)
    assert json.loads(dump(AgentWindow(id="a", hwnd=None)))["hwnd"] is None
    assert json.loads(dump(AgentWindow(id="a", hwnd=132456)))["hwnd"] == 132456
    with pytest.raises(ValueError):
        AgentWindow(id="a", hwnd=0)
    with pytest.raises(ValueError):
        PresenceMsg(state="nope")
    assert get_args(ActivityKind) == ACTIVITY_KINDS
    assert get_args(PresenceMsg.model_fields["state"].annotation) == get_args(presence.State)
    test_committed_schema_matches()


def test_agent_activity_label_is_capped_at_40():
    from riya.core.protocol import AgentActivity
    assert AgentActivity(session="s", label="x" * 40, kind="edit", at=1.0).label == "x" * 40
    with pytest.raises(ValueError):
        AgentActivity(session="s", label="x" * 41, kind="edit", at=1.0)
    with pytest.raises(ValueError):
        AgentActivity(session="", label="x", kind="edit", at=1.0)
    with pytest.raises(ValueError):
        AgentActivity(session="s", label="x", kind="dance", at=1.0)
