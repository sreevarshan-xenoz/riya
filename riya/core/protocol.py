"""The WebSocket protocol between the core and the body: pydantic models, parsing, and the committed JSON schema."""
import argparse
import json
from pathlib import Path
from typing import Annotated, Literal, Union, get_args

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

PROTOCOL_VERSION = 1
EXPRESSIONS = ("neutral", "happy", "excited", "missed_you", "pouty", "flustered", "teasing", "worried", "sad",
               "sleepy", "curious", "thinking", "surprised", "covering_eyes")


class ProtocolError(ValueError):
    pass


# core -> body
class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    version: int
    expressions: list[str]


class SettingsMsg(BaseModel):
    type: Literal["settings"] = "settings"
    settings: dict


class StateMsg(BaseModel):
    type: Literal["state"] = "state"
    presence: str
    expression: str
    feelings: dict[str, float]


class ExpressionMsg(BaseModel):
    type: Literal["expression"] = "expression"
    name: str


class Say(BaseModel):
    type: Literal["say"] = "say"
    utterance_id: str
    text: str
    spoken: bool
    kind: Literal["reply", "greet", "bid", "report"]


class SayEnd(BaseModel):
    type: Literal["say_end"] = "say_end"
    utterance_id: str


class Mouth(BaseModel):
    type: Literal["mouth"] = "mouth"
    level: float = Field(ge=0, le=1)


class EpisodeInfo(BaseModel):
    id: int
    started_at: float
    ended_at: float
    summary: str


class ThreadInfo(BaseModel):
    id: int
    text: str
    due_at: float


class GlanceInfo(BaseModel):
    """A note from one look at the screen (the screenshot itself is never kept)."""
    id: int
    at: float
    app: str
    title: str
    activity: str
    details: str
    notable: str | None


class NoteInfo(BaseModel):
    """A note she took for the user ("note that ..."): never a fact."""
    id: int
    at: float
    text: str


class MemoriesMsg(BaseModel):
    type: Literal["memories"] = "memories"
    facts: list[dict]
    transcript_days: list[str]
    episodes: list[EpisodeInfo] = Field(default_factory=list)  # newest first
    threads: list[ThreadInfo] = Field(default_factory=list)  # open follow-ups, soonest due first
    glances: list[GlanceInfo] = Field(default_factory=list)  # newest first, kept 7 days
    notes: list[NoteInfo] = Field(default_factory=list)  # newest first


class Look(BaseModel):
    """The side of her the active window is on (null: none, or her own) and whether she's watching it."""
    type: Literal["look"] = "look"
    side: Literal["left", "right", "center"] | None
    watching: bool


class EyesMsg(BaseModel):
    type: Literal["eyes"] = "eyes"
    enabled: bool
    dont_look: bool
    glances_today: int
    cap: int


class AgentInfo(BaseModel):
    """One of the user's agents (a Claude Code session or a Colab run); detail is a sanitized summary, <= 80 chars."""
    id: str
    kind: str
    name: str
    status: str
    since: float
    detail: str
    needs_attention: bool


class AgentsMsg(BaseModel):
    """The agents list (attention first, at most 20), sent on change and on hello. hooks_seen_at: when a Claude Code
    hook last reached her (None: never), for the settings page's install status. receiver_ok: the hook receiver is
    serving (False while port 8766 is busy; she retries every 60 s). rejected_recent: hook requests turned away for a
    wrong token in the last 24 h."""
    type: Literal["agents"] = "agents"
    items: list[AgentInfo]
    hooks_seen_at: float | None = None
    receiver_ok: bool = True
    rejected_recent: int = 0


class ErrorMsg(BaseModel):
    type: Literal["error"] = "error"
    message: str


class ConfirmRequest(BaseModel):
    """A T2 (or raised T1) action waiting for the user's yes: the templated summary, and when it expires (epoch s)."""
    type: Literal["confirm_request"] = "confirm_request"
    id: str
    summary: str
    expires_at: float


class ConfirmDone(BaseModel):
    """A confirmation is over: the body drops its bubble and buttons. outcome is what actually happened: ran (and
    worked), started (a command went to the worker), failed (it ran and failed, or couldn't run: the tool was switched
    off since, or a command is still running), declined, expired, replaced or cancelled."""
    type: Literal["confirm_done"] = "confirm_done"
    id: str
    outcome: Literal["ran", "started", "failed", "declined", "expired", "replaced", "cancelled"]


class ReminderInfo(BaseModel):
    id: int
    kind: Literal["timer", "reminder"]
    due_at: float
    text: str


class RemindersMsg(BaseModel):
    """The active reminders and timers, soonest first (on change and on hello)."""
    type: Literal["reminders"] = "reminders"
    items: list[ReminderInfo]


class ActionInfo(BaseModel):
    id: int
    at: float
    tool: str
    tier: int
    summary: str
    outcome: str  # ran | failed | refused | declined | expired | cancelled | replaced


class ActionsMsg(BaseModel):
    """Her action log for the settings page, newest first."""
    type: Literal["actions"] = "actions"
    items: list[ActionInfo]


class DialInfo(BaseModel):
    value: float
    lo: float
    hi: float


class TraitChangeInfo(BaseModel):
    at: float
    dial: str
    old: float
    new: float
    reason: str
    source: Literal["explicit", "drift", "reset"]


class StyleNoteInfo(BaseModel):
    """A style note in her own words ("They like it when I ask about their day."): never a fact."""
    id: int
    at: float
    text: str
    polarity: Literal["like", "dislike", "rule"]


class TwinInfo(BaseModel):
    """One decision-twin bucket: p is the decayed engagement estimate, n the decayed evidence above the prior."""
    kind: str
    bucket: str
    p: float
    n: float


class LearningMsg(BaseModel):
    """What she learned, for the settings page (on change, on hello and on get_memories): the dials with their bands,
    the last 20 changes (newest first), the style notes (newest first), the twin's buckets and the master toggle."""
    type: Literal["learning"] = "learning"
    dials: dict[str, DialInfo]
    history: list[TraitChangeInfo]
    notes: list[StyleNoteInfo]
    twin: list[TwinInfo]
    enabled: bool


# v0.7: the Health page and the welcome. The action names are setup.ACTIONS and the gated four setup.CONFIRM_GATED,
# written out in the same order (no import of setup here)
SetupActionName = Literal["online_key_set", "online_key_clear", "online_test", "voice_sample", "voice_choose",
                          "mic_check", "eyes_test", "hooks_connect", "hooks_disconnect", "autostart", "onboarding_done",
                          "use_base", "use_riya_v1"]


class HealthItemInfo(BaseModel):
    """One Health row: its status, a one-line explanation (never the key, only its mask) and at most one fix."""
    key: str
    status: Literal["ok", "warn", "off"]
    detail: str = Field(max_length=120)
    fix: SetupActionName | None = None


class HealthMsg(BaseModel):
    """The Health checklist, sent only in reply to health_get. key_source: where the online key comes from."""
    type: Literal["health"] = "health"
    items: list[HealthItemInfo]
    key_source: Literal["app", ".env", "env"] | None = None


class SetupResult(BaseModel):
    type: Literal["setup_result"] = "setup_result"
    action: SetupActionName
    ok: bool
    detail: str = Field(max_length=120)


class SetupConfirm(BaseModel):
    """Asks the body to confirm an outward-facing action with this exact text; the answer is setup_action{nonce}."""
    type: Literal["setup_confirm"] = "setup_confirm"
    nonce: str
    action: Literal["online_test", "eyes_test", "hooks_connect", "hooks_disconnect"]
    text: str


class VoicesMsg(BaseModel):
    """Her feminine built-in voices ([] until they are loaded) and the chosen one; sent only in reply to health_get."""
    type: Literal["voices"] = "voices"
    items: list[str]
    current: str


# body -> core
class BodyReady(BaseModel):
    type: Literal["body_ready"] = "body_ready"
    mode: str


class UserText(BaseModel):
    type: Literal["user_text"] = "user_text"
    text: str = Field(min_length=1, max_length=4000)


class TalkHotkey(BaseModel):
    type: Literal["talk_hotkey"] = "talk_hotkey"


class GetMemories(BaseModel):
    type: Literal["get_memories"] = "get_memories"


class MemoryDelete(BaseModel):
    type: Literal["memory_delete"] = "memory_delete"
    id: int
    delete_episodes: bool = False


class MemoryCorrect(BaseModel):
    type: Literal["memory_correct"] = "memory_correct"
    id: int
    text: str


class TranscriptDelete(BaseModel):
    type: Literal["transcript_delete"] = "transcript_delete"
    day: str


class EpisodeDelete(BaseModel):
    """Deletes the episode outright (no forgotten marker: its facts are deleted separately)."""
    type: Literal["episode_delete"] = "episode_delete"
    id: int


class ThreadResolve(BaseModel):
    type: Literal["thread_resolve"] = "thread_resolve"
    id: int


class DontLook(BaseModel):
    type: Literal["dont_look"] = "dont_look"
    on: bool = Field(strict=True)  # "yes" or 1 is a body bug, not a choice


class GlanceDelete(BaseModel):
    type: Literal["glance_delete"] = "glance_delete"
    id: int


class QuitCore(BaseModel):
    """The user chose Quit in the tray: the core stops too."""
    type: Literal["quit_core"] = "quit_core"


class SetSettings(BaseModel):
    type: Literal["set_settings"] = "set_settings"
    patch: dict


class Confirm(BaseModel):
    """The confirmation bubble's Yes/No button for confirmation `id`."""
    type: Literal["confirm"] = "confirm"
    id: str
    yes: bool = Field(strict=True)


class ReminderCancel(BaseModel):
    type: Literal["reminder_cancel"] = "reminder_cancel"
    id: int


class NoteDelete(BaseModel):
    type: Literal["note_delete"] = "note_delete"
    id: int


class ActionsClear(BaseModel):
    type: Literal["actions_clear"] = "actions_clear"


class StyleNoteDelete(BaseModel):
    type: Literal["note_delete_style"] = "note_delete_style"
    id: int


class TwinReset(BaseModel):
    """Forget one kind of the decision twin (the kinds are twin.KINDS, written out: no import of the twin here)."""
    type: Literal["twin_reset"] = "twin_reset"
    kind: Literal["bid", "game_offer", "report", "greeting"]


class LearningReset(BaseModel):
    """Reset to her defaults: the dials, the notes and the twin (the body confirms first)."""
    type: Literal["learning_reset"] = "learning_reset"


class HealthGet(BaseModel):
    type: Literal["health_get"] = "health_get"


class SetupRequest(BaseModel):
    """A click on a setup button. The key (online_key_set) is a plain str in args, with no length rule here, so no
    validation error can quote it."""
    type: Literal["setup_request"] = "setup_request"
    action: SetupActionName
    args: dict[str, str | bool] = Field(default_factory=dict)


class SetupAction(BaseModel):
    """The user's Yes or No to a setup_confirm."""
    type: Literal["setup_action"] = "setup_action"
    nonce: str = Field(min_length=1, max_length=64)
    yes: bool = Field(strict=True)


class OnboardingDone(BaseModel):
    """The welcome was finished or skipped."""
    type: Literal["onboarding_done"] = "onboarding_done"


ActivityKind = Literal["edit", "run", "read", "search", "web", "delegate", "think", "wave", "done", "fail", "oops",
                       "sparkle"]  # written out (no hub import); a test keeps it equal to hub.ACTIVITY_KINDS


class AgentActivity(BaseModel):
    """What one agent session is doing, for Little Riya's bubble and pose: never spoken. label: a tool name plus a
    redacted file basename or a command's first word, at most 40 chars. at: the core clock when it was made."""
    type: Literal["agent_activity"] = "agent_activity"
    session: str = Field(min_length=1, max_length=128)
    label: str = Field(max_length=40)
    kind: ActivityKind
    at: float


class PresenceMsg(BaseModel):
    """The user's presence changed (sent only on change, never replayed: hello's state carries it for a new body)."""
    type: Literal["presence"] = "presence"
    state: Literal["away", "focused", "break", "talking", "gaming"]


class AgentWindow(BaseModel):
    """The answer to focus_agent: the agent's window at click time, or null when none was found."""
    type: Literal["agent_window"] = "agent_window"
    id: str = Field(min_length=1, max_length=128)
    hwnd: int | None = Field(default=None, ge=1)


class FocusAgent(BaseModel):
    """The user clicked Little Riya while an agent needed them: which window is it? Answered with agent_window to the
    asking connection only. The core never focuses anything itself."""
    type: Literal["focus_agent"] = "focus_agent"
    id: str = Field(min_length=1, max_length=128)


class RoutineCellInfo(BaseModel):
    """One routine she has learned, in her own words (never anything the user typed): the line, her confidence and
    the decayed evidence behind it."""
    line: str
    confidence: float
    n: float


class RoutineHonestyInfo(BaseModel):
    """Her honesty numbers over the last 30 days: claims in the user reading (a disputed claim counts false)."""
    claims_true: int
    claims_scored: int
    hunches_right: int
    hunches_scored: int
    unverifiable: int
    expired: int
    claims_enabled: bool


class RoutineMsg(BaseModel):
    """What she has noticed about the user's routines, for the settings page (on hello, on change and on
    get_memories)."""
    type: Literal["routine"] = "routine"
    enabled: bool
    cells: list[RoutineCellInfo]
    honesty: RoutineHonestyInfo


class RoutineForget(BaseModel):
    """Forget every routine she has learned (the body confirms first). Her lifetime honesty tally is kept."""
    type: Literal["routine_forget"] = "routine_forget"


CoreMessage = Annotated[Union[Hello, SettingsMsg, StateMsg, ExpressionMsg, Say, SayEnd, Mouth, MemoriesMsg, ErrorMsg,
                              Look, EyesMsg, AgentsMsg, ConfirmRequest, ConfirmDone, RemindersMsg, ActionsMsg,
                              LearningMsg, HealthMsg, SetupResult, SetupConfirm, VoicesMsg, AgentActivity, PresenceMsg,
                              AgentWindow, RoutineMsg],
                        Field(discriminator="type")]
BodyMessage = Annotated[Union[BodyReady, UserText, TalkHotkey, GetMemories, MemoryDelete, MemoryCorrect,
                              TranscriptDelete, EpisodeDelete, ThreadResolve, SetSettings, DontLook, GlanceDelete,
                              QuitCore, Confirm, ReminderCancel, NoteDelete, ActionsClear, StyleNoteDelete, TwinReset,
                              LearningReset, HealthGet, SetupRequest, SetupAction, OnboardingDone, FocusAgent,
                              RoutineForget],
                        Field(discriminator="type")]
_CORE = TypeAdapter(CoreMessage)
_BODY = TypeAdapter(BodyMessage)


_BODY_TYPES = frozenset(m.model_fields["type"].default for m in get_args(get_args(BodyMessage)[0]))


def parse_body_message(raw: str) -> BodyMessage:
    """A message that doesn't parse gets a fixed error: pydantic's and json's texts quote the input, and a body bug
    could put the key into any message. Only a known type name (a constant here, not input) is named."""
    try:
        data = json.loads(raw)
    except ValueError:
        raise ProtocolError("invalid message: not JSON") from None
    try:
        return _BODY.validate_python(data)
    except ValidationError:
        kind = data.get("type") if isinstance(data, dict) else None
        known = isinstance(kind, str) and kind in _BODY_TYPES  # a list "type" is unhashable: never a set lookup
        raise ProtocolError(f"invalid {kind}" if known else "invalid message") from None


def dump(msg: BaseModel) -> str:
    return msg.model_dump_json()


def schema() -> dict:
    return {"version": PROTOCOL_VERSION, "core_to_body": _CORE.json_schema(), "body_to_core": _BODY.json_schema(),
            "expressions": list(EXPRESSIONS)}


def schema_json() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the protocol JSON schema, or write it to a file as UTF-8.")
    ap.add_argument("--write", metavar="PATH")
    args = ap.parse_args()
    if args.write:
        Path(args.write).write_text(schema_json(), encoding="utf-8", newline="\n")
    else:
        print(schema_json(), end="")
