"""The WebSocket protocol between the core and the body: pydantic models, parsing, and the committed JSON schema."""
import argparse
import json
from pathlib import Path
from typing import Annotated, Literal, Union

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
    """A confirmation is over: the body drops its bubble and buttons."""
    type: Literal["confirm_done"] = "confirm_done"
    id: str
    outcome: Literal["ran", "declined", "expired", "replaced", "cancelled"]


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


CoreMessage = Annotated[Union[Hello, SettingsMsg, StateMsg, ExpressionMsg, Say, SayEnd, Mouth, MemoriesMsg, ErrorMsg,
                              Look, EyesMsg, AgentsMsg, ConfirmRequest, ConfirmDone, RemindersMsg, ActionsMsg],
                        Field(discriminator="type")]
BodyMessage = Annotated[Union[BodyReady, UserText, TalkHotkey, GetMemories, MemoryDelete, MemoryCorrect,
                              TranscriptDelete, EpisodeDelete, ThreadResolve, SetSettings, DontLook, GlanceDelete,
                              QuitCore, Confirm, ReminderCancel, NoteDelete, ActionsClear],
                        Field(discriminator="type")]
_CORE = TypeAdapter(CoreMessage)
_BODY = TypeAdapter(BodyMessage)


def parse_body_message(raw: str) -> BodyMessage:
    try:
        return _BODY.validate_python(json.loads(raw))
    except (ValueError, ValidationError) as e:  # JSONDecodeError and ValidationError are both ValueErrors
        raise ProtocolError(str(e)) from e


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
