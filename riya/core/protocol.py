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
    kind: Literal["reply", "greet", "bid"]


class SayEnd(BaseModel):
    type: Literal["say_end"] = "say_end"
    utterance_id: str


class Mouth(BaseModel):
    type: Literal["mouth"] = "mouth"
    level: float = Field(ge=0, le=1)


class MemoriesMsg(BaseModel):
    type: Literal["memories"] = "memories"
    facts: list[dict]
    transcript_days: list[str]


class ErrorMsg(BaseModel):
    type: Literal["error"] = "error"
    message: str


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


class SetSettings(BaseModel):
    type: Literal["set_settings"] = "set_settings"
    patch: dict


CoreMessage = Annotated[Union[Hello, SettingsMsg, StateMsg, ExpressionMsg, Say, SayEnd, Mouth, MemoriesMsg, ErrorMsg],
                        Field(discriminator="type")]
BodyMessage = Annotated[Union[BodyReady, UserText, TalkHotkey, GetMemories, MemoryDelete, MemoryCorrect,
                              TranscriptDelete, SetSettings], Field(discriminator="type")]
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
