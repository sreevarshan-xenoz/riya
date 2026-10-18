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
