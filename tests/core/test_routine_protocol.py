"""The routine messages on the wire (v0.12 §4.8)."""
import json
from pathlib import Path

from pydantic import TypeAdapter

from riya.core.protocol import (CoreMessage, RoutineCellInfo, RoutineForget, RoutineHonestyInfo, RoutineMsg, dump,
                                parse_body_message, schema)


def test_routine_messages_round_trip():
    m = RoutineMsg(enabled=True, cells=[RoutineCellInfo(line="Agent runs (riya): about 10 min, 14 runs, sure",
                                                        confidence=0.78, n=14.0)],
                   honesty=RoutineHonestyInfo(claims_true=47, claims_scored=48, hunches_right=31, hunches_scored=52,
                                              unverifiable=2, expired=1, claims_enabled=True))
    assert TypeAdapter(CoreMessage).validate_json(dump(m)) == m
    assert json.loads(dump(m))["type"] == "routine"


def test_routine_forget_is_a_body_message():
    assert parse_body_message('{"type":"routine_forget"}') == RoutineForget()


def test_the_committed_schema_is_current():
    assert json.loads(Path("docs/protocol.schema.json").read_text(encoding="utf-8")) == schema()
