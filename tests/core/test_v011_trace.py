"""Task 1 of v0.12: the v0.11 golden trace. With routine off the engine must still be v0.11 exactly (A12), so this
trace is recorded once from v0.11's engine and replayed by every later task."""
import json
import os

from tests.routine_script import GOLDEN, dumps, trace

OFF = {"routine_enabled": False}


def test_the_script_is_deterministic(tmp_path):
    assert trace(tmp_path / "a", OFF) == trace(tmp_path / "b", OFF)


def test_the_trace_covers_every_turn_type(tmp_path):
    t = trace(tmp_path, OFF)
    says = [(step["beat"], m) for step in t["steps"] for m in step["emitted"] if m.get("type") == "say"]
    kinds = {m["kind"] for _, m in says}
    assert {"reply", "greet", "bid", "report"} <= kinds

    gaming = [m for beat, m in says if beat == "day1 16:00 gaming report"]
    assert any(m["kind"] == "report" and m["spoken"] is False for m in gaming), "the gaming bubble"

    assert any(step["expression"] == "pouty" for step in t["steps"]), "the ignored bid's pout"

    # The 20-questions opening is the situation the engine builds for her first game turn.
    game = [step for step in t["steps"] if step["beat"] == "day2 08:45 work then game offer"]
    assert any("picked your secret" in json.dumps(prompt) for step in game for prompt in step["prompts"]), "the opening"

    assert any(m["kind"] == "report" and "call mom" in m["text"] for _, m in says)

    for beat in ("day1 12:00 lock and 15:30 greeting", "day2 08:40 morning greeting"):
        first = next(m for step in t["steps"] if step["beat"] == beat for m in step["emitted"]
                     if m.get("type") == "say")
        assert first["kind"] == "greet", beat


def test_this_engine_matches_the_v011_trace(tmp_path):
    if os.environ.get("RIYA_RECORD_V011_TRACE") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(dumps(trace(tmp_path / "record", OFF)), encoding="utf-8", newline="\n")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = trace(tmp_path / "check", OFF)
    if got != want:
        for a, b in zip(got["steps"], want["steps"]):
            if a != b:
                for key in sorted(set(a) | set(b)):
                    if a.get(key) != b.get(key):
                        raise AssertionError(f"first difference at beat {a['beat']!r} key {key!r}")
        if len(got["steps"]) != len(want["steps"]):
            raise AssertionError(f"step count {len(got['steps'])} != {len(want['steps'])}")
        raise AssertionError("first difference in transcripts")
