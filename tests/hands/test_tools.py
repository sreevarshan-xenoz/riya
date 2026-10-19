"""Registry, ParamSpec validation, validate_call and the runs that need no effects."""
import copy
import math
from zoneinfo import ZoneInfo

import pytest

from riya.core.clock import FakeClock
from riya.core.hands.tools import (DEFAULT_OFF, REGISTRY, RESULT_MAX, TIERS, Call, ParamSpec, ToolCtx, ToolResult,
                                   _check, validate_call)
from riya.core.settings import Settings

T0 = {"time_now", "timer_set", "reminder_set", "reminder_list", "reminder_cancel", "note_add", "note_list", "calc",
      "weather", "lookup"}
T1 = {"open_url", "open_app", "media", "volume", "clipboard_copy"}
T2 = {"run_command", "write_file"}
CMD = {"name": "backup", "argv": ["robocopy", "C:/a", "D:/b"], "cwd": "C:/a", "timeout_s": 30}


@pytest.fixture
def s():
    return Settings(apps=[{"name": "Notepad", "target": "notepad.exe"}], commands=[copy.deepcopy(CMD)])


def v(s, tool, **args):
    return validate_call(Call(tool, args), s)


def test_registry_matches_tier_lists():
    assert set(REGISTRY) == T0 | T1 | T2 == set(TIERS)
    for names, tier in ((T0, 0), (T1, 1), (T2, 2)):
        assert all(REGISTRY[n].tier == tier == TIERS[n] and REGISTRY[n].name == n for n in names)
    assert DEFAULT_OFF == {"run_command", "write_file"} and RESULT_MAX == 200


def test_toolresult_cuts_text():
    assert len(ToolResult(True, "x" * 500).text) == 200 and ToolResult(True, "hi").data is None
    assert ParamSpec("str").max_len == 200 and ParamSpec("int").required is True


def test_valid_basic_calls_pass(s):
    assert v(s, "calc", expr="1+1") == Call("calc", {"expr": "1+1"})
    assert v(s, "time_now") == Call("time_now", {})
    assert v(s, "weather") == Call("weather", {}) and v(s, "weather", place="Paris").args == {"place": "Paris"}
    assert v(s, "media", action="play_pause") and v(s, "volume", action="mute")
    assert v(s, "reminder_set", when="at 5", text="call mom")


def test_unknown_tool_key_missing_and_wrong_type(s):
    assert v(s, "teleport") is None
    assert v(s, "calc", expr="1", extra="x") is None
    assert v(s, "calc") is None
    assert v(s, "calc", expr=5) is None and v(s, "calc", expr="  ") is None
    assert validate_call(Call("calc", ["1"]), s) is None


def test_enum_and_lengths(s):
    assert v(s, "media", action="explode") is None and v(s, "volume", action="UP") is None
    assert v(s, "reminder_set", when="x" * 101, text="y") is None and v(s, "reminder_set", when="x" * 100, text="y")
    assert v(s, "weather", place="p" * 81) is None and v(s, "weather", place="p" * 80)
    assert v(s, "note_add", text="n" * 1001) is None and v(s, "note_add", text="n" * 1000)
    assert v(s, "calc", expr="1" * 201) is None
    assert v(s, "clipboard_copy", text="c" * 4001) is None and v(s, "clipboard_copy", text="c" * 4000)
    assert v(s, "open_url", url="a.com/" + "x" * 2000) is None


def test_int_param_rejects_bool_and_bounds(s):
    assert v(s, "reminder_cancel", id=3) == Call("reminder_cancel", {"id": 3})
    for bad in (True, 3.0, "3", 0, -1):
        assert v(s, "reminder_cancel", id=bad) is None


def test_one_of(s):
    assert v(s, "reminder_cancel") is None
    assert v(s, "reminder_cancel", id=3) and v(s, "reminder_cancel", text="mom")


def test_duration_types(s):
    assert v(s, "timer_set", duration="10 minutes").args == {"duration": 600.0}
    out = v(s, "timer_set", duration=600.0).args["duration"]
    assert out == 600.0 and isinstance(out, float)
    assert isinstance(v(s, "timer_set", duration=600).args["duration"], float)
    for bad in (0, -5, float("inf"), float("nan"), True, "forever", "", "0 minutes", None, [5], 86401, "2 days"):
        assert v(s, "timer_set", duration=bad) is None, bad
    assert v(s, "timer_set", duration=86400) and v(s, "timer_set", duration="24 hours")


def test_float_spec_and_bounds_directly():
    assert _check(ParamSpec("float", min=0, max=1), 1) == 1.0
    for bad in (True, "1", 2, -0.1, math.inf):
        with pytest.raises(ValueError):
            _check(ParamSpec("float", min=0, max=1), bad)


def test_open_url_is_normalized_or_refused(s):
    assert v(s, "open_url", url="evil.com").args == {"url": "https://evil.com"}
    for bad in ("file:///C:/x", "javascript:alert(1)", "https://u:p@evil.com", "a b.com"):
        assert v(s, "open_url", url=bad) is None


def test_open_app_only_known_names(s):
    assert v(s, "open_app", name="Notepad").args == {"name": "Notepad"}
    assert v(s, "open_app", name="notepad").args == {"name": "Notepad"}  # normalized to the stored spelling
    assert v(s, "open_app", name="calc") is None
    assert v(Settings(), "open_app", name="Notepad") is None


def test_run_command_snapshot(s):
    c = v(s, "run_command", name="backup")
    assert c.args == {"name": "backup", "argv": CMD["argv"], "cwd": "C:/a", "timeout_s": 30}
    c.args["argv"].append("--evil")
    assert s.commands[0]["argv"] == CMD["argv"]  # a copy
    assert v(s, "run_command", name="nope") is None and v(Settings(), "run_command", name="backup") is None
    for extra in ({"argv": ["cmd"]}, {"cwd": "C:/"}, {"timeout_s": 5}):
        assert v(s, "run_command", name="backup", **extra) is None


def test_run_command_snapshot_is_fixed_at_validation(s):
    c = v(s, "run_command", name="backup")
    s.commands[0]["argv"][0] = "changed"
    assert c.args["argv"][0] == "robocopy"


def test_write_file_limits(s):
    assert v(s, "write_file", filename="a.txt", text="x" * 65_536)
    assert v(s, "write_file", filename="a.txt", text="x" * 65_537) is None
    assert v(s, "write_file", filename="f" * 201, text="x") is None
    assert v(s, "write_file", filename="a.txt") is None


def test_summaries(s):
    c = v(s, "run_command", name="backup")
    text = REGISTRY["run_command"].summary(c.args)
    assert '"backup"' in text and "robocopy C:/a D:/b" in text
    w = v(s, "write_file", filename="n.txt", text="é" + "a" * 199).args
    text = REGISTRY["write_file"].summary(w)
    assert '"n.txt"' in text and "201 bytes" in text and "a" * 79 in text and "a" * 81 not in text
    for name in ("open_url", "open_app", "clipboard_copy", "note_add"):  # templates never use *stage directions*
        assert "*" not in REGISTRY[name].summary({"url": "https://a.com", "name": "x", "text": "t"})


def test_every_tool_has_a_summary():
    ok = {"time_now": {}, "timer_set": {"duration": 90.0}, "reminder_set": {"when": "at 5", "text": "t"},
          "reminder_list": {}, "reminder_cancel": {"id": 1}, "note_add": {"text": "t"}, "note_list": {},
          "calc": {"expr": "1"}, "weather": {}, "lookup": {"query": "q"}, "open_url": {"url": "https://a.com"},
          "open_app": {"name": "n"}, "media": {"action": "next"}, "volume": {"action": "up"},
          "clipboard_copy": {"text": "t"}, "run_command": {"name": "b", "argv": ["a"], "cwd": None, "timeout_s": 1},
          "write_file": {"filename": "a.txt", "text": "t"}}
    assert set(ok) == set(REGISTRY)
    for n, a in ok.items():
        assert REGISTRY[n].summary(a)


def test_calc_and_time_now_runs():
    ctx = ToolCtx(clock=FakeClock(1_800_000_000.0))
    r = REGISTRY["calc"].run({"expr": "12 * 7"}, ctx)
    assert r.ok and "84" in r.text
    assert not REGISTRY["calc"].run({"expr": "2**101"}, ctx).ok
    r = REGISTRY["time_now"].run({}, ctx)
    assert r.ok and r.text.startswith("It") and len(r.text) <= 200


def test_time_now_uses_the_zone():
    ctx = ToolCtx(clock=FakeClock(1_800_000_000.0), tz=ZoneInfo("Asia/Kolkata"))
    text = REGISTRY["time_now"].run({}, ctx).text
    assert " on " in text and "January" in text and ":" in text and ("AM" in text or "PM" in text)


def test_unwritten_runs_raise_until_their_task():
    with pytest.raises(NotImplementedError):
        REGISTRY["note_add"].run({"text": "x"}, ToolCtx())
