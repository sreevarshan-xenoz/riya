"""Pure request parsing: durations, times, match, command form, forbidden."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from riya.core.hands.intent import (SITES, is_command_form, is_forbidden, match, parse_duration,
                                    parse_when)
from riya.core.hands.types import Call

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")


def ts(y, mo, d, h=0, mi=0, tz=IST):
    return datetime(y, mo, d, h, mi, tzinfo=tz).timestamp()


NOW = ts(2026, 10, 19, 16, 0)


def m(text, now=NOW, **kw):
    kw.setdefault("tz", IST)
    return match(text, now, **kw)


# ---- durations ----
@pytest.mark.parametrize("text,secs", [
    ("10 minutes", 600), ("an hour and a half", 5400), ("half an hour", 1800), ("2 hrs 15 min", 8100),
    ("a minute", 60), ("90 seconds", 90), ("1.5 hours", 5400), ("ten minutes", 600), ("5m", 300),
])
def test_parse_duration(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text", ["0 minutes", "forever", "", "soon"])
def test_parse_duration_none(text):
    assert parse_duration(text) is None


# ---- times ----
def test_at_5_after_1730_is_5am_tomorrow():
    assert parse_when("at 5", ts(2026, 10, 19, 17, 30), IST) == ts(2026, 10, 20, 5)
    assert parse_when("at 5", ts(2026, 10, 19, 16, 0), IST) == ts(2026, 10, 19, 17)
    assert parse_when("at 5", ts(2026, 10, 20, 5, 0), IST) == ts(2026, 10, 20, 17)


def test_other_times():
    assert parse_when("at 5:30 pm", ts(2026, 10, 19, 18), IST) == ts(2026, 10, 20, 17, 30)
    assert parse_when("at 12:15 am", ts(2026, 10, 19, 23, 50), IST) == ts(2026, 10, 19, 23, 50) + 25 * 60
    assert parse_when("in 20 minutes", NOW, IST) == NOW + 1200
    assert parse_when("tomorrow at 9", NOW, IST) == ts(2026, 10, 20, 9)
    assert parse_when("tomorrow at 5", NOW, IST) == ts(2026, 10, 20, 17)
    assert parse_when("tomorrow at 11", NOW, IST) == ts(2026, 10, 20, 11)
    assert parse_when("tomorrow at 12", NOW, IST) == ts(2026, 10, 20, 12)
    assert parse_when("tomorrow at 8 pm", NOW, IST) == ts(2026, 10, 20, 20)
    assert parse_when("at 17:45", NOW, IST) == ts(2026, 10, 19, 17, 45)
    assert parse_when("at noon", NOW, IST) == ts(2026, 10, 20, 12)


@pytest.mark.parametrize("text", ["at blah", "at 25", "at 13 pm", "in forever", "tomorrow", "sometime", "at 5:75", ""])
def test_parse_when_rejects(text):
    assert parse_when(text, NOW, IST) is None


def test_tomorrow_at_9_across_spring_forward():
    now = ts(2026, 3, 7, 20, tz=NY)
    assert parse_when("tomorrow at 9", now, NY) == datetime(2026, 3, 8, 9, tzinfo=NY).timestamp()
    assert parse_when("in 2 hours", ts(2026, 3, 8, 1, 30, tz=NY), NY) == ts(2026, 3, 8, 1, 30, tz=NY) + 7200
    # the first (EDT) 1:30 of the fall-back night
    first = datetime(2026, 11, 1, 1, 30, tzinfo=NY).timestamp()
    assert parse_when("at 1:30", ts(2026, 11, 1, 0, 30, tz=NY), NY) == first
    # a nonexistent wall time lands after now, never before
    gap = parse_when("at 2:30", ts(2026, 3, 8, 0, 10, tz=NY), NY)
    assert gap > ts(2026, 3, 8, 0, 10, tz=NY)


def test_parse_when_default_zone_does_not_crash():
    assert parse_when("at 5", NOW) > NOW


# ---- match positives ----
def test_timer():
    assert m("set a timer for 10 minutes") == Call("timer_set", {"duration": 600.0})
    assert m("set a 5 minute timer") == Call("timer_set", {"duration": 300.0})
    assert m("set a timer") is None


def test_reminders():
    want = Call("reminder_set", {"when": "at 5", "text": "call mom"})
    assert m("remind me at 5 to call mom") == want
    assert m("remind me to call mom at 5") == want
    assert m("remind me tomorrow at 9 to x") == Call("reminder_set", {"when": "tomorrow at 9", "text": "x"})
    assert m("Remind me in 20 minutes that the tea is ready") == Call(
        "reminder_set", {"when": "in 20 minutes", "text": "the tea is ready"})
    assert m("remind me to book tomorrow at 9") == Call("reminder_set", {"when": "tomorrow at 9", "text": "book"})
    assert m("remind me to meet at the cafe at 5") == Call(
        "reminder_set", {"when": "at 5", "text": "meet at the cafe"})


def test_open_sites_and_domains():
    assert m("open youtube") == Call("open_url", {"url": "https://www.youtube.com"})
    for alias, url in SITES.items():
        assert m(f"open {alias}") == Call("open_url", {"url": url})
    assert SITES["x"] == SITES["twitter"] == "https://x.com"
    assert m("could you open github please") == Call("open_url", {"url": SITES["github"]})
    assert m("hey riya, open YouTube") == Call("open_url", {"url": SITES["youtube"]})
    assert m("open evil.com") == Call("open_url", {"url": "https://evil.com"})
    assert m("open Evil.com/Path") == Call("open_url", {"url": "https://evil.com/Path"})
    assert m("open notes.txt") is None


def test_open_app_beats_alias():
    assert m("open spotify", apps=["Spotify"]) == Call("open_app", {"name": "Spotify"})
    assert m("launch visual studio code", apps=["Visual Studio Code"]) == Call(
        "open_app", {"name": "Visual Studio Code"})
    assert m("open spotify").tool == "open_url"


@pytest.mark.parametrize("text,action", [("pause the music", "play_pause"), ("play", "play_pause"),
                                         ("resume the music", "play_pause"), ("skip", "next"),
                                         ("next track", "next"), ("previous track", "previous"),
                                         ("skip this song", "next")])
def test_media(text, action):
    assert m(text) == Call("media", {"action": action})


@pytest.mark.parametrize("text,action", [("volume up", "up"), ("turn the volume down", "down"),
                                         ("mute", "mute"), ("turn up the volume", "up"),
                                         ("turn it down", "down")])
def test_volume(text, action):
    assert m(text) == Call("volume", {"action": action})


def test_simple_tools():
    assert m("what time is it") == Call("time_now", {})
    assert m("what's the time?") == Call("time_now", {})
    assert m("weather in Pune") == Call("weather", {"place": "Pune"})
    assert m("what's the weather like") == Call("weather", {})
    assert m("note that I parked on level 3") == Call("note_add", {"text": "I parked on level 3"})
    assert m("what are my notes") == Call("note_list", {})
    assert m("copy hello world") == Call("clipboard_copy", {"text": "hello world"})
    assert m("run backup", commands=["backup"]) == Call("run_command", {"name": "backup"})
    assert m("Run the Backup command", commands=["backup"]) == Call("run_command", {"name": "backup"})
    assert m("what's 12 * 7") == Call("calc", {"expr": "12 * 7"})
    assert m("calculate 12 * 7") == Call("calc", {"expr": "12 * 7"})
    assert m("what is 2 plus 2") == Call("calc", {"expr": "2 + 2"})
    assert m("what are my reminders") == Call("reminder_list", {})
    assert m("cancel the reminder about mom") == Call("reminder_cancel", {"text": "mom"})
    assert m("cancel my timer") == Call("reminder_cancel", {"text": "timer"})
    assert m("write a file called groceries saying eggs and milk") == Call(
        "write_file", {"filename": "groceries", "text": "eggs and milk"})
    assert m("tell me about Ada Lovelace") == Call("lookup", {"query": "Ada Lovelace"})
    assert m("who is Ada Lovelace?") == Call("lookup", {"query": "Ada Lovelace"})


@pytest.mark.parametrize("text", [
    "don't open youtube", "I opened the app", "can you not open youtube", "open the door", "run fast",
    "who is my sister", "what is this error", "I love youtube", "remind me at blah to x",
    "what's up", "who are you", "what is it", "hello there", "remind me to call mom",
    "copy that", "run the tests",
])
def test_match_negatives(text):
    assert m(text, apps=["Spotify"]) is None


def test_run_needs_a_known_command():
    assert m("run backup") is None
    assert m("run backup", commands=["deploy"]) is None


# ---- command form ----
@pytest.mark.parametrize("text", ["launch my editor", "could you look up boiling points", "please search for cats",
                                  "hey riya, open the pod bay doors", "Can you turn the lights on?",
                                  "set the oven timer"])
def test_command_form_true(text):
    assert is_command_form(text)


@pytest.mark.parametrize("text", ["the play was great", "what's up", "don't open youtube", "setup is hard",
                                  "I want to open it", ""])
def test_command_form_false(text):
    assert not is_command_form(text)


# ---- forbidden (T3) ----
@pytest.mark.parametrize("text", ["delete my downloads folder", "send an email to bob", "buy me a keyboard",
                                  "what's my bank password", "can you please delete all the files",
                                  "run a shell command", "open powershell", "sudo rm -rf /",
                                  "I want you to send a message to mom", "tell me my password"])
def test_forbidden_true(text):
    assert is_forbidden(text)


@pytest.mark.parametrize("text", ["open youtube", "what time is it", "I bought a keyboard", "I changed my password",
                                  "run backup", "the shell was pretty", "remind me to send mom flowers at 5"])
def test_forbidden_false(text):
    assert not is_forbidden(text)
