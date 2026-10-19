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


# ---- fix round 1: strict durations ----
@pytest.mark.parametrize("text,secs", [
    ("two and a half hours", 9000), ("seventy five minutes", 4500), ("a hundred and twenty seconds", 120),
    ("1 and a half hours", 5400), ("ninety seconds", 90), ("eighty minutes", 4800), ("two hundred seconds", 200),
    ("forty-five minutes", 2700), ("an hour and a half", 5400), ("2 days", 172800),
])
def test_parse_duration_words(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text", ["ten thousand minutes", "five and a quarter hours", "twenty minutes and thirty",
                                  "set a timer for seventy-five and a bit minutes", "3 minutes 20"])
def test_garbled_durations_are_none(text):
    assert parse_duration(text) is None
    assert m("set a timer for " + text) is None


def test_timer_phrases_with_spoken_numbers():
    assert m("set a timer for two and a half hours") == Call("timer_set", {"duration": 9000.0})
    assert m("set a timer for seventy five minutes") == Call("timer_set", {"duration": 4500.0})
    assert m("set a timer for a hundred and twenty seconds") == Call("timer_set", {"duration": 120.0})
    assert m("set a timer for 5 minutes and 30 seconds") == Call("timer_set", {"duration": 330.0})


# ---- bounds ----
def test_non_finite_and_huge_values_are_none():
    big = "9" * 400
    assert parse_duration(f"{big} hours") is None
    assert m(f"set a timer for {big} hours") is None
    assert m(f"remind me in {big} hours to x") is None
    assert parse_when(f"in {big} hours", NOW, IST) is None
    assert parse_when("in 99999 hours", NOW, IST) is None
    assert m("remind me in 99999 hours to x") is None


def test_timer_cap_is_24_hours():
    assert m("set a timer for 24 hours") == Call("timer_set", {"duration": 86400.0})
    assert m("set a timer for 1 day") == Call("timer_set", {"duration": 86400.0})
    assert m("set a timer for 24 hours and 1 second") is None
    assert m("set a timer for 25 hours") is None


def test_reminder_cap_is_366_days():
    assert parse_when("in 366 days", NOW, IST) == NOW + 366 * 86400
    assert parse_when("in 8784 hours", NOW, IST) == NOW + 366 * 86400
    assert parse_when("in 366 days and 1 second", NOW, IST) is None
    assert parse_when("in 367 days", NOW, IST) is None
    assert m("remind me in 366 days to renew") == Call("reminder_set", {"when": "in 366 days", "text": "renew"})
    assert m("remind me in 367 days to renew") is None


# ---- input length ----
def test_long_junk_is_fast():
    import time
    for junk in ("open youtube" + "," * 10000 + "x", " ," * 5000 + "x", "!" * 10000 + "x", "remind me " + "to a " * 3000,
                 "," * 10000):
        t0 = time.perf_counter()
        m(junk)
        is_forbidden(junk)
        is_command_form(junk)
        parse_when(junk, NOW, IST)
        assert time.perf_counter() - t0 < 0.2


def test_only_first_500_chars_are_read():
    assert m("open youtube" + " " * 600 + "x") == Call("open_url", {"url": SITES["youtube"]})
    assert m(" " * 600 + "open youtube") is None


# ---- file extensions ----
@pytest.mark.parametrize("name", ["notes.rtf", "index.php", "resume.docx", "main.rs", "photo.bmp", "a.py", "x.7z",
                                  "movie.mkv", "data.csv", "readme.md"])
def test_file_names_are_not_domains(name):
    assert m(f"open {name}") is None


# ---- is_forbidden narrowing ----
@pytest.mark.parametrize("text", ["send a message to mom", "email john", "text sarah hi", "dm alex", "send an email to bob",
                                  "send mom a message", "send the report to bob", "whatsapp priya", "tweet hello world",
                                  "message dad", "can you please text sarah"])
def test_messaging_someone_else_is_forbidden(text):
    assert is_forbidden(text)


@pytest.mark.parametrize("text", ["send me the weather", "text me when it's done", "send me a reminder to call mom at 5",
                                  "message me when the timer ends", "text is hard to read", "message received",
                                  "write a text file called x saying hi", "send me a message", "email is down",
                                  "send it to me", "send a reminder to me"])
def test_messaging_yourself_or_chat_is_not_forbidden(text):
    assert not is_forbidden(text)


# ---- review minors ----
@pytest.mark.parametrize("text", ["note taking is hard", "note how nice", "notes are useful", "note thatcher"])
def test_loose_note_does_not_match(text):
    assert m(text) is None


@pytest.mark.parametrize("text,saved", [("note that I parked on level 3", "I parked on level 3"),
                                        ("note: buy milk", "buy milk"), ("note to self: call mom", "call mom"),
                                        ("take a note buy milk", "buy milk"), ("make a note that it rains", "it rains")])
def test_note_forms(text, saved):
    assert m(text) == Call("note_add", {"text": saved})


@pytest.mark.parametrize("text", ["what's the date today", "what's going on with the build", "what's 9**9**9",
                                  "what's 2020-01-05", "what's on screen", "skip this", "skip a"])
def test_noise_is_not_a_tool(text):
    assert m(text) is None


# ---- final review: I3, lookup takes only its own forms ------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "what's a good name for a cat", "what's the difference between TCP and UDP", "what is the best way to learn python",
    "what are some good movies", "who is the best football player", "what is claude doing",
    "what's the capital of the united kingdom", "what's better, tea or coffee", "what is a good laptop",
    "what are the odds", "who was the better player",
])
def test_question_and_opinion_turns_are_not_lookups(text):
    assert m(text) is None


@pytest.mark.parametrize("text, query", [
    ("look up Alan Turing", "Alan Turing"), ("search for Ada Lovelace", "Ada Lovelace"),
    ("search Ada Lovelace", "Ada Lovelace"), ("can you look up black holes", "black holes"),
    ("who was Alan Turing", "Alan Turing"), ("what is a black hole", "black hole"),
    ("what was the Ming dynasty", "Ming dynasty"), ("what are quasars", "quasars"),
    ("tell me about Ada Lovelace", "Ada Lovelace"),
])
def test_lookup_forms(text, query):
    assert m(text) == Call("lookup", {"query": query})


@pytest.mark.parametrize("text", ["search my files for taxes", "look up my password", "look it up", "search for it"])
def test_lookup_keeps_its_guards(text):
    assert m(text) is None or m(text).tool != "lookup"


# ---- final review: I4, the 500-char window bounds only the detection --------------------------------------------
def words(n):
    s = " ".join(f"word{i}" for i in range(n))
    return s


@pytest.mark.parametrize("size", [501, 3000, 5000])
def test_write_file_text_is_the_whole_turn(size):
    text = words(2000)[:size].rstrip()
    assert m(f"write a file called essay.txt saying {text}") == Call("write_file",
                                                                     {"filename": "essay.txt", "text": text})


def test_note_and_copy_text_is_the_whole_turn():
    text = words(400)[:900].rstrip()
    assert m(f"note that {text}") == Call("note_add", {"text": text})
    big = words(1000)[:3000].rstrip()
    assert m(f"copy {big} to the clipboard") == Call("clipboard_copy", {"text": big})


# ---- residual 2 (N1): agent-status questions are chat, not lookups ------------------------------------------------
@pytest.mark.parametrize("text", [
    "what's claude working on", "what is the agent waiting for", "what's colab saying", "what is my build doing",
    "what is the training run doing", "who is the session for", "what was codex saying", "what are the tests doing",
    "what's the job waiting on", "who is their manager", "what is the agent running", "what is claude",
])
def test_agent_status_questions_are_not_lookups(text):
    c = m(text)
    assert c is None or c.tool != "lookup"


@pytest.mark.parametrize("text, query", [
    ("what is a black hole", "black hole"), ("who is Ada Lovelace", "Ada Lovelace"),
    ("look up claude shannon", "claude shannon"), ("search for the training of dogs", "training of dogs"),
    ("what is a wedding", "wedding"), ("what is swimming", "swimming"),
])
def test_lookups_that_stay(text, query):
    assert m(text) == Call("lookup", {"query": query})
