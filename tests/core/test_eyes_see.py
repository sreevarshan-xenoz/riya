import base64
import json

import pytest

from riya.core.eyes.see import NOTE_PROMPT, GlanceNote, describe, is_screen_question, parse_note, vision_messages
from riya.core.models import ModelUnavailable
from tests.fakes import FakeModel

JPEG = b"\xff\xd8\xff\xe0fakejpegbytes\xff\xd9"


def note_json(**kw):
    return json.dumps({"app": "VS Code", "activity": "fixing a failing test", "details": "pytest shows one red test",
                       "notable": None, **kw})


def test_vision_messages_are_content_parts_with_a_data_uri():
    msgs = vision_messages(JPEG, "what is this?")
    assert msgs == [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()}}]}]


def test_vision_messages_put_the_system_prompt_first():
    msgs = vision_messages(JPEG, "q", system="You are Riya.")
    assert msgs[0] == {"role": "system", "content": "You are Riya."} and msgs[1]["role"] == "user" and len(msgs) == 2


def test_note_prompt_states_the_rules():
    p = NOTE_PROMPT.lower()
    for needle in ("json", "app", "activity", "details", "notable", "12 words", "30 words", "password",
                   "something private"):
        assert needle in p, needle


def test_parse_note_reads_the_fields():
    assert parse_note(note_json()) == GlanceNote("VS Code", "fixing a failing test", "pytest shows one red test", None)
    assert parse_note(note_json(notable="TypeError in test_x")).notable == "TypeError in test_x"


def test_parse_note_takes_the_outermost_object_from_fenced_or_chatty_text():
    assert parse_note("```json\n" + note_json() + "\n```").app == "VS Code"
    assert parse_note("Sure! Here you go: " + note_json() + " Hope that helps.").activity == "fixing a failing test"
    nested = '{"app": "A", "activity": "b", "details": "c", "notable": null, "extra": {"x": 1}}'
    assert parse_note(nested) == GlanceNote("A", "b", "c", None)


def test_parse_note_clips_to_the_word_limits():
    n = parse_note(note_json(activity=" ".join(f"a{i}" for i in range(20)), details=" ".join(f"d{i}" for i in range(50))))
    assert len(n.activity.split()) == 12 and n.activity.endswith("a11")
    assert len(n.details.split()) == 30 and n.details.endswith("d29")


def test_parse_note_normalises_optional_fields():
    n = parse_note('{"app": "Chrome", "activity": "reading docs"}')
    assert n == GlanceNote("Chrome", "reading docs", "", None)
    assert parse_note(note_json(notable="  ")).notable is None
    assert parse_note(note_json(details=None)).details == ""


@pytest.mark.parametrize("text", [
    "", "no json here", "{broken", "[1, 2]", '{"app": "x"}', '{"activity": "y"}', '{"app": "", "activity": "y"}',
    '{"app": "x", "activity": "  "}', '{"app": 3, "activity": "y"}', '{"app": "x", "activity": ["y"]}',
    "} {", "{not json} and {also not}", None, 5])
def test_parse_note_rejects_bad_input(text):
    assert parse_note(text) is None


@pytest.mark.parametrize("text", [
    "what's wrong with this", "can you see this error", "what am I looking at", "read this", "look at my screen",
    "What’s wrong with this?", "can you see my screen", "why is this error happening", "what's this error",
    "look at this", "check this out", "what's going on here", "what does this say", "help me with this code",
    "is there anything on screen", "tell me what's on my screen", "do you see this?", "explain this"])
def test_screen_questions(text):
    assert is_screen_question(text), text


@pytest.mark.parametrize("text", [
    "what's wrong with you", "this weekend was fun", "", "hi riya", "how was your day", "I missed you",
    "I think that's right", "that was fun", "what are you doing", "can you see me smiling", "look at you go",
    "I see that you're tired"])
def test_not_screen_questions(text):
    assert not is_screen_question(text), text


def test_describe_returns_a_note_and_sends_the_image():
    m = FakeModel("online", [note_json(notable="red test")])
    assert describe(m, JPEG) == GlanceNote("VS Code", "fixing a failing test", "pytest shows one red test", "red test")
    content = m.calls[0][-1]["content"]
    assert content[0]["text"] == NOTE_PROMPT and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_describe_never_raises():
    assert describe(FakeModel("o", fail=ModelUnavailable("429")), JPEG) is None
    assert describe(FakeModel("o", fail=RuntimeError("boom")), JPEG) is None
    assert describe(FakeModel("o", fail=ModelUnavailable("x")), b"") is None
    assert describe(FakeModel("o", ["I can't describe that."]), JPEG) is None
    assert describe(FakeModel("o", ["{\"app\": \"x\", \"activity\": \"y\"", ], fail_after=1), JPEG) is None
