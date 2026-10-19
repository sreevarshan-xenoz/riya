import time

import pytest

from riya.core.learn.dials import BANDS, DELTA
from riya.core.learn.feedback import ACKS, MAX_WORDS, NOTE_ACKS, NOTE_MAX, Feedback, ack_line, detect_feedback

PHRASES = [
    ("stop teasing me", "teasing", -1), ("less teasing please", "teasing", -1),
    ("riya, stop teasing me!", "teasing", -1),
    ("tease me more", "teasing", 1), ("you can tease me", "teasing", 1), ("you can tease more", "teasing", 1),
    ("shorter answers please", "verbosity", -1), ("talk less", "verbosity", -1), ("that's too long", "verbosity", -1),
    ("keep it short", "verbosity", -1),
    ("longer answers", "verbosity", 1), ("talk more", "verbosity", 1),
    ("no emojis", "emoji", -1), ("fewer emojis please", "emoji", -1), ("don't use emojis", "emoji", -1),
    ("use more emojis", "emoji", 1),
    ("be sweeter", "sweetness", 1), ("be cuter", "sweetness", 1), ("be more cute", "sweetness", 1),
    ("less cutesy", "sweetness", -1), ("talk normally", "sweetness", -1),
    ("you're too clingy", "clinginess", -1), ("I like that you miss me", "clinginess", 1),
    ("stop teasing me, it's getting kind of annoying when I'm trying to work", "teasing", -1),
    ("that’s too long", "verbosity", -1), ("that's way too short", "verbosity", 1),
]


@pytest.mark.parametrize("text,dial,sign", PHRASES)
def test_dial_phrases(text, dial, sign):
    fb = detect_feedback(text)
    assert fb is not None and fb.dial == dial and fb.note is None and fb.polarity is None
    assert fb.delta == pytest.approx(sign * DELTA)


def test_style_notes():
    assert detect_feedback("I like it when you ask about my day") == Feedback(
        None, 0.0, "They like it when I ask about their day.", "like")
    assert detect_feedback("I don't like it when you call me cute") == Feedback(
        None, 0.0, "They don't like it when I call them cute.", "dislike")
    assert detect_feedback("please don't call me babe") == Feedback(
        None, 0.0, "They asked me not to call them babe.", "rule")
    assert detect_feedback("please always say goodnight to me") == Feedback(
        None, 0.0, "They asked me to always say goodnight to them.", "rule")
    assert detect_feedback("I like it when you remind me about your day").note == (
        "They like it when I remind them about my day.")
    assert detect_feedback("hey riya, I really like it when you hum along!").note == "They like it when I hum along."


def test_a_long_clause_is_cut():
    fb = detect_feedback("I like it when you " + "tell me about your wonderful day " * 10 + "ok")
    assert fb.note.endswith(".") and len(fb.note) <= NOTE_MAX and fb.note.startswith("They like it when I tell them")


@pytest.mark.parametrize("text", [
    "give me a shorter summary of this page", "make this paragraph shorter",
    "write longer answers for my essay questions", "talk less about the intro and more about the results",
    "there are no emojis in this doc", "the movie was too long", "I should talk less",
    "stop teasing the dog", "talk more slowly",
])
def test_requests_are_not_feedback(text):
    assert detect_feedback(text) is None


@pytest.mark.parametrize("text", [
    "don't stop teasing me", "I never said stop teasing",
    "do you like teasing?", "do you like teasing", "is that too long?", "can you tease me more?",
    "she said stop teasing", "he told me to talk less", "my friend says you're too clingy",
    'she said "stop teasing me"', '"stop teasing me" she said',
    "please don't go", "please always", "what if I like it when you tease me", "do you like it when I tease you",
    "", "   ", "do you tease me", "do not stop teasing me",
])
def test_negations_questions_and_narration(text):
    assert detect_feedback(text) is None


def test_do_not_rule():
    assert detect_feedback("Please do not call me sir").polarity == "rule"


def test_long_input_is_bounded():
    t = time.perf_counter()
    assert detect_feedback("stop teasing me " + "x " * 5000) is None
    assert detect_feedback("a " * 5000 + "stop teasing me") is None
    assert detect_feedback("I like it when you " + "z " * 5000).note.endswith(".")
    assert time.perf_counter() - t < 0.5


def test_acks():
    assert ack_line(Feedback("teasing", -0.15, None)) == "Okay~ less teasing, got it."
    assert ack_line(Feedback("teasing", 0.15, None)) == "Ooh, more teasing? You asked for it~"
    for dial in BANDS:
        assert (dial, 1) in ACKS and (dial, -1) in ACKS
    for pol in ("like", "dislike", "rule"):
        assert pol in NOTE_ACKS
    assert ack_line(Feedback(None, 0.0, "x y.", "dislike")) == "Okay, I won't do that anymore."
    assert MAX_WORDS == 10
