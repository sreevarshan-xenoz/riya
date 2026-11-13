"""The claim guard and the routine, goodbye and dispute detectors (v0.12 §4.5.3, §4.5.4, §4.12.1, A11)."""
import json
from pathlib import Path

import pytest

from riya.core.learn.feedback import detect_feedback
from riya.core.routine.detect import TOPICS, claim_guard, disputes, goodbye, routine_question

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "claim_guard.json"

QUESTIONS = {
    "stretch": ["how long have i been at it?", "how long have I been on here??", "how long have i been coding",
                "since when have i been on the computer", "how long was i coding for", "how long ive been here lol",
                "riya how long have i been sitting here"],
    "runs": ["how long do my claude runs take?", "how long does a colab job usually take",
             "how long do my agent runs take", "how long are my runs usually", "what's my usual run time",
             "how long do my riya runs normally take?"],
    "run_left": ["when will my agent be done?", "how much longer?", "when will claude finish",
                 "how long until my run is done", "is my agent almost done", "how long till colab finishes",
                 "how much longer for my agent"],
    "gaming": ["when do i usually game?", "what time do i usually play games", "which days do i game",
               "when do i usually play?", "what nights do i mostly game", "when's my usual game time"],
}
NOT_QUESTIONS = ["when's my 9 am lab?", "how long is a marathon?", "what time is it?", "how long do eggs take to boil",
                 "when does the game start", "how long have you been alive", "how much longer is the movie",
                 "do you like games?", "i've been coding all day", "when will dinner be ready", "how long is the run to the store",
                 "how long does a job interview take", "how long have i been working at this company",
                 "how long have i been going to the gym", "when do i play with my cat",
                 "what time do i usually play piano", "how much longer do i have left"]

GOODBYES = ["bye!", "brb", "gtg", "gotta go, back at 6", "good night riya", "ok see you later", "heading out now~",
            "off to class", "byeee"]
NOT_GOODBYES = ["bye the way, can you help me plan my week tomorrow", "say bye to my sister for me", "hello!",
                "i'm going to keep coding", "what does gtg mean in texting slang and where did it come from",
                "I can see you're excited", "oh I see you've changed it", "this week is off to a rough start",
                "what a good night", "had a good night's sleep", "do I have to go to school tomorrow"]

DISPUTES = ["that's wrong", "not true", "no I wasn't", "I wasn't", "that's not right", "no, i didn't",
            "nope, that's not true", "you're wrong", "wrong!", "i wasn't coding, i was watching a video"]
NOT_DISPUTES = ["what's wrong?", "nothing's wrong with that", "is that wrong?", "not true love, just a crush",
                "you're right", "that's true", "i was coding", "nope, thanks", "nope not hungry", "nope, I'm good",
                "I didn't sleep well", "I haven't eaten yet", "I wasn't able to find it", "it's wrong to lie",
                "this is false advertising", "it's not right now"]


def test_the_guard_is_exact_on_the_fixture():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert data["version"] == 1
    assert sum(c["keep"] for c in cases) >= 40 and sum(not c["keep"] for c in cases) >= 40
    for c in cases:
        assert claim_guard(c["sentence"], c["offered"], c["user_text"]) == c["keep"], c["why"]


def test_gate_probe_facts_are_kept_when_offered_and_dropped_when_not():
    lab = "You have a 9 am lab on Mondays!"
    assert claim_guard(lab, "- The user has a 9 am lab on Mondays (user said)", None)
    assert not claim_guard(lab, "", None)
    assert claim_guard("Your reminder's at 5 pm~", '### Your reminders: "call mom" at 5:00 pm', None)
    assert claim_guard("You're up at six every day for your jog, right?",
                       "- The user is up at six daily for a jog (user said)", None)


def test_a_habit_with_no_time_is_never_dropped():
    assert claim_guard("You always make me laugh~", "", None)
    assert claim_guard("You never let me win!", "", None)


def test_routine_question_topics():
    assert set(QUESTIONS) == set(TOPICS)
    for topic, texts in QUESTIONS.items():
        assert len(texts) >= 6
        for t in texts:
            assert routine_question(t) == topic, t
    assert len(NOT_QUESTIONS) >= 10
    for t in NOT_QUESTIONS:
        assert routine_question(t) is None, t


def test_goodbye_is_short_and_final():
    for t in GOODBYES:
        assert goodbye(t), t
    for t in NOT_GOODBYES:
        assert not goodbye(t), t


def test_disputes():
    assert len(DISPUTES) >= 6
    for t in DISPUTES:
        assert disputes(t), t
    for t in NOT_DISPUTES:
        assert not disputes(t), t


def _variants(t):
    return [t.upper(), t.replace("'", "’"), t + "??", t + "~", t.upper().replace("'", "’") + "??"]


@pytest.mark.parametrize("detector,texts,want", [
    (goodbye, GOODBYES, True), (goodbye, NOT_GOODBYES, False),
    (disputes, DISPUTES, True), (disputes, NOT_DISPUTES, False),
])
def test_case_and_curly_apostrophes_never_change_a_verdict(detector, texts, want):
    for t in texts:
        for v in _variants(t):
            assert detector(v) is want, v


def test_routine_question_ignores_case_and_curly_apostrophes():
    for topic, texts in QUESTIONS.items():
        for t in texts:
            for v in _variants(t):
                assert routine_question(v) == topic, v
    for t in NOT_QUESTIONS:
        for v in _variants(t):
            assert routine_question(v) is None, v


def test_feedback_never_reads_a_dispute():
    for t in DISPUTES:
        assert detect_feedback(t) is None, t
