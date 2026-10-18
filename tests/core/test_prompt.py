from riya.core.memory import Fact, Retrieved, Thread
from riya.core.persona import load_traits_file, persona_prompt
from riya.core.prompt import (HONESTY_LINE, NO_MEMORY_LINE, SPEAK_FIRST_TURN, build_messages,
                              greeting_situation)

NOW = 1_800_000_000.0
EXAM = Fact(1, "The user has an exam on Thursday", "user.exam", "user_said", 0.9, NOW, None, "active", None)
THREAD = Thread(1, "Ask how the exam went", NOW, "open", None)


def test_traits_file_is_versioned():
    t = load_traits_file()
    assert t["version"] >= 1 and t["history"][0]["reason"]
    assert set(t["traits"]) == {"sweetness", "teasing", "clinginess", "verbosity"}


def test_persona_has_base_and_companion_layers():
    p = persona_prompt().lower()
    assert "you are riya" in p and "teasing" in p and "clingy" in p and "you are an ai" in p


def test_messages_include_memories_with_sources():
    msgs = build_messages(persona="P", feelings_tag="[F]", retrieved=Retrieved([EXAM], [], []), turns=[],
                          user_text="hi", situation=None, now=NOW)
    system = msgs[0]["content"]
    assert msgs[0]["role"] == "system" and "P" in system and "[F]" in system
    assert "exam on Thursday (user said)" in system and HONESTY_LINE in system
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_no_memories_tells_her_not_to_guess():
    msgs = build_messages(persona="P", feelings_tag="[F]", retrieved=Retrieved([], [], []), turns=[],
                          user_text="what's my sister's name?", situation=None, now=NOW)
    assert NO_MEMORY_LINE in msgs[0]["content"]


def test_greeting_situation():
    s = greeting_situation(away_s=2 * 86400, missing=0.85, thread=THREAD)
    assert "2 days" in s and "missed" in s and "exam" in s


def test_speaking_first():
    msgs = build_messages(persona="P", feelings_tag="[F]", retrieved=Retrieved([], [], [THREAD]), turns=[],
                          user_text=None, situation="S", now=NOW)
    assert msgs[-1] == {"role": "user", "content": SPEAK_FIRST_TURN}
    assert "[Situation] S" in msgs[0]["content"] and "Follow up: Ask how the exam went" in msgs[0]["content"]


def test_fact_and_episode_line_formats():
    from riya.core.memory import Episode
    conflict = Fact(2, "Lives in Pune", None, "inferred", 0.5, NOW, None, "conflict", None)
    eps = [Episode(1, NOW - 90000, NOW - 86400, "Talked about exams", None, None),
           Episode(2, NOW - 3000, NOW - 60, "Fixed a bug", None, None),
           Episode(3, NOW - 400000, NOW - 3 * 86400 - 5, "Watched a film", None, None)]
    system = build_messages(persona="P", feelings_tag="[F]", retrieved=Retrieved([conflict], eps, []), turns=[],
                            user_text="hi", situation=None, now=NOW)[0]["content"]
    assert "- Lives in Pune (you inferred, CONFLICTING)" in system
    assert "- 1 day ago: Talked about exams" in system and "- today: Fixed a bug" in system
    assert "- 3 days ago: Watched a film" in system and NO_MEMORY_LINE not in system


def test_turns_sit_between_system_and_user():
    turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    msgs = build_messages(persona="P", feelings_tag="[F]", retrieved=Retrieved([], [], []), turns=turns,
                          user_text="c", situation=None, now=NOW)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_greeting_durations_and_mood():
    assert "3 hours" in greeting_situation(3 * 3600, 0.1, None)
    assert "1 hour" in greeting_situation(3600, 0.1, None) and "1 hours" not in greeting_situation(3600, 0.1, None)
    s = greeting_situation(3 * 3600, 0.1, None)
    assert "missed" not in s and "how their day went" in s


def test_bid_situation_is_short_and_playful():
    from riya.core.prompt import bid_situation
    s = bid_situation("long work session", THREAD)
    assert "playful" in s and "exam" in s and "No lecture" in s


def test_honesty_line_says_to_admit_what_is_not_listed():
    assert HONESTY_LINE.startswith("Only mention memories listed here.")
    assert "If what they ask about isn't listed, say you don't remember." in HONESTY_LINE
