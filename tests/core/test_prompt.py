from pathlib import Path

import pytest

from riya.core.memory import Episode, Fact, Retrieved, Thread
from riya.core.persona import STYLE_HEADER, load_traits_file, persona_prompt, style_examples
from riya.core.prompt import (HARD_RULES as HARD, HONESTY_LINE, MEMORY_END, MEMORY_START, NO_ABOUT_ME_LINE,
                              NO_MEMORY_LINE, SPEAK_FIRST_TURN, build_messages, greeting_situation, reply_rule)

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


# ---- v0.2 task 3: persona layers, delimited memory block, reply rules ----
CORE_DIR = Path(__file__).resolve().parents[2] / "riya" / "core"
EP = Episode(1, NOW - 90000, NOW - 86400, "Talked about exams", None, None)


def _system(retrieved, **kw):
    return build_messages(persona="P", feelings_tag="[F]", retrieved=retrieved, turns=[], user_text="hi",
                          situation=kw.pop("situation", None), now=NOW, **kw)[0]["content"]


def _block(system):
    start = system.index(MEMORY_START)
    return system[start:system.index(MEMORY_END, start) + len(MEMORY_END)]


def test_local_persona_is_short_second_person_and_honest():
    text = (CORE_DIR / "persona_local.md").read_text(encoding="utf-8")
    assert len(text.split()) <= 180
    p = persona_prompt("local").lower()
    assert "you are riya" in p and "you're an ai" in p and "memory block" in p
    assert "you are riya" in persona_prompt("online").lower() and p != persona_prompt("online").lower()
    assert persona_prompt() == persona_prompt("online")


def test_companion_layer_lost_its_example_quote():
    assert "three hours on one bug" not in (CORE_DIR / "companion.md").read_text(encoding="utf-8")
    assert "three hours on one bug" not in persona_prompt("online")


def test_style_examples_are_labelled_made_up_and_six_to_eight_exchanges():
    text = style_examples()
    assert STYLE_HEADER == "Made-up examples of your style. They never happened. Never mention them."
    assert text.startswith(STYLE_HEADER)
    lines = text.splitlines()
    users, riyas = [ln for ln in lines if ln.startswith("User: ")], [ln for ln in lines if ln.startswith("Riya: ")]
    assert 6 <= len(users) <= 8 and len(riyas) == len(users)
    assert all(len(ln.split()) >= 7 for ln in riyas)  # 6+ words after "Riya:", so the scorecard can spot a leak


def test_memory_block_is_delimited():
    system = _system(Retrieved([EXAM], [EP], [THREAD]))
    assert MEMORY_START == "### What you know about them (the only true things you know):" and MEMORY_END == "### End"
    block = _block(system)
    assert "- The user has an exam on Thursday (user said)" in block and "Talked about exams" in block
    assert "Follow up: Ask how the exam went" in block


def test_empty_memory_block_is_still_delimited():
    system = _system(Retrieved([], [], []))
    assert NO_MEMORY_LINE in _block(system)


def test_system_sections_are_in_order():
    system = _system(Retrieved([EXAM], [], []), situation="S", examples="EXAMPLES", kind="small")
    order = [system.index(s) for s in ["P", "EXAMPLES", "[F]", MEMORY_START, MEMORY_END, HONESTY_LINE,
                                       "[Situation] S", "### Reply rules:"]]
    assert order == sorted(order)


def test_style_examples_never_sit_inside_the_memory_block():
    ex = style_examples()
    for retrieved in (Retrieved([], [], []), Retrieved([EXAM], [EP], [THREAD])):
        for kind in (None, "small", "about_me"):
            system = _system(retrieved, examples=ex, kind=kind)
            assert ex in system and ex not in _block(system)
            assert system.index(ex) < system.index(MEMORY_START)


@pytest.mark.parametrize("kind, rule", [
    ("small", "### Reply rules: Reply in at most 3 sentences. " + HARD),
    ("explain", "### Reply rules: Explain clearly in at most 8 sentences. No ~ in explanations. " + HARD),
    ("about_me", "### Reply rules: List what's in the memory block, nothing else. " + HARD)])
def test_reply_rules_line_per_turn_kind(kind, rule):
    assert HARD == ("You're an AI and say so if asked. Only the memory block is true about them; if something "
                    "isn't there, say you don't know.")
    assert reply_rule(kind) == rule
    assert _system(Retrieved([], [], []), kind=kind).endswith(rule)


def test_no_reply_rules_line_without_a_turn_kind():
    assert "### Reply rules" not in _system(Retrieved([], [], [])) and reply_rule(None) is None


def test_about_me_block_holds_only_facts():
    system = _system(Retrieved([EXAM], [EP], [THREAD]), kind="about_me")
    block = _block(system)
    assert "exam on Thursday" in block and "Talked about exams" not in block and "Follow up" not in block
    assert system.index("Talked about exams") > system.index(MEMORY_END)


def test_about_me_with_no_facts_says_she_knows_little():
    block = _block(_system(Retrieved([], [EP], []), kind="about_me"))
    assert NO_ABOUT_ME_LINE in block and "Talked about exams" not in block


def test_local_route_gets_local_persona_and_online_route_gets_the_full_one():
    from tests.fakes import FakeModel, make_core
    online = FakeModel("online", ["Recursion is a function calling itself."])
    core, _, _, local = make_core(local_replies=["Hey you."], online=online)
    list(core.reply("hi riya"))
    list(core.reply("explain recursion to me"))
    local_sys, online_sys = local.calls[-1][0]["content"], online.calls[-1][0]["content"]
    assert local_sys.startswith(persona_prompt("local")) and online_sys.startswith(persona_prompt("online"))
    assert style_examples() in local_sys and style_examples() in online_sys
    assert local_sys.endswith(reply_rule("small")) and online_sys.endswith(reply_rule("explain"))


def test_no_memory_line_is_neutral():
    assert NO_MEMORY_LINE == "(nothing stored about this yet)"


def test_remember_situation_quotes_the_saved_fact():
    from riya.core.prompt import remember_situation
    assert remember_situation("The user said: I hate cilantro") == (
        'You just saved this to memory: "The user said: I hate cilantro". '
        "Confirm in one short sentence that you'll remember it.")
