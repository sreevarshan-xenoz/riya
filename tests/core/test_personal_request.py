"""v0.11 Task 4: the personal-request detector (§4.1), written from the frozen dev set (tests/data/personal_set.json),
and the naming-word fix (§4.3): "called" and "named" say nothing about a topic, so they're never shared content."""
import json
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.memory import Memory, about_me, personal_request
from riya.core.store import open_db
from tests.embed_stubs import TableEmbedder

DATA = Path(__file__).resolve().parents[1] / "data" / "personal_set.json"


def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_fires_on_the_dev_set_s_requests():
    reqs = _load()["requests"]
    misses = [(r["id"], r["q"]) for r in reqs if not personal_request(r["q"])]
    assert (len(reqs) - len(misses)) / len(reqs) >= 0.9, misses


def test_never_fires_on_the_dev_set_s_negatives():
    negatives = _load()["negatives"]
    assert [n["id"] for n in negatives if personal_request(n["q"])] == []


def test_about_me_wins():
    for q in ("what do you remember about me?", "tell me everything you know about me", "what have i told you"):
        assert about_me(q), q
        assert not personal_request(q), q


def test_the_spec_s_never_forms():
    never = (
        "what's my cat's name?", "who's my best friend?", "when's my birthday?", "how old is my brother?",
        "what did i name my dog?", "what did i tell you about my flatmate?", "remember when i told you about the trip?",
        "you know my cat's name, right?", "do you know my sister's name?", "do you happen to know my favourite colour?",
        "are you an ai?", "what should we do tonight?", "remember that i hate mornings",
    )
    assert [q for q in never if personal_request(q)] == []


def test_the_spec_s_fire_forms():
    fire = (
        "pick a snack for me, you know what i like", "you know how i am", "you know my taste, choose one",
        "what should i eat tonight?", "what should i listen to?", "what should i ask them for?",
        "find something i'd actually enjoy", "is this jacket very me?", "what can't i eat?",
        "what do you know about my sister?", "what did you remember about where i work?",
        "what have you picked up about me?", "describe me in three words", "what kind of person am i?",
    )
    assert [q for q in fire if not personal_request(q)] == []


def test_case_and_curly_apostrophes_never_change_a_verdict():
    d = _load()
    changed = []
    for row in d["requests"] + d["negatives"]:
        q, verdict = row["q"], personal_request(row["q"])
        for variant in (q.upper(), q.replace("'", "’"), q + " pls~"):
            if personal_request(variant) != verdict:
                changed.append((row["id"], variant))
    assert changed == []


PEREGRINE = "The user's best friend is called Peregrine"
ROLO = "The user's dog is called Rolo"
BUBBLES = "The user's goldfish is called Bubbles"


def _mem(sims: dict[str, float]) -> Memory:
    mem = Memory(open_db(":memory:"), TableEmbedder(sims), FakeClock())
    for text in sims:
        mem.add_fact(text, source="user_said")
    return mem


def test_naming_words_are_not_shared_content():
    query = "what's my neighbour's dog called?"
    mem = _mem({PEREGRINE: 0.65, ROLO: 0.65})
    assert [f.text for f in mem.retrieve(query).facts] == [ROLO]
    strong = _mem({PEREGRINE: 0.65, ROLO: 0.65, BUBBLES: 0.85})  # shares only "called", kept by the strong branch
    assert sorted(f.text for f in strong.retrieve(query).facts) == sorted([ROLO, BUBBLES])
