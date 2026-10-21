"""v0.11 Task 4: the personal-request detector (§4.1), written from the frozen dev set (tests/data/personal_set.json),
and the naming-word fix (§4.3): "called" and "named" say nothing about a topic, so they're never shared content."""
import json
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.memory import PERSONAL_RE, Memory, about_me, personal_request
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


def _fires_raw(q: str) -> bool:
    """PERSONAL_RE alone, before the exclusions and the about-me guard."""
    return PERSONAL_RE.search(q.replace("’", "'").lower()) is not None


def test_about_me_wins():
    for q in ("what do you remember about me?", "tell me everything you know about me", "what have i told you"):
        assert about_me(q), q
        assert not personal_request(q), q
    both = ("what do you remember about me? pick something for me", "what have i told you? describe me")
    assert [q for q in both if not (about_me(q) and _fires_raw(q))] == []
    assert [q for q in both if personal_request(q)] == []


def test_exclusions_beat_a_fire_form():
    """Each line holds a fire form and an exclusion: only NOT_PERSONAL_RE keeps it from firing."""
    mixed = (
        "what's my dog called? you know what i like", "how old is my brother? as far as you know",
        "what did i name my dog? you know how i am", "what did i tell you about my flatmate, as far as you know?",
        "do you know my favourite snack? pick one for me", "you know my cat's name, right? pick one for me",
        "now that you know what i like, pick a film", "don't forget to pick something for me",
        "let's pick something for me", "are you an ai? pick a film for me",
    )
    assert [q for q in mixed if not _fires_raw(q)] == []
    assert [q for q in mixed if personal_request(q)] == []


def test_leading_typed_and_identity_exclusions():
    """Fix round 1: "do you know what i like" is a leading question (§4.1); apostrophes dropped or typed as ‘ or ʼ
    still hit the exclusions; "human", "real person", "sentient" and "conscious" are identity words."""
    mixed = (
        "do you know what i like to eat?", "do you know what i like?", "do you know how i am?",
        "did you remember where i like to eat? pick one for me",
        "whats my mums name, as far as you know?", "whos my best friend, as far as you know?",
        "wheres my brother these days, as far as you can tell?", "what‘s my dog called? you know what i like",
        "whatʼs my dog called? you know what i like", "dont forget to pick something for me",
        "lets pick something for me", "let‘s pick something for me",
        "are you human? pick a film for me", "are you a real person? pick a film for me",
        "are you sentient? suggest something", "are you conscious? pick a film for me",
    )
    assert [q for q in mixed if not _fires_raw(q)] == []
    assert [q for q in mixed if personal_request(q)] == []


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


def test_plural_direct_attribute_questions_never_fire():
    """Final review I1 (a post-freeze tightening, §4.1): "what are my kids called", "who are my best friends" and
    "where are my parents from" are direct attribute questions, so a hedge or "describe me" never makes them fire.
    A topical plural ("what are my pet peeves, as far as you can tell") still does."""
    plural = (
        "what are my kids called, as far as you know?", "who are my best friends, as far as you know",
        "where are my parents from, as far as you know", "what are my pets called? describe me",
        "what're my cats named, as far as you can tell?", "who were my exes, as far as you know?",
        "what were my grandparents' names, as far as you know?", "where were my parents from? describe me",
        "whatre my kids called, as far as you know?", "who’re my best friends, as far as you know?",
    )
    assert [q for q in plural if not _fires_raw(q)] == []
    assert [q for q in plural if personal_request(q)] == []
    assert personal_request("what are my pet peeves, as far as you can tell")
    assert personal_request("what are my habits like, as far as you know?")


def test_case_and_curly_apostrophes_never_change_a_verdict():
    d = _load()
    changed = []
    for row in d["requests"] + d["negatives"]:
        q, verdict = row["q"], personal_request(row["q"])
        for variant in (q.upper(), q.replace("'", "’"), q + " pls~"):
            if personal_request(variant) != verdict:
                changed.append((row["id"], variant))
    assert changed == []


OTHER_KINDS = {"unknown", "ai", "ai_offpath", "small", "explain", "remember", "greeting"}


def _bank_turns(kinds: set[str]) -> list[tuple[str, str]]:
    from evaluation.persona_eval import load_probes
    return [(f"{c['id']}/{t['id']}", t["text"]) for c in load_probes()["conversations"] for t in c["turns"]
            if t["kind"] in kinds and t.get("text")]


def test_never_fires_on_the_bank_s_other_turns():
    """A2: the bank is a held-out report; the detector was frozen before this test was written."""
    turns = _bank_turns(OTHER_KINDS)
    assert turns
    assert [pid for pid, text in turns if personal_request(text)] == []


def test_bank_about_me_hits_are_reported():
    turns = _bank_turns({"about_me"})
    hits = [pid for pid, text in turns if personal_request(text)]
    print(f"\npersonal_request fires on {len(hits)} of {len(turns)} bank about-me turns: {hits}")
    assert isinstance(hits, list)


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
