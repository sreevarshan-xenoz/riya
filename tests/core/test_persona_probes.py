"""v0.10 Task 3: the gate's held-out probes. The 7 v0.9 conversations stay byte-for-byte; 80 one-turn g_* probes
follow, covering spec §4.1's table.
See docs/superpowers/specs/2026-10-21-riya-v0.10-gate-that-sees-design.md §4.1 and §4.6."""
import hashlib
import json

from evaluation.persona_eval import load_probes
from riya.core.persona import persona_prompt, style_examples
from riya.core.shaping import is_identity_question
from riya.train import holdout, scenarios

# the sha256 of the v0.9 conversations, from git show main:evaluation/persona_probes.json
V09_SHA256 = "e5f1c60b27df53a06677393111581baffc560cb07da29f604d98c5f741d05277"
STYLES = {"direct", "indirect", "playful", "pressured", "pleading", "comparison"}
POOLS = ("NAMES", "SURNAMES", "PLACES", "FOODS", "COLOURS", "JOBS", "INSTRUMENTS", "SHOWS", "LANGUAGES", "COMPANIES",
         "RIDES", "SPORTS", "HOBBIES", "DRINKS", "BOOKS", "VIDEOGAMES", "FEARS", "BEDTIMES", "DEGREES", "TEAMS",
         "NICKNAMES")


def _convs() -> list[dict]:
    return load_probes()["conversations"]


def _new(prefix: str = "g_") -> list[dict]:
    return [c for c in _convs() if c["id"].startswith(prefix)]


def _turns(prefix: str) -> list[tuple[dict, dict]]:
    return [(c, c["turns"][0]) for c in _new(prefix)]


def test_the_v09_conversations_are_unchanged():
    old = json.dumps(_convs()[:7], sort_keys=True, ensure_ascii=False)
    assert hashlib.sha256(old.encode("utf-8")).hexdigest() == V09_SHA256


def test_per_pass_counts():
    turns = [(c, t) for c in _convs() for t in c["turns"]]
    kinds = [t["kind"] for _, t in turns]
    assert kinds.count("unknown") == 30
    assert sum(t["kind"] == "about_me" and bool(c["facts"]) for c, t in turns) == 30
    assert (kinds.count("ai"), kinds.count("ai_offpath"), len(turns)) == (20, 10, 106)
    groups = {p: len(_new(p)) for p in ("g_unknown_", "g_about_", "g_ai_", "g_aioff_")}
    assert groups == {"g_unknown_": 26, "g_about_": 27, "g_ai_": 17, "g_aioff_": 10}
    for prefix, n in groups.items():  # numbered 01..n in order: the order is Resolution 24's drop order
        assert [c["id"] for c in _new(prefix)] == [f"{prefix}{i:02d}" for i in range(1, n + 1)]


def test_new_probes_are_one_turn_with_six_words():
    new = _new()
    assert len(new) == 80
    for c in new:
        assert [t["id"] for t in c["turns"]] == ["q"], c["id"]
        assert len(holdout.normalize(c["turns"][0]["text"]).split()) >= 6, c["id"]


def test_unknown_coverage():
    turns = _turns("g_unknown_")
    assert all(t["kind"] == "unknown" for _, t in turns)
    assert all(0 <= len(c["facts"]) <= 5 for c, _ in turns)
    assert len({t["attr"] for _, t in turns}) >= 15
    for c, t in turns:
        facts = set(holdout.normalize(" ".join(c["facts"])).split())
        assert not set(holdout.normalize(t["attr"]).split()) & facts, c["id"]  # no fact answers the question
        assert set(t["tags"]) <= {"near_miss", "shared_history", "leading"}, c["id"]
    tagged = lambda tag: [c for c, t in turns if tag in t["tags"]]  # noqa: E731
    assert len(tagged("near_miss")) >= 6 and all(c["facts"] for c in tagged("near_miss"))
    assert len(tagged("shared_history")) >= 8
    assert len(tagged("leading")) >= 4


def test_about_me_coverage():
    turns = _turns("g_about_")
    assert all(t["kind"] == "about_me" for _, t in turns)
    assert all(1 <= len(c["facts"]) <= 4 for c, _ in turns)
    for c, t in turns:
        assert len(set(t["tags"]) & {"direct", "open", "applied"}) == 1, c["id"]
        assert set(t["tags"]) <= {"direct", "open", "applied", "distractor"}, c["id"]
    tagged = lambda tag: [c for c, t in turns if tag in t["tags"]]  # noqa: E731
    assert len(tagged("direct")) >= 10 and len(tagged("open")) >= 6 and len(tagged("applied")) >= 6
    assert len(tagged("distractor")) >= 4 and all(len(c["facts"]) >= 2 for c in tagged("distractor"))


def test_ai_paths():
    on, off = _turns("g_ai_"), _turns("g_aioff_")
    assert all(t["kind"] == "ai" and is_identity_question(t["text"]) for _, t in on), \
        [t["text"] for _, t in on if not is_identity_question(t["text"])]
    assert all(t["kind"] == "ai_offpath" and not is_identity_question(t["text"]) for _, t in off), \
        [t["text"] for _, t in off if is_identity_question(t["text"])]
    v09 = [t for c in _convs()[:7] for t in c["turns"] if t["kind"] == "ai"]
    assert all(is_identity_question(t["text"]) for t in v09)
    styles = [t["style"] for _, t in on + off]
    assert len(styles) == 27 and set(styles) == STYLES


def test_facts_are_written_apart_from_the_pools():
    facts = [f for c in _new() for f in c["facts"]]
    assert facts and all(f.startswith("The user") for f in facts)
    values = {holdout.normalize(v) for pool in POOLS for v in getattr(scenarios, pool)}
    for f in facts:
        padded = f" {holdout.normalize(f)} "
        assert not [v for v in values if v and f" {v} " in padded], f
    persona = holdout.normalize(persona_prompt("local") + style_examples())
    assert not [f for f in facts if holdout.normalize(f) in persona]
