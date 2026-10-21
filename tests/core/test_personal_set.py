"""v0.11 Task 1: the personal-request development set (tests/data/personal_set.json), written blind. It must be shaped
as the brief says and stay disjoint from the persona probe file; a hit is fixed by rewriting the item, never the test."""
import json
import math
import re
from pathlib import Path

from riya.core import embed
from riya.train.holdout import normalize, overlaps, probe_facts, probe_texts

DATA = Path(__file__).resolve().parents[1] / "data" / "personal_set.json"
NEGATIVE_KINDS = {"direct", "shared_history", "leading", "near_miss", "small_talk", "we"}
OPTIONAL_KINDS = {"identity", "remember"}


def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def _count(rows: list[dict], kind: str) -> int:
    return sum(1 for r in rows if r["kind"] == kind)


def test_counts_and_layout():
    d = _load()
    facts, reqs, negs = d["facts"], d["requests"], d["negatives"]
    assert d["version"] == 1
    assert len(facts) >= 30
    assert all(f["text"].startswith("The user") for f in facts)
    assert len(reqs) >= 24
    assert _count(reqs, "applied") >= 14
    assert _count(reqs, "topical") >= 10
    assert {r["kind"] for r in reqs} <= {"applied", "topical"}
    assert len(negs) >= 20
    for kind in NEGATIVE_KINDS:
        assert _count(negs, kind) >= 3, kind
    assert {n["kind"] for n in negs} <= NEGATIVE_KINDS | OPTIONAL_KINDS
    ids = [r["id"] for r in facts + reqs + negs]
    assert len(ids) == len(set(ids))
    fact_ids = {f["id"] for f in facts}
    for r in reqs + negs:
        assert set(r["memory"]) <= fact_ids, r["id"]
    for r in reqs:
        assert r["relevant"], r["id"]
        assert set(r["relevant"]) <= set(r["memory"]), r["id"]


def test_half_the_requests_carry_a_distractor():
    reqs = _load()["requests"]
    carrying = sum(1 for r in reqs if set(r["memory"]) - set(r["relevant"]))
    assert carrying >= math.ceil(len(reqs) / 2)


def test_near_misses_hold_a_sibling_fact():
    near = [n for n in _load()["negatives"] if n["kind"] == "near_miss"]
    for n in near:
        assert n["memory"], n["id"]


def _pairs(text: str) -> set[tuple[str, str]]:
    toks = [t for t in embed.tokenize(text) if t not in embed.STOPWORDS]
    return set(zip(toks, toks[1:]))


def _probe_names(probe_fact_texts: list[str]) -> set[str]:
    names = set()
    for fact in probe_fact_texts:
        for m in re.finditer(r"[^\W\d_]+", fact):
            word = m.group()
            before = fact[:m.start()].rstrip()
            initial = not before or before[-1] in ".!?"
            if len(word) >= 2 and word[0].isupper() and not initial:
                names.add(word.casefold())
    return names


def test_written_apart_from_the_probe_file():
    d = _load()
    texts, facts = probe_texts(), probe_facts()
    probes = texts + facts
    names = _probe_names(facts)
    assert texts and facts and names, "the probe file read as empty, so this check would pass vacuously"
    probe_pairs = set()
    for f in facts:
        probe_pairs |= _pairs(f)
    problems = []
    for item in d["facts"]:
        if overlaps(item["text"], probes):
            problems.append(f"{item['id']}: fact text overlaps a probe: {item['text']!r}")
        hit = _pairs(item["text"]) & probe_pairs
        if hit:
            problems.append(f"{item['id']}: fact shares pair(s) with a probe fact: {sorted(hit)}")
    for item in d["requests"] + d["negatives"]:
        if overlaps(item["q"], probes):
            problems.append(f"{item['id']}: question overlaps a probe: {item['q']!r}")
    texts_by_id = [(f["id"], f["text"]) for f in d["facts"]] + [(r["id"], r["q"]) for r in d["requests"] + d["negatives"]]
    for item_id, text in texts_by_id:
        hit = names & set(embed.tokenize(text))
        if hit:
            problems.append(f"{item_id}: uses probe proper name(s) {sorted(hit)}")
    assert not problems, "\n".join(problems)


def test_every_question_has_four_words():
    d = _load()
    for r in d["requests"] + d["negatives"]:
        assert len(normalize(r["q"]).split()) >= 4, r["id"]


# Pairs of near-synonyms (any order) that a positive may not use against its relevant facts: the request would then
# echo the fact in other words. Compared on the stems below.
NEAR_SYNONYMS = [
    ("bro", "brother"), ("flick", "movie"), ("flick", "film"), ("movie", "film"), ("pastime", "hobby"),
    ("mate", "friend"), ("tunes", "music"), ("mum", "mother"), ("mum", "mom"), ("finances", "budget"),
    ("money", "budget"), ("cash", "budget"), ("weekend", "saturday"), ("weekend", "sunday"), ("scary", "horror"),
    ("veggie", "vegetarian"), ("pup", "dog"), ("kitty", "cat"), ("curry", "spicy"), ("novel", "book"),
    ("telly", "tv"), ("alarm", "wake"), ("snack", "food"), ("doctor", "health"),
]


def _stems(word: str) -> set[str]:
    """The word and the forms left by stripping a trailing "ing", "es" or "s" (all of them, so the match is generous)."""
    out = {word}
    for suffix in ("ing", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            out.add(word[:-len(suffix)])
    return out


def _content_stems(text: str) -> set[str]:
    out: set[str] = set()
    for tok in embed.tokenize(text):
        if tok not in embed.STOPWORDS:
            out |= _stems(tok)
    return out


def test_positives_share_no_content_word_with_their_relevant_facts():
    d = _load()
    fact_text = {f["id"]: f["text"] for f in d["facts"]}
    problems = []
    for r in d["requests"]:
        q = _content_stems(r["q"])
        for fid in r["relevant"]:
            f = _content_stems(fact_text[fid])
            if q & f:
                problems.append(f"{r['id']} / {fid}: shares {sorted(q & f)}")
            for a, b in NEAR_SYNONYMS:
                if (a in q and b in f) or (b in q and a in f):
                    problems.append(f"{r['id']} / {fid}: near-synonyms {a}/{b}")
    assert not problems, "\n".join(problems)


def test_no_two_questions_open_the_same_way():
    d = _load()
    seen: dict[str, str] = {}
    for r in d["requests"] + d["negatives"]:
        opening = " ".join(normalize(r["q"]).split()[:3])
        assert opening not in seen, f"{r['id']} opens like {seen[opening]}: {opening!r}"
        seen[opening] = r["id"]
