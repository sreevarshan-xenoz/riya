"""v0.11 Task 6: retrieval_eval --personal. The dev set and the probe bank, measured through Memory.retrieve(personal=True)
on the recorded bge vectors: A3 and A6 (recall, dragging), A4 (a printed report), A5 (bounds), A8 (non-personal retrieval
unchanged), A10 (the honesty probes hold no fact) and the six `relevant` annotations on the bank.

The report functions are also run on small memories built from recorded texts, where every number is known and
non-zero (the positive controls): an "is it empty" assertion alone would pass if a metric were never filled."""
import dataclasses
import json

import pytest

from evaluation import retrieval_eval
from evaluation.persona_eval import PERSONAL_PROBES, load_probes
from evaluation.recorded_vectors import RecordedEmbedder
from evaluation.retrieval_eval import DEV, HONESTY_KINDS, NEVER_FIRE_KINDS, bank_report, format_personal, \
    personal_report, seeded
from riya.core.memory import Memory, personal_request

ANNOTATED = {"g_about_04": [0], "g_about_05": [0], "g_about_08": [1], "g_about_12": [0], "g_about_16": [0],
             "g_about_21": [0]}


@pytest.fixture(scope="module")
def embedder():
    return RecordedEmbedder()


@pytest.fixture(scope="module")
def dev():
    return json.loads(DEV.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reports(dev, embedder):
    probes = load_probes()
    return (personal_report(dev, embedder), personal_report(dev, embedder, pooled=True), bank_report(probes, embedder))


def test_the_bank_annotations():
    convs = {c["id"]: c for c in load_probes()["conversations"]}
    annotated = {c["id"]: t["relevant"] for c in convs.values() for t in c["turns"] if "relevant" in t}
    assert annotated == ANNOTATED
    for cid, indices in annotated.items():
        assert all(0 <= i < len(convs[cid]["facts"]) for i in indices), cid
        assert "distractor" in convs[cid]["turns"][0]["tags"], cid
    assert len(PERSONAL_PROBES) == len(set(PERSONAL_PROBES)) == 17
    for cid in PERSONAL_PROBES:
        assert cid.startswith("g_about_") and cid in convs, cid
        tags = convs[cid]["turns"][0]["tags"]
        assert "applied" in tags or "open" in tags or cid == "g_about_12", cid


def test_seeded_holds_every_text_at_one_time(embedder):
    texts = ["The user is a college student", "The user listens to lo-fi music while working"]  # recorded texts
    mem, ids = seeded(texts, embedder)
    assert sorted(ids.values()) == sorted(texts)
    assert {f.id: f.text for f in mem.list_facts()} == ids
    assert len({f.valid_from for f in mem.list_facts()}) == 1


def test_dev_recall_and_dragging(reports):
    own, pooled, _ = reports
    assert own["personal_recall"] >= 0.80, own["misses"]
    assert own["dragged_mean"] <= 1.0
    assert pooled["personal_recall"] >= 0.50, pooled["misses"]
    assert pooled["dragged_mean"] <= 2.5
    assert own["negatives_fired"] == [] and pooled["negatives_fired"] == []
    assert own["fire_rate"] >= 0.9


def test_the_dev_numbers_are_pinned(reports):
    """The dev set and the recorded vectors are frozen, so the numbers are too (Task 5 measured the same ones): a metric
    that drifts, or a pooled run that stops differing from the own-memory one, shows here."""
    own, pooled, _ = reports
    assert (own["requests"], own["fired"], own["negatives_fired"]) == (35, 33, [])
    assert own["misses"] == ["r09", "r17", "t05", "t15"] and own["personal_recall"] == 31 / 35
    assert round(own["dragged_mean"], 2) == 0.23 and own["dragged_max"] == 2
    assert pooled["personal_recall"] == 22 / 35 and len(pooled["misses"]) == 13
    assert round(pooled["dragged_mean"], 2) == 1.86 and pooled["dragged_max"] == 3 and pooled["negatives_fired"] == []
    assert (own["max_facts"], own["max_filled"], pooled["max_facts"], pooled["max_filled"]) == (3, 3, 3, 3)
    assert (own["nonpersonal_n"], pooled["nonpersonal_n"]) == (28, 28)


def test_bounds_on_the_dev_set_and_the_bank(reports):
    for r in reports:
        assert r["max_facts"] <= 5 and r["max_filled"] <= 3


def test_nonpersonal_retrieval_is_unchanged(reports):
    for r in reports:
        assert r["nonpersonal_identical"] == 1.0 and r["nonpersonal_n"] > 0


def test_honesty_probes_hold_no_fact(reports):
    held = reports[2]["honesty_facts"]
    assert held == {}, "; ".join(f"{p}: {facts}" for p, facts in held.items())
    assert {"unknown", "ai", "ai_offpath"} == set(HONESTY_KINDS)


def test_the_bank_report(reports):
    bank = reports[2]
    for kind in NEVER_FIRE_KINDS:
        assert bank["fired"][kind] == [], kind
    assert bank["dragged_total"] <= 2
    assert bank["personal_total"] == 17 and set(bank["personal_hits"]) <= set(PERSONAL_PROBES)
    print(f"bank personal recall {len(bank['personal_hits'])}/{bank['personal_total']} (prototype 16)")


def test_the_report_functions_measure_what_they_say(embedder):
    """A tiny dev set: one hit with one dragged fact, one miss, a negative that must not fire and one that does."""
    data = {"facts": [{"id": "a", "text": "The user is obsessed with salted caramel popcorn"},
                      {"id": "b", "text": "The user hates mushrooms"}],
            "requests": [{"id": "r1", "q": "pick a snack for me, you know what i like", "relevant": ["a"],
                          "memory": ["a", "b"]},
                         {"id": "r2", "q": "pick a snack for me, you know what i like", "relevant": ["b"],
                          "memory": ["a"]}],
            "negatives": [{"id": "n1", "q": "what's my dog called again", "memory": ["a", "b"]},
                          {"id": "n2", "q": "pick a snack for me, you know what i like", "memory": ["a"]}]}
    rep = personal_report(data, embedder)
    assert rep["requests"] == 2 and rep["fired"] == 2 and rep["fire_rate"] == 1.0
    assert rep["negatives_fired"] == ["n2"] and rep["misses"] == ["r2"] and rep["personal_recall"] == 0.5
    # r1 drags "mushrooms" and r2 drags "popcorn": neither is the fact its request wants
    assert rep["dragged_max"] == 1 and rep["dragged_mean"] == 1.0
    assert rep["max_filled"] == 2 and rep["max_facts"] == 2
    assert rep["nonpersonal_identical"] == 1.0 and rep["nonpersonal_n"] == 1
    assert personal_request(data["requests"][0]["q"]) and not personal_request(data["negatives"][0]["q"])
    pooled = personal_report(data, embedder, pooled=True)  # every fact for every question: r2 now finds b
    assert pooled["misses"] == [] and pooled["personal_recall"] == 1.0 and pooled["negatives_fired"] == ["n2"]


# --- positive controls: each number is known and non-zero on a small memory built from recorded texts ---

def _dev_set() -> dict:
    return json.loads(DEV.read_text(encoding="utf-8"))


def _dev_texts() -> dict:
    return {f["id"]: f["text"] for f in _dev_set()["facts"]}


def _questions() -> dict:
    d = _dev_set()
    return {q["id"]: q for q in d["requests"] + d["negatives"]}


def _conv(cid: str, kind: str, req: dict, *, annotate: bool = False) -> dict:
    """A bank conversation holding a dev question's own memory, asked as one turn of `kind`."""
    text = _dev_texts()
    turn = {"id": "q", "kind": kind, "tags": ["applied"], "text": req["q"]}
    if annotate:
        turn["relevant"] = [req["memory"].index(i) for i in req["relevant"]]
    return {"id": cid, "facts": [text[i] for i in req["memory"]], "turns": [turn]}


@pytest.fixture(scope="module")
def toy_bank(embedder):
    """r01 (2 facts filled, both relevant), t13 (3 filled, 1 relevant: 2 dragged), r09 (nothing found), d1 (a plain
    question) and a greeting whose stage note is no recorded text (so retrieval must never touch it)."""
    qs = _questions()
    greet = {"id": "toy_greet", "facts": [], "turns": [{"id": "back", "kind": "greeting", "speak_first": True,
                                                         "away_hours": 30, "text": "stage note: pick a snack for me"}]}
    probes = {"conversations": [
        _conv("g_about_02", "about_me", qs["r01"], annotate=True),  # one of the 17, a hit
        _conv("g_about_05", "about_me", qs["t13"], annotate=True),  # one of the 17, a hit that drags 2
        _conv("g_about_03", "about_me", qs["r09"], annotate=True),  # one of the 17, a miss
        _conv("toy_hit", "about_me", qs["r01"], annotate=True),     # a hit, but not one of the 17
        _conv("toy_unknown", "unknown", qs["r01"]),
        _conv("toy_ai", "ai", qs["r01"]),
        _conv("toy_aioff", "ai_offpath", qs["r01"]),
        _conv("toy_small", "small", qs["d1"]),
        greet]}
    return probes, bank_report(probes, embedder)


def test_bank_report_a10_names_every_honesty_probe_with_its_facts(toy_bank):
    """A10's positive control: each honesty kind, holding facts its question reaches, is named with those facts."""
    _, rep = toy_bank
    text = _dev_texts()
    assert set(rep["honesty_facts"]) == {"toy_unknown.q", "toy_ai.q", "toy_aioff.q"}
    for facts in rep["honesty_facts"].values():
        assert sorted(facts) == sorted([text["f14"], text["f10"]])


def test_bank_report_a2_lists_what_the_detector_fires_on(toy_bank):
    """A2's positive control: the unknown, ai and off-path turns here are personal requests, so they are named; the
    plain small-talk question is not; a speak_first turn is still checked (its stage note is no recorded text, so
    retrieval never touched it)."""
    _, rep = toy_bank
    assert rep["fired"]["unknown"] == ["toy_unknown.q"] and rep["fired"]["ai"] == ["toy_ai.q"]
    assert rep["fired"]["ai_offpath"] == ["toy_aioff.q"] and rep["fired"]["greeting"] == ["toy_greet.back"]
    assert rep["fired"]["small"] == []
    assert set(rep["fired"]["about_me"]) == {"g_about_02.q", "g_about_05.q", "g_about_03.q", "toy_hit.q"}


def test_bank_report_counts_hits_dragging_and_bounds(toy_bank):
    _, rep = toy_bank
    assert rep["personal_total"] == 17
    assert rep["personal_hits"] == ["g_about_02", "g_about_05"]  # not the miss g_about_03, not toy_hit
    assert rep["dragged_total"] == 2  # t13's two filled facts that aren't its annotated one
    assert (rep["max_facts"], rep["max_filled"]) == (3, 3)


def test_dragged_counts_only_the_filled_facts(embedder):
    """An eligible fact the gate passes (wasabi peas, for a snack request) that isn't the relevant one is not dragged: only
    what the fill adds is. Here the fill adds popcorn (relevant) and mushrooms (dragged): 1, not 2."""
    wasabi, popcorn, mushrooms = ("The user's favourite snack is wasabi peas",
                                  "The user is obsessed with salted caramel popcorn", "The user hates mushrooms")
    q = "pick a snack for me, you know what i like"
    mem, _ = seeded([wasabi, popcorn, mushrooms], embedder)
    on = mem.retrieve(q, personal=True)
    assert [f.text for f in on.facts][0] == wasabi and len(on.facts) == 3 and len(on.filled) == 2
    data = {"facts": [{"id": "w", "text": wasabi}, {"id": "p", "text": popcorn}, {"id": "m", "text": mushrooms}],
            "requests": [{"id": "r", "q": q, "relevant": ["p"], "memory": ["w", "p", "m"]}], "negatives": []}
    assert personal_report(data, embedder)["dragged_mean"] == 1.0
    probes = {"conversations": [{"id": "toy", "facts": [wasabi, popcorn, mushrooms], "turns": [
        {"id": "q", "kind": "about_me", "tags": ["applied"], "text": q, "relevant": [1]}]}]}
    assert bank_report(probes, embedder)["dragged_total"] == 1


def test_bank_report_a8_has_a_real_denominator(toy_bank):
    _, rep = toy_bank
    assert rep["nonpersonal_n"] == 1 and rep["nonpersonal_identical"] == 1.0  # the d1 question only


def test_a8_notices_a_leak_on_a_non_personal_turn(monkeypatch, dev, embedder, toy_bank):
    """A8's positive control: if a non-personal question retrieved differently with the fill on, the share drops."""
    real = Memory.retrieve

    def leaky(self, query, *a, personal=False, **k):
        r = real(self, query, *a, personal=personal, **k)
        if personal and not personal_request(query):  # the bug: another fact gets in under personal=True
            extra = [f for f in self.list_facts() if f.id not in {x.id for x in r.facts}]
            return dataclasses.replace(r, facts=r.facts + extra[:1]) if extra else dataclasses.replace(r, facts=[])
        return r

    monkeypatch.setattr(Memory, "retrieve", leaky)
    assert personal_report(dev, embedder)["nonpersonal_identical"] < 1.0
    assert bank_report(toy_bank[0], embedder)["nonpersonal_identical"] < 1.0


def test_an_empty_denominator_is_not_a_hundred_percent(embedder):
    """Every question here is a personal request: nothing to compare, which must read n/a and never 100%."""
    r01 = _questions()["r01"]
    data = {"facts": [{"id": i, "text": t} for i, t in _dev_texts().items()], "requests": [r01], "negatives": []}
    rep = personal_report(data, embedder)
    assert rep["nonpersonal_n"] == 0 and rep["nonpersonal_identical"] is None
    bank = bank_report({"conversations": [_conv("toy", "about_me", r01)]}, embedder)
    assert bank["nonpersonal_n"] == 0 and bank["nonpersonal_identical"] is None
    text = format_personal(rep, rep, bank)
    assert "non-personal identical (own memory) n/a of 0" in text and "bank non-personal identical n/a of 0" in text


def test_format_personal_prints_one_line_per_number(reports):
    own, pooled, bank = reports
    text = format_personal(own, pooled, bank)
    assert f"bank personal recall {len(bank['personal_hits'])}/17 (prototype 16)" in text
    assert "bank personal probes without a relevant fact: g_about_20, g_about_21, g_about_24" in text
    assert "personal recall (own memory) 31/35 (89%); misses: r09, r17, t05, t15" in text
    dev_only = format_personal(own, pooled, None)
    assert "bank" not in dev_only and "personal recall" in dev_only
    assert text.startswith(dev_only)


def test_format_personal_names_no_missed_probe_when_all_hit(reports):
    own, pooled, bank = reports
    full = {**bank, "personal_hits": list(PERSONAL_PROBES)}
    assert "bank personal probes without a relevant fact: none" in format_personal(own, pooled, full)


def test_main_personal_prints_and_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert retrieval_eval.main(["--personal"]) == 0
    assert "bank personal recall" in capsys.readouterr().out
    lines = (tmp_path / "scratch_runs" / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["embedder"] == "recorded" and {"dev", "dev_pooled", "bank"} <= set(row)


def test_dev_only_shows_no_bank_numbers(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def boom(*a, **k):
        raise AssertionError("--dev-only must not touch the bank")

    monkeypatch.setattr(retrieval_eval, "bank_report", boom)
    monkeypatch.setattr(retrieval_eval, "load_probes", boom)  # nor read the probe file
    assert retrieval_eval.main(["--personal", "--dev-only", "--personal-sim", "0.40"]) == 0
    out = capsys.readouterr().out
    assert "personal recall" in out and "bank" not in out
    assert "/17" not in out
    lines = (tmp_path / "scratch_runs" / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "bank" not in row and row["personal_sim"] == 0.40


def test_dev_only_needs_personal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        retrieval_eval.main(["--dev-only"])


def _dev_row(tmp_path, monkeypatch, *argv) -> dict:
    monkeypatch.chdir(tmp_path)
    assert retrieval_eval.main(["--personal", "--dev-only", *argv]) == 0
    return json.loads((tmp_path / "scratch_runs" / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines()[-1])


def test_personal_sim_reaches_the_report(tmp_path, monkeypatch, embedder, dev):
    """A lower floor fills more, a higher one fewer (strictly, on the frozen dev set), and the default is 0.45."""
    default = _dev_row(tmp_path, monkeypatch)
    low = _dev_row(tmp_path, monkeypatch, "--personal-sim", "0.40")
    high = _dev_row(tmp_path, monkeypatch, "--personal-sim", "0.50")
    assert (default["personal_sim"], low["personal_sim"], high["personal_sim"]) == (0.45, 0.40, 0.50)
    assert low["dev"]["dragged_mean"] > default["dev"]["dragged_mean"] > high["dev"]["dragged_mean"]
    assert default["dev"] == personal_report(dev, embedder)


def test_personal_sim_is_refused_without_a_personal_floor(tmp_path, monkeypatch):
    """The hash embedder has no floor, so it gets no fill: an override would silently switch the fill on."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        retrieval_eval.main(["--personal", "--embedder", "hash", "--personal-sim", "0.40"])
    with pytest.raises(SystemExit):
        retrieval_eval.main(["--personal-sim", "0.40"])
    assert not (tmp_path / "scratch_runs").exists()
