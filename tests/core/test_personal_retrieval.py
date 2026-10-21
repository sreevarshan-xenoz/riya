"""v0.11 Task 6: retrieval_eval --personal. The dev set and the probe bank, measured through Memory.retrieve(personal=True)
on the recorded bge vectors: A3 and A6 (recall, dragging), A4 (a printed report), A5 (bounds), A8 (non-personal retrieval
unchanged), A10 (the honesty probes hold no fact) and the six `relevant` annotations on the bank."""
import json

import pytest

from evaluation import retrieval_eval
from evaluation.persona_eval import PERSONAL_PROBES, load_probes
from evaluation.recorded_vectors import RecordedEmbedder
from evaluation.retrieval_eval import DEV, HONESTY_KINDS, NEVER_FIRE_KINDS, bank_report, format_personal, \
    personal_report, seeded
from riya.core.memory import personal_request

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


def test_bounds_on_the_dev_set_and_the_bank(reports):
    for r in reports:
        assert r["max_facts"] <= 5 and r["max_filled"] <= 3


def test_nonpersonal_retrieval_is_unchanged(reports):
    for r in reports:
        assert r["nonpersonal_identical"] == 1.0


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
    """A tiny dev set: one hit with one dragged fact, one miss, one negative that must not fire."""
    data = {"facts": [{"id": "a", "text": "The user is obsessed with salted caramel popcorn"},
                      {"id": "b", "text": "The user hates mushrooms"}],
            "requests": [{"id": "r1", "q": "pick a snack for me, you know what i like", "relevant": ["a"],
                          "memory": ["a", "b"]},
                         {"id": "r2", "q": "pick a snack for me, you know what i like", "relevant": ["b"],
                          "memory": ["a"]}],
            "negatives": [{"id": "n1", "q": "what's my dog called again", "memory": ["a", "b"]}]}
    rep = personal_report(data, embedder)
    assert rep["requests"] == 2 and rep["fired"] == 2 and rep["fire_rate"] == 1.0
    assert rep["negatives_fired"] == [] and rep["misses"] == ["r2"] and rep["personal_recall"] == 0.5
    assert rep["max_facts"] <= 5 and rep["max_filled"] <= 3 and rep["nonpersonal_identical"] == 1.0
    assert personal_request(data["requests"][0]["q"]) and not personal_request(data["negatives"][0]["q"])


def test_format_personal_prints_one_line_per_number(reports):
    own, pooled, bank = reports
    text = format_personal(own, pooled, bank)
    assert f"bank personal recall {len(bank['personal_hits'])}/17 (prototype 16)" in text
    dev_only = format_personal(own, pooled, None)
    assert "bank" not in dev_only and "personal recall" in dev_only
    assert text.startswith(dev_only)


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
        raise AssertionError("bank_report must not run under --dev-only")

    monkeypatch.setattr(retrieval_eval, "bank_report", boom)
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


def test_personal_sim_overrides_the_floor(embedder, dev):
    """A lower floor fills at least as many facts: the override reaches the embedder."""
    low = RecordedEmbedder(personal_sim=0.40)
    assert low.personal_sim == 0.40 and embedder.personal_sim == 0.45
    a, b = personal_report(dev, low), personal_report(dev, embedder)
    assert a["dragged_mean"] >= b["dragged_mean"]
