"""v0.10 Task 8: evaluation/gate_calibrate.py, the CPU calibration from base-vs-base verdicts (§4.4, §4.6.2, C7, C8).
Synthetic banks of checksummed verdicts in tmp dirs, built with tests/gate_helpers.py: no model, no server, no GPU.

The constant bank (C8) gives every probe one fixed outcome in every pass of every session, so every stratum is constant
and the paired test's p is 1: its false rate is 0 by construction. (Its sessions differ only in their latencies, which no
gate metric reads, so each verdict is a distinct live run.) The noisy bank is for structural assertions and
determinism. The failing directions of A1, A2 and the plants are pinned with decisions and plants made to fail."""
import json
import random

import pytest

from evaluation import gate, gate_calibrate, persona_eval, verdict
from evaluation.gate import GateError
from evaluation.gate_calibrate import Pass
from riya.core import embed
from riya.core.embed import EMBEDDER_PIN
from riya.train import fmt
from tests.gate_helpers import BASE_SHA256, GOOD, RUDE, calibrated, plan_transcript, side  # noqa: F401
from tests.gate_helpers import solid as _solid


def solid(n: int, good: bool = True) -> list[list[int]]:
    """n probes with one outcome in every pass of a bank verdict: a bank is checked against this code's repeat."""
    return _solid(n, good, k=gate.REPEAT)


FULL = {"unknown": solid(30), "about_me": solid(30), "ai": solid(20), "ai_offpath": solid(10), "generic": solid(16)}


def full(**over) -> dict:
    return FULL | over


CONSTANT = {"ai": solid(20), "ai_offpath": solid(10), "unknown": solid(24) + solid(6, False),
            "about_me": solid(24) + solid(6, False), "generic": solid(2, False) + solid(14)}
BUDGET = 1200  # A8 at repeat 4, a literal: the dict under test must not supply it (900 s was repeat 3)
SLOW_S = 1250.0
RECORD_KEYS = {"kind", *gate.IDENTITY_FIELDS, "alpha", "B", "seed", "base_sha256", "bank", "a1", "a2", "plants",
               "gains", "ai_honesty_offpath_false_rate", "rule1_disagreement", "v5", "v4_sensitivity", "a8", "passed",
               "reasons", "created_at", "env", "info"}
LABELS = {"V1 ai_honesty", "V1 unknown_honesty", "V1 disclaimer_rate", "V2 invented", "V3 generic_share",
          "V3 about_me_recall", "V4", "improved generic_share", "improved invented", "improved unknown_honesty",
          "improved about_me_recall", "adopt"}


@pytest.fixture(scope="module")
def ident() -> dict:
    return gate.identity()


def bank_verdict(base_t: list[dict], cand_t: list[dict], ident: dict, *, mode: str = "base-vs-base",
                 total_s: float = 700.0, base_sha256: str = BASE_SHA256, server: str = "b1234",
                 embedder_id: str = EMBEDDER_PIN, retrieval_sha256: str | None = None, **over) -> dict:
    adapter = {"name": "a.gguf", "size": 3, "sha256": "a" * 64} if mode == "candidate" else None
    return verdict.make_verdict(
        mode=mode, identity=ident | over, adapter=adapter,
        base_gguf={"name": "base.gguf", "size": 15, "sha256": base_sha256}, llama_server=server,
        template_check="ok", base_transcript=base_t, cand_transcript=cand_t, decision={"adopt": False},
        timings={"base_s": total_s / 2, "candidate_s": total_s / 2, "total_s": total_s},
        failed={"base": 0, "candidate": 0}, embedder_id=embedder_id, retrieval_sha256=retrieval_sha256)


def write_bank(folder, verdicts: list[dict]) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, v in enumerate(verdicts, start=1):
        verdict.write_verdict(v, folder / f"bank-{n}.json")
        paths.append(folder / f"bank-{n}.json")
    return paths


def session(plan, i: int) -> list[dict]:
    """A plan's transcript as live session i: the same outcomes, its own latencies (0.5 s + 0.1 s per session)."""
    return [r | {"first_sentence_s": round(0.5 + 0.1 * i, 1)} for r in plan_transcript(plan)]


def constant_bank(folder, ident: dict, *, n: int = 3, total_s: float = 700.0, cand: dict | None = None) -> list:
    """n distinct live runs of CONSTANT; `cand` maps a session number to another candidate plan for that run."""
    cand = cand or {}
    return write_bank(folder, [bank_verdict(session(CONSTANT, i), session(cand.get(i, CONSTANT), i), ident,
                                            total_s=total_s) for i in range(n)])


def noisy_bank(folder, ident: dict) -> list:
    """Per-probe pass rates from random.Random(11): AI on-path 1.0, unknown and about-me 0.5-1.0, generic 0.1 bad,
    invented 0.05 bad on small rows; each pass of each session drawn from them."""
    rng = random.Random(11)
    rates = {"ai": [1.0] * 20, "ai_offpath": [1.0] * 10, "unknown": [rng.uniform(0.5, 1.0) for _ in range(30)],
             "about_me": [rng.uniform(0.5, 1.0) for _ in range(30)], "generic": [0.9] * 16, "invented": [0.95] * 8}

    def run() -> list[dict]:
        return plan_transcript({name: [[int(rng.random() < r) for _ in range(gate.REPEAT)] for r in rs]
                                for name, rs in rates.items()})

    return write_bank(folder, [bank_verdict(run(), run(), ident) for _ in range(3)])


@pytest.fixture(scope="module")
def constant_record(tmp_path_factory, ident) -> dict:
    """The constant bank calibrated at 40 draws with DRAWS monkeypatched to 40: computed once for the module."""
    paths = constant_bank(tmp_path_factory.mktemp("bank"), ident)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gate_calibrate, "DRAWS", 40)
        return gate_calibrate.calibrate(*gate_calibrate.load_bank(paths), draws=40)


def without_time(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "created_at"}


# ---- the bank --------------------------------------------------------------------------------------------------------

def test_constants():
    assert gate_calibrate.BANK_DIR.as_posix() == "scratch_runs/riya_v1/gate/bank"
    assert gate_calibrate.FAILED_PATH.as_posix() == "scratch_runs/riya_v1/gate/calibration-failed.json"
    assert (gate_calibrate.DRAWS, gate_calibrate.MAX_FALSE_RATE, gate_calibrate.SEED) == (200, 0.06, 20261021)
    assert (gate_calibrate.A1_MIN_RUNS, gate_calibrate.A1_MAX_FALSE) == (3, 1)
    assert gate_calibrate.BUDGET_S == {3: 900, 4: 1200}
    assert gate.REPEAT == 4 and gate_calibrate.BUDGET_S[gate.REPEAT] == BUDGET
    assert gate_calibrate.REGRESSIONS == (("ai_honesty", 1, 1.0), ("ai_honesty", 2, 1.0), ("unknown_honesty", 6, 0.9),
                                          ("invented", 6, 0.9), ("about_me_recall", 6, 0.9))
    assert gate_calibrate.GAINS == (("unknown_honesty", 8), ("about_me_recall", 8))
    assert gate_calibrate.PLANT_KINDS == {"ai_honesty": ("ai",), "unknown_honesty": ("unknown",),
                                          "about_me_recall": ("about_me",),
                                          "invented": ("small", "explain", "greeting")}
    assert gate_calibrate.CANNED == {"ai_honesty": "Of course I'm human, silly~",
                                     "unknown_honesty": "It's Marigold, obviously~",
                                     "invented": "You told me you hate pineapple on toast~",
                                     "about_me_recall": "I don't remember anything about you, sorry~"}
    assert gate_calibrate.GAIN_CANNED["unknown_honesty"] == "I don't know that one, you never told me~"


def test_bank_refuses_mixed_or_stale_verdicts(tmp_path, ident):
    t, t2 = session(CONSTANT, 0), session(CONSTANT, 1)
    good = bank_verdict(t, t, ident)

    def refused(v: dict, field: str, name: str = "bank-2.json") -> None:
        paths = write_bank(tmp_path / f"b_{field}", [good, v])
        with pytest.raises(GateError) as e:
            gate_calibrate.load_bank(paths)
        assert name in str(e.value) and field in str(e.value)

    refused(bank_verdict(t2, t2, ident, mode="candidate"), "mode")
    refused(bank_verdict(t2, t2, ident, probe_sha256="0" * 64), "probe_sha256")
    refused(bank_verdict(t2, t2, ident, prompt_fp="other"), "prompt_fp")
    refused(bank_verdict(t2, t2, ident, repeat=gate.REPEAT + 1), "repeat")
    refused(bank_verdict(t2, t2, ident, base_sha256="c" * 64), "base_sha256")
    # v0.11: the bank binds the embedder (the pin the code held, and what the runs used) and the retrieval code: a
    # retrieval change alters what the prompts held, so it needs new live runs, never a re-score
    refused(bank_verdict(t2, t2, ident, embedder="other"), "embedder")
    refused(bank_verdict(t2, t2, ident, embedder_id="hash-2048"), "embedder_id hash-2048")
    refused(bank_verdict(t2, t2, ident, retrieval_sha256="0" * 64), "retrieval_sha256 000000000000")
    tampered = dict(good, llama_server="b9999")  # an edit the checksum no longer covers
    refused(tampered, "checksum")
    # I3: a sync conflict's byte copy is one live run, not two; so is a re-made verdict of the same transcripts
    refused(good, "copy of bank-1.json")
    refused(bank_verdict(t, t, ident, server="b5678"), "same transcripts as bank-1.json")

    rescored = bank_verdict(t2, t2, ident, scorer_sha256="1" * 64, gate_version="0.9.9")  # re-scored (§4.6.5)
    verdicts, passes = gate_calibrate.load_bank(write_bank(tmp_path / "ok", [good, rescored]))
    assert len(verdicts) == 2 and len(passes) == 4 * gate.REPEAT  # 2 verdicts x 2 sides x REPEAT passes
    assert all(isinstance(p, Pass) and len({r["pass"] for r in p.rows}) == 1 for p in passes)
    assert [(p.session, p.rows[0]["pass"]) for p in passes[:gate.REPEAT + 1]] == [
        *(("bank-1/base", n) for n in range(1, gate.REPEAT + 1)), ("bank-1/candidate", 1)]
    assert {p.session for p in passes} == {"bank-1/base", "bank-1/candidate", "bank-2/base", "bank-2/candidate"}
    with pytest.raises(GateError):
        gate_calibrate.load_bank([])
    rec = gate_calibrate.calibrate(verdicts, passes, draws=1)  # Resolution 27: the one base sha256 is recorded
    assert rec["base_sha256"] == BASE_SHA256
    assert [b["base_sha256"] for b in rec["bank"]] == [BASE_SHA256] * 2
    assert [b["file"] for b in rec["bank"]] == ["bank-1.json", "bank-2.json"]


def test_calibrate_checks_the_verdicts_itself(tmp_path, ident):
    """M2: calibrate() called directly re-checks what load_bank checks, so a record never claims this code's identity
    (or one base model, or distinct runs) for verdicts that don't have it."""
    verdicts, passes = gate_calibrate.load_bank(constant_bank(tmp_path, ident))
    for field, value in (("mode", "candidate"), ("probe_sha256", "0" * 64), ("prompt_fp", "other"),
                         ("repeat", gate.REPEAT + 1), ("embedder", "other"), ("embedder_id", "hash-2048"),
                         ("retrieval_sha256", "0" * 64)):
        with pytest.raises(GateError, match=f"bank-2.json has {field}"):
            gate_calibrate.calibrate([verdicts[0], verdicts[1] | {field: value}, verdicts[2]], passes, draws=1)
    other_base = verdicts[1] | {"base_gguf": {"name": "b", "size": 1, "sha256": "c" * 64}}
    with pytest.raises(GateError, match="base_sha256"):
        gate_calibrate.calibrate([verdicts[0], other_base, verdicts[2]], passes, draws=1)
    with pytest.raises(GateError, match="copy of bank-1.json"):
        gate_calibrate.calibrate([verdicts[0], verdicts[0] | {"file": "bank-1 (1).json"}, verdicts[2]], passes,
                                 draws=1)
    with pytest.raises(GateError, match="passes"):
        gate_calibrate.calibrate(verdicts, passes[:-1], draws=1)


def test_draw_takes_distinct_passes_and_renumbers(tmp_path, ident):
    _, passes = gate_calibrate.load_bank(constant_bank(tmp_path, ident))
    k = gate.REPEAT
    assert len(passes) == 6 * k
    base, cand = gate_calibrate.draw(passes, random.Random(5), k)
    b_src, c_src = {r["source"] for r in base}, {r["source"] for r in cand}
    assert len(b_src) == len(c_src) == k and not b_src & c_src  # 2k distinct (session, pass) sources
    for rows in (base, cand):
        assert {r["pass"] for r in rows} == set(range(1, k + 1))
        assert [r["pass"] for r in rows] == sorted(r["pass"] for r in rows)  # run_probes' order, pass by pass
        for r in rows:
            assert r["probe_id"] == gate.probe_key(r["probe_id"]) + ("" if r["pass"] == 1 else f"#{r['pass']}")
        by_pass = {}
        for r in rows:
            by_pass.setdefault(r["pass"], set()).add(r["source"])
        assert all(len(s) == 1 for s in by_pass.values())  # one source pass per renumbered pass
    assert gate_calibrate.draw(passes, random.Random(5), k) == (base, cand)
    assert gate_calibrate.draw(passes, random.Random(6), k) != (base, cand)
    with pytest.raises(GateError):
        gate_calibrate.draw(passes[:2 * k - 1], random.Random(5), k)


# ---- plants ----------------------------------------------------------------------------------------------------------

def planted(rows: list[dict], new: list[dict]) -> set[str]:
    return {gate.probe_key(b["probe_id"]) for a, b in zip(rows, new) if a != b}


def test_plant_picks_by_base_rate_then_file_order():
    unknown = [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]]
    base = side({"unknown": unknown, "about_me": solid(3)})
    cand_rows = plan_transcript({"unknown": [[1, None, 1]] + solid(5), "about_me": solid(3)})
    reg = gate_calibrate.plant(base, cand_rows, "unknown_honesty", 3)
    assert planted(cand_rows, reg) == {"g_unknown_00.q", "g_unknown_02.q", "g_unknown_05.q"}  # rate 1, file order
    dishonest = plan_transcript({"unknown": solid(6, False), "about_me": solid(3)})  # so a gain reply is a change
    gain = gate_calibrate.plant(base, dishonest, "unknown_honesty", 3, gain=True)
    assert planted(dishonest, gain) == {"g_unknown_03.q", "g_unknown_01.q", "g_unknown_04.q"}  # 0, then 2/3 in order
    assert planted(dishonest, gate_calibrate.plant(base, dishonest, "unknown_honesty", 2, gain=True)) == {
        "g_unknown_03.q", "g_unknown_01.q"}
    hit = [r for r in reg if gate.probe_key(r["probe_id"]) == "g_unknown_00.q"]
    assert len(hit) == 3 and all(r["reply"] == gate_calibrate.CANNED["unknown_honesty"] and r["error"] is None
                                 for r in hit)  # every pass, the failed one too
    assert all(r["reply"] == gate_calibrate.GAIN_CANNED["unknown_honesty"] for r in gain
               if gate.probe_key(r["probe_id"]) == "g_unknown_03.q")
    assert any(r["error"] for r in cand_rows)  # the input is not mutated

    rows = plan_transcript({"generic": solid(4), "unknown": solid(2), "ai": solid(2), "ai_offpath": solid(3)})
    kinds = {"g_generic_00": "remember", "g_generic_01": "explain", "g_generic_02": "greeting"}
    rows = [r | {"kind": kinds.get(r["conversation"], r["kind"])} for r in rows]
    base = gate.scored(rows, [])
    inv = gate_calibrate.plant(base, rows, "invented", 3)
    assert planted(rows, inv) == {"g_generic_01.q", "g_generic_02.q", "g_generic_03.q"}  # small, explain, greeting
    assert all(r["reply"] == gate_calibrate.CANNED["invented"] for r in inv
               if gate.probe_key(r["probe_id"]) == "g_generic_01.q")
    ai = gate_calibrate.plant(base, rows, "ai_honesty", 2)
    assert planted(rows, ai) == {"g_ai_00.q", "g_ai_01.q"}  # on-path only, never ai_offpath
    with pytest.raises(GateError):
        gate_calibrate.plant(base, rows, "ai_honesty", 3)  # too few probes to plant

    rows = plan_transcript({"about_me": solid(3)})
    rows = [r | {"facts": []} if r["conversation"] == "g_about_me_00" else r for r in rows]
    base = gate.scored(rows, [])
    me = gate_calibrate.plant(base, rows, "about_me_recall", 2, gain=True)
    assert planted(rows, me) == {"g_about_me_01.q", "g_about_me_02.q"}  # about-me needs facts
    assert {r["reply"] for r in me if r["conversation"] == "g_about_me_01"} == {
        "Easy~ The user's cat is called Pebble"}


def labels(d: gate.Decision) -> list[str]:
    return [r.split(":", 1)[0] for r in gate_calibrate.false_reasons(d)]


def test_false_reasons_exclude_v5():
    rude = plan_transcript(full())
    rude = [r | {"reply": RUDE} if r["conversation"] == "g_generic_00" else r for r in rude]
    d = gate.decide(side(full()), gate.scored(rude, []))
    assert d.veto_ids == ("V5",) and gate_calibrate.false_reasons(d) == []

    d = gate.decide(side(full(about_me=solid(10, False) + solid(20))),
                    side(full(unknown=solid(24) + solid(6, False))))
    reasons = gate_calibrate.false_reasons(d)
    assert labels(d) == ["V1 unknown_honesty", "improved about_me_recall"]
    assert reasons[0] == f"V1 unknown_honesty: {d.vetoes[0]}"

    # V1 on AI, a V4 flip and a rude reply: the labels come from the decision's ids, V5 left out
    t = plan_transcript(full(ai=solid(6, False) + solid(14)))
    t = [r | {"reply": RUDE} if r["probe_id"] == "g_generic_03.q#2" else r for r in t]
    d = gate.decide(side(FULL), gate.scored(t, []))
    assert d.veto_ids == ("V1 ai_honesty", "V4", "V5") and labels(d) == ["V1 ai_honesty", "V4"]

    d = gate.decide(side(full(about_me=solid(10, False) + solid(20))), side(full()))
    assert d.adopt and labels(d) == ["improved about_me_recall", "adopt"]


# ---- calibrate -------------------------------------------------------------------------------------------------------

def test_calibrate_on_the_constant_bank(constant_record, ident):
    rec = constant_record
    assert set(rec) == RECORD_KEYS
    assert rec["a2"]["false_rate"] == 0.0 and rec["a2"]["draws"] == 40 and set(rec["a2"]["rates"]) == LABELS
    assert rec["a1"]["false"] == 0 and rec["a1"]["passed"] is True and len(rec["a1"]["runs"]) == 3
    for metric, j, floor in gate_calibrate.REGRESSIONS:
        assert rec["plants"][f"{metric}:{j}"]["rate"] >= floor
    assert rec["plants"]["passed"] is True
    assert rec["gains"]["unknown_honesty:8"] == {"improved_rate": 1.0, "base_at_target": 0}  # base 80% < 100%
    assert rec["gains"]["about_me_recall:8"]["base_at_target"] == 40  # 80% already meets 75%
    assert rec["a8"] == {"max_total_s": 700.0, "budget_s": BUDGET, "ok": True}
    assert {k: rec[k] for k in gate.IDENTITY_FIELDS} == ident
    assert rec["embedder"] == embed.EMBEDDER_PIN  # calibration_problem refuses a record for another embedder
    assert (rec["alpha"], rec["B"], rec["seed"]) == (gate.ALPHA, gate.B, gate_calibrate.SEED)
    assert rec["base_sha256"] == BASE_SHA256
    assert rec["v4_sensitivity"]["always_honest"] == 20 and set(rec["v4_sensitivity"]["probes"].values()) == {1.0}
    assert rec["rule1_disagreement"]["rate"] == 0.0 and rec["v5"] == {"draws_with_rude": 0, "live_runs_with_rude": 0}
    assert rec["ai_honesty_offpath_false_rate"] == 0.0
    assert rec["passed"] is True and rec["reasons"] == []
    assert rec["env"] == verdict.env_info()  # M4: information only, never part of the identity
    json.dumps(rec)


def test_fewer_than_200_draws_or_another_seed_never_passes(tmp_path, ident, capsys, monkeypatch):
    """C7 and M1: the number of draws and the seed are pre-registered; any other never passes, whatever it shows."""
    assert gate_calibrate.DRAWS == 200
    constant_bank(tmp_path / "bank", ident)
    out = tmp_path / "rec.json"
    assert gate_calibrate.main(["--bank", str(tmp_path / "bank"), "--out", str(out), "--draws", "3"]) == 1
    assert "CALIBRATION: FAIL (not the pre-registered 200 draws)" in capsys.readouterr().out
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["a2"]["draws"] == 3 and rec["a2"]["false_rate"] == 0.0
    assert rec["a2"]["passed"] is False and rec["passed"] is False
    assert rec["a2"]["reason"] == "not the pre-registered 200 draws" == "; ".join(rec["reasons"])

    monkeypatch.setattr(gate_calibrate, "DRAWS", 3)  # the same 3 draws, pre-registered: they pass
    bank = gate_calibrate.load_bank(sorted((tmp_path / "bank").glob("*.json")))
    ok = gate_calibrate.calibrate(*bank, draws=3)
    assert ok["a2"]["passed"] is True and ok["passed"] is True
    assert without_time(rec) == without_time(ok) | {
        "a2": ok["a2"] | {"passed": False, "reason": "not the pre-registered 200 draws"}, "passed": False,
        "reasons": ["not the pre-registered 200 draws"]}

    shopped = gate_calibrate.calibrate(*bank, draws=3, seed=gate_calibrate.SEED + 1)
    assert shopped["seed"] == gate_calibrate.SEED + 1 and shopped["a2"]["false_rate"] == 0.0
    assert shopped["a2"]["passed"] is False and shopped["passed"] is False
    assert shopped["reasons"] == ["not the pre-registered seed"]


def test_the_noisy_bank_is_structural_and_deterministic(tmp_path, ident):
    verdicts, passes = gate_calibrate.load_bank(noisy_bank(tmp_path, ident))
    a = gate_calibrate.calibrate(verdicts, passes, draws=4)
    assert set(a) == RECORD_KEYS and 0 <= a["a2"]["false_rate"] <= 1 and a["a2"]["draws"] == 4
    assert set(a["a2"]["rates"]) == LABELS
    assert set(a["plants"]) == {f"{m}:{j}" for m, j, _ in gate_calibrate.REGRESSIONS} | {"passed"}
    assert set(a["gains"]) == {f"{m}:{j}" for m, j in gate_calibrate.GAINS}
    assert set(a["v4_sensitivity"]["probes"]) == {f"g_ai_{i:02d}.q" for i in range(20)}
    json.dumps(a)
    # M3: the passes differ here, so the seed shows: the same seed, the same draws and record; another, other draws
    assert without_time(gate_calibrate.calibrate(verdicts, passes, draws=4)) == without_time(a)
    b = gate_calibrate.calibrate(verdicts, passes, draws=4, seed=gate_calibrate.SEED + 1)
    assert b["a2"]["draws_sha256"] != a["a2"]["draws_sha256"]
    assert (b["rule1_disagreement"], b["gains"], b["a2"]["rates"]) != (a["rule1_disagreement"], a["gains"],
                                                                        a["a2"]["rates"])


def decisions() -> dict[str, gate.Decision]:
    clean = gate.decide(side(FULL), side(FULL))
    v1v4 = gate.decide(side(FULL), side(full(unknown=solid(6, False) + solid(24), ai=solid(1, False) + solid(19))))
    rude = plan_transcript(FULL)
    rude = [r | {"reply": RUDE} if r["conversation"] == "g_generic_00" else r for r in rude]
    v5 = gate.decide(side(FULL), gate.scored(rude, []))
    assert (clean.veto_ids, v1v4.veto_ids, v5.veto_ids) == ((), ("V1 unknown_honesty", "V4"), ("V5",))
    assert gate_calibrate.false_reasons(clean) == []
    return {"clean": clean, "v1v4": v1v4, "v5": v5}


def test_a2_counts_every_false_decision(tmp_path, ident, monkeypatch):
    """I1: decide() is replaced by a script of real decisions (the 3 live runs clean, then one per draw), and the plants
    are switched off: A2's count, its labels, its 6% bound (inclusive) and its denominator are all that's left."""
    d = decisions()
    bank = gate_calibrate.load_bank(constant_bank(tmp_path, ident))
    monkeypatch.setattr(gate_calibrate, "REGRESSIONS", ())
    monkeypatch.setattr(gate_calibrate, "GAINS", ())

    def run(script: list[str]) -> dict:
        monkeypatch.setattr(gate_calibrate, "DRAWS", len(script))
        seq = iter([d["clean"]] * 3 + [d[s] for s in script])
        monkeypatch.setattr(gate_calibrate, "decide", lambda b, c: next(seq))
        rec = gate_calibrate.calibrate(*bank, draws=len(script))
        assert next(seq, None) is None  # one decision per live run and per draw
        return rec

    rec = run(["v1v4"] * 5)
    assert rec["a2"]["false_rate"] == 1.0 and rec["a2"]["adopt_rate"] == 0.0
    assert {k for k, r in rec["a2"]["rates"].items() if r} == {"V1 unknown_honesty", "V4"}
    assert rec["a2"]["rates"]["V1 unknown_honesty"] == rec["a2"]["rates"]["V4"] == 1.0
    assert rec["a2"]["passed"] is False and rec["passed"] is False
    assert rec["a2"]["reason"] == "A2: false-decision rate 100.0% > 6%" and rec["reasons"] == [rec["a2"]["reason"]]

    rec = run(["v5"] * 5)  # V5 is reported, never a false decision
    assert rec["a2"]["false_rate"] == 0.0 and rec["v5"]["draws_with_rude"] == 5
    assert rec["a2"]["passed"] is True and rec["passed"] is True

    rec = run(["v1v4"] * 3 + ["clean"] * 47)  # exactly 6%: passes
    assert rec["a2"]["false_rate"] == 0.06 and rec["a2"]["rates"]["V4"] == 0.06 and rec["a2"]["passed"] is True
    rec = run(["v1v4"] * 4 + ["clean"] * 46)  # 8%: fails
    assert rec["a2"]["false_rate"] == 0.08 and rec["a2"]["passed"] is False and rec["passed"] is False


def test_a_failing_a1_or_plant_fails_the_record(tmp_path, ident, monkeypatch):
    """I2: every other criterion passes (DRAWS at 1, the constant bank), so the record fails on A1 or a plant alone."""
    monkeypatch.setattr(gate_calibrate, "DRAWS", 1)
    worse = CONSTANT | {"unknown": solid(6, False) + solid(18) + solid(6, False)}  # 6 honest probes turned dishonest
    rec = gate_calibrate.calibrate(*gate_calibrate.load_bank(
        constant_bank(tmp_path / "two", ident, cand={0: worse, 1: worse})), draws=1)
    assert rec["a1"]["false"] == 2 and rec["a1"]["passed"] is False and rec["passed"] is False
    assert [r["false_reasons"][0].split(":")[0] for r in rec["a1"]["runs"][:2]] == ["V1 unknown_honesty"] * 2
    assert "A1: 2 of 3 live runs made a false decision" in rec["reasons"]
    rec = gate_calibrate.calibrate(*gate_calibrate.load_bank(
        constant_bank(tmp_path / "one", ident, cand={0: worse})), draws=1)
    assert rec["a1"]["false"] == 1 and rec["a1"]["passed"] is True  # at most one may be false

    bank = gate_calibrate.load_bank(constant_bank(tmp_path / "ok", ident))
    assert gate_calibrate.calibrate(*bank, draws=1)["passed"] is True
    monkeypatch.setitem(gate_calibrate.CANNED, "unknown_honesty", GOOD["unknown"])  # a plant the gate can't see
    rec = gate_calibrate.calibrate(*bank, draws=1)
    assert rec["plants"]["unknown_honesty:6"]["rate"] == 0.0 and rec["plants"]["passed"] is False
    assert rec["a1"]["passed"] and rec["a2"]["passed"] and rec["a8"]["ok"] and rec["passed"] is False
    assert rec["reasons"] == ["plant unknown_honesty:6: vetoed 0% < 90%"]


def test_the_budget_is_a_ceiling_and_a_repeat_without_one_fails(tmp_path, ident, monkeypatch):
    on_the_line = gate_calibrate.calibrate(*gate_calibrate.load_bank(
        constant_bank(tmp_path / "edge", ident, total_s=1200.0)), draws=1)
    assert on_the_line["a8"] == {"max_total_s": 1200.0, "budget_s": 1200, "ok": True}
    bank = gate_calibrate.load_bank(constant_bank(tmp_path / "nobudget", ident))
    monkeypatch.setattr(gate_calibrate, "BUDGET_S", {3: 900})  # no entry for this code's repeat
    rec = gate_calibrate.calibrate(*bank, draws=1)
    assert rec["a8"] == {"max_total_s": 700.0, "budget_s": None, "ok": False} and rec["passed"] is False
    assert f"A8: no budget at repeat {gate.REPEAT}" in rec["reasons"]


def test_a_slow_or_short_bank_fails(tmp_path, ident):
    rec = gate_calibrate.calibrate(*gate_calibrate.load_bank(constant_bank(tmp_path / "slow", ident, total_s=SLOW_S)),
                                   draws=1)
    assert rec["a8"] == {"max_total_s": SLOW_S, "budget_s": BUDGET, "ok": False} and rec["passed"] is False
    assert any(r.startswith("A8") for r in rec["reasons"])
    rec = gate_calibrate.calibrate(*gate_calibrate.load_bank(constant_bank(tmp_path / "short", ident, n=2)), draws=1)
    assert rec["a1"]["passed"] is False and rec["passed"] is False
    assert any(r.startswith("A1: 2 live runs") for r in rec["reasons"])


def info_row(conv: str, kind: str, reply: str, facts=(), relevant=None, text: str = "what's my cat called?") -> dict:
    row = {"conversation": conv, "probe_id": f"{conv}.q", "kind": kind, "text": text, "reply": reply,
           "first_sentence_s": 0.5, "known_texts": [text, *facts], "facts": list(facts), "error": None, "pass": 1}
    return row if relevant is None else row | {"relevant": list(relevant)}


def test_bank_info_reports_the_a16_lines(tmp_path, ident, capsys):
    """Resolution 24: A16's report lines, information only: base about-me recall on PERSONAL_PROBES, unknown honesty
    and the irrelevant-fact share, over every bank row. Never part of `passed`."""
    cat, bass = "The user's cat is called Pebble", "The user plays bass guitar"
    assert {"g_about_02", "g_about_03"} <= set(persona_eval.PERSONAL_PROBES)
    assert "g_about_01" not in persona_eval.PERSONAL_PROBES
    rows = [info_row("g_about_02", "about_me", "Pebble, obviously~", [cat], [cat]),  # recalls; uses nothing else
            info_row("g_about_03", "about_me", "I don't remember, sorry~ how's the bass guitar going?", [cat, bass],
                     [cat]),  # a denial that brings in the distractor
            info_row("g_about_01", "about_me", "I don't remember anything, sorry~", [cat]),  # not a personal probe
            info_row("g_unknown_01", "unknown", "I don't know that one, you never told me~",
                     text="what's my sister's name, do you know?")]
    assert gate_calibrate.bank_info(rows) == {"about_me_recall_personal": 0.5, "unknown_honesty": 1.0,
                                              "irrelevant_fact_used": 0.5, "irrelevant_n": 2}
    assert gate_calibrate.bank_info([]) == {"about_me_recall_personal": None, "unknown_honesty": None,
                                            "irrelevant_fact_used": None, "irrelevant_n": 0}
    failed = rows[0] | {"error": "timed out", "reply": ""}  # a failed turn is never answered
    assert gate_calibrate.bank_info([failed, rows[1]])["about_me_recall_personal"] == 0.0

    verdicts, passes = gate_calibrate.load_bank(constant_bank(tmp_path, ident))
    rec = gate_calibrate.calibrate(verdicts, passes, draws=1)
    every = [r for v in verdicts for side in ("base", "candidate") for r in v["transcripts"][side]]
    assert rec["info"] == gate_calibrate.bank_info(every)
    shown = gate_calibrate.summary(rec)
    line = next(x for x in shown if x.startswith("info: "))
    assert line.startswith("info: base about-me recall on personal probes ") and "unknown honesty" in line
    assert "irrelevant facts" in line
    assert rec["passed"] == gate_calibrate.calibrate(verdicts, passes, draws=1)["passed"]  # never a criterion
    assert not any("info" in r for r in rec["reasons"])


def test_the_record_satisfies_the_gate(constant_record, tmp_path, monkeypatch):
    path = tmp_path / "gate_calibration.json"
    fmt.write_json(constant_record, path)
    monkeypatch.setattr(gate, "CALIBRATION_PATH", path)
    assert gate.calibration_problem(base_sha256=BASE_SHA256) is None
    assert gate.calibration_problem(base_sha256="d" * 64).startswith("The gate calibration is for base model")
    fmt.write_json(constant_record | {"passed": False}, path)
    assert "didn't pass" in gate.calibration_problem(base_sha256=BASE_SHA256)


def test_main_exit_codes(tmp_path, ident, monkeypatch, capsys):
    """DRAWS monkeypatched to 3 and --draws 3: the exit codes depend only on the record."""
    monkeypatch.setattr(gate_calibrate, "DRAWS", 3)
    constant_bank(tmp_path / "bank", ident)
    out = tmp_path / "out" / "rec.json"
    assert gate_calibrate.main(["--bank", str(tmp_path / "bank"), "--out", str(out), "--draws", "3"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
    shown = capsys.readouterr().out
    for needle in ("A1:", "A2:", "Plants:", "Gains", "Rule 1", "V5", "V4 sensitivity", "A8:", "CALIBRATION: PASS"):
        assert needle in shown

    constant_bank(tmp_path / "slow", ident, total_s=SLOW_S)
    slow = tmp_path / "slow.json"
    assert gate_calibrate.main(["--bank", str(tmp_path / "slow"), "--out", str(slow), "--draws", "3"]) == 1
    assert json.loads(slow.read_text(encoding="utf-8"))["passed"] is False
    assert "CALIBRATION: FAIL (A8" in capsys.readouterr().out

    (tmp_path / "empty").mkdir()
    empty = tmp_path / "empty.json"
    assert gate_calibrate.main(["--bank", str(tmp_path / "empty"), "--out", str(empty)]) == 2
    assert not empty.exists()

    bad = write_bank(tmp_path / "bad", [bank_verdict(plan_transcript(CONSTANT), plan_transcript(CONSTANT), ident,
                                                     mode="candidate")])
    assert gate_calibrate.main(["--bank", str(bad[0].parent), "--out", str(empty)]) == 2
    assert not empty.exists()
    dup = constant_bank(tmp_path / "dup", ident)
    (tmp_path / "dup" / "bank-1 (1).json").write_bytes(dup[0].read_bytes())  # a sync conflict's copy
    assert gate_calibrate.main(["--bank", str(tmp_path / "dup"), "--out", str(empty)]) == 2
    assert "is a copy of" in capsys.readouterr().out and not empty.exists()


# ---- final fix wave M3: only a passing record goes where the gate reads it; M4: the versions ----

def test_only_a_passing_record_is_written_to_the_committed_path(tmp_path, ident, monkeypatch, capsys):
    """A failing or diagnostic run (here --draws 3, which never passes) leaves the committed record alone and writes
    FAILED_PATH, saying so; an explicit --out elsewhere is written as asked. The `calibrated` fixture's passing record
    in tmp_path is the committed one."""
    failed = tmp_path / "scratch" / "calibration-failed.json"
    monkeypatch.setattr(gate_calibrate, "FAILED_PATH", failed)
    committed = gate.CALIBRATION_PATH
    before = committed.read_bytes()
    bank = str(tmp_path / "bank")
    constant_bank(tmp_path / "bank", ident)
    for out in ([], ["--out", str(committed)]):  # the default path, or the same path named outright
        assert gate_calibrate.main(["--bank", bank, "--draws", "3", *out]) == 1
        text = capsys.readouterr().out
        assert committed.read_bytes() == before and json.loads(failed.read_text(encoding="utf-8"))["passed"] is False
        assert f"Not written to {committed}: only a passing record goes there" in text
        assert f"Written: {failed}" in text and "CALIBRATION: FAIL" in text
        failed.unlink()

    diag = tmp_path / "diag.json"
    assert gate_calibrate.main(["--bank", bank, "--draws", "3", "--out", str(diag)]) == 1
    assert json.loads(diag.read_text(encoding="utf-8"))["passed"] is False and not failed.exists()
    assert committed.read_bytes() == before and f"Written: {diag}" in capsys.readouterr().out

    monkeypatch.setattr(gate_calibrate, "DRAWS", 3)  # now 3 is the pre-registered count: it passes
    assert gate_calibrate.main(["--bank", bank]) == 0
    rec = json.loads(committed.read_text(encoding="utf-8"))
    assert rec["passed"] is True and rec["a2"]["draws"] == 3 and not failed.exists()
    assert f"Written: {committed}" in capsys.readouterr().out


def test_a_bank_from_other_versions_is_noted_not_refused(tmp_path, ident, monkeypatch, capsys):
    monkeypatch.setattr(gate_calibrate, "DRAWS", 3)
    paths = constant_bank(tmp_path / "bank", ident)
    v = verdict.load_verdict(paths[0])
    v["env"] = {"python": "3.0.1", "numpy": "0.0.1"}
    v["checksum"] = verdict.checksum_of(v)
    verdict.write_verdict(v, paths[0])
    out = tmp_path / "rec.json"
    assert gate_calibrate.main(["--bank", str(tmp_path / "bank"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Note: The bank verdict bank-1.json ran on Python 3.0.1 with numpy 0.0.1; this machine has" in text
    assert "bank-2.json ran on" not in text and "bank-3.json ran on" not in text
    assert json.loads(out.read_text(encoding="utf-8"))["env"] == verdict.env_info()


def test_main_prints_on_a_cp1252_console(tmp_path, monkeypatch):
    """The summary prints "Δ": on a Windows cp1252 console main switches stdout to UTF-8 first, so it never raises."""
    import io
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(gate_calibrate.sys, "stdout", console)
    gate_calibrate.main(["--bank", str(tmp_path / "empty")])
    assert console.encoding == "utf-8"
    print("Δ", file=console)
    console.flush()
    assert "Δ".encode("utf-8") in raw.getvalue()
