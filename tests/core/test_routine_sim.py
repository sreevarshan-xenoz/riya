"""The routine simulator: deterministic, on the real engine, with its oracle and criteria (v0.12 §4.10, §4.12.4)."""
import json
import random
from pathlib import Path

import pytest

import riya.core.routine as routine
from evaluation import routine_sim as rs
from riya.core.engine import RiyaCore
from riya.core.initiative import Initiative

DAY, H = 86400.0, 3600.0


def test_a_short_run_is_deterministic():
    a, b = rs.simulate("office", 0, weeks=1), rs.simulate("office", 0, weeks=1)
    assert a.statements == b.statements and a.coverage == b.coverage and a.absences == b.absences
    assert a.ticks > 1000


def test_it_drives_the_real_engine(monkeypatch):
    counts = {"tick": 0, "break": 0, "came_back": 0}

    def wrap(cls, name, key):
        real = getattr(cls, name)

        def counted(self, *a, **k):
            counts[key] += 1
            return real(self, *a, **k)
        monkeypatch.setattr(cls, name, counted)
    wrap(RiyaCore, "tick", "tick")
    wrap(Initiative, "on_break", "break")
    wrap(Initiative, "on_came_back", "came_back")
    rs.simulate("office", 0, weeks=1)
    assert all(counts.values()), counts
    source = Path(rs.__file__).read_text(encoding="utf-8")
    assert "GREET_MIN_AWAY_S" not in source and "in_quiet_hours" not in source


def _truth():
    seg = rs.Segment
    t0 = rs.SIM_START + 9 * H
    segs = [seg(t0 - 10 * H, t0, "away", label="night"), seg(t0, t0 + 3 * H, "present", work=True),
            seg(t0 + 3 * H, t0 + 4 * H, "away", label="lunch")]
    table = {("P1", "afternoon/weekday"): 2700.0, ("P3", "claude:riya"): 480.0, ("P4", "evening/weekday"): 0.75,
             ("P4", "afternoon/weekday"): 0.1}
    return {"segments": segs, "tables": [(rs.SIM_START, table)], "breaks": [(t0 + H, t0 + H + 300)],
            "break_ends": [], "window_end": {}}, t0


def test_the_oracle_rules():
    truth, t0 = _truth()
    p = rs.PERSONAS["office"]
    s1 = {"kind": "S1", "claim": True, "at": t0 + 50 * 60, "value": {"t0": t0}}
    assert rs.oracle(s1, p, truth)
    assert not rs.oracle({**s1, "at": t0 + 2 * H}, p, truth)  # a break in the window
    assert not rs.oracle({**s1, "value": {"t0": t0 + 30 * 60}}, p, truth)  # no transition near t0
    s2 = {"kind": "S2", "claim": True, "at": t0, "value": {"predictor": "P1", "bucket": "afternoon/weekday",
                                                           "x": 9000.0, "direction": "late"}}
    assert rs.oracle(s2, p, truth) and not rs.oracle({**s2, "value": {**s2["value"], "x": 1000.0}}, p, truth)
    s4 = {"kind": "S4", "claim": True, "at": t0, "value": {"predictor": "P3", "bucket": "claude:riya", "x": 100.0,
                                                           "direction": "shorter"}}
    assert rs.oracle(s4, p, truth) and not rs.oracle({**s4, "value": {**s4["value"], "x": 900.0}}, p, truth)
    s5 = {"kind": "S5", "claim": True, "at": t0, "value": {"predictor": "P3", "bucket": "claude:riya",
                                                           "median_s": 600.0}}
    assert rs.oracle(s5, p, truth) and not rs.oracle({**s5, "value": {**s5["value"], "median_s": 2000.0}}, p, truth)
    s7 = {"kind": "S7", "claim": True, "at": t0, "value": {"predictor": "P4", "bucket": "evening/weekday"}}
    assert rs.oracle(s7, p, truth)
    assert not rs.oracle({**s7, "value": {"predictor": "P4", "bucket": "afternoon/weekday"}}, p, truth)
    hunch = {"kind": "S3", "claim": False, "at": t0, "value": {}}
    assert rs.oracle({**hunch, "sensor": "true"}, p, truth) and not rs.oracle({**hunch, "sensor": "false"}, p, truth)


def _result(persona="office", variant="full", **kw):
    return rs.RunResult(persona, 0, variant, **kw)


def _claims(n_true, n_false, week=3, disputed=0):
    out = [{"kind": "S1", "claim": True, "oracle": True, "disputed": False, "sensor": "true", "week": week}
           for _ in range(n_true)]
    out += [{"kind": "S1", "claim": True, "oracle": False, "disputed": i < disputed, "sensor": "true", "week": week}
            for i in range(n_false)]
    return out


def test_criteria_on_hand_made_results():
    good_cov = [("P1", 3, i < 8) for i in range(10)] + [("P3", 3, i < 8) for i in range(10)]
    surprises = [("P1", 3.0, True)] * 5 + [("P1", 0.0, False)] * 50
    full = [_result(name, statements=_claims(99, 1) if name in ("office", "night_owl") else _claims(9, 1),
                    coverage=good_cov, surprises=surprises,
                    chances=[{"week": w, "kind": "greet"} for w in range(2, 6) for _ in range(3)])
            for name in (*rs.FIVE, "late")]
    for r in full:
        r.changes = [{"cell": "P1:x", "at": rs.SIM_START, "change": False, "entered": False}]
        r.cells_seen = 1
    full[3].shift_obs_to_change = 3
    plant = [_result(n, "plant", statements=_claims(5, 5)) for n in rs.FIVE]
    clingy = [_result(n, "clingy") for n in ("office", "late")]
    c = rs.criteria(full, plant, clingy)
    assert {k: v["pass"] for k, v in c.items() if v["blocking"]} == {k: True for k in ("A3", "A4", "A5", "A6", "A7",
                                                                                          "A9")}
    assert not c["A8"]["blocking"] and not c["A20"]["blocking"]

    full[0].statements = _claims(90, 10)  # office + night_owl under 95%
    assert not rs.criteria(full, plant, clingy)["A6"]["pass"]
    full[0].statements = _claims(99, 1)
    assert not rs.criteria(full, [_result(n, "plant", statements=_claims(9, 1)) for n in rs.FIVE], clingy)["A7"]["pass"]
    full[3].shift_obs_to_change = 6
    assert not rs.criteria(full, plant, clingy)["A5"]["pass"]
    full[3].shift_obs_to_change = 3
    clingy[0].absences = [{"at": 0.0, "remark": True, "quiet": False, "missing_bonus": 0.1, "concern_bump": 0.1}]
    assert not rs.criteria(full, plant, clingy)["A9"]["pass"]
    clingy[0].absences = []
    full[0].coverage = [("P1", 3, True)] * 10
    assert not rs.criteria(full, plant, clingy)["A3"]["pass"]
    full[0].coverage = good_cov
    for r in full:
        if r.persona == "office":
            r.chances = []
    assert rs.criteria(full, plant, clingy)["A8"]["note"] == "unreachable, reported"


def test_disputes_follow_the_persona(monkeypatch):
    monkeypatch.setattr(routine, "CLAIMS_ENABLED", True)  # the claim path itself; claims ship off (held-out record)
    monkeypatch.setattr(rs, "oracle", lambda row, p, truth: not row.get("claim"))  # every claim is "false"
    r = rs.simulate("office", 3)
    claims = [s for s in r.statements if s["claim"]]
    assert len(claims) >= 10
    rate = sum(s["disputed"] for s in claims) / len(claims)
    p = rs.PERSONAS["office"].p_d
    sd = (p * (1 - p) / len(claims)) ** 0.5
    assert abs(rate - p) <= 3 * sd + 0.05


def test_grid_order_and_admissibility():
    assert rs.admissible({"STARTLE_FACTOR": 4, "CUSUM_H": 3}) is False
    assert rs.admissible({}) is True
    base = rs.current_constants()
    singles = [p for p in rs._grid_points() if len(p) == 1]
    want = [{k: v} for k, v in rs.GRID_ORDER if base[k] != v and rs.admissible({k: v})]
    assert singles == want
    assert all(rs.admissible(p) for p in rs._grid_points())


def test_held_out_refuses_overrides_and_a_stale_dev_record(tmp_path, monkeypatch, capsys):
    assert rs.main(["--seeds", "heldout", "--set", "CUSUM_H=3"]) == 2
    assert rs.main(["--seeds", "heldout", "--grid"]) == 2
    record = tmp_path / "dev.json"
    monkeypatch.setattr(rs, "DEV_RECORD", record)
    assert rs.main(["--seeds", "heldout"]) == 2  # missing
    record.write_text(json.dumps({"constants": {**rs.current_constants(), "CUSUM_H": 6.0}}), encoding="utf-8")
    assert rs.main(["--seeds", "heldout"]) == 2  # stale
    assert "differ" in capsys.readouterr().out


def test_the_shipping_flags_follow_the_held_out_record():
    if not rs.HELDOUT_RECORD.exists():
        pytest.skip("no held-out record yet (Task 15)")
    record = json.loads(rs.HELDOUT_RECORD.read_text(encoding="utf-8"))
    assert routine.CLAIMS_ENABLED == record["blocking_pass"]
    assert routine.ABSENCE_REMARKS_ENABLED == record["criteria"]["A9"]["pass"]
