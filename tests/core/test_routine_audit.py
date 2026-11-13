"""The audit, the two readings and the honesty numbers (v0.12 §4.6, A15, A16)."""
from riya.core.memory import Glance
from riya.core.routine import audit as au
from riya.core.routine.records import TALLY_KEY, RoutineDB
from riya.core.routine.statements import Statement
from riya.core.store import KV, open_db

DAY = 86400.0
H = 3600.0
T0 = 100 * DAY


def _db():
    return RoutineDB(open_db(":memory:"))


def _say(db, kind, value, at, claim=True, where="reply"):
    sid = db.add_statement(Statement(kind=kind, claim=claim, text=f"{kind} line", value=value, evidence={},
                                     where=where, due_at=None), at)
    return db.statement(sid)


def _no_glances(a, b):
    return []


def _verdict(db, row, now, glances=_no_glances):
    return au.score(db.statement(row.id), db, now, glances)


def _g(at, app="VS Code", activity="editing python"):
    return Glance(id=0, at=at, app=app, title="", activity=activity, details="", notable=None)


def test_s1_needs_an_observed_start_and_no_break_in_its_window():
    db = _db()
    db.log_presence(T0, "back")
    said = T0 + 2 * H
    row = _say(db, "S1", {"t0": T0, "coding": False}, said)
    assert _verdict(db, row, said + 60) == "true"

    db2 = _db()
    db2.log_presence(T0, "back")
    db2.log_presence(T0 + H, "break")
    db2.log_presence(T0 + H + 300, "break_end")
    row = _say(db2, "S1", {"t0": T0, "coding": False}, said)
    assert _verdict(db2, row, said + 60) == "false"

    db3 = _db()
    db3.log_presence(T0 - 601, "back")
    row = _say(db3, "S1", {"t0": T0, "coding": False}, said)
    assert _verdict(db3, row, said + 60) == "false"

    db4 = _db()  # a bid said 150 s into a break that started at w: the window ends at w
    w = T0 + 2 * H
    db4.log_presence(T0, "back")
    db4.log_presence(w, "break")
    row = _say(db4, "S1", {"t0": T0, "coding": False}, w + 150, where="bid")
    assert _verdict(db4, row, w + 200) == "true"


def test_s1_coding_is_unverifiable_without_its_glances():
    db = _db()
    db.log_presence(T0, "back")
    said = T0 + 2 * H
    row = _say(db, "S1", {"t0": T0, "coding": True}, said)
    assert _verdict(db, row, said + 60) == "unverifiable"
    assert _verdict(db, row, said + 60, lambda a, b: [_g(T0 + 60), _g(T0 + 600)]) == "true"
    assert _verdict(db, row, said + 60, lambda a, b: [_g(T0 + 60), _g(T0 + 600, "Firefox", "watching a video")]) \
        == "false"
    assert _verdict(db, _say(db, "S1", {}, said), said + 60) == "unverifiable"
    assert _verdict(db, _say(db, "S1", {"t0": T0}, said, claim=False), said + 60) == "unverifiable"


def _s2_setup(direction, x, others, later=()):
    db = _db()
    for i, v in enumerate(others):
        db.add_obs(T0 - (i + 1) * DAY, "P1", "b", v)
    oid = db.add_obs(T0, "P1", "b", x)
    for i, v in enumerate(later):
        db.add_obs(T0 + (i + 1) * DAY, "P1", "b", v)
    row = _say(db, "S2", {"predictor": "P1", "bucket": "b", "obs_id": oid, "x": x, "direction": direction},
               T0 + 60, where="greeting")
    return db, row


def test_s2_and_s4_against_the_median_of_the_others():
    db, row = _s2_setup("early", 1000.0, [5000.0, 6000.0, 7000.0, 8000.0])
    assert _verdict(db, row, T0 + DAY) is None  # no 3 later observations yet
    assert _verdict(db, row, T0 + 14 * DAY + 60) == "true"
    db, row = _s2_setup("late", 1000.0, [5000.0, 6000.0, 7000.0, 8000.0])
    assert _verdict(db, row, T0 + 14 * DAY + 60) == "false"
    db, row = _s2_setup("late", 9000.0, [5000.0, 6000.0], later=[7000.0, 8000.0, 5500.0])
    assert _verdict(db, row, T0 + 4 * DAY) == "true"  # scored early: 3 later observations exist
    db, row = _s2_setup("late", 9000.0, [5000.0, 6000.0, 7000.0])
    assert _verdict(db, row, T0 + 5 * DAY) is None
    assert _verdict(db, row, T0 + 14 * DAY + 60) == "unverifiable"
    db = _db()
    oid = db.add_obs(T0, "P3", "claude", 1500.0)
    for i, v in enumerate([180.0, 200.0, 240.0, 300.0]):
        db.add_obs(T0 - (i + 1) * DAY, "P3", "claude", v)
    row = _say(db, "S4", {"predictor": "P3", "bucket": "claude", "obs_id": oid, "x": 1500.0, "direction": "longer"},
               T0 + 10, where="report")
    assert _verdict(db, row, T0 + 15 * DAY) == "true"


def test_s5_against_the_next_runs():
    def run(values, now):
        db = _db()
        row = _say(db, "S5", {"predictor": "P3", "bucket": "claude", "median_s": 600.0}, T0)
        for i, v in enumerate(values):
            db.add_obs(T0 + (i + 1) * H, "P3", "claude", v)
        return _verdict(db, row, now)
    assert run([600, 400, 1000, 9000, 9000], T0 + DAY) == "true"
    assert run([600, 9000, 9000, 9000, 100], T0 + DAY) == "false"
    assert run([600, 600, 600], T0 + DAY) is None
    assert run([600, 600, 600], T0 + 14 * DAY + 1) == "true"
    assert run([600, 600], T0 + 14 * DAY + 1) == "expired"


def test_s7_against_the_next_five_day_bands():
    def run(values, now):
        db = _db()
        row = _say(db, "S7", {"predictor": "P4", "bucket": "weekday/evening"}, T0)
        for i, v in enumerate(values):
            db.add_obs(T0 + (i + 1) * DAY, "P4", "weekday/evening", v)
        return _verdict(db, row, now)
    assert run([1, 1, 0, 1, 0], T0 + 6 * DAY) == "true"
    assert run([1, 0, 0, 1, 0], T0 + 6 * DAY) == "false"
    assert run([1, 1], T0 + 6 * DAY) is None
    assert run([1, 1], T0 + 28 * DAY + 1) == "expired"


def test_the_hunches_are_scored_on_their_events():
    # S3: back within ± max(30 min, 25% of the guessed absence)
    def s3(back_after, now):
        db = _db()
        row = _say(db, "S3", {"back_at": T0 + 4 * H, "left_at": T0}, T0, claim=False, where="goodbye")
        if back_after is not None:
            db.log_presence(T0 + back_after, "back")
        return _verdict(db, row, now)
    assert s3(4 * H + 3500, T0 + 5 * H) == "true"  # slack is 25% of 4 h = 1 h
    assert s3(4 * H + 3700, T0 + 6 * H) == "false"
    assert s3(None, T0 + DAY - 1) is None
    assert s3(None, T0 + DAY) == "expired"

    # S6: the run ends within [min(0.5g, g - 120), max(2g, g + 120)] of when she guessed
    def s6(end_after, now, g=600.0):
        db = _db()
        row = _say(db, "S6", {"predictor": "P3", "bucket": "claude", "left_s": g}, T0, claim=False)
        if end_after is not None:
            db.add_obs(T0 + end_after, "P3", "claude", 900.0)
        return _verdict(db, row, now)
    assert s6(1000, T0 + 2 * H) == "true"
    assert s6(1300, T0 + 2 * H) == "false"
    assert s6(200, T0 + 2 * H, g=60.0) == "false" and s6(150, T0 + 2 * H, g=60.0) == "true"  # ± 2 min at least
    assert s6(None, T0 + DAY) == "expired"

    # the S7 bid hunch: gaming in that band, that day
    def s7(gamed, now):
        db = _db()
        row = _say(db, "S7", {"predictor": "P4", "bucket": "weekday/evening", "band_start": T0 + H,
                              "band_end": T0 + 5 * H}, T0, claim=False, where="bid")
        if gamed:
            db.log_presence(T0 + 2 * H, "gaming")
        return _verdict(db, row, now)
    assert s7(True, T0 + 5 * H) == "true"
    assert s7(False, T0 + 5 * H) == "false"
    assert s7(True, T0 + 4 * H) is None

    # S8: the next 3 observations stay outside the pre-change 50% interval
    def s8(values, now):
        db = _db()
        row = _say(db, "S8", {"predictor": "P1", "bucket": "b", "lo": 3000.0, "hi": 5000.0}, T0, claim=False)
        for i, v in enumerate(values):
            db.add_obs(T0 + (i + 1) * DAY, "P1", "b", v)
        return _verdict(db, row, now)
    assert s8([9000, 10000, 200], T0 + 4 * DAY) == "true"
    assert s8([9000, 4000], T0 + 4 * DAY) == "false"
    assert s8([9000], T0 + 4 * DAY) is None
    assert s8([9000], T0 + 28 * DAY) == "expired"


def test_a_dispute_is_final_for_the_user_reading_and_keeps_the_sensor_verdict():
    db = _db()
    db.log_presence(T0, "back")
    row = _say(db, "S1", {"t0": T0, "coding": False}, T0 + 2 * H, where="bid")
    assert au.dispute(db, row.id)
    assert not au.dispute(db, row.id)  # once
    assert au.audit(db, T0 + 3 * H, frozen=False, glances=_no_glances) == 1
    assert db.statement(row.id).verdict == "true"  # the sensor reading
    h = au.honesty(db, T0 + 3 * H, enabled=True, claims_enabled=True)
    assert (h.claims_true, h.claims_scored) == (0, 1)  # the user reading
    assert au.tally(db)["S1"]["disputed"] == 1


def test_honesty_two_readings_window_and_counts():
    db = _db()
    now = T0
    rows = [
        _say(db, "S1", {}, now - 30 * DAY - 1),       # outside the window
        _say(db, "S2", {}, now - DAY),                # true claim
        _say(db, "S4", {}, now - DAY),                # false claim
        _say(db, "S5", {}, now - DAY),                # unverifiable claim
        _say(db, "S3", {}, now - DAY, claim=False),   # right hunch
        _say(db, "S6", {}, now - DAY, claim=False),   # wrong hunch
        _say(db, "S8", {}, now - DAY, claim=False),   # expired hunch
        _say(db, "S7", {}, now - DAY),                # pending claim: not scored yet
    ]
    for r, v in zip(rows, ["true", "true", "false", "unverifiable", "true", "false", "expired"]):
        db.set_verdict(r.id, v, now - 60)
    h = au.honesty(db, now, enabled=True, claims_enabled=False)
    assert h == au.Honesty(enabled=True, claims_enabled=False, claims_true=1, claims_scored=2, hunches_right=1,
                           hunches_scored=2, unverifiable=1, expired=1)


def test_learning_off_expires_every_pending_row():
    db = _db()
    for k in ("S1", "S3", "S5"):
        _say(db, k, {}, T0)
    assert au.audit(db, T0 + 1, frozen=True, glances=_no_glances) == 3
    assert db.pending() == [] and {r.verdict for r in db.statements_since(0)} == {"expired"}


def test_forget_mid_audit_leaves_nothing_behind():
    db = _db()
    for k in ("S1", "S3", "S5"):
        _say(db, k, {}, T0)  # empty values: S1 and S5 score at once, as unverifiable
    seen = []

    def on_scored(row):
        seen.append(row.id)
        db.forget()
    n = au.audit(db, T0 + 1, frozen=False, glances=_no_glances, on_scored=on_scored)
    assert n == 1 and len(seen) == 1
    for t in ("routine_cells", "routine_p2", "routine_obs", "statements", "presence_log"):
        assert db.conn.execute(f"select count(*) from {t}").fetchone()[0] == 0
    t = au.tally(db)
    assert sum(sum(e["claim"].values()) + sum(e["hunch"].values()) for e in t.values()) == 1


def test_the_tally_counts_and_survives_forget():
    db = _db()
    row = _say(db, "S5", {}, T0)
    au.tally_said(db, row)
    au.audit(db, T0 + 1, frozen=False, glances=_no_glances)
    hunch = _say(db, "S3", {}, T0, claim=False)
    au.tally_said(db, hunch)
    au.dispute(db, hunch.id)
    db.forget()
    t = au.tally(db)
    assert t["S5"] == {"claim": {"unverifiable": 1}, "hunch": {}, "said": 1, "disputed": 0}
    assert t["S3"] == {"claim": {}, "hunch": {}, "said": 1, "disputed": 1}
    assert KV(db.conn, db.lock).keys_with("routine.") == [TALLY_KEY]


def test_odd_values_never_stop_the_audit():
    db = _db()
    huge = _say(db, "S1", {"t0": 10 ** 400}, T0)
    text = _say(db, "S3", {"back_at": "six", "left_at": None}, T0)
    fine = _say(db, "S5", {}, T0)
    assert au.audit(db, T0 + 1, frozen=False, glances=_no_glances) == 3
    assert {db.statement(r.id).verdict for r in (huge, text, fine)} == {"unverifiable"}


def test_an_s7_bid_hunch_counts_a_game_that_began_before_the_band():
    db = _db()
    row = _say(db, "S7", {"predictor": "P4", "bucket": "weekday/evening", "band_start": T0 + H,
                          "band_end": T0 + 5 * H}, T0, claim=False, where="bid")
    db.log_presence(T0 + 45 * 60, "gaming")  # started 15 minutes before the band, still on at its start
    assert _verdict(db, row, T0 + 5 * H) == "true"
    db2 = _db()
    row = _say(db2, "S7", {"predictor": "P4", "bucket": "weekday/evening", "band_start": T0 + H,
                           "band_end": T0 + 5 * H}, T0, claim=False, where="bid")
    db2.log_presence(T0 + 30 * 60, "gaming")
    db2.log_presence(T0 + 50 * 60, "gaming_end")  # over before the band
    assert _verdict(db2, row, T0 + 5 * H) == "false"
