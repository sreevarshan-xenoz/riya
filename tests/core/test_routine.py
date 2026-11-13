"""The Routine facade: the switch, the observations, the statements, the caps and the readouts (v0.12 §4.2-§4.8)."""
from datetime import datetime, timedelta, timezone

import pytest

import riya.core.routine as rt
from riya.agents.hub import AgentEvent, RunInfo
from riya.core.clock import FakeClock
from riya.core.learn.twin import MemoryCells, Twin
from riya.core.presence import PresenceEvent
from riya.core.routine import appraisal as ap
from riya.core.routine import predictors as pr
from riya.core.settings import Settings, apply_patch
from riya.core.store import KV, open_db
from tests.routine_helpers import mature_beta, mature_lognormal, sensed

TZ = timezone(timedelta(hours=5, minutes=30))
DAY = 86400.0
H = 3600.0
TABLES = ("routine_cells", "routine_p2", "routine_obs", "statements", "presence_log")


def at(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


MON_9 = at(2026, 11, 9, 9)  # a Monday


class Box:
    def __init__(self, **kw):
        self.s = Settings(**kw)

    def __call__(self):
        return self.s


def make(t=MON_9, *, started_at=None, evidence=True, **settings):
    clock = FakeClock(t)
    conn = open_db(":memory:")
    kv = KV(conn)
    box = Box(**{"routine_enabled": True, **settings})
    twin = Twin(MemoryCells(), clock)
    r = rt.Routine(conn, None, kv, clock, TZ, twin, box, started_at=t - 30 * DAY if started_at is None else started_at,
                   evidence=evidence)
    return r, clock, box


def rows(r):
    return {t: r.db.conn.execute(f"select count(*) from {t}").fetchone()[0] for t in TABLES}


def _done(agent="a1", run_s=600.0, kind="finished"):
    return AgentEvent(kind, agent, "riya a1", "done", run_s=run_s)


def test_routine_enabled_defaults_on_and_validates():
    assert Settings().routine_enabled is True
    with pytest.raises(ValueError):
        apply_patch(Settings(), {"routine_enabled": "yes"})
    assert apply_patch(Settings(), {"routine_enabled": False}).routine_enabled is False


def test_off_observes_and_says_nothing():
    r, clock, box = make(routine_enabled=False)
    now = clock.now()
    assert r.observe_return(3 * H, now) is None
    assert r.observe_run(_done(), "claude:riya", now) is None
    assert r.observe_engagement("bid", "b", True) is None
    assert r.observe_presence(now, "gaming", [PresenceEvent("came_back", 3 * H)]) == []
    s = sensed(now=now, stretch_start=now - 2 * H)
    assert r.greeting(None, 3 * H, s) is None and r.bid(s, True) is None
    assert r.report(_done(), None, s) is None and r.question("how long have i been at it") is None
    assert r.answer("stretch", "how long have i been at it", s) is None and r.farewell("bye!", s) is None
    assert r.step(now) is False
    assert set(rows(r).values()) == {0}


def test_learning_off_freezes_and_turns_claims_into_hunches():
    r, clock, box = make(learning_enabled=False)
    now = clock.now()
    assert r.observe_return(3 * H, now) is None and r.observe_run(_done(), "claude:riya", now) is None
    assert r.observe_presence(now, "focused", [PresenceEvent("came_back", 3 * H)]) == []
    assert set(rows(r).values()) == {0}
    s = r.answer("stretch", "how long have i been at it", sensed(now=now, stretch_start=now - 2 * H))
    assert s.kind == "S1" and s.claim is False


def test_the_flags_give_hunch_only_and_no_absence_remarks(monkeypatch):
    r, clock, _ = make()
    now = clock.now()
    monkeypatch.setattr(rt, "CLAIMS_ENABLED", False)
    s = r.answer("stretch", "how long have i been at it", sensed(now=now, stretch_start=now - 2 * H))
    assert s.kind == "S1" and not s.claim
    monkeypatch.setattr(rt, "CLAIMS_ENABLED", True)
    s = r.answer("stretch", "how long have i been at it", sensed(now=now, stretch_start=now - 2 * H))
    assert s.claim

    monkeypatch.setattr(rt, "ABSENCE_REMARKS_ENABLED", False)
    r, clock, _ = make()
    a = _late_return(r, clock)
    assert a.direction == "late" and a.surprised
    assert r.greeting(a, 30 * H, sensed(now=clock.now(), clinginess=0.9)) is None


def _late_return(r, clock, away=30 * H):
    """A mature morning cell (2 h usual), then an absence far longer than that."""
    leave = clock.now()
    for b in ("morning/weekday", "morning"):
        mature_lognormal(r.db, "P1", b, 2 * H, n=20, now=leave)
    clock.advance(away)
    return r.observe_return(away, clock.now())


def test_p1_buckets_back_off_and_censor():
    r, clock, _ = make()
    leave = clock.now()
    clock.advance(3 * H)
    a = r.observe_return(3 * H, clock.now())
    assert a.bucket == "morning"  # the fine cell is still young
    assert r.db.get_lognormal("P1", "morning/weekday").n == pytest.approx(1.0)
    assert r.db.get_lognormal("P1", "morning").n == pytest.approx(1.0)
    mature_lognormal(r.db, "P1", "morning/weekday", 3 * H, n=6, now=clock.now())
    clock.advance(DAY - 3 * H)
    clock.advance(3 * H)
    assert r.observe_return(3 * H, clock.now()).bucket == "morning/weekday"

    r2, clock2, _ = make(started_at=MON_9 + H)
    clock2.advance(3 * H)
    assert r2.observe_return(3 * H, clock2.now()) is None  # left before she started: not seen
    assert rows(r2)["routine_cells"] == 0


def test_absurd_absences_and_runs_are_censored():
    r, clock, _ = make()
    now = clock.now()
    for away in (-5, 899, 31 * DAY, float("nan")):
        assert r.observe_return(away, now) is None
    for run_s in (None, 29, -1, float("inf")):
        assert r.observe_run(_done(run_s=run_s), "claude:riya", now) is None
    assert r.observe_run(_done(kind="waiting"), "claude:riya", now) is None
    assert r.observe_run(_done(), None, now) is None
    assert set(rows(r).values()) == {0}


def test_p3_takes_finished_runs_and_backs_off():
    r, clock, _ = make()
    a = r.observe_run(_done(run_s=600), "claude:riya", clock.now())
    assert a.topic == "run" and a.bucket == "claude:*"
    assert r.db.get_lognormal("P3", "claude:riya").n == pytest.approx(1.0)
    assert r.db.get_lognormal("P3", "claude:*").n == pytest.approx(1.0)
    mature_lognormal(r.db, "P3", "claude:riya", 600, n=6, now=clock.now())
    a = r.observe_run(_done(run_s=3000, kind="failed"), "claude:riya", clock.now())
    assert a.bucket == "claude:riya" and a.direction == "longer"
    a = r.observe_run(_done(run_s=600), "colab:w3", clock.now())
    assert a.bucket == "colab:*"


def _tick(r, clock, state, seconds, step=60):
    out = []
    for _ in range(int(seconds // step)):
        clock.advance(step)
        out += r.observe_presence(clock.now(), state, [])
    return out


def test_p4_tracks_bands_coverage_and_the_night():
    r, clock, _ = make(at(2026, 11, 9, 17))  # Monday evening
    r.observe_presence(clock.now(), "focused", [])
    out = _tick(r, clock, "gaming", 660)
    assert [a.direction for a in out] == ["gamed"]
    assert _tick(r, clock, "gaming", 1200) == []  # each band once
    assert r.db.get_beta("P4", "evening/weekday") is not None

    r, clock, _ = make(at(2026, 11, 10, 12))  # Tuesday afternoon, covered 40%
    r.observe_presence(clock.now(), "focused", [])
    _tick(r, clock, "focused", 0.4 * 5 * H)
    clock.t = at(2026, 11, 10, 17, 1)
    assert r.observe_presence(clock.now(), "focused", []) == []
    assert r.db.get_beta("P4", "afternoon/weekday") is None

    r, clock, _ = make(at(2026, 11, 15, 22))  # Sunday 22:00: a weekend night, even after midnight
    r.observe_presence(clock.now(), "focused", [])
    _tick(r, clock, "focused", 3 * H)
    out = _tick(r, clock, "gaming", 660)
    assert [a.bucket for a in out] == ["night"] and r.db.get_beta("P4", "night/weekend") is not None

    r, clock, _ = make(at(2026, 11, 10, 12))  # a covered afternoon without gaming is observed False at its end
    r.observe_presence(clock.now(), "focused", [])
    _tick(r, clock, "focused", 5 * H - 60)
    clock.t = at(2026, 11, 10, 17, 0, )
    out = r.observe_presence(clock.now(), "focused", [])
    assert [a.direction for a in out] == ["didnt"]


def test_presence_events_are_logged():
    r, clock, _ = make()
    r.observe_presence(clock.now(), "focused", [PresenceEvent("came_back", 3 * H), PresenceEvent("break_started")])
    clock.advance(60)
    r.observe_presence(clock.now(), "gaming", [])
    clock.advance(60)
    r.observe_presence(clock.now(), "focused", [])
    assert [e for _, e in r.db.presence_between(0, clock.now())] == ["back", "break", "gaming", "gaming_end"]


def test_p2_reads_p_engage_before_the_twin_records():
    r, clock, _ = make()
    for _ in range(10):
        r.twin.record("bid", "b", False)
    p = r.twin.p_engage("bid", "b")
    assert p < 0.4 and r.expected_ignore("bid", "b")
    a = r.observe_engagement("bid", "b", True)
    assert a.predictor == "P2" and a.value == pytest.approx(p) and a.direction == "answered"
    assert r.db.get_p2("bid", "b") is not None
    assert r.observe_engagement("report", "b", True) is None
    assert r.clause(rt.Statement("S5", True, "x", {}, {}, "reply", None), clock.now()) == rt.st.CLAUSES["answered"]


def _step(change, n=20.0, cusum=0.0):
    cell = pr.LogNormalCell(n, 0.0, 0.0, ap.AppraisalState(0.25, cusum, ()), 0.0)
    return pr.Step(cell=cell, surprise=0.0, ratio=1.0, s=0.0, surprised=False, startled=(), change=change,
                   learned=True, n_eff_before=n, confidence_before=0.5, sign=1.0)


def test_unsure_has_hysteresis_and_one_bump():
    r, clock, _ = make()
    mature_lognormal(r.db, "P3", "claude:riya", 600, n=20, now=clock.now())
    mature_lognormal(r.db, "P3", "claude:*", 600, n=20, now=clock.now())
    far = []
    for _ in range(3):
        clock.advance(60)
        far.append(r.observe_run(_done(run_s=600 * 40), "claude:riya", clock.now()))
    assert [a.change for a in far].count(True) == 1  # entering unsure bumps once
    assert "P3:claude:riya" in r.db.kv_get(rt.UNSURE_KEY)

    cell = "P1:x"
    now = clock.now()
    assert r._unsure(cell, _step(True), now) is True  # entering
    assert r._unsure(cell, _step(True), now) is False  # a second change while unsure
    assert r._unsure(cell, _step(False, n=20, cusum=0.0), now) is False
    assert cell in r.db.kv_get(rt.UNSURE_KEY)  # one confident observation isn't enough
    assert r._unsure(cell, _step(False, n=2, cusum=0.0), now) is False  # C = 2/6 < 0.5 resets the count
    r._unsure(cell, _step(False, n=20, cusum=0.0), now)
    assert cell in r.db.kv_get(rt.UNSURE_KEY)
    r._unsure(cell, _step(False, n=20, cusum=0.0), now)
    assert cell not in r.db.kv_get(rt.UNSURE_KEY)  # two in a row at C >= 0.5
    assert r._unsure(cell, _step(True), now) is True  # it can enter again


def test_s8_is_only_for_the_day_cells():
    r, clock, _ = make()
    mature_lognormal(r.db, "P3", "claude:riya", 600, n=20, now=clock.now())
    for _ in range(3):
        clock.advance(60)
        r.observe_run(_done(run_s=600 * 40), "claude:riya", clock.now())
    assert r._s8 is None
    s = sensed(now=clock.now())
    assert r.bid(s, False) is None

    r, clock, _ = make()
    for b in ("morning/weekday", "morning"):
        mature_lognormal(r.db, "P1", b, 2 * H, n=20, now=clock.now())
    changes = []
    for i in range(3):
        clock.advance(DAY)
        changes.append(r.observe_return(2 * H * 60, clock.now()).change)
    assert any(changes) and r._s8 is not None and r._s8["predictor"] == "P1"
    g = r.greeting(None, 3 * H, sensed(now=clock.now()))
    assert g.kind == "S8" and not g.claim and r.clause(g, clock.now()) == rt.st.CLAUSES["unsure"]


def test_the_statement_order_per_place(monkeypatch):
    r, clock, _ = make()
    a = _late_return(r, clock)
    s = sensed(now=clock.now(), clinginess=0.9)
    g = r.greeting(a, 30 * H, s)
    assert g.kind == "S2" and g.value["direction"] == "late" and r.clause(g, clock.now()) == rt.st.CLAUSES["late"]
    assert r.greeting(a, 30 * H, sensed(now=clock.now(), clinginess=0.9, news=True)) is None

    b = r.bid(sensed(now=clock.now(), stretch_start=clock.now() - 2 * H), True)
    assert b.kind == "S1" and b.where == "bid" and "you'd" in b.text.casefold()

    mature_lognormal(r.db, "P3", "claude:riya", 600, n=20, now=clock.now())
    mature_lognormal(r.db, "P3", "claude:*", 600, n=20, now=clock.now())
    ev = _done(run_s=600 * 30)
    a = r.observe_run(ev, "claude:riya", clock.now())
    rep = r.report(ev, a, sensed(now=clock.now()))
    assert rep.kind == "S4" and rep.where == "report" and "5 hours" in rep.text

    assert r.question("how long do my claude runs take?") == "runs"
    ans = r.answer("runs", "how long do my riya runs take", sensed(now=clock.now()))
    assert ans.kind == "S5" and ans.where == "reply"
    run = RunInfo("a2", "riya a2", "claude:riya", clock.now() - 120, 0.0)
    left = r.answer("run_left", "when will my agent be done", sensed(now=clock.now(), runs=(run,)))
    assert left.kind == "S6" and not left.claim
    assert r.answer("gaming", "when do i usually game", sensed(now=clock.now())).kind == "unknown"
    mature_beta(r.db, "P4", "evening/weekday", 0.9, days=20, now=clock.now())
    assert r.answer("gaming", "when do i usually game", sensed(now=clock.now())).kind == "S7"

    for b_ in ("evening/weekday", "evening"):
        mature_lognormal(r.db, "P1", b_, 3 * H, n=20, now=clock.now())
    clock.t = at(2026, 11, 10, 18)
    bye = r.farewell("gotta go, bye!", sensed(now=clock.now()))
    assert bye.kind == "S3" and bye.where == "goodbye" and "9:00 pm" in bye.text
    assert r.farewell("say bye to my sister for me", sensed(now=clock.now())) is None


def test_the_absence_and_volume_caps_hold_across_places():
    r, clock, _ = make()
    a = _late_return(r, clock)
    s = sensed(now=clock.now(), clinginess=0.9)
    g = r.greeting(a, 30 * H, s)
    r.said(g, clock.now())
    assert r.greeting(a, 30 * H, s) is None  # one S2 a day, one absence remark a day
    assert r.greeting(a, 30 * H, sensed(now=clock.now(), clinginess=0.6)) is None
    for i in range(3):
        clock.advance(10)
        b = r.bid(sensed(now=clock.now(), stretch_start=clock.now() - 2 * H - i), True)
        if b is not None:
            r.said(b, clock.now())
    assert len([x for x in r.db.statements_since(0) if x.kind == "S1"]) == 1  # one per kind per day
    ans = r.answer("stretch", "how long have i been at it", sensed(now=clock.now(), stretch_start=clock.now() - 2 * H))
    assert ans is not None and ans.kind == "S1"  # a reply is never capped


def test_said_logs_one_row_and_the_scrub_hides_it():
    r, clock, _ = make()
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    assert r.scrub(turns) is turns
    s = r.answer("stretch", "how long have i been at it", sensed(now=clock.now(), stretch_start=clock.now() - 2 * H))
    sid = r.said(s, clock.now())
    assert sid is not None and rows(r)["statements"] == 1
    assert r.said(rt.Statement("unknown", False, "dunno", {}, {}, "reply", None), clock.now()) is None
    turns = [{"role": "user", "content": "how long?"}, {"role": "assistant", "content": s.text + " Wow, right?~"},
             {"role": "assistant", "content": "Something else."}]
    out = r.scrub(turns)
    assert out[1]["content"] == rt.PLACEHOLDER and out[0] == turns[0] and out[2] == turns[2]
    assert turns[1]["content"] != rt.PLACEHOLDER  # a copy


def test_a_dispute_counts_only_on_the_next_turn_within_120_s():
    r, clock, _ = make()
    s = r.answer("stretch", "x", sensed(now=clock.now(), stretch_start=clock.now() - 2 * H))
    sid = r.said(s, clock.now())
    clock.advance(121)
    r.on_user_turn("that's wrong", clock.now())
    assert not r.db.statement(sid).disputed
    sid = r.said(r.answer("stretch", "x", sensed(now=clock.now(), stretch_start=clock.now() - 3 * H)), clock.now())
    clock.advance(30)
    r.on_user_turn("ok cool", clock.now())
    r.on_user_turn("that's wrong", clock.now())
    assert not r.db.statement(sid).disputed  # only the first turn after it
    sid = r.said(r.answer("stretch", "x", sensed(now=clock.now(), stretch_start=clock.now() - 4 * H)), clock.now())
    clock.advance(30)
    r.on_user_turn("no I wasn't", clock.now())
    assert r.db.statement(sid).disputed


def test_the_plant_marks_every_hunch_a_claim():
    r, clock, _ = make(evidence=False)
    s = r.answer("stretch", "x", sensed(now=clock.now(), stretch_start=clock.now() - 2 * H, stretch_censored=True))
    assert s.kind == "S1" and s.claim is True and rt.st.HEDGE_RE.search(s.text)


def test_msg_lines_and_honesty():
    r, clock, _ = make()
    mature_lognormal(r.db, "P3", "claude:riya", 600, n=14, now=clock.now())
    mature_lognormal(r.db, "P3", "claude:*", 600, n=14, now=clock.now())
    mature_lognormal(r.db, "P1", "morning/weekday", 3 * H, n=6, now=clock.now())
    mature_beta(r.db, "P4", "evening/weekday", 0.75, days=8, now=clock.now())
    m = r.msg(clock.now())
    lines = [c.line for c in m.cells]
    assert "Agent runs (riya): about 10 min, 14 runs, sure" in lines
    assert "Coming back after leaving in the morning on weekdays: usually about 3 hours, 6 times, fairly sure" in lines
    assert any(x.startswith("Gaming on weekday evenings: ") and x.endswith("8 days, fairly sure") for x in lines)
    assert m.enabled and m.honesty.claims_scored == 0
    r2, _, _ = make(routine_enabled=False)
    assert r2.msg(clock.now()).enabled is False and r2.msg(clock.now()).cells == []


def test_forget_and_reset_p2():
    r, clock, _ = make()
    r.observe_engagement("bid", "b", True)
    r.observe_engagement("greeting", "b", True)
    r.reset_p2("bid")
    assert r.db.get_p2("bid", "b") is None and r.db.get_p2("greeting", "b") is not None
    r.observe_run(_done(), "claude:riya", clock.now())
    r.forget()
    assert set(rows(r).values()) == {0}


def test_the_hourly_audit_runs_once_an_hour():
    r, clock, _ = make()
    s = r.answer("stretch", "x", sensed(now=clock.now(), stretch_start=clock.now() - 2 * H))
    r.said(s, clock.now())
    assert r.step(clock.now()) is True
    assert r.step(clock.now() + 60) is False
    assert r.db.pending() == []


def test_a_twin_reset_clears_p2_unsure_and_odd_inputs_never_raise():
    r, clock, _ = make()
    r.db.kv_set(rt.UNSURE_KEY, {"P2:bid:b": {"since": 0, "ok": 0}, "P2:greeting:b": {"since": 0, "ok": 0},
                                "P1:morning": {"since": 0, "ok": 0}})
    r.reset_p2("bid")
    assert set(r.db.kv_get(rt.UNSURE_KEY)) == {"P2:greeting:b", "P1:morning"}
    r.reset_p2()
    assert set(r.db.kv_get(rt.UNSURE_KEY)) == {"P1:morning"}
    nan = float("nan")
    assert r.observe_return(3 * H, nan) is None and r.observe_run(_done(), "claude:riya", nan) is None
    assert r.observe_presence(nan, "focused", None) == [] and r.observe_presence(clock.now(), "focused", None) == []
    assert r.scrub(["oops", {"role": "assistant"}]) == ["oops", {"role": "assistant"}]


def test_a_run_end_without_a_time_clears_the_last_run():
    r, clock, _ = make()
    a = r.observe_run(_done(run_s=600), "claude:riya", clock.now())
    assert "a1" in r._runs
    assert r.observe_run(_done(run_s=None), "claude:riya", clock.now()) is None
    assert "a1" not in r._runs
    assert r.report(_done(run_s=None), a, sensed(now=clock.now())) is None  # the old run's time is never told
