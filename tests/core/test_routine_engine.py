"""Engine I: what she observes and feels with routine on (v0.12 §4.4, §4.9, A12, A16, Review Focus 2 and 3)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import riya.core.routine as rt
from riya.core.clock import FakeClock
from riya.core.engine import BID_IGNORED_S, RiyaCore
from riya.core.initiative import Action
from riya.core.protocol import ExpressionMsg, GetMemories, RoutineForget, RoutineMsg, TwinReset, LearningReset
from riya.core.routine import audit as au
from tests.fakes import make_core
from tests.routine_helpers import mature_lognormal
from tests.routine_script import GOLDEN, trace

TZ = timezone(timedelta(hours=5, minutes=30))
MON_9 = datetime(2026, 11, 9, 9, tzinfo=TZ).timestamp()
H = 3600.0
TABLES = ("routine_cells", "routine_p2", "routine_obs", "statements", "presence_log")


def core_on(t=MON_9, **kw):
    core, emitted, src, local = make_core(clock=FakeClock(t), tz=TZ, routine=True, **kw)
    return core, emitted, src, local


def tick(core):
    it = core.tick()
    if it:
        list(it)


def run(core, seconds, step=60):
    for _ in range(int(seconds // step)):
        core.clock.advance(step)
        tick(core)


def leave_and_return(core, src, away):
    src.idle = 1000
    core.clock.advance(60)
    tick(core)
    assert core.presence.state == "away"
    core.clock.advance(away)
    src.idle = 0
    tick(core)


def rows(core, table):
    return core.routine.db.conn.execute(f"select count(*) from {table}").fetchone()[0]


def _says(t):
    return [(m["text"], m["kind"], m["spoken"]) for step in t["steps"] for m in step["emitted"]
            if m.get("type") == "say"]


def test_routine_on_adds_nothing_but_statements(tmp_path, monkeypatch):
    for name in ("greeting", "bid", "report", "answer", "farewell", "question"):
        monkeypatch.setattr(rt.Routine, name, lambda self, *a, **k: None)
    monkeypatch.setattr(rt.detect, "claim_guard", lambda *a, **k: True)
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = trace(tmp_path, {"routine_enabled": True})
    assert _says(got) == _says(want)


def test_p1_is_observed_at_came_back_with_its_leave_time():
    core, emitted, src, _ = core_on()
    run(core, 1800)
    leave_and_return(core, src, 3 * H)
    obs = core.routine.db.conn.execute("select bucket, value from routine_obs where predictor='P1'").fetchall()
    assert sorted(b for b, _ in obs) == ["morning", "morning/weekday"]
    assert core._return_appraisal is None  # the greeting used it

    src.is_fullscreen = True  # a greeting deferred by a game still observes the return at came_back
    leave_and_return(core, src, 3 * H)
    assert rows(core, "routine_obs") == 4


def test_a_startle_on_an_early_return_shows_surprised_first():
    core, emitted, src, _ = core_on()
    run(core, 1800)
    leave = core.clock.now() + 60 - 1000
    for b in ("morning/weekday", "morning"):
        mature_lognormal(core.routine.db, "P1", b, 20 * H, n=20, now=leave)
    leave_and_return(core, src, H)
    a = core._return_appraisal
    assert a.direction == "early" and a.startled
    faces = [m.name for m in emitted if isinstance(m, ExpressionMsg)]
    assert "surprised" in faces
    i = faces.index("surprised")
    assert any(f in ("missed_you", "happy") for f in faces[i:]) or core.feelings._after is not None


def test_p3_is_fed_from_agent_events_and_kept_for_the_report():
    core, emitted, src, _ = core_on()
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    core.clock.advance(600)
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "Stop", "cwd": "/w/riya"})
    tick(core)
    assert core._run_appraisals[sid].predictor == "P3"
    assert core.routine.db.get_lognormal("P3", "claude:riya").n == pytest.approx(1.0)


def test_presence_is_logged_and_p4_bands_close():
    core, emitted, src, _ = core_on(datetime(2026, 11, 9, 17, tzinfo=TZ).timestamp())
    tick(core)
    src.is_fullscreen = True
    run(core, 700)
    events = [e for _, e in core.routine.db.presence_between(0, core.clock.now())]
    assert "gaming" in events
    assert core.routine.db.get_beta("P4", "evening/weekday") is not None
    assert any(isinstance(m, RoutineMsg) for m in emitted)


def _bid_core(p_engaged, routine=True):
    core, emitted, src, _ = make_core(clock=FakeClock(MON_9), tz=TZ, routine=routine)
    b = core._bucket()
    for i in range(20):
        core.twin.record("bid", b, i < round(20 * p_engaged))
    core._bid(Action("bid", False, None, "a long stretch"), bucket=b)
    run(core, BID_IGNORED_S + 60)
    return [m.name for m in emitted if isinstance(m, ExpressionMsg)]


def test_p2_and_the_grounded_pout():
    assert "pouty" not in _bid_core(0.1)
    assert "pouty" in _bid_core(0.9)
    assert "pouty" in _bid_core(0.1, routine=False)
    assert "pouty" in _bid_core(0.9, routine=False)


def test_twin_reset_and_learning_reset_clear_p2():
    core, emitted, src, _ = core_on()
    for kind in ("bid", "greeting"):
        core._engage(kind, "b", True)
    assert core.routine.db.get_p2("bid", "b") is not None
    core.handle(TwinReset(kind="bid"))
    assert core.routine.db.get_p2("bid", "b") is None and core.routine.db.get_p2("greeting", "b") is not None
    core.handle(LearningReset())
    assert rows(core, "routine_p2") == 0


def test_forget_empties_the_tables_even_mid_audit():
    core, emitted, src, _ = core_on()
    run(core, 1800)
    leave_and_return(core, src, 3 * H)
    for i in range(3):
        core.routine.said(core.routine._make("S5", True, "claim", {"agent": "riya", "median": "10 minutes"}, {},
                                             {}, "reply", core.clock.now() + i), core.clock.now())
    core.routine.on_scored = lambda row: core.handle(RoutineForget())
    core.routine._last_audit = None
    core.clock.advance(60)
    tick(core)
    for t in TABLES:
        assert rows(core, t) == 0, t
    assert sum(sum(e["claim"].values()) for e in au.tally(core.routine.db).values()) == 1


def test_learning_off_observes_nothing():
    core, emitted, src, _ = core_on()
    core.patch_settings({"learning_enabled": False})
    run(core, 1800)
    leave_and_return(core, src, 3 * H)
    core._engage("bid", "b", True)
    for t in TABLES:
        assert rows(core, t) == 0, t


def test_a_restart_censors_the_stretch_the_run_and_the_band(tmp_path):
    t0 = datetime(2026, 11, 9, 17, tzinfo=TZ).timestamp()
    clock = FakeClock(t0)
    core, emitted, src, _ = make_core(clock=clock, tz=TZ, routine=True, tmp_path=tmp_path)
    run(core, 600)
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    src.is_fullscreen = True
    run(core, 300)  # mid-band, a gaming span under 600 s
    again, emitted2, src2, _ = make_core(clock=clock, tz=TZ, routine=True, tmp_path=tmp_path)
    assert again.presence.censored is True
    clock.advance(2 * H)  # a long stretch, but it began at the restart: never a claim
    s = again.routine.answer("stretch", "how long have i been at it", again._sensed(clock.now()))
    assert s.kind == "S1" and not s.claim
    clock.advance(-2 * H)
    clock.advance(300)
    again.agent_inbox.put({"session_id": sid, "hook_event_name": "Stop", "cwd": "/w/riya"})
    tick(again)
    assert again.routine.db.get_lognormal("P3", "claude:riya") is None  # the run in flight has no time
    clock.t = datetime(2026, 11, 9, 22, 1, tzinfo=TZ).timestamp()
    tick(again)
    assert again.routine.db.get_beta("P4", "evening/weekday") is None  # the restarted band is censored


def test_the_audit_runs_hourly_and_the_purge_daily(monkeypatch):
    core, emitted, src, _ = core_on()
    audits, purges = [], []
    monkeypatch.setattr(rt.au, "audit", lambda *a, **k: audits.append(1) or 0)
    real = core.routine.db.purge
    core.routine.db.purge = lambda now: purges.append(now) or real(now)
    run(core, 2 * H + 60)
    assert len(audits) == 3  # at the first tick, then hourly
    assert len(purges) >= 3  # the audit purges; _maybe_purge too, daily


def test_a_failing_routine_never_stops_the_tick(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("broken routine")
    for name in ("observe_presence", "observe_return", "observe_run", "observe_engagement", "step", "purge", "msg",
                 "greeting", "bid", "report", "answer", "farewell", "question", "said", "clause", "scrub",
                 "on_user_turn"):
        monkeypatch.setattr(rt.Routine, name, boom)
    core, emitted, src, _ = core_on()
    run(core, 1800)
    leave_and_return(core, src, 3 * H)
    assert any(getattr(m, "kind", None) == "greet" for m in emitted)
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    core.clock.advance(600)
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "StopFailure", "cwd": "/w/riya", "error": "x"})
    run(core, 120)
    assert any(getattr(m, "kind", None) == "report" for m in emitted)  # a failure is urgent: said at once


def test_hello_survives_a_broken_routine():
    core, emitted, src, _ = core_on()
    db = core.routine.db
    db.put_cell("P1", "morning/weekday", rt.pr.LogNormalCell(5.0, 40.0, 330.0, rt.ap.fresh("lognormal"), 0.0))
    db.conn.execute("update routine_cells set startled_json='{', n=-1")
    sid = db.add_statement(rt.Statement("S1", True, "x", {}, {}, "bid", None), 0.0)
    db.conn.execute("pragma ignore_check_constraints=1")
    db.conn.execute("update statements set verdict='maybe', value_json='nope' where id=?", (sid,))
    db.conn.execute("pragma ignore_check_constraints=0")
    db.conn.commit()
    msgs = core.hello_messages()
    assert any(isinstance(m, RoutineMsg) for m in msgs)
    run(core, 120)

    core.routine.msg = lambda now: (_ for _ in ()).throw(RuntimeError("x"))
    m = next(m for m in core.hello_messages() if isinstance(m, RoutineMsg))
    assert m.enabled is True and m.cells == []


def test_routine_msg_on_hello_get_memories_and_change():
    core, emitted, src, _ = core_on()
    assert any(isinstance(m, RoutineMsg) and m.enabled for m in core.hello_messages())
    emitted.clear()
    core.handle(GetMemories())
    assert any(isinstance(m, RoutineMsg) for m in emitted)
    emitted.clear()
    core.patch_settings({"routine_enabled": False})
    assert [m.enabled for m in emitted if isinstance(m, RoutineMsg)] == [False]
    emitted.clear()
    core.handle(RoutineForget())
    assert any(isinstance(m, RoutineMsg) for m in emitted)


def test_routine_off_sends_only_its_switch():
    core, emitted, src, _ = make_core(clock=FakeClock(MON_9), tz=TZ)
    assert not any(isinstance(m, RoutineMsg) for m in core.hello_messages())  # the settings carry the switch
    run(core, 1800)
    leave_and_return(core, src, 3 * H)
    assert not any(isinstance(m, RoutineMsg) for m in emitted)
    for t in TABLES:
        assert rows(core, t) == 0, t
