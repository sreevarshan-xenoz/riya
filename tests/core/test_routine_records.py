"""Schema v6 and the routine records (v0.12 §4.8, §4.12.7, A16, A17)."""
import hashlib
import json
import math
import sqlite3

import pytest

from riya.core import store
from riya.core.routine import appraisal as ap
from riya.core.routine.predictors import BetaCell, LogNormalCell
from riya.core.routine.records import TALLY_KEY, RoutineDB
from riya.core.routine.statements import Statement
from riya.core.store import KV, open_db

DAY = 86400.0
TABLES = {"routine_cells", "routine_p2", "routine_obs", "statements", "presence_log"}
INDEXES = {"routine_obs_at", "statements_at", "presence_log_at"}


def _db():
    return RoutineDB(open_db(":memory:"))


def _st(kind="S1", text="You've been at it since 9:05 am.", where="bid", claim=True):
    return Statement(kind=kind, claim=claim, text=text, value={"since": 1000.0}, evidence={"n_eff": 9.0},
                     where=where, due_at=2000.0)


def _v5_db(p):
    conn = sqlite3.connect(str(p))
    for i, script in enumerate([store._V1, store._V2, store._V3, store._V4, store._V5]):
        conn.executescript(f"begin; {script} pragma user_version={i + 1}; commit;")
    conn.executescript(
        "insert into facts(text, source, confidence, valid_from, status, created_at) "
        "values('likes tea', 'user_said', 1.0, 0, 'active', 0);"
        "insert into twin(kind, bucket, alpha, beta, updated_at) values('bid', 'morning/active/short', 3, 2, 0);"
        "insert into kv(key, value) values('settings', '{\"onboarded\": true}');")
    conn.close()


def test_a_new_db_has_the_v6_tables():
    conn = open_db(":memory:")
    assert conn.execute("pragma user_version").fetchone()[0] == 6
    names = {r[0] for r in conn.execute("select name from sqlite_master")}
    assert TABLES <= names and INDEXES <= names


def test_a_v5_db_migrates_to_v6_with_its_backup(tmp_path):
    p = tmp_path / "riya.db"
    _v5_db(p)
    conn = open_db(p)
    assert conn.execute("pragma user_version").fetchone()[0] == 6
    assert conn.execute("select count(*) from facts").fetchone()[0] == 1
    assert conn.execute("select alpha from twin").fetchone()[0] == 3
    assert json.loads(conn.execute("select value from kv where key='settings'").fetchone()[0]) == {"onboarded": True}
    bak = tmp_path / "riya.db.pre-v6.bak"
    old = sqlite3.connect(str(bak))
    assert old.execute("pragma user_version").fetchone()[0] == 5
    old.close()
    conn.close()
    before = bak.stat().st_mtime_ns
    open_db(p).close()
    assert bak.stat().st_mtime_ns == before
    assert sorted(x.name for x in tmp_path.iterdir() if x.name.endswith(".bak")) == ["riya.db.pre-v6.bak"]


def test_a_v7_db_is_refused_untouched(tmp_path):
    p = tmp_path / "riya.db"
    conn = sqlite3.connect(str(p))
    conn.executescript("create table future(x); insert into future values(1); pragma user_version=7;")
    conn.close()
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="newer Riya.*schema 7"):
        open_db(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["riya.db"]


def test_a_failed_v6_migration_leaves_the_v5_database_untouched(tmp_path, monkeypatch):
    p = tmp_path / "riya.db"
    _v5_db(p)
    monkeypatch.setattr(store, "_MIGRATIONS", [store._V1, store._V2, store._V3, store._V4, store._V5,
                                               "create table routine_cells(x); create table broken(;"])
    with pytest.raises(sqlite3.Error):
        open_db(p)
    monkeypatch.undo()
    conn = sqlite3.connect(str(p))
    assert conn.execute("pragma user_version").fetchone()[0] == 5
    assert conn.execute("select count(*) from sqlite_master where name='routine_cells'").fetchone()[0] == 0
    assert conn.execute("select count(*) from facts").fetchone()[0] == 1
    conn.close()
    assert open_db(p).execute("pragma user_version").fetchone()[0] == 6


def test_cells_round_trip():
    db = _db()
    st = ap.AppraisalState(0.3, 1.25, (1.0, 2.0, 3.0))
    ln = LogNormalCell(5.5, 44.0, 360.0, st, 100.0)
    be = BetaCell(2.5, 4.0, ap.AppraisalState(0.5, 0.0, ()), 200.0)
    db.put_cell("P1", "weekday/morning", ln)
    db.put_cell("P4", "weekday/evening", be)
    assert db.get_lognormal("P1", "weekday/morning") == ln
    assert db.get_beta("P4", "weekday/evening") == be
    assert db.get_beta("P1", "weekday/morning") is None and db.get_lognormal("P4", "weekday/evening") is None
    assert db.get_lognormal("P1", "missing") is None
    assert [(p, b) for p, b, _ in db.cells()] == [("P1", "weekday/morning"), ("P4", "weekday/evening")]
    assert [c for _, _, c in db.cells("P4")] == [be]

    def bump(cell):
        new = LogNormalCell(cell.n + 1, cell.s1, cell.s2, cell.state, 300.0)
        return new, "done"
    assert db.update_cell("P1", "weekday/morning", bump) == "done"
    assert db.get_lognormal("P1", "weekday/morning").n == 6.5


def test_corrupt_rows_read_as_the_prior_or_are_skipped():
    db = _db()
    good = LogNormalCell(5.0, 40.0, 330.0, ap.AppraisalState(0.3, 0.0, (1.0,)), 100.0)
    for bucket in ("nan_is_null", "neg", "brace", "inf", "s1_inf", "s2_null", "deep"):
        db.put_cell("P1", bucket, good)
    c = db.conn
    c.execute("update routine_cells set n=? where bucket='nan_is_null'", (math.nan,))  # SQLite stores NaN as NULL
    c.execute("update routine_cells set s1=? where bucket='s1_inf'", (math.inf,))
    c.execute("update routine_cells set s2=NULL where bucket='s2_null'")
    c.execute("update routine_cells set startled_json=? where bucket='deep'", ("[" * 100000,))
    c.execute("update routine_cells set n=-1 where bucket='neg'")
    c.execute("update routine_cells set startled_json='{' where bucket='brace'")
    c.execute("update routine_cells set updated_at=? where bucket='inf'", (math.inf,))
    c.commit()
    assert db.get_lognormal("P1", "nan_is_null") is None
    assert db.get_lognormal("P1", "s1_inf") is None and db.get_lognormal("P1", "s2_null") is None
    assert db.get_lognormal("P1", "deep").state.startled == ()
    assert db.get_lognormal("P1", "neg") is None
    assert db.get_lognormal("P1", "inf") is None
    assert db.get_lognormal("P1", "brace").state.startled == ()
    assert [b for _, b, _ in db.cells()] == ["brace", "deep"]
    db.put_p2("bid", "b", ap.AppraisalState(0.5, 0.0, ()), 0.0)
    c.execute("update routine_p2 set startled_json='nope'")
    c.commit()
    assert db.get_p2("bid", "b").startled == ()

    ok = db.add_statement(_st(), 10.0)
    bad_json = db.add_statement(_st(text="x"), 11.0)
    bad_verdict = db.add_statement(_st(text="y"), 12.0)
    c.execute("update statements set value_json='nope' where id=?", (bad_json,))
    c.execute("pragma ignore_check_constraints=1")
    c.execute("update statements set verdict='maybe' where id=?", (bad_verdict,))
    c.execute("pragma ignore_check_constraints=0")
    c.commit()
    assert [s.id for s in db.statements_since(0)] == [ok]
    assert [s.id for s in db.pending()] == [ok]
    assert db.statement(bad_json) is None and db.statement(bad_verdict) is None


def test_statements_obs_and_presence_round_trip():
    db = _db()
    sid = db.add_statement(_st(), 10.0)
    row = db.statement(sid)
    assert (row.kind, row.claim, row.text, row.value, row.evidence, row.where, row.due_at) == \
        ("S1", True, "You've been at it since 9:05 am.", {"since": 1000.0}, {"n_eff": 9.0}, "bid", 2000.0)
    assert (row.verdict, row.disputed, row.scored_at) == ("pending", False, None)
    assert db.set_disputed(sid) and db.statement(sid).disputed
    assert db.set_verdict(sid, "true", 50.0)
    assert db.statement(sid).verdict == "true" and db.statement(sid).scored_at == 50.0
    assert db.pending() == []
    assert [s.id for s in db.scored_since(40.0)] == [sid] and db.scored_since(51.0) == []
    assert not db.set_verdict(sid, "maybe", 60.0)
    assert not db.set_verdict(999, "true", 60.0) and not db.set_disputed(999)

    i = db.add_obs(5.0, "P1", "weekday/morning", 3600.0)
    db.add_obs(15.0, "P1", "weekday/morning", 1800.0)
    db.add_obs(6.0, "P1", "weekend/morning", 60.0)
    assert db.obs("P1", "weekday/morning", 0.0, 10.0) == [(i, 5.0, 3600.0)]
    assert [v for _, _, v in db.obs("P1", "weekday/morning", 0.0, 20.0)] == [3600.0, 1800.0]

    for at, ev in ((1.0, "away"), (2.0, "back"), (3.0, "break"), (4.0, "break_end")):
        db.log_presence(at, ev)
    assert db.presence_between(2.0, 3.0) == [(2.0, "back"), (3.0, "break")]
    with pytest.raises(ValueError):
        db.log_presence(5.0, "sleeping")

    db.kv_set("routine.x", {"a": 1})
    assert db.kv_get("routine.x") == {"a": 1} and db.kv_get("routine.none", 7) == 7
    with pytest.raises(ValueError):
        db.kv_set("settings", {})
    with pytest.raises(ValueError):
        db.kv_get("learn.dials")


def test_purge_keeps_what_the_audit_needs():
    db = _db()
    now = 100 * DAY
    db.add_obs(now - 30 * DAY - 1, "P1", "b", 1.0)
    keep_obs = db.add_obs(now - 30 * DAY + 1, "P1", "b", 2.0)
    old = db.add_statement(_st(text="old"), now - 40 * DAY)
    db.set_verdict(old, "true", now - 30 * DAY - 1)
    recent = db.add_statement(_st(text="recent"), now - 40 * DAY)
    db.set_verdict(recent, "false", now - 30 * DAY + 1)
    pending = db.add_statement(_st(text="pending"), now - 90 * DAY)
    db.log_presence(now - 14 * DAY - 1, "away")
    db.log_presence(now - 14 * DAY + 1, "back")
    db.purge(now)
    assert [i for i, _, _ in db.obs("P1", "b", 0, now)] == [keep_obs]
    assert {s.id for s in db.statements_since(0)} == {recent, pending}
    assert db.presence_between(0, now) == [(now - 14 * DAY + 1, "back")]


def test_forget_empties_everything_but_the_tally():
    db = _db()
    kv = KV(db.conn, db.lock)
    db.put_cell("P1", "b", LogNormalCell(1.0, 1.0, 1.0, ap.fresh("lognormal"), 0.0))
    db.put_p2("bid", "b", ap.fresh("beta"), 0.0)
    db.add_obs(0.0, "P1", "b", 1.0)
    db.add_statement(_st(), 0.0)
    db.log_presence(0.0, "away")
    db.kv_set(TALLY_KEY, {"true": 3})
    db.kv_set("routine.frozen", True)
    kv.set("settings", {"onboarded": True})
    db.forget()
    for t in TABLES:
        assert db.conn.execute(f"select count(*) from {t}").fetchone()[0] == 0, t
    assert kv.keys_with("routine.") == ["routine.tally"]
    assert db.kv_get(TALLY_KEY) == {"true": 3}
    assert kv.get("settings") == {"onboarded": True}


def test_reset_p2_by_kind_or_all():
    db = _db()
    for kind in ("bid", "greeting"):
        for b in ("x", "y"):
            db.put_p2(kind, b, ap.fresh("beta"), 0.0)
    db.reset_p2("bid")
    assert db.get_p2("bid", "x") is None and db.get_p2("greeting", "x") is not None
    db.reset_p2()
    assert db.get_p2("greeting", "y") is None


def test_no_column_holds_user_text():
    conn = open_db(":memory:")
    text_cols = set()
    for t in TABLES:
        for col in conn.execute(f"pragma table_info({t})"):
            if col[2].lower() == "text":
                text_cols.add(col[1])
    assert text_cols == {"predictor", "bucket", "startled_json", "kind", "text", "value_json", "evidence_json",
                         "where_", "verdict", "event"}
    db = RoutineDB(conn)
    db.add_obs(0.0, "P3", "claude:my folder!", 1.0)
    assert conn.execute("select bucket from routine_obs").fetchone()[0] == "claude:my folder_"
