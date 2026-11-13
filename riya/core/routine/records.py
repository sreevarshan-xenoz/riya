"""RoutineDB: the routine tables of schema v6, behind the memory's lock (the LearnDB pattern; every write commits).

A corrupt or hand-edited row never raises: a bad cell reads as the prior (None), a bad startled buffer as (), and a
statement with bad JSON or an unknown verdict is skipped. No table holds anything the user typed.
"""
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from typing import Callable

from riya.core.prompt import _safe
from riya.core.routine.appraisal import AppraisalState
from riya.core.routine.predictors import BetaCell, LogNormalCell
from riya.core.routine.statements import Statement

VERDICTS = ("pending", "true", "false", "unverifiable", "expired")
PRESENCE_EVENTS = ("away", "back", "break", "break_end", "gaming", "gaming_end")
OBS_KEEP_S = 30 * 86400.0
SCORED_KEEP_S = 30 * 86400.0
PRESENCE_KEEP_S = 14 * 86400.0
KV_PREFIX = "routine."
TALLY_KEY = "routine.tally"

_TABLES = ("routine_cells", "routine_p2", "routine_obs", "statements", "presence_log")


@dataclass(frozen=True)
class StatementRow:
    id: int
    at: float
    kind: str
    claim: bool
    text: str
    value: dict
    evidence: dict
    where: str
    due_at: float | None
    verdict: str
    disputed: bool
    scored_at: float | None


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _startled(raw) -> tuple[float, ...]:
    try:
        vals = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return ()
    if not isinstance(vals, list) or not all(_finite(v) for v in vals):
        return ()
    return tuple(float(v) for v in vals)


def _state(row) -> AppraisalState | None:
    if not (_finite(row["scale"]) and _finite(row["cusum"]) and _finite(row["updated_at"])):
        return None
    return AppraisalState(float(row["scale"]), float(row["cusum"]), _startled(row["startled_json"]))


def _cell(row) -> LogNormalCell | BetaCell | None:
    st = _state(row)
    if st is None:
        return None
    if row["n"] is not None:
        if not all(_finite(row[k]) for k in ("n", "s1", "s2")) or row["n"] < 0:
            return None
        return LogNormalCell(float(row["n"]), float(row["s1"]), float(row["s2"]), st, float(row["updated_at"]))
    if not (_finite(row["alpha"]) and _finite(row["beta"])) or row["alpha"] <= 0 or row["beta"] <= 0:
        return None
    return BetaCell(float(row["alpha"]), float(row["beta"]), st, float(row["updated_at"]))


def _statement(row) -> StatementRow | None:
    try:
        value, evidence = json.loads(row["value_json"]), json.loads(row["evidence_json"])
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or not isinstance(evidence, dict) or row["verdict"] not in VERDICTS:
        return None
    return StatementRow(id=row["id"], at=row["at"], kind=row["kind"], claim=bool(row["claim"]), text=row["text"],
                        value=value, evidence=evidence, where=row["where_"], due_at=row["due_at"],
                        verdict=row["verdict"], disputed=bool(row["disputed"]), scored_at=row["scored_at"])


class RoutineDB:
    """lock: share the memory's lock, so a commit here never lands inside a memory transaction."""

    def __init__(self, conn: sqlite3.Connection, lock=None):
        self.conn, self.lock = conn, lock or threading.RLock()

    def _rows(self, sql: str, args=()) -> list:
        cur = self.conn.execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ---- cells ----
    def _get(self, predictor: str, bucket: str):
        rows = self._rows("select * from routine_cells where predictor=? and bucket=?", (predictor, bucket))
        return _cell(rows[0]) if rows else None

    def get_lognormal(self, predictor: str, bucket: str) -> LogNormalCell | None:
        with self.lock:
            c = self._get(predictor, bucket)
        return c if isinstance(c, LogNormalCell) else None

    def get_beta(self, predictor: str, bucket: str) -> BetaCell | None:
        with self.lock:
            c = self._get(predictor, bucket)
        return c if isinstance(c, BetaCell) else None

    def _put(self, predictor: str, bucket: str, cell: LogNormalCell | BetaCell) -> None:
        st = cell.state
        if isinstance(cell, LogNormalCell):
            stats = (cell.n, cell.s1, cell.s2, None, None)
        else:
            stats = (None, None, None, cell.alpha, cell.beta)
        self.conn.execute("insert or replace into routine_cells(predictor, bucket, n, s1, s2, alpha, beta, scale, cusum,"
                          " startled_json, updated_at) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (predictor, bucket, *stats, st.scale, st.cusum, json.dumps(list(st.startled)),
                           cell.updated_at))

    def put_cell(self, predictor: str, bucket: str, cell: LogNormalCell | BetaCell) -> None:
        with self.lock:
            self._put(predictor, bucket, cell)
            self.conn.commit()

    def update_cell(self, predictor: str, bucket: str, fn: Callable) -> object:
        """Reads the cell (None when missing or corrupt), calls fn(cell) -> (new_cell, result) and writes new_cell, all
        under one hold of the lock. Returns result."""
        with self.lock:
            new, result = fn(self._get(predictor, bucket))
            if new is not None:
                self._put(predictor, bucket, new)
                self.conn.commit()
            return result

    def cells(self, predictor: str | None = None) -> list[tuple[str, str, LogNormalCell | BetaCell]]:
        with self.lock:
            rows = self._rows("select * from routine_cells" + (" where predictor=?" if predictor is not None else "")
                              + " order by predictor, bucket", (predictor,) if predictor is not None else ())
        out = []
        for r in rows:
            c = _cell(r)
            if c is not None:
                out.append((r["predictor"], r["bucket"], c))
        return out

    # ---- P2's appraisal state ----
    def get_p2(self, kind: str, bucket: str) -> AppraisalState | None:
        with self.lock:
            rows = self._rows("select * from routine_p2 where kind=? and bucket=?", (kind, bucket))
        return _state(rows[0]) if rows else None

    def put_p2(self, kind: str, bucket: str, state: AppraisalState, now: float) -> None:
        with self.lock:
            self.conn.execute("insert or replace into routine_p2(kind, bucket, scale, cusum, startled_json, updated_at)"
                              " values(?, ?, ?, ?, ?, ?)",
                              (kind, bucket, state.scale, state.cusum, json.dumps(list(state.startled)), now))
            self.conn.commit()

    def reset_p2(self, kind: str | None = None) -> None:
        with self.lock:
            if kind is None:
                self.conn.execute("delete from routine_p2")
            else:
                self.conn.execute("delete from routine_p2 where kind=?", (kind,))
            self.conn.commit()

    # ---- observations ----
    def add_obs(self, at: float, predictor: str, bucket: str, value: float) -> int:
        with self.lock:
            cur = self.conn.execute("insert into routine_obs(at, predictor, bucket, value) values(?, ?, ?, ?)",
                                    (at, predictor, _safe(bucket), value))
            self.conn.commit()
            return cur.lastrowid

    def obs(self, predictor: str, bucket: str, start: float, end: float) -> list[tuple[int, float, float]]:
        with self.lock:
            rows = self.conn.execute("select id, at, value from routine_obs where predictor=? and bucket=? and at>=? "
                                     "and at<=? order by at", (predictor, _safe(bucket), start, end)).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    # ---- statements ----
    def add_statement(self, st: Statement, at: float) -> int:
        with self.lock:
            cur = self.conn.execute(
                "insert into statements(at, kind, claim, text, value_json, evidence_json, where_, due_at) "
                "values(?, ?, ?, ?, ?, ?, ?, ?)",
                (at, st.kind, int(st.claim), st.text, json.dumps(st.value), json.dumps(st.evidence), st.where,
                 st.due_at))
            self.conn.commit()
            return cur.lastrowid

    def _statements(self, where: str, args=()) -> list[StatementRow]:
        with self.lock:
            rows = self._rows("select * from statements " + where, args)
        return [s for s in (_statement(r) for r in rows) if s is not None]

    def statement(self, id: int) -> StatementRow | None:
        rows = self._statements("where id=?", (id,))
        return rows[0] if rows else None

    def pending(self) -> list[StatementRow]:
        return self._statements("where verdict='pending' order by at")

    def set_verdict(self, id: int, verdict: str, at: float) -> bool:
        if verdict not in VERDICTS:
            return False
        with self.lock:
            cur = self.conn.execute("update statements set verdict=?, scored_at=? where id=?", (verdict, at, id))
            self.conn.commit()
            return cur.rowcount > 0

    def set_disputed(self, id: int) -> bool:
        with self.lock:
            cur = self.conn.execute("update statements set disputed=1 where id=?", (id,))
            self.conn.commit()
            return cur.rowcount > 0

    def statements_since(self, at: float) -> list[StatementRow]:
        return self._statements("where at>=? order by at", (at,))

    def scored_since(self, at: float) -> list[StatementRow]:
        return self._statements("where verdict!='pending' and scored_at>=? order by scored_at", (at,))

    # ---- presence ----
    def log_presence(self, at: float, event: str) -> None:
        if event not in PRESENCE_EVENTS:
            raise ValueError(f"unknown presence event: {event!r}")
        with self.lock:
            self.conn.execute("insert into presence_log(at, event) values(?, ?)", (at, event))
            self.conn.commit()

    def presence_between(self, start: float, end: float) -> list[tuple[float, str]]:
        with self.lock:
            rows = self.conn.execute("select at, event from presence_log where at>=? and at<=? order by at, id",
                                     (start, end)).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ---- kv ----
    @staticmethod
    def _key(key: str) -> str:
        if not isinstance(key, str) or not key.startswith(KV_PREFIX):
            raise ValueError(f"not a routine key: {key!r}")
        return key

    def kv_get(self, key: str, default=None):
        with self.lock:
            row = self.conn.execute("select value from kv where key=?", (self._key(key),)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (ValueError, RecursionError):
            return default

    def kv_set(self, key: str, value) -> None:
        with self.lock:
            self.conn.execute("insert or replace into kv(key, value) values(?, ?)", (self._key(key), json.dumps(value)))
            self.conn.commit()

    # ---- housekeeping ----
    def purge(self, now: float) -> None:
        with self.lock:
            self.conn.execute("delete from routine_obs where at < ?", (now - OBS_KEEP_S,))
            self.conn.execute("delete from statements where verdict!='pending' and scored_at is not null "
                              "and scored_at < ?", (now - SCORED_KEEP_S,))
            self.conn.execute("delete from presence_log where at < ?", (now - PRESENCE_KEEP_S,))
            self.conn.commit()

    def forget(self) -> None:
        """Every routine row and every routine.* KV key but the honesty tally, in one transaction."""
        with self.lock:
            try:
                for t in _TABLES:
                    self.conn.execute(f"delete from {t}")
                self.conn.execute("delete from kv where substr(key, 1, ?) = ? and key != ?",
                                  (len(KV_PREFIX), KV_PREFIX, TALLY_KEY))
                self.conn.commit()
            except sqlite3.Error:
                self.conn.rollback()
                raise
