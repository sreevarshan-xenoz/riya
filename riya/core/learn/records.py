"""Learning records in SQLite (schema v4): the live dials (KV), their history, her style notes, the decision twin's
cells and a 30-day signal log. Times are epoch seconds. Note and event text is user-derived data: it is only ever bound
as a parameter."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass

from riya.core.learn.dials import DIALS, clamp
from riya.core.learn.twin import PRIOR, Cell
from riya.core.persona import load_traits

SOURCES = ("explicit", "drift", "reset")
POLARITIES = ("like", "dislike", "rule")
EVENT_KINDS = ("bid", "game_offer", "report", "greeting", "barge", "mood")
EVENT_OUTCOMES = ("engaged", "ignored", "suppressed", "barged", "up", "down", "flat")
MAX_NOTES = 30
EVENTS_KEEP_S = 30 * 86400
DIALS_KEY = "learn.dials"
DRIFT_KEY = "learn.drift_at"
RESET_REASON = "reset to her defaults"
FALLBACK = {"sweetness": 0.85, "teasing": 0.5, "clinginess": 0.8, "verbosity": 0.35, "emoji": 0.3}  # a missing default


@dataclass(frozen=True)
class TraitChange:
    id: int
    at: float
    dial: str
    old: float
    new: float
    reason: str
    source: str


@dataclass(frozen=True)
class StyleNote:
    id: int
    at: float
    text: str
    polarity: str
    active: bool


@dataclass(frozen=True)
class Event:
    id: int
    at: float
    kind: str
    outcome: str
    context: dict


def _num(v) -> bool:
    """A real, finite number (a bool, or an int too big for a float, is not)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _event(r: sqlite3.Row) -> Event:
    try:
        ctx = json.loads(r["context_json"])
    except ValueError:
        ctx = {}
    return Event(r["id"], r["at"], r["kind"], r["outcome"], ctx if isinstance(ctx, dict) else {})


def _cell(r: sqlite3.Row) -> Cell:
    """A row whose alpha, beta or updated_at is not a finite number (alpha and beta positive) reads as the prior."""
    a, b, t = r["alpha"], r["beta"], r["updated_at"]
    if not (_num(a) and _num(b) and _num(t) and a > 0 and b > 0):
        return Cell(PRIOR, PRIOR, t if _num(t) else 0.0)
    return Cell(a, b, t)


class LearnDB:
    """lock: share the memory's lock (like KV and HandsDB), so a commit here never lands inside a memory transaction.
    defaults: her defaults (traits.json when None)."""

    def __init__(self, conn: sqlite3.Connection, lock=None, defaults: dict[str, float] | None = None):
        self.conn, self.lock = conn, lock or threading.RLock()
        base = defaults if defaults is not None else load_traits()
        self.defaults = {d: clamp(d, base[d] if _num(base.get(d)) else FALLBACK[d]) for d in DIALS}  # v1 had no emoji

    # ---- dials ----
    def _dials(self) -> dict[str, float]:
        row = self.conn.execute("select value from kv where key=?", (DIALS_KEY,)).fetchone()
        try:
            stored = json.loads(row[0]) if row else {}
        except ValueError:
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        return {d: clamp(d, stored[d]) if _num(stored.get(d)) else self.defaults[d] for d in DIALS}

    def dials(self) -> dict[str, float]:
        """Every dial: the stored value where it is a finite number, else her default; each within its band."""
        with self.lock:
            return self._dials()

    def _apply(self, values: dict[str, float], reason: str, source: str, now: float) -> dict[str, tuple[float, float]]:
        """Under the lock: set the dials and log each real change, in one commit. Returns (old, new) per dial."""
        if source not in SOURCES:
            raise ValueError(f"unknown source: {source!r}")
        cur = self._dials()
        new = {d: clamp(d, v) for d, v in values.items()}
        changed = {d: (cur[d], v) for d, v in new.items() if v != cur[d]}
        if changed:
            self.conn.execute("insert or replace into kv(key, value) values(?, ?)",
                              (DIALS_KEY, json.dumps({**cur, **new})))
            self.conn.executemany("insert into trait_history(at, dial, old, new, reason, source) values(?, ?, ?, ?, ?, ?)",
                                  [(now, d, o, n, reason, source) for d, (o, n) in changed.items()])
            self.conn.commit()
        return {d: (cur[d], new[d]) for d in new}

    def nudge(self, dial: str, delta: float, reason: str, source: str, now: float) -> tuple[float, float]:
        """(old, new) after moving one dial by delta, clamped to its band; history only when it moved."""
        with self.lock:
            clamp(dial, 0.0)  # an unknown dial is a bug
            return self._apply({dial: self._dials()[dial] + delta}, reason, source, now)[dial]

    def set_dials(self, values: dict[str, float], reason: str, source: str, now: float) -> None:
        with self.lock:
            self._apply(values, reason, source, now)

    def reset_dials(self, now: float) -> None:
        self.set_dials(self.defaults, RESET_REASON, "reset", now)

    def history(self, limit: int = 20) -> list[TraitChange]:
        with self.lock:
            rows = self.conn.execute("select * from trait_history order by id desc limit ?", (limit,)).fetchall()
        return [TraitChange(r["id"], r["at"], r["dial"], r["old"], r["new"], r["reason"], r["source"]) for r in rows]

    # ---- style notes ----
    def add_note(self, text: str, polarity: str, now: float) -> int:
        """The note's id. The same active text again refreshes its time; past MAX_NOTES the oldest is deleted."""
        if polarity not in POLARITIES:
            raise ValueError(f"unknown polarity: {polarity!r}")
        with self.lock:
            row = self.conn.execute("select id from style_notes where active=1 and text=?", (text,)).fetchone()
            if row:
                self.conn.execute("update style_notes set at=? where id=?", (now, row[0]))
                self.conn.commit()
                return row[0]
            nid = self.conn.execute("insert into style_notes(at, text, polarity) values(?, ?, ?)",
                                    (now, text, polarity)).lastrowid
            self.conn.execute("delete from style_notes where id in (select id from style_notes where active=1 "
                              "order by at desc, id desc limit -1 offset ?)", (MAX_NOTES,))
            self.conn.commit()
            return nid

    def notes(self) -> list[StyleNote]:
        """Active notes, the newest first."""
        with self.lock:
            rows = self.conn.execute("select * from style_notes where active=1 order by at desc, id desc").fetchall()
        return [StyleNote(r["id"], r["at"], r["text"], r["polarity"], bool(r["active"])) for r in rows]

    def delete_note(self, id: int) -> bool:
        with self.lock:
            n = self.conn.execute("delete from style_notes where id=?", (id,)).rowcount
            self.conn.commit()
        return n > 0

    def clear_notes(self) -> None:
        with self.lock:
            self.conn.execute("delete from style_notes")
            self.conn.commit()

    # ---- decision twin ----
    def get_cell(self, kind: str, bucket: str) -> Cell | None:
        """The stored cell; one whose alpha, beta or updated_at is corrupt (see _cell) reads as the prior."""
        with self.lock:
            r = self.conn.execute("select alpha, beta, updated_at from twin where kind=? and bucket=?",
                                  (kind, bucket)).fetchone()
        return None if r is None else _cell(r)

    def put_cell(self, kind: str, bucket: str, cell: Cell) -> None:
        with self.lock:
            self.conn.execute("insert or replace into twin(kind, bucket, alpha, beta, updated_at) values(?, ?, ?, ?, ?)",
                              (kind, bucket, cell.alpha, cell.beta, cell.updated_at))
            self.conn.commit()

    def update_cell(self, kind: str, bucket: str, fn) -> None:
        """get_cell, fn, put_cell under one hold of the lock, so a reset_twin is never undone by a record."""
        with self.lock:
            self.put_cell(kind, bucket, fn(self.get_cell(kind, bucket)))

    def cells(self) -> list[tuple[str, str, Cell]]:
        with self.lock:
            rows = self.conn.execute("select kind, bucket, alpha, beta, updated_at from twin "
                                     "order by kind, bucket").fetchall()
        return [(r["kind"], r["bucket"], _cell(r)) for r in rows]

    def reset_twin(self, kind: str | None) -> None:
        """Forget one kind, or every kind for None."""
        with self.lock:
            if kind is None:
                self.conn.execute("delete from twin")
            else:
                self.conn.execute("delete from twin where kind=?", (kind,))
            self.conn.commit()

    # ---- events ----
    def log_event(self, at: float, kind: str, outcome: str, context: dict) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {kind!r}")
        if outcome not in EVENT_OUTCOMES:
            raise ValueError(f"unknown event outcome: {outcome!r}")
        with self.lock:
            self.conn.execute("insert into events(at, kind, outcome, context_json) values(?, ?, ?, ?)",
                              (at, kind, outcome, json.dumps(context, sort_keys=True)))
            self.conn.commit()

    def events(self, limit: int = 100) -> list[Event]:
        with self.lock:
            rows = self.conn.execute("select * from events order by id desc limit ?", (limit,)).fetchall()
        return [_event(r) for r in rows]

    def purge(self, now: float) -> None:
        """Drop events older than EVENTS_KEEP_S."""
        with self.lock:
            self.conn.execute("delete from events where at < ?", (now - EVENTS_KEEP_S,))
            self.conn.commit()

    # ---- verbosity drift ----
    def drift_at(self) -> float | None:
        with self.lock:
            row = self.conn.execute("select value from kv where key=?", (DRIFT_KEY,)).fetchone()
        try:
            v = json.loads(row[0]) if row else None
        except ValueError:
            return None
        return float(v) if _num(v) else None

    def clear_drift_at(self) -> None:
        with self.lock:
            self.conn.execute("delete from kv where key=?", (DRIFT_KEY,))
            self.conn.commit()

    def set_drift_at(self, now: float) -> None:
        with self.lock:
            self.conn.execute("insert or replace into kv(key, value) values(?, ?)", (DRIFT_KEY, json.dumps(now)))
            self.conn.commit()
