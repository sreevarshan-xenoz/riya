"""Hands records in SQLite (schema v3): reminders and timers, the user's notes and the action log.

Times are epoch seconds. Notes are never facts. A reminder stays "active" (and so is returned by `due`) until it is
marked fired or cancelled, so a due reminder is never silently lost.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

MAX_ACTIVE = 50
PURGE_S = 30 * 86400
OUTCOMES = ("ran", "failed", "refused", "declined", "expired", "cancelled", "replaced")
SUMMARY_MAX = 500
_ACTIVE = "fired_at is null and cancelled_at is null"


@dataclass(frozen=True)
class Reminder:
    id: int
    kind: str
    due_at: float
    text: str
    created_at: float


@dataclass(frozen=True)
class Note:
    id: int
    at: float
    text: str


@dataclass(frozen=True)
class Action:
    id: int
    at: float
    tool: str
    tier: int
    summary: str
    outcome: str


def _rem(r: sqlite3.Row) -> Reminder:
    return Reminder(r["id"], r["kind"], r["due_at"], r["text"], r["created_at"])


class HandsDB:
    """lock: share the memory's lock (like KV), so a commit here never lands inside a memory transaction."""

    def __init__(self, conn: sqlite3.Connection, lock=None):
        self.conn, self.lock = conn, lock or threading.RLock()

    # ---- reminders and timers ----
    def add_reminder(self, kind: str, due_at: float, text: str, now: float) -> int | None:
        """The new id, or None when MAX_ACTIVE are already active."""
        with self.lock:
            if self.conn.execute(f"select count(*) from reminders where {_ACTIVE}").fetchone()[0] >= MAX_ACTIVE:
                return None
            cur = self.conn.execute("insert into reminders(kind, due_at, text, created_at) values(?, ?, ?, ?)",
                                    (kind, due_at, text, now))
            self.conn.commit()
            return cur.lastrowid

    def active(self) -> list[Reminder]:
        """Every reminder neither fired nor cancelled, the soonest first."""
        with self.lock:
            rows = self.conn.execute(f"select * from reminders where {_ACTIVE} order by due_at, id").fetchall()
        return [_rem(r) for r in rows]

    def due(self, now: float) -> list[Reminder]:
        """Active reminders with due_at <= now, the oldest first. However late, a due one stays until marked fired."""
        with self.lock:
            rows = self.conn.execute(f"select * from reminders where {_ACTIVE} and due_at <= ? order by due_at, id",
                                     (now,)).fetchall()
        return [_rem(r) for r in rows]

    def mark_fired(self, ids, now: float) -> None:
        """Idempotent: an already fired or cancelled one keeps its first stamp."""
        with self.lock:
            self.conn.executemany(f"update reminders set fired_at=? where id=? and {_ACTIVE}",
                                  [(now, int(i)) for i in ids])
            self.conn.commit()

    def cancel(self, id: int, now: float) -> bool:
        """True when it was active and is now cancelled."""
        with self.lock:
            cur = self.conn.execute(f"update reminders set cancelled_at=? where id=? and {_ACTIVE}", (now, id))
            self.conn.commit()
            return cur.rowcount > 0

    def find_active(self, text: str) -> Reminder | None:
        """The soonest active reminder whose text contains `text` (case-insensitive). "timer" prefers a real timer
        (kind 'timer', the newest first) before any text match, so "cancel my timer" never hits "check the timer"."""
        needle = text.strip().casefold()
        if not needle:
            return None
        items = self.active()
        if needle == "timer" and (timers := [r for r in items if r.kind == "timer"]):
            return max(timers, key=lambda r: (r.created_at, r.id))
        return next((r for r in items if needle in r.text.casefold()), None)

    # ---- notes ----
    def add_note(self, text: str, now: float) -> int:
        with self.lock:
            cur = self.conn.execute("insert into notes(at, text) values(?, ?)", (now, text))
            self.conn.commit()
            return cur.lastrowid

    def notes(self) -> list[Note]:
        """Newest first."""
        with self.lock:
            rows = self.conn.execute("select * from notes order by at desc, id desc").fetchall()
        return [Note(r["id"], r["at"], r["text"]) for r in rows]

    def delete_note(self, id: int) -> bool:
        with self.lock:
            cur = self.conn.execute("delete from notes where id=?", (id,))
            self.conn.commit()
            return cur.rowcount > 0

    # ---- the action log ----
    def log_action(self, at: float, tool: str, tier: int, summary: str, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        with self.lock:
            self.conn.execute("insert into actions(at, tool, tier, summary, outcome) values(?, ?, ?, ?, ?)",
                              (at, tool, tier, summary[:SUMMARY_MAX], outcome))
            self.conn.commit()

    def actions(self, limit: int = 100) -> list[Action]:
        """Newest first."""
        with self.lock:
            rows = self.conn.execute("select * from actions order by at desc, id desc limit ?", (limit,)).fetchall()
        return [Action(r["id"], r["at"], r["tool"], r["tier"], r["summary"], r["outcome"]) for r in rows]

    def clear_actions(self) -> None:
        with self.lock:
            self.conn.execute("delete from actions")
            self.conn.commit()

    def purge(self, now: float) -> None:
        """Drop actions, and reminders fired or cancelled, older than 30 days. Active reminders and notes stay."""
        cut = now - PURGE_S
        with self.lock:
            self.conn.execute("delete from actions where at < ?", (cut,))
            self.conn.execute("delete from reminders where coalesce(fired_at, cancelled_at) < ?", (cut,))
            self.conn.commit()
