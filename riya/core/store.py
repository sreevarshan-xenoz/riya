"""SQLite store: schema migrations, connection setup and a JSON key-value table."""
import json
import os
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 3

_V1 = """
create table facts(id integer primary key, text text not null, key text, source text not null
  check(source in ('user_said','inferred','looked_up')), confidence real not null, valid_from real not null,
  valid_until real, status text not null check(status in ('active','superseded','conflict')),
  supersedes_id integer, episode_id integer, embedding blob, created_at real not null);
create table episodes(id integer primary key, started_at real not null, ended_at real not null, summary text not null,
  user_mood text, outcome text, embedding blob);
create table threads(id integer primary key, text text not null, due_at real not null, status text not null
  check(status in ('open','followed_up','resolved')), last_raised_at real, created_at real not null, episode_id integer);
create table transcripts(id integer primary key, at real not null, role text not null, text text not null,
  consolidated integer not null default 0);
create table forgotten(id integer primary key, description text not null, key text, embedding blob, at real not null);
create table kv(key text primary key, value text not null);
"""

# v0.3: her glance notes (never the screenshots)
_V2 = """
create table glances(id integer primary key, at real not null, app text not null, title text not null,
  activity text not null, details text not null, notable text);
create index glances_at on glances(at);
"""

# v0.5: her hands (reminders and timers, the user's notes, the action log)
_V3 = """
create table reminders(id integer primary key, kind text not null check(kind in ('timer','reminder')),
  due_at real not null, text text not null, created_at real not null, fired_at real, cancelled_at real);
create index reminders_due on reminders(due_at);
create table notes(id integer primary key, at real not null, text text not null);
create table actions(id integer primary key, at real not null, tool text not null, tier integer not null,
  summary text not null, outcome text not null);
create index actions_at on actions(at);
"""

_MIGRATIONS = [_V1, _V2, _V3]


def _backup(conn: sqlite3.Connection, path: Path) -> None:
    """A copy of the database as it is (WAL included) at <db>.pre-v<SCHEMA_VERSION>.bak, made once: an existing backup
    is never overwritten."""
    bak = path.with_name(f"{path.name}.pre-v{SCHEMA_VERSION}.bak")
    if bak.exists():
        return
    tmp = bak.with_name(bak.name + ".tmp")
    dst = sqlite3.connect(str(tmp))
    try:
        conn.backup(dst)
    finally:
        dst.close()
    os.replace(tmp, bak)


def open_db(path: str | Path) -> sqlite3.Connection:
    """Opens (creating) the database and brings its schema up to date; each migration and its version bump commit
    together, so a failure leaves the previous version intact. An existing database is copied to
    <db>.pre-v<N>.bak before it's migrated, and one from a newer Riya (a schema above SCHEMA_VERSION) is refused
    before anything is written to it."""
    memory = str(path) == ":memory:"
    if not memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    version = conn.execute("pragma user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"{path} is from a newer Riya (schema {version}; this one knows up to {SCHEMA_VERSION}). "
                           "It was left untouched: run the newer Riya, or move the file aside to start fresh.")
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    if 0 < version < len(_MIGRATIONS) and not memory:
        try:
            _backup(conn, Path(path))
        except (OSError, sqlite3.Error):  # no backup, no migration: the database stays as it was
            conn.close()
            raise
    for i in range(version, len(_MIGRATIONS)):
        try:
            conn.executescript(f"begin; {_MIGRATIONS[i]} pragma user_version={i + 1}; commit;")
        except sqlite3.Error:
            conn.rollback()
            conn.close()
            raise
    return conn


class KV:
    """lock: share the memory's lock, so a commit here never lands in the middle of a memory transaction."""

    def __init__(self, conn: sqlite3.Connection, lock=None):
        self.conn, self.lock = conn, lock or threading.RLock()

    def get(self, key: str, default=None):
        with self.lock:
            row = self.conn.execute("select value from kv where key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        with self.lock:
            self.conn.execute("insert or replace into kv(key, value) values(?, ?)", (key, json.dumps(value)))
            self.conn.commit()

    def keys_with(self, prefix: str) -> list[str]:
        """Every key starting with prefix (taken literally)."""
        with self.lock:
            rows = self.conn.execute("select key from kv where substr(key, 1, ?) = ?", (len(prefix), prefix))
            return [r[0] for r in rows]

    def delete(self, key: str) -> None:
        with self.lock:
            self.conn.execute("delete from kv where key=?", (key,))
            self.conn.commit()
