"""SQLite store: schema migrations, connection setup and a JSON key-value table."""
import json
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 2

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

_MIGRATIONS = [_V1, _V2]


def open_db(path: str | Path) -> sqlite3.Connection:
    """Opens (creating) the database and brings its schema up to date; each migration and its version bump commit
    together, so a failure leaves the previous version intact."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    version = conn.execute("pragma user_version").fetchone()[0]
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
