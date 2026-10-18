"""Riya's long-term memory: facts, episodes, follow-up threads and transcripts, with hybrid retrieval."""
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from riya.core.embed import Embedder

RANK = {"user_said": 2, "looked_up": 1, "inferred": 0}
QUALITY = {"user_said": 1.0, "looked_up": 0.7, "inferred": 0.5}
LIVE = "('active','conflict')"
EPISODE_FORGET_SIM = 0.6
DAY = 86400.0


@dataclass
class Fact:
    id: int
    text: str
    key: str | None
    source: str
    confidence: float
    valid_from: float
    valid_until: float | None
    status: str
    supersedes_id: int | None


@dataclass
class Episode:
    id: int
    started_at: float
    ended_at: float
    summary: str
    user_mood: str | None
    outcome: str | None


@dataclass
class Thread:
    id: int
    text: str
    due_at: float
    status: str
    last_raised_at: float | None


@dataclass
class Retrieved:
    facts: list[Fact]
    episodes: list[Episode]
    threads: list[Thread]


_FACT_COLS = "id, text, key, source, confidence, valid_from, valid_until, status, supersedes_id"
_EPISODE_COLS = "id, started_at, ended_at, summary, user_mood, outcome"
_THREAD_COLS = "id, text, due_at, status, last_raised_at"


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(**{k: row[k] for k in _FACT_COLS.split(", ")})


def _episode(row: sqlite3.Row) -> Episode:
    return Episode(**{k: row[k] for k in _EPISODE_COLS.split(", ")})


def _thread(row: sqlite3.Row) -> Thread:
    return Thread(**{k: row[k] for k in _THREAD_COLS.split(", ")})


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Memory:
    def __init__(self, db: sqlite3.Connection, embedder: Embedder, clock,
                 dup_sim: float = 0.92, forgotten_sim: float = 0.85, min_sim: float = 0.2):
        self.db, self.embedder, self.clock = db, embedder, clock
        self.dup_sim, self.forgotten_sim, self.min_sim = dup_sim, forgotten_sim, min_sim

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    def _max_sim(self, vec: np.ndarray, rows) -> float:
        # rows hold L2-normalized (or all-zero) blobs, so a dot product is the cosine and never divides by zero;
        # blobs from another embedder (different length) never match
        sims = []
        for r in rows:
            v = _vec(r["embedding"]) if r["embedding"] else None
            if v is not None and v.shape == vec.shape:
                sims.append(float(np.dot(vec, v)))
        return max(sims, default=0.0)

    # --- facts ---

    def add_fact(self, text: str, source: str, confidence: float = 0.9, key: str | None = None,
                 episode_id: int | None = None, *, check_forgotten: bool | None = None) -> Fact | None:
        """check_forgotten: None checks forgotten markers unless the source is user_said, True always checks."""
        if source not in RANK:
            raise ValueError(f"unknown source {source!r}")
        vec = self._embed(text)
        if check_forgotten or (check_forgotten is None and source != "user_said"):
            # a live, explicit user statement overrides a forgotten marker; a model's reading of a transcript doesn't
            forgotten = self.db.execute("select key, embedding from forgotten").fetchall()
            if key is not None and any(r["key"] == key for r in forgotten):
                return None
            if self._max_sim(vec, forgotten) >= self.forgotten_sim:
                return None
        live = self.db.execute(f"select text, embedding from facts where status in {LIVE}").fetchall()
        if any(r["text"] == text for r in live) or self._max_sim(vec, live) >= self.dup_sim:
            return None
        now = self.clock.now()
        status, supersedes_id = "active", None
        olds = [] if key is None else self.db.execute(
            f"select id, source from facts where key=? and status in {LIVE} order by id", (key,)).fetchall()
        with self.db:
            for old in olds:
                if source == "user_said" or RANK[source] > RANK[old["source"]]:
                    self.db.execute("update facts set status='superseded', valid_until=? where id=?", (now, old["id"]))
                    supersedes_id = old["id"]
                else:
                    self.db.execute("update facts set status='conflict' where id=?", (old["id"],))
                    status = "conflict"
            cur = self.db.execute(
                "insert into facts(text, key, source, confidence, valid_from, status, supersedes_id, episode_id,"
                " embedding, created_at) values(?,?,?,?,?,?,?,?,?,?)",
                (text, key, source, confidence, now, status, supersedes_id, episode_id, vec.tobytes(), now))
        return self.get_fact(cur.lastrowid)

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.db.execute(f"select {_FACT_COLS} from facts where id=?", (fact_id,)).fetchone()
        return _fact(row) if row else None

    def correct(self, fact_id: int, new_text: str) -> Fact:
        """An explicit user statement: skips the forgotten check and supersedes every live fact on the key."""
        old = self.get_fact(fact_id)
        if old is None:
            raise KeyError(f"no fact {fact_id}")
        if old.status not in ("active", "conflict"):
            raise ValueError(f"fact {fact_id} is {old.status}; correct its current version instead")
        now = self.clock.now()
        vec = self._embed(new_text)
        with self.db:
            self.db.execute(
                f"update facts set status='superseded', valid_until=? where id=? "
                f"or (? is not null and key=? and status in {LIVE})", (now, fact_id, old.key, old.key))
            cur = self.db.execute(
                "insert into facts(text, key, source, confidence, valid_from, status, supersedes_id, embedding,"
                " created_at) values(?,?,?,?,?,?,?,?,?)",
                (new_text, old.key, "user_said", 1.0, now, "active", fact_id, vec.tobytes(), now))
        return self.get_fact(cur.lastrowid)

    def _chain(self, fact_id: int) -> list[int]:
        """The supersedes_id chain of fact_id (both directions) plus superseded rows sharing a key with it."""
        seen, todo = set(), [fact_id]
        while todo:
            i = todo.pop()
            if i in seen:
                continue
            seen.add(i)
            row = self.db.execute("select supersedes_id from facts where id=?", (i,)).fetchone()
            if row and row["supersedes_id"] is not None:
                todo.append(row["supersedes_id"])
            todo += [r[0] for r in self.db.execute("select id from facts where supersedes_id=?", (i,))]
        keys = {r[0] for r in self.db.execute(
            f"select key from facts where id in ({','.join('?' * len(seen))}) and key is not null", list(seen))}
        for k in keys:  # versions superseded in bulk are linked to the new fact only through the shared key
            seen.update(r[0] for r in self.db.execute("select id from facts where key=? and status='superseded'", (k,)))
        return sorted(seen)

    def delete_fact(self, fact_id: int, delete_episodes: bool = False) -> None:
        if self.get_fact(fact_id) is None:
            return
        ids = self._chain(fact_id)
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"select text, key, episode_id from facts where id in ({marks})", ids).fetchall()
        vecs = [self._embed(r["text"]) for r in rows]
        now = self.clock.now()
        with self.db:
            self.db.execute(f"delete from facts where id in ({marks})", ids)
            for r, vec in zip(rows, vecs):
                self.db.execute(
                    "insert into forgotten(description, key, embedding, at) values(?,?,?,?)",
                    (r["key"] or r["text"][:60], r["key"], vec.tobytes(), now))
            if delete_episodes:
                doomed = {r["episode_id"] for r in rows if r["episode_id"] is not None}
                for ep in self.db.execute("select id, embedding from episodes").fetchall():
                    if any(self._max_sim(v, [ep]) >= EPISODE_FORGET_SIM for v in vecs):
                        doomed.add(ep["id"])
                self.db.executemany("delete from episodes where id=?", [(i,) for i in doomed])

    def list_facts(self, include_superseded: bool = False) -> list[Fact]:
        where = "" if include_superseded else "where status != 'superseded'"
        return [_fact(r) for r in self.db.execute(f"select {_FACT_COLS} from facts {where} order by id")]

    def history(self, fact_id: int) -> list[Fact]:
        out, seen, cur = [], set(), self.get_fact(fact_id)
        while cur is not None and cur.id not in seen:
            out.append(cur)
            seen.add(cur.id)
            cur = self.get_fact(cur.supersedes_id) if cur.supersedes_id is not None else None
        return out

    # --- episodes and threads ---

    def add_episode(self, started_at: float, ended_at: float, summary: str, user_mood: str | None = None,
                    outcome: str | None = None) -> Episode:
        with self.db:
            cur = self.db.execute(
                "insert into episodes(started_at, ended_at, summary, user_mood, outcome, embedding) values(?,?,?,?,?,?)",
                (started_at, ended_at, summary, user_mood, outcome, self._embed(summary).tobytes()))
        return _episode(self.db.execute(f"select {_EPISODE_COLS} from episodes where id=?", (cur.lastrowid,)).fetchone())

    def recent_episodes(self, n: int = 3) -> list[Episode]:
        rows = self.db.execute(f"select {_EPISODE_COLS} from episodes order by ended_at desc, id desc limit ?", (n,))
        return [_episode(r) for r in rows]

    def add_thread(self, text: str, due_at: float, episode_id: int | None = None) -> Thread:
        with self.db:
            cur = self.db.execute(
                "insert into threads(text, due_at, status, created_at, episode_id) values(?,?,?,?,?)",
                (text, due_at, "open", self.clock.now(), episode_id))
        return _thread(self.db.execute(f"select {_THREAD_COLS} from threads where id=?", (cur.lastrowid,)).fetchone())

    def due_threads(self) -> list[Thread]:
        rows = self.db.execute(
            f"select {_THREAD_COLS} from threads where status='open' and due_at<=? order by due_at desc, id desc",
            (self.clock.now(),))
        return [_thread(r) for r in rows]

    def mark_thread(self, thread_id: int, status: str | None = None, raised_at: float | None = None) -> None:
        with self.db:
            self.db.execute("update threads set status=coalesce(?, status), last_raised_at=coalesce(?, last_raised_at)"
                            " where id=?", (status, raised_at, thread_id))

    # --- transcripts ---

    def append_transcript(self, role: str, text: str) -> int:
        with self.db:
            cur = self.db.execute("insert into transcripts(at, role, text) values(?,?,?)",
                                  (self.clock.now(), role, text))
        return cur.lastrowid

    def unconsolidated(self) -> list[dict]:
        rows = self.db.execute("select id, at, role, text from transcripts where consolidated=0 order by id")
        return [dict(r) for r in rows]

    def mark_consolidated(self, ids: list[int]) -> None:
        with self.db:
            self.db.executemany("update transcripts set consolidated=1 where id=?", [(i,) for i in ids])

    def purge_transcripts(self, max_age_s: float = 30 * DAY) -> int:
        with self.db:
            return self.db.execute("delete from transcripts where at < ?", (self.clock.now() - max_age_s,)).rowcount

    @staticmethod
    def _day(at: float) -> str:
        return datetime.fromtimestamp(at).strftime("%Y-%m-%d")

    def transcript_days(self) -> list[str]:
        return sorted({self._day(r["at"]) for r in self.db.execute("select at from transcripts")})

    def delete_transcript_day(self, day: str) -> int:
        ids = [(r["id"],) for r in self.db.execute("select id, at from transcripts") if self._day(r["at"]) == day]
        with self.db:
            self.db.executemany("delete from transcripts where id=?", ids)
        return len(ids)

    # --- retrieval ---

    @staticmethod
    def _score(sim: float, age_s: float, quality: float) -> float:
        return 0.6 * sim + 0.25 * math.exp(-max(age_s, 0.0) / DAY / 30) + 0.15 * quality

    def retrieve(self, query: str, k_facts: int = 5, k_episodes: int = 2) -> Retrieved:
        vec, now = self._embed(query), self.clock.now()
        facts, episodes = [], []
        for r in self.db.execute(f"select {_FACT_COLS}, embedding from facts where status in {LIVE}"):
            sim = self._max_sim(vec, [r])
            if sim >= self.min_sim:
                facts.append((self._score(sim, now - r["valid_from"], QUALITY[r["source"]]), _fact(r)))
        for r in self.db.execute(f"select {_EPISODE_COLS}, embedding from episodes"):
            sim = self._max_sim(vec, [r])
            if sim >= self.min_sim:
                episodes.append((self._score(sim, now - r["ended_at"], QUALITY["inferred"]), _episode(r)))
        facts.sort(key=lambda p: p[0], reverse=True)
        episodes.sort(key=lambda p: p[0], reverse=True)
        return Retrieved([f for _, f in facts[:k_facts]], [e for _, e in episodes[:k_episodes]], self.due_threads())

    # --- legacy import ---

    def import_legacy_json(self, path: str | Path) -> Episode | None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        summary = data.get("summary") if isinstance(data, dict) else None
        if not isinstance(summary, str) or not summary.strip():
            return None
        now = self.clock.now()
        return self.add_episode(now, now, summary.strip(), outcome="imported")
