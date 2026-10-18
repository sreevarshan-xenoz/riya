"""Riya's long-term memory: facts with provenance, supersession and forgetting."""
import sqlite3
from dataclasses import dataclass

import numpy as np

from riya.core.embed import Embedder

RANK = {"user_said": 2, "looked_up": 1, "inferred": 0}
LIVE = "('active','conflict')"


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


_FACT_COLS = "id, text, key, source, confidence, valid_from, valid_until, status, supersedes_id"


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(**{k: row[k] for k in row.keys()})


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
                 episode_id: int | None = None) -> Fact | None:
        if source not in RANK:
            raise ValueError(f"unknown source {source!r}")
        vec = self._embed(text)
        if source != "user_said":  # an explicit user statement always overrides a forgotten marker
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
        now = self.clock.now()
        with self.db:
            self.db.execute(f"delete from facts where id in ({marks})", ids)
            for r in rows:
                if delete_episodes and r["episode_id"] is not None:
                    self.db.execute("delete from episodes where id=?", (r["episode_id"],))
                self.db.execute(
                    "insert into forgotten(description, key, embedding, at) values(?,?,?,?)",
                    (r["key"] or r["text"][:60], r["key"], self._embed(r["text"]).tobytes(), now))

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
