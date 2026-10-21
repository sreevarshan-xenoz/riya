"""Riya's long-term memory: facts, episodes, follow-up threads and transcripts, with hybrid retrieval."""
import functools
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from riya.core.embed import Embedder, content_tokens

RANK = {"user_said": 2, "looked_up": 1, "inferred": 0}
QUALITY = {"user_said": 1.0, "looked_up": 0.7, "inferred": 0.5}
LIVE = "('active','conflict')"
EPISODE_FORGET_SIM = 0.6
DAY = 86400.0
ABOUT_ME_FACTS = 8
PERSONAL_EXTRA = 3  # a personal request's fill: at most this many facts beyond the eligible ones (v0.11 §4.2)
MAX_EPISODES = 3
RECENT_EPISODE_S = 7 * DAY  # the second-latest episode is always shown while it's this fresh
STALE_THREAD_S = 7 * DAY  # an open thread this far past due is dropped by the daily purge
ABOUT_ME_RE = re.compile(
    r"\bwhat (?:do|did) you (?:remember|know) about me\b|\beverything you (?:remember|know) about me\b"
    r"|\btell me (?:everything |all |what )?(?:that )?you (?:remember|know)"
    r"(?: about me\b|(?:,?\s+riya)?\s*(?:[.?!]|$))"
    r"|\bwhat have i (?:told|said to) you\b", re.IGNORECASE)


def about_me(query: str) -> bool:
    """What do you remember about me? and the like: answered from her best facts, not by similarity. "Tell me what
    you remember" counts only at the end of the sentence (", Riya" may follow) or followed by "about me", not
    "...about my sister's X"."""
    return ABOUT_ME_RE.search(query) is not None


NAMING = frozenset({"call", "calls", "called", "name", "names", "named"})  # never shared content (v0.11 §4.3)
_HEDGE = (r"as far as you (?:know|can tell)|going by what you(?:'ve| have)? (?:know|seen|heard)"
          r"|from what you(?:'ve| have)? (?:know|seen|heard|can tell)|as you see it")
PERSONAL_RE = re.compile("|".join((
    # applied: an action for them, or "me" as the object
    r"\b(?:pick|choose|suggest|recomm?end|plan|make(?! it\b)|find|book|sort out|cook|order) (?:\w+ ){0,3}?for me\b",
    r"\b(?:pick|choose|book|sort|plan|find|cook|order)(?: out)? me (?:out )?(?:a|an|some|something|anything|one)\b",
    r"\bwhich (?:\w+ ){1,3}(?:would|should|do) you (?:book|pick|choose|order|recomm?end) (?:for )?me\b",
    r"\bhelp me (?:pick|choose|decide|plan|find)\b",
    r"(?:^|[,.!?;]\s*|\b(?:please|pls|can you|could you|would you|will you|just) )(?:suggest|recomm?end|recomend)\b",
    r"\b(?:what|which \w+) (?:would|do|can|could) you (?:suggest|recomm?end|recomend)\b",
    r"\bany (?:picks|suggestions|recommendations|recs)\b",
    r"\bgive me (?:a |an |some )?(?:\w+ ){0,2}(?:routine|plan|idea|ideas|recipe|playlist|suggestions?)\b",
    r"\bneed (?:some |a )(?:music|tunes|songs|playlist|film|movie|book|show)\b",
    r"\byou know (?:what i (?:like|enjoy|love)|how i am|my tastes?)\b",
    r"\b(?:what|how|when|where|which \w+) (?:should|shall) i (?:eat|get|order|watch|listen to|wear|cook|drink|have"
    r"|make|read|play|buy|book|travel|go|start|do)\b",
    r"\b(?:what|which \w+) (?:should|shall|do) i (?:ask (?:\w+ ){0,2}for|tell (?:\w+ ){0,2}to get me)\b",
    r"\bshould i (?:buy|book)\b",
    r"\b(?:no (?:clue|idea)|thoughts on|any ideas? (?:on|for)) what to (?:eat|get|order|watch|listen to|wear|cook"
    r"|drink|make|read|buy)\b",
    r"\b(?:something|anything|that|stuff|things?) (?:i'd|i would|i'll|i will) (?:actually |really )?"
    r"(?:like|enjoy|love)\b",
    r"\b(?:is|are|would|does) (?:\w+ ){0,4}(?:very|so) me\b",
    r"\bwhat (?:can't|cannot|can not|shouldn't) i (?:eat|drink)\b",
    # topical: what she knows about a part of their life
    r"\bwhat (?:do|did) you (?:actually |really )?(?:know|remember) about (?:my|where i|how i|what i)\b",
    r"\b(?:anything|what) (?:else )?you (?:know|remember) about (?:my\b|(?:the )?(?:\w+ ){0,4}(?:to|of|around) me\b)",
    r"\b(?:you(?:'ve|ve| have)?|have you) (?:\w+ )?(?:noticed|figured out|picked up|learned|learnt|gathered"
    r"|worked out) about (?:me|my|them|how i|what i|where i)\b",
    r"\bdescribe (?:me|my)\b",
    r"\bwhat (?:kind|sort|type) of (?:person|guy|girl|man|woman) am i\b",
    r"\b(?:what|which) (?:sort|kind|type)s? of (?:\w+ ){1,2}i (?:go for|tend to)\b",
    r"\b(?:your|ur) (?:read|take|impression|sense) (?:on|of) (?:me|my)\b",
    r"\bhow (?:do|does) my\b(?: \S+){1,8}? (?:usually|normally|typically) go\s*(?:[?.!,~]|$)",
    rf"\b(?:{_HEDGE})\b",
)))
_APOS = "['‘ʼ]?"  # the exclusions also catch phone typing: "whats", "dont", "lets", and ‘ or ʼ
NOT_PERSONAL_RE = re.compile("|".join((
    # direct attribute questions
    rf"\b(?:what|who|when|where)(?:{_APOS}s| is| was) my\b", r"\bhow old (?:is|was|am)\b",
    r"\bwhat (?:did|do) i (?:name|call)\b",
    # shared history and remember-that turns
    r"\bwhat did i (?:tell|say|mention)\b", r"\bdid i (?:ever )?(?:tell|say|mention)\b", r"\bremember (?:when|that)\b",
    r"\bkeep in mind\b", rf"\bdon{_APOS}t forget\b",
    # leading questions
    r"\b(?:do|did) you (?:happen to |even )?(?:know|remember) my\b",
    r"\b(?:do|did) you (?:happen to |even )?(?:know|remember) (?:what|how|where|which|who) i\b",
    r"\byou (?:already )?(?:know|remember) my\b(?! tastes?\b)",
    r"\bnow that you know\b", r"\bsince you (?:already )?know\b", r",? right\?",
    # "we", and identity or AI questions
    r"\b(?:should|shall|can|could) we\b", rf"\blet{_APOS}s\b", r"\byou and i\b", r"\bthe two of us\b", r"\bjust us\b",
    r"\b(?:ai|a\.i\.|bot|chatbot|robot|language model|llm|chatgpt|programmed|human|real person|sentient|conscious)\b",
)))


def personal_request(query: str) -> bool:
    """Asks her to apply what she knows about them ("pick a snack for me, you know what i like") or about a part of
    their life ("what have you noticed about my nights"): v0.11 §4.1. Never a direct attribute, shared-history,
    leading, "we", identity or remember-that question, and never an about-me one (that path wins). Case and curly
    apostrophes don't matter."""
    q = query.replace("’", "'").lower()
    if about_me(query) or about_me(q):
        return False
    return PERSONAL_RE.search(q) is not None and NOT_PERSONAL_RE.search(q) is None


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
class Glance:
    """A note from one look at their screen (the screenshot itself is never kept)."""
    id: int
    at: float
    app: str
    title: str
    activity: str
    details: str
    notable: str | None


@dataclass
class Retrieved:
    facts: list[Fact]
    episodes: list[Episode]
    threads: list[Thread]
    filled: tuple[int, ...] = ()  # the ids a personal request's fill added, in rank order (they end `facts`)


_FACT_COLS = "id, text, key, source, confidence, valid_from, valid_until, status, supersedes_id"
_EPISODE_COLS = "id, started_at, ended_at, summary, user_mood, outcome"
_THREAD_COLS = "id, text, due_at, status, last_raised_at"
_GLANCE_COLS = "id, at, app, title, activity, details, notable"
GLANCE_KEEP_S = 7 * DAY


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(**{k: row[k] for k in _FACT_COLS.split(", ")})


def _episode(row: sqlite3.Row) -> Episode:
    return Episode(**{k: row[k] for k in _EPISODE_COLS.split(", ")})


def _thread(row: sqlite3.Row) -> Thread:
    return Thread(**{k: row[k] for k in _THREAD_COLS.split(", ")})


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _locked(method):
    """Every public method holds the memory's RLock, so the socket, tick and reply threads never interleave."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class Memory:
    def __init__(self, db: sqlite3.Connection, embedder: Embedder, clock,
                 dup_sim: float = 0.92, forgotten_sim: float = 0.85, min_sim: float | None = None):
        """min_sim, when given, overrides the embedder's strong/weak retrieval thresholds."""
        self.db, self.embedder, self.clock = db, embedder, clock
        self.dup_sim, self.forgotten_sim, self.min_sim = dup_sim, forgotten_sim, min_sim
        self.lock = threading.RLock()

    def eligible(self, sim: float, query_tokens: set[str], text: str) -> bool:
        """Strongly similar, or weakly similar and sharing at least one content word with the query. Naming words
        ("called", "named") aren't shared content: "...dog called?" never reaches "...friend is called X"."""
        if self.min_sim is not None:
            return sim >= self.min_sim
        strong = getattr(self.embedder, "strong_sim", 0.2)
        weak = getattr(self.embedder, "weak_sim", strong)
        return sim >= strong or (sim >= weak and not (query_tokens - NAMING).isdisjoint(content_tokens(text) - NAMING))

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

    @_locked
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
        if any(r["text"] == text or self._duplicate(vec, text, r) for r in live):
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

    def _duplicate(self, vec: np.ndarray, text: str, row) -> bool:
        """Close in meaning and made of the same words: "exam on Thursday" vs "...Friday" is a correction, kept."""
        return self._max_sim(vec, [row]) >= self.dup_sim and content_tokens(text) == content_tokens(row["text"])

    @_locked
    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.db.execute(f"select {_FACT_COLS} from facts where id=?", (fact_id,)).fetchone()
        return _fact(row) if row else None

    @_locked
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

    @_locked
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

    @_locked
    def list_facts(self, include_superseded: bool = False) -> list[Fact]:
        where = "" if include_superseded else "where status != 'superseded'"
        return [_fact(r) for r in self.db.execute(f"select {_FACT_COLS} from facts {where} order by id")]

    @_locked
    def history(self, fact_id: int) -> list[Fact]:
        out, seen, cur = [], set(), self.get_fact(fact_id)
        while cur is not None and cur.id not in seen:
            out.append(cur)
            seen.add(cur.id)
            cur = self.get_fact(cur.supersedes_id) if cur.supersedes_id is not None else None
        return out

    # --- episodes and threads ---

    @_locked
    def add_episode(self, started_at: float, ended_at: float, summary: str, user_mood: str | None = None,
                    outcome: str | None = None) -> Episode:
        with self.db:
            cur = self.db.execute(
                "insert into episodes(started_at, ended_at, summary, user_mood, outcome, embedding) values(?,?,?,?,?,?)",
                (started_at, ended_at, summary, user_mood, outcome, self._embed(summary).tobytes()))
        return _episode(self.db.execute(f"select {_EPISODE_COLS} from episodes where id=?", (cur.lastrowid,)).fetchone())

    @_locked
    def recent_episodes(self, n: int = 3) -> list[Episode]:
        rows = self.db.execute(f"select {_EPISODE_COLS} from episodes order by ended_at desc, id desc limit ?", (n,))
        return [_episode(r) for r in rows]

    @_locked
    def list_episodes(self) -> list[Episode]:
        return [_episode(r) for r in self.db.execute(
            f"select {_EPISODE_COLS} from episodes order by ended_at desc, id desc")]

    @_locked
    def delete_episode(self, episode_id: int) -> None:
        with self.db:
            self.db.execute("delete from episodes where id=?", (episode_id,))

    @_locked
    def add_thread(self, text: str, due_at: float, episode_id: int | None = None) -> Thread:
        with self.db:
            cur = self.db.execute(
                "insert into threads(text, due_at, status, created_at, episode_id) values(?,?,?,?,?)",
                (text, due_at, "open", self.clock.now(), episode_id))
        return _thread(self.db.execute(f"select {_THREAD_COLS} from threads where id=?", (cur.lastrowid,)).fetchone())

    @_locked
    def open_threads(self) -> list[Thread]:
        return [_thread(r) for r in self.db.execute(
            f"select {_THREAD_COLS} from threads where status='open' order by due_at, id")]

    @_locked
    def due_threads(self) -> list[Thread]:
        rows = self.db.execute(
            f"select {_THREAD_COLS} from threads where status='open' and due_at<=? order by due_at desc, id desc",
            (self.clock.now(),))
        return [_thread(r) for r in rows]

    def _relevant_threads(self, vec: np.ndarray, qtokens: set[str], k: int) -> list[Thread]:
        due = self.due_threads()
        if not due:
            return []
        vecs = self.embedder.embed([t.text for t in due])
        sims = [float(np.dot(vec, v)) if v.shape == vec.shape else 0.0 for v in vecs]
        ranked = sorted(zip(sims, due), key=lambda p: p[0], reverse=True)
        return [t for sim, t in ranked if self.eligible(sim, qtokens, t.text)][:k]

    @_locked
    def resolve_stale_threads(self, max_overdue_s: float = STALE_THREAD_S) -> int:
        with self.db:
            return self.db.execute("update threads set status='resolved' where status='open' and due_at < ?",
                                   (self.clock.now() - max_overdue_s,)).rowcount

    @_locked
    def mark_thread(self, thread_id: int, status: str | None = None, raised_at: float | None = None) -> None:
        with self.db:
            self.db.execute("update threads set status=coalesce(?, status), last_raised_at=coalesce(?, last_raised_at)"
                            " where id=?", (status, raised_at, thread_id))

    # --- transcripts ---

    @_locked
    def append_transcript(self, role: str, text: str, consolidate: bool = True) -> int:
        """consolidate=False keeps the row out of consolidation (her agent reports, game turns): it's stored as
        already consolidated, so it shows in the transcript but never becomes a fact, an episode or a thread."""
        with self.db:
            cur = self.db.execute("insert into transcripts(at, role, text, consolidated) values(?,?,?,?)",
                                  (self.clock.now(), role, text, 0 if consolidate else 1))
        return cur.lastrowid

    @_locked
    def last_transcript_at(self) -> float | None:
        row = self.db.execute("select at from transcripts order by id desc limit 1").fetchone()
        return row["at"] if row else None

    @_locked
    def unconsolidated(self) -> list[dict]:
        rows = self.db.execute("select id, at, role, text from transcripts where consolidated=0 order by id")
        return [dict(r) for r in rows]

    @_locked
    def mark_consolidated(self, ids: list[int]) -> None:
        with self.db:
            self.db.executemany("update transcripts set consolidated=1 where id=?", [(i,) for i in ids])

    @_locked
    def purge_transcripts(self, max_age_s: float = 30 * DAY) -> int:
        with self.db:
            return self.db.execute("delete from transcripts where at < ?", (self.clock.now() - max_age_s,)).rowcount

    @staticmethod
    def _day(at: float) -> str:
        return datetime.fromtimestamp(at).strftime("%Y-%m-%d")

    @_locked
    def transcript_days(self) -> list[str]:
        return sorted({self._day(r["at"]) for r in self.db.execute("select at from transcripts")})

    @_locked
    def delete_transcript_day(self, day: str) -> int:
        ids = [(r["id"],) for r in self.db.execute("select id, at from transcripts") if self._day(r["at"]) == day]
        with self.db:
            self.db.executemany("delete from transcripts where id=?", ids)
        return len(ids)

    # --- glances ---

    @_locked
    def add_glance(self, app: str, title: str, activity: str, details: str, notable: str | None) -> Glance:
        """Keeps a glance note, dropping notes older than GLANCE_KEEP_S on the way."""
        now = self.clock.now()
        with self.db:
            self.db.execute("delete from glances where at < ?", (now - GLANCE_KEEP_S,))
            cur = self.db.execute("insert into glances(at, app, title, activity, details, notable) values(?,?,?,?,?,?)",
                                  (now, app, title, activity, details, notable))
        return Glance(cur.lastrowid, now, app, title, activity, details, notable)

    def _glances(self, where: str = "", args: tuple = (), limit: int = -1) -> list[Glance]:
        rows = self.db.execute(f"select {_GLANCE_COLS} from glances {where} order by at desc, id desc limit ?",
                               (*args, limit))
        return [Glance(**dict(r)) for r in rows]

    @_locked
    def recent_glances(self, n: int = 3) -> list[Glance]:
        """The newest n notes, newest first."""
        return self._glances(limit=n)

    @_locked
    def list_glances(self) -> list[Glance]:
        return self._glances()

    @_locked
    def glances_between(self, start: float, end: float) -> list[Glance]:
        return self._glances("where at >= ? and at <= ?", (start, end))

    @_locked
    def delete_glance(self, glance_id: int) -> None:
        with self.db:
            self.db.execute("delete from glances where id=?", (glance_id,))

    @_locked
    def purge_glances(self, max_age_s: float = GLANCE_KEEP_S) -> int:
        with self.db:
            return self.db.execute("delete from glances where at < ?", (self.clock.now() - max_age_s,)).rowcount

    # --- retrieval ---

    @staticmethod
    def _score(sim: float, age_s: float, quality: float) -> float:
        return 0.6 * sim + 0.25 * math.exp(-max(age_s, 0.0) / DAY / 30) + 0.15 * quality

    @_locked
    def retrieve(self, query: str, k_facts: int = 5, k_episodes: int = 2, k_threads: int = 2,
                 personal: bool = True) -> Retrieved:
        """Facts and episodes that pass the hybrid gate, ranked; plus the latest episode, and the one before it while
        it's under a week old, so "what did we talk about last time?" has an answer. An about-me question gets the
        facts that pass the gate first, then is filled to 8 with her best facts by trust and recency. Due threads
        pass the same gate as facts.

        A personal request ("pick a snack for me, you know what i like") gets the eligible facts first, then up to
        PERSONAL_EXTRA more live facts whose cosine is at least the embedder's personal_sim, ranked by score, never
        more than k_facts in all; `filled` names them. It ignores min_sim, and an embedder without personal_sim
        (HashEmbedder) gets no fill. personal=False turns it off: the engine does that on the online route, so an
        online-routed request ("plan my week for me" goes online on "plan") gets no fill, a known limit (v0.11
        §4.7.9). Every other query retrieves exactly what it did before."""
        vec, now, qtokens = self._embed(query), self.clock.now(), content_tokens(query)
        mine = about_me(query)
        floor = getattr(self.embedder, "personal_sim", None)
        fill = personal and not mine and floor is not None and personal_request(query)
        facts, rest, extra, episodes = [], [], [], []
        for r in self.db.execute(f"select {_FACT_COLS}, embedding from facts where status in {LIVE}"):
            sim = self._max_sim(vec, [r])
            if self.eligible(sim, qtokens, r["text"]):
                facts.append((self._score(sim, now - r["valid_from"], QUALITY[r["source"]]), _fact(r)))
            elif mine:
                rest.append((self._score(0.0, now - r["valid_from"], QUALITY[r["source"]]), _fact(r)))
            elif fill and sim >= floor:
                extra.append((self._score(sim, now - r["valid_from"], QUALITY[r["source"]]), _fact(r)))
        for r in self.db.execute(f"select {_EPISODE_COLS}, embedding from episodes"):
            sim = self._max_sim(vec, [r])
            if self.eligible(sim, qtokens, r["summary"]):
                episodes.append((self._score(sim, now - r["ended_at"], QUALITY["inferred"]), _episode(r)))
        facts.sort(key=lambda p: p[0], reverse=True)
        facts += sorted(rest, key=lambda p: p[0], reverse=True)  # empty unless about-me
        episodes.sort(key=lambda p: p[0], reverse=True)
        latest = self.recent_episodes(2)
        picked = latest[:1] + [e for e in latest[1:] if now - e.ended_at <= RECENT_EPISODE_S] + [
            e for _, e in episodes[:k_episodes]]
        unique = list({e.id: e for e in picked}.values())  # in order of first appearance
        chosen = sorted(unique[:MAX_EPISODES], key=lambda e: (e.ended_at, e.id), reverse=True)
        kept = [f for _, f in facts[:ABOUT_ME_FACTS if mine else k_facts]]
        extra.sort(key=lambda p: p[0], reverse=True)  # empty unless a personal request
        added = [f for _, f in extra[:min(PERSONAL_EXTRA, k_facts - len(kept))]]
        return Retrieved(kept + added, chosen, self._relevant_threads(vec, qtokens, k_threads),
                         tuple(f.id for f in added))

    # --- legacy import ---

    @_locked
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
