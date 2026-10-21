"""The hold-out: no training row may share a probe with the champion gate. A user message that equals a persona probe or
shares a run of HOLDOUT_N words with one, or a system message that holds a probe fact, is a hit. Stdlib only, so the
Colab trainer can import it without riya.core (or numpy, or torch). scenarios, build_sft, filter_sft and the trainer
all draw their hold-out from here.
See docs/superpowers/specs/2026-10-21-riya-v0.10-gate-that-sees-design.md §4.1 and §4.6."""
import contextlib
import functools
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

HOLDOUT_N = 6  # a request shares no run of this many words with a persona probe
PROBES_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "persona_probes.json"


def normalize(text: str) -> str:
    """Casefold, no punctuation, single spaces: the form every hold-out comparison uses."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


@functools.cache
def _conversations(path: Path) -> tuple[dict, ...]:
    """The probe conversations, read and parsed once per path (nobody edits the file mid-run)."""
    return tuple(json.loads(path.read_text(encoding="utf-8"))["conversations"])


def probe_texts(path: Path | None = None) -> list[str]:
    """The user text of every probe turn, except the speak-first ones (they carry a stage note, not user text). The
    path is read at call time (PROBES_PATH when not given), so probes_at can swap it."""
    return [t["text"] for c in _conversations(path or PROBES_PATH) for t in c["turns"] if not t.get("speak_first")]


def probe_facts(path: Path | None = None) -> list[str]:
    """Every fact of every probe conversation."""
    return [f for c in _conversations(path or PROBES_PATH) for f in c["facts"]]


@contextlib.contextmanager
def probes_at(path: Path) -> Iterator[None]:
    """Reads the probes from path inside the block (a test's tmp file), then restores PROBES_PATH."""
    global PROBES_PATH
    before, PROBES_PATH = PROBES_PATH, Path(path)
    try:
        yield
    finally:
        PROBES_PATH = before


def _windows(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + HOLDOUT_N]) for i in range(len(words) - HOLDOUT_N + 1)}


def overlaps(text: str, probes: Iterable[str]) -> bool:
    """text equals a probe, or shares a run of HOLDOUT_N words with one (after normalize)."""
    t = normalize(text)
    tw = t.split()
    wins = _windows(tw)
    for p in probes:
        n = normalize(p)
        if t == n or (wins and wins & _windows(n.split())):
            return True
    return False


def first_hit(row: dict, texts: list[str], facts: list[str]) -> str | None:
    """The probe text that one of the row's user messages overlaps, else the probe fact that its normalised system
    message holds, else None."""
    for message in row["messages"]:
        if message["role"] == "user":
            for text in texts:
                if overlaps(message["content"], [text]):
                    return text
    system = normalize(" ".join(m["content"] for m in row["messages"] if m["role"] == "system"))
    return next((f for f in facts if (n := normalize(f)) and n in system), None)


def holdout_violations(rows: list[dict], probes: list[str], facts: list[str]) -> list[str]:
    """The ids of rows where a user message overlaps a probe (equal, or a shared 6-word window), or the system message
    holds a probe fact (both normalised)."""
    return [row["id"] for row in rows if first_hit(row, probes, facts) is not None]


def drop_holdout(rows: list[dict], path: Path | None = None) -> tuple[list[dict], list[str]]:
    """(the rows without a hit, in order; the ids dropped). Works on requests and on SFT rows: both have an id and
    messages. The probes are the file's (path, else PROBES_PATH)."""
    texts, facts = probe_texts(path), probe_facts(path)
    kept, dropped = [], []
    for row in rows:
        if first_hit(row, texts, facts) is None:
            kept.append(row)
        else:
            dropped.append(row["id"])
    return kept, dropped
