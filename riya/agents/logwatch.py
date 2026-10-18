"""Colab log watcher (spec §4.3): tails the user's synced run logs and reports running/done/fail/lost/stalled.

Each log file is one run named "<entry name>:<parent folder>/<file stem>". Only appended bytes are read; offsets and run
state live in KV under `agents.logwatch.<path>` so a restart carries on without re-reporting. At most the 200 most
recently modified files per entry are read each poll, and keys of files no longer matched are pruned once a day.
`poll()` never raises.
"""
import glob
import logging
import os
import re

log = logging.getLogger("riya.logwatch")

PREFIX = "agents.logwatch."
POLL_KEY = PREFIX + "#poll"
PRUNE_KEY = PREFIX + "#pruned"
PRUNE_EVERY_S = 24 * 3600  # keys of files no glob matches anymore are dropped this often
MAX_FILES = 200          # per entry per poll: only the most recently modified matching files are read
STALL_S = 30 * 60        # a running log with no growth this long is stalled
MAX_READ = 1 << 20       # bytes read per file per poll
LINE_MAX = 120
FAIL_QUIET_S = 5 * 60    # after fail/lost, growth this soon is the rest of the trace, not a new run
SLEEP_GAP_S = 5 * 60     # a longer gap between polls means the machine slept: no stall verdict on that poll
TAIL_PROBE = 64 * 1024   # how far back from EOF to look for the last newline of a pre-existing file
DEFAULT_ENTRIES = [{
    "name": "colab",
    "glob": "G:/My Drive/jarvis/runs/**/*.out",
    "done_re": "STAGE0_TRAIN_DONE|ROLE_DONE|BENCH_DONE|GEN_DONE|TRAIN_DONE|PASS1_DONE",
    "fail_re": "FAILED|Traceback",
    "lost_re": "LOST",
}]


def _root(pattern: str) -> str:
    """The literal directory a glob starts from (so a missing root can be told from an empty one)."""
    cut = min((i for i in (pattern.find(c) for c in "*?[") if i >= 0), default=len(pattern))
    head = pattern[:cut]
    return head if head.endswith(("/", "\\")) else os.path.dirname(head)


class LogWatch:
    def __init__(self, entries, kv, clock):
        self.entries, self.kv, self.clock = list(entries), kv, clock
        self._rebase = False
        self._matched: set[str] = set()  # every path matched this poll
        self._unsure: list[str] | None = []  # roots that were missing or failed this poll; None: can't tell, keep all

    def poll(self) -> list[tuple[str, str, str]]:
        out, now = [], self.clock.now()
        try:
            prev = self.kv.get(POLL_KEY)
            self._rebase = isinstance(prev, (int, float)) and now - prev > SLEEP_GAP_S
        except Exception:
            self._rebase = False
        self._matched, self._unsure = set(), []
        for entry in self.entries:
            try:
                self._poll_entry(entry, out)
            except Exception:
                self._doubt(entry.get("glob") if isinstance(entry, dict) else None)
                log.exception("logwatch entry %r failed", entry.get("name") if isinstance(entry, dict) else entry)
        try:
            self.kv.set(POLL_KEY, now)
        except Exception:
            log.exception("logwatch could not save its poll time")
        try:
            self._prune(now)
        except Exception:
            log.exception("logwatch could not prune its keys")
        return out

    def _doubt(self, pattern) -> None:
        """This poll can't tell which files under pattern's root still exist (root missing, or the entry failed)."""
        root = _root(pattern).replace("\\", "/").rstrip("/") if isinstance(pattern, str) else ""
        if not root:
            self._unsure = None
        elif self._unsure is not None:
            self._unsure.append(root + "/")

    def _prune(self, now: float) -> None:
        """Once a day, drop the state of files no entry matches anymore. Files under a root that was missing (Drive
        not mounted) or whose entry failed are kept: there "not matched" doesn't mean gone."""
        last = self.kv.get(PRUNE_KEY)
        if self._unsure is None or (isinstance(last, (int, float)) and 0 <= now - last < PRUNE_EVERY_S):
            return
        for key in self.kv.keys_with(PREFIX):
            path = key[len(PREFIX):]
            if not path.startswith("#") and path not in self._matched and not path.startswith(tuple(self._unsure)):
                self.kv.delete(key)
        self.kv.set(PRUNE_KEY, now)

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("-inf")

    def _poll_entry(self, entry, out) -> None:
        """Appends to `out` as it goes, so an error part-way never discards events already found."""
        name, pattern = entry["name"], entry["glob"]
        res = {k: re.compile(entry[k]) if entry.get(k) else None for k in ("lost_re", "fail_re", "done_re")}
        seen_key = f"agents.logwatch.#seen.{name}"
        seen_at = self.kv.get(seen_key)     # when this entry was first polled: older files are history
        paths = glob.glob(pattern, recursive=True)
        self._matched.update(p.replace("\\", "/") for p in paths)
        if not os.path.isdir(_root(pattern)):
            self._doubt(pattern)
        for path in sorted(sorted(paths, key=self._mtime, reverse=True)[:MAX_FILES]):
            try:
                path = path.replace("\\", "/")
                parent = os.path.basename(os.path.dirname(path))
                stem = os.path.splitext(os.path.basename(path))[0]
                out += self._poll_file(f"{name}:{parent}/{stem}" if parent else f"{name}:{stem}", path, seen_at, res)
            except Exception:
                log.exception("logwatch file %s failed", path)
        if seen_at is None and os.path.isdir(_root(pattern)):
            self.kv.set(seen_key, self.clock.now())  # a missing root (Drive not mounted) is not "polled"

    @staticmethod
    def _baseline(path, size) -> int:
        """Where a pre-existing file starts: just after its last newline, so a half-written tail is read whole later."""
        n = min(size, TAIL_PROBE)
        with open(path, "rb") as f:
            f.seek(size - n)
            i = f.read(n).rfind(b"\n")
        return size - n + i + 1 if i >= 0 else (0 if size <= TAIL_PROBE else size)

    def _poll_file(self, run, path, seen_at, res) -> list[tuple[str, str, str]]:
        key, now = f"agents.logwatch.{path}", self.clock.now()
        size = os.path.getsize(path)
        old = self.kv.get(key)
        if old:
            st = dict(old)
        else:
            # Pre-existing (before the entry's first poll, even if that poll failed on this file) = history: baseline.
            # Anything modified after that first poll is a new run and is read from the start.
            history = seen_at is None or os.path.getmtime(path) <= seen_at
            st = {"off": self._baseline(path, size) if history else 0, "size": size if history else 0,
                  "running": False, "ended": False, "stalled": False, "last": now}
        out, grew = [], False
        # Size-only detection: a rewrite that ends up at least as large as before is not noticed.
        if size < st["size"] or size < st["off"]:   # rewritten or truncated: jump to the new EOF, report nothing
            st.update(off=size, size=size, running=False, ended=False, stalled=False, last=now)
        else:
            hits = {}
            if size > st["off"]:
                with open(path, "rb") as f:
                    f.seek(st["off"])
                    data = f.read(MAX_READ)
                end = data.rfind(b"\n") + 1
                if not end and len(data) >= MAX_READ:
                    end = len(data)                    # one endless line: don't stall on it forever
                if end:
                    st["off"] += end
                    grew = True
                    for line in data[:end].decode("utf-8", errors="replace").split("\n"):
                        line = line.rstrip("\r").rsplit("\r", 1)[-1]   # CRLF, and progress bars: the last redraw counts
                        for kind in ("lost", "fail", "done"):
                            rx = res[f"{kind}_re"]
                            if rx and rx.search(line):
                                hits[kind] = line.strip()[:LINE_MAX]
                                break
            grew = grew or size > st["size"]
            st["size"] = size
            quiet = (st["ended"] and st.get("end_kind") in ("fail", "lost")
                     and now - st.get("end_at", float("-inf")) < FAIL_QUIET_S)
            if grew and not quiet:
                back = st["stalled"] or st["ended"]    # growth after stalled/done/fail/lost: running again
                st["last"], st["stalled"] = now, False
                if not st["running"] or back:
                    st["running"], st["ended"] = True, False
                    out.append((run, "running", ""))
            for kind in ("lost", "fail", "done"):
                if kind in hits:
                    out.append((run, kind, hits[kind]))
                    if kind != "done" or not st["ended"] or st.get("end_at") != now:   # fail/lost outrank done
                        st["end_kind"], st["end_at"] = kind, now
                    st["ended"] = True
            if st["ended"]:
                st["running"] = st["stalled"] = False
            elif st["running"] and not st["stalled"] and not grew:
                if self._rebase:
                    st["last"] = now                   # we slept through the gap: judge stalls from now on
                elif now - st["last"] >= STALL_S:
                    st["stalled"] = True
                    out.append((run, "stalled", ""))
        if st != old:
            self.kv.set(key, st)
        return out
