"""Colab log watcher (spec §4.3): tails the user's synced run logs and reports running/done/fail/lost/stalled.

Each log file is one run named "<entry name>:<file stem>". Only appended bytes are read; offsets and run state live
in KV under `agents.logwatch.<path>` so a restart carries on without re-reporting. `poll()` never raises.
"""
import glob
import logging
import os
import re

log = logging.getLogger("riya.logwatch")

STALL_S = 30 * 60        # a running log with no growth this long is stalled
MAX_READ = 1 << 20       # bytes read per file per poll
LINE_MAX = 120
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

    def poll(self) -> list[tuple[str, str, str]]:
        out = []
        for entry in self.entries:
            try:
                out += self._poll_entry(entry)
            except Exception:
                log.exception("logwatch entry %r failed", entry.get("name") if isinstance(entry, dict) else entry)
        return out

    def _poll_entry(self, entry) -> list[tuple[str, str, str]]:
        name, pattern = entry["name"], entry["glob"]
        res = {k: re.compile(entry[k]) if entry.get(k) else None for k in ("lost_re", "fail_re", "done_re")}
        seen_key = f"agents.logwatch.#seen.{name}"
        seen = bool(self.kv.get(seen_key))
        out = []
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                out += self._poll_file(f"{name}:{os.path.splitext(os.path.basename(path))[0]}",
                                       path.replace("\\", "/"), seen, res)
            except Exception:
                log.exception("logwatch file %s failed", path)
        if not seen and os.path.isdir(_root(pattern)):
            self.kv.set(seen_key, True)  # a missing root (Drive not mounted) is not "polled": old logs never replay
        return out

    def _poll_file(self, run, path, seen, res) -> list[tuple[str, str, str]]:
        key, now = f"agents.logwatch.{path}", self.clock.now()
        size = os.path.getsize(path)
        old = self.kv.get(key)
        st = dict(old) if old else {"off": 0 if seen else size, "size": 0 if seen else size, "running": False,
                                    "ended": False, "stalled": False, "last": now}
        out, grew = [], False
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
                        for kind in ("lost", "fail", "done"):
                            rx = res[f"{kind}_re"]
                            if rx and rx.search(line):
                                hits[kind] = line.strip()[:LINE_MAX]
                                break
            grew = grew or size > st["size"]
            st["size"] = size
            if grew:
                revived = st["stalled"]
                st["last"], st["stalled"] = now, False
                if (not st["running"] or revived) and not st["ended"]:
                    st["running"] = True
                    out.append((run, "running", ""))
            for kind in ("lost", "fail", "done"):
                if kind in hits:
                    out.append((run, kind, hits[kind]))
                    st["ended"] = True
            if st["ended"]:
                st["running"] = st["stalled"] = False
            elif st["running"] and not st["stalled"] and not grew and now - st["last"] >= STALL_S:
                st["stalled"] = True
                out.append((run, "stalled", ""))
        if st != old:
            self.kv.set(key, st)
        return out
