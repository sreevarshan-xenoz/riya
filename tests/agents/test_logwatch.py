import os
import time

import pytest

from riya.agents.logwatch import DEFAULT_ENTRIES, LogWatch
from riya.core.clock import FakeClock


class KV(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def set(self, key, value):
        self[key] = value


@pytest.fixture
def env(tmp_path):
    clock, kv = FakeClock(time.time() + 3600), KV()   # real mtimes of files made now are older than "now"
    d = tmp_path / "run1"
    d.mkdir()
    entry = {"name": "colab", "glob": str(tmp_path / "**" / "*.out"), "done_re": "TRAIN_DONE|ROLE_DONE",
             "fail_re": "FAILED|Traceback", "lost_re": "LOST"}
    return d, clock, kv, entry


def watch(env, **kw):
    _, clock, kv, entry = env
    return LogWatch([{**entry, **kw}], kv, clock)


def add(path, text):
    with open(path, "ab") as f:
        f.write(text.encode())


def fresh(path, text, clock):
    """A file created after the first poll: its mtime is later than the watcher's first-poll time."""
    add(path, text)
    os.utime(path, (clock.now() + 1, clock.now() + 1))


def run_for(w, clock, seconds, step=60):
    """Poll every `step` s (like the engine) for `seconds`, collecting the events."""
    out = []
    for _ in range(seconds // step):
        clock.advance(step)
        out += w.poll()
    return out


def test_default_entry():
    e = DEFAULT_ENTRIES[0]
    assert e["name"] == "colab" and e["glob"] == "G:/My Drive/jarvis/runs/**/*.out"
    assert e["fail_re"] == "FAILED|Traceback" and e["lost_re"] == "LOST"
    assert e["done_re"] == "STAGE0_TRAIN_DONE|ROLE_DONE|BENCH_DONE|GEN_DONE|TRAIN_DONE|PASS1_DONE"


def test_missing_root_returns_empty(tmp_path):
    w = LogWatch([{"name": "c", "glob": str(tmp_path / "nope" / "**" / "*.out")}], KV(), FakeClock())
    assert w.poll() == []


def test_old_history_is_not_replayed_and_later_files_start_at_zero(env):
    d, clock, *_ = env
    add(d / "old.out", "ROLE_DONE long ago\n")
    w = watch(env)
    assert w.poll() == []
    fresh(d / "new.out", "step 1\n", clock)    # appears after the first poll: read from offset 0
    assert w.poll() == [("colab:run1/new", "running", "")]
    add(d / "old.out", "step 2\n")
    assert w.poll() == [("colab:run1/old", "running", "")]


def test_running_once_then_done(env):
    d, clock, *_ = env
    f = d / "w3-belief.out"
    add(f, "boot\n")
    w = watch(env)
    w.poll()
    add(f, "step 1\n")
    assert w.poll() == [("colab:run1/w3-belief", "running", "")]
    add(f, "step 2\n")
    assert w.poll() == []                      # no repeat running
    add(f, "step 3\n  TRAIN_DONE ok  \nafter\n")
    assert w.poll() == [("colab:run1/w3-belief", "done", "TRAIN_DONE ok")]
    assert run_for(w, clock, 3 * 3600) == []   # an ended run does not stall


def test_fail_lost_precedence_and_one_event_per_kind(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "FAILED one\nTraceback two\nLOST FAILED both\nROLE_DONE\n")
    got = w.poll()
    assert ("colab:run1/a", "fail", "Traceback two") in got
    assert ("colab:run1/a", "lost", "LOST FAILED both") in got
    assert ("colab:run1/a", "done", "ROLE_DONE") in got
    assert len(got) == 4 and got[0][1] == "running"


def test_line_truncated_to_120(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "FAILED " + "y" * 500 + "\n")
    assert len(w.poll()[-1][2]) == 120


def test_partial_line_waits_for_newline(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "TRAIN_")
    assert w.poll() == [("colab:run1/a", "running", "")]
    add(f, "DONE\n")
    assert w.poll() == [("colab:run1/a", "done", "TRAIN_DONE")]


def test_truncation_resets_without_rereporting(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\nFAILED early\n")
    w.poll()
    f.write_bytes(b"FAILED early\n")           # rewritten smaller: new EOF, nothing re-reported
    assert w.poll() == []
    assert w.poll() == []
    add(f, "step\n")                           # a fresh run in the same file
    assert w.poll() == [("colab:run1/a", "running", "")]


def test_stalled_once_then_running_again(env):
    d, clock, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\n")
    w.poll()
    assert run_for(w, clock, 29 * 60) == []
    assert run_for(w, clock, 2 * 60) == [("colab:run1/a", "stalled", "")]
    assert run_for(w, clock, 10 * 60) == []    # once
    add(f, "step\n")
    assert w.poll() == [("colab:run1/a", "running", "")]


def test_never_ran_never_stalls(env):
    d, clock, *_ = env
    add(d / "a.out", "x\n")
    w = watch(env)
    w.poll()
    assert run_for(w, clock, 5 * 3600) == []


def test_restart_keeps_offsets_and_state(env):
    d, clock, kv, entry = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\nTRAIN_DONE\n")
    assert [e[1] for e in w.poll()] == ["running", "done"]
    fresh(d / "b.out", "s\n", clock)
    add(f, "tail\n")
    w2 = LogWatch([entry], kv, clock)          # restart: same kv
    assert w2.poll() == [("colab:run1/a", "running", ""), ("colab:run1/b", "running", "")]
    add(d / "b.out", "s2\n")
    assert w2.poll() == []


def test_restart_does_not_replay_unread_history(env):
    d, clock, kv, entry = env
    add(d / "a.out", "TRAIN_DONE\n")
    LogWatch([entry], kv, clock).poll()
    assert LogWatch([entry], kv, clock).poll() == []


def test_reads_at_most_1mb_per_poll(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, ("pad\n" * 300_000) + "TRAIN_DONE\n")     # ~1.2 MB
    assert [e[1] for e in w.poll()] == ["running"]
    assert w.poll() == [("colab:run1/a", "done", "TRAIN_DONE")]


def test_never_raises(env, monkeypatch):
    d, clock, kv, entry = env
    add(d / "a.out", "x\n")
    add(d / "b.out", "x\n")
    w = LogWatch([{**entry, "done_re": "("}, {**entry, "name": "ok"}, "junk"], kv, clock)
    assert w.poll() == []
    add(d / "a.out", "FAILED\n")
    real = os.path.getsize

    def boom(p):
        if str(p).endswith("a.out"):
            raise PermissionError
        return real(p)

    monkeypatch.setattr("riya.agents.logwatch.os.path.getsize", boom)
    add(d / "b.out", "s\n")
    assert w.poll() == [("ok:run1/b", "running", "")]


def test_invalid_utf8_is_replaced(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    f.write_bytes(f.read_bytes() + b"FAILED \xff\xfe\n")
    assert w.poll()[-1] == ("colab:run1/a", "fail", "FAILED \ufffd\ufffd")


def test_growth_after_done_is_running_again(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\nSTAGE ROLE_DONE\n")
    assert [e[1] for e in w.poll()] == ["running", "done"]
    add(f, "stage 2 begins\n")                 # the done tags are per stage
    assert w.poll() == [("colab:run1/a", "running", "")]
    add(f, "stage 2 TRAIN_DONE\n")
    assert w.poll() == [("colab:run1/a", "done", "stage 2 TRAIN_DONE")]


def test_same_stem_in_two_run_folders_differ(env):
    d, clock, *_ = env
    add(d / "train.out", "x\n")
    w = watch(env)
    w.poll()
    (d.parent / "run2").mkdir()
    add(d / "train.out", "s\n")
    fresh(d.parent / "run2" / "train.out", "s\n", clock)
    assert sorted(e[0] for e in w.poll()) == ["colab:run1/train", "colab:run2/train"]


def test_sleep_gap_is_not_a_stall(env):
    d, clock, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\n")
    w.poll()
    clock.advance(31 * 60)                      # one jump: the machine slept
    assert w.poll() == []
    assert run_for(w, clock, 29 * 60) == []     # the clock restarted at the wake-up poll
    assert run_for(w, clock, 2 * 60) == [("colab:run1/a", "stalled", "")]


def test_sleep_gap_persists_across_restart(env):
    d, clock, kv, entry = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\n")
    w.poll()
    clock.advance(2 * 3600)
    assert LogWatch([entry], kv, clock).poll() == []


def test_first_poll_error_does_not_cause_replay(env, monkeypatch):
    d, clock, kv, entry = env
    add(d / "old.out", "FAILED long ago\nROLE_DONE\n")
    real = os.path.getsize

    def hiccup(p):
        if str(p).endswith("old.out"):
            raise OSError("drive hiccup")
        return real(p)

    monkeypatch.setattr("riya.agents.logwatch.os.path.getsize", hiccup)
    w = watch(env)
    assert w.poll() == []
    monkeypatch.setattr("riya.agents.logwatch.os.path.getsize", real)
    assert w.poll() == []                       # old.out predates the first poll: baseline, not offset 0
    fresh(d / "new.out", "s\n", clock)
    assert w.poll() == [("colab:run1/new", "running", "")]
    add(d / "old.out", "step\n")
    assert w.poll() == [("colab:run1/old", "running", "")]


def test_preexisting_half_line_is_read_whole_later(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "old\nTRAIN_")
    w = watch(env)
    assert w.poll() == []
    add(f, "DONE\n")
    assert w.poll() == [("colab:run1/a", "running", ""), ("colab:run1/a", "done", "TRAIN_DONE")]


def test_progress_bar_reports_the_last_redraw(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "10% ok\r50% ok\r60% FAILED oom\n")
    assert w.poll()[-1] == ("colab:run1/a", "fail", "60% FAILED oom")
    add(f, "FAILED early\r90% fine\n")          # overdrawn by the last redraw: not news
    assert w.poll() == [("colab:run1/a", "running", "")]


def test_crlf_lines(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\r\n")
    w = watch(env)
    w.poll()
    add(f, "step\r\nROLE_DONE\r\n")
    assert w.poll()[-1] == ("colab:run1/a", "done", "ROLE_DONE")


def test_endless_line_is_consumed_in_chunks(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "z" * 1_500_000)                     # no newline at all
    assert w.poll() == [("colab:run1/a", "running", "")]
    assert w.poll() == []                       # the 0.5 MB rest waits for its newline
    add(f, "\nTRAIN_DONE\n")
    assert w.poll() == [("colab:run1/a", "done", "TRAIN_DONE")]


def test_a_failing_seen_save_keeps_the_events(env):
    d, clock, kv, entry = env

    class Flaky(KV):
        def set(self, key, value):
            if "#seen" in key:
                raise RuntimeError("disk")
            super().set(key, value)

    add(d / "a.out", "x\n")
    w = LogWatch([entry], Flaky(), clock)
    w.poll()
    add(d / "a.out", "FAILED\n")
    assert [e[1] for e in w.poll()] == ["running", "fail"]
