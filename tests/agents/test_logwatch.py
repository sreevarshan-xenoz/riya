import os

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
    clock, kv = FakeClock(), KV()
    entry = {"name": "colab", "glob": str(tmp_path / "**" / "*.out"), "done_re": "TRAIN_DONE|ROLE_DONE",
             "fail_re": "FAILED|Traceback", "lost_re": "LOST"}
    return tmp_path, clock, kv, entry


def watch(env, **kw):
    _, clock, kv, entry = env
    return LogWatch([{**entry, **kw}], kv, clock)


def add(path, text):
    with open(path, "ab") as f:
        f.write(text.encode())


def test_default_entry():
    e = DEFAULT_ENTRIES[0]
    assert e["name"] == "colab" and e["glob"] == "G:/My Drive/jarvis/runs/**/*.out"
    assert e["fail_re"] == "FAILED|Traceback" and e["lost_re"] == "LOST"
    assert e["done_re"] == "STAGE0_TRAIN_DONE|ROLE_DONE|BENCH_DONE|GEN_DONE|TRAIN_DONE|PASS1_DONE"


def test_missing_root_returns_empty(tmp_path):
    w = LogWatch([{"name": "c", "glob": str(tmp_path / "nope" / "**" / "*.out")}], KV(), FakeClock())
    assert w.poll() == []


def test_old_history_is_not_replayed_and_later_files_start_at_zero(env):
    d, *_ = env
    add(d / "old.out", "ROLE_DONE long ago\n")
    w = watch(env)
    assert w.poll() == []
    add(d / "new.out", "step 1\n")            # appears after the first poll: read from offset 0
    assert w.poll() == [("colab:new", "running", "")]
    add(d / "old.out", "step 2\n")
    assert w.poll() == [("colab:old", "running", "")]


def test_running_once_then_done(env):
    d, clock, *_ = env
    f = d / "w3-belief.out"
    add(f, "boot\n")
    w = watch(env)
    w.poll()
    add(f, "step 1\n")
    assert w.poll() == [("colab:w3-belief", "running", "")]
    add(f, "step 2\n")
    assert w.poll() == []                      # no repeat running
    add(f, "step 3\n  TRAIN_DONE ok  \nafter\n")
    assert w.poll() == [("colab:w3-belief", "done", "TRAIN_DONE ok")]
    add(f, "more\n")
    clock.advance(3 * 3600)
    assert w.poll() == []                      # an ended run neither re-runs nor stalls


def test_fail_lost_precedence_and_one_event_per_kind(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "FAILED one\nTraceback two\nLOST FAILED both\nROLE_DONE\n")
    got = w.poll()
    assert ("colab:a", "fail", "Traceback two") in got
    assert ("colab:a", "lost", "LOST FAILED both") in got
    assert ("colab:a", "done", "ROLE_DONE") in got
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
    assert w.poll() == [("colab:a", "running", "")]
    add(f, "DONE\n")
    assert w.poll() == [("colab:a", "done", "TRAIN_DONE")]


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
    assert w.poll() == [("colab:a", "running", "")]


def test_stalled_once_then_running_again(env):
    d, clock, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\n")
    w.poll()
    clock.advance(29 * 60)
    assert w.poll() == []
    clock.advance(2 * 60)
    assert w.poll() == [("colab:a", "stalled", "")]
    clock.advance(60)
    assert w.poll() == []                      # once
    add(f, "step\n")
    assert w.poll() == [("colab:a", "running", "")]


def test_never_ran_never_stalls(env):
    d, clock, *_ = env
    add(d / "a.out", "x\n")
    w = watch(env)
    w.poll()
    clock.advance(5 * 3600)
    assert w.poll() == []


def test_restart_keeps_offsets_and_state(env):
    d, clock, kv, entry = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    add(f, "step\nTRAIN_DONE\n")
    assert [e[1] for e in w.poll()] == ["running", "done"]
    add(d / "b.out", "s\n")
    add(f, "tail\n")
    w2 = LogWatch([entry], kv, clock)          # restart: same kv
    assert w2.poll() == [("colab:b", "running", "")]
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
    assert w.poll() == [("colab:a", "done", "TRAIN_DONE")]


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
    assert w.poll() == [("ok:b", "running", "")]


def test_invalid_utf8_is_replaced(env):
    d, *_ = env
    f = d / "a.out"
    add(f, "x\n")
    w = watch(env)
    w.poll()
    f.write_bytes(f.read_bytes() + b"FAILED \xff\xfe\n")
    assert w.poll()[-1] == ("colab:a", "fail", "FAILED \ufffd\ufffd")
