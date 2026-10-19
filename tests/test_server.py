"""The llama-server wrapper with her adapter: --lora, slot caches keyed by variant, the fall back to the base model and
live following of the local_adapter setting. No real llama-server is ever started: popen, health and the clock are
fakes; the settings DBs are in tmp_path."""
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from riya import brain, server
from riya.core.settings import Settings, apply_patch, save_settings
from riya.core.store import KV, open_db
from riya.server import GameModePolicy, RiyaServer


class FakeProc:
    def __init__(self, exit_code=None, wait_raises=()):
        self.exit_code, self.calls, self.wait_raises = exit_code, [], list(wait_raises)

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        self.calls.append("wait")
        if self.wait_raises:
            raise self.wait_raises.pop(0)
        return 0


class Popen:
    """A recording popen: each call returns the next scripted proc (an int means "exits with that code"; the last
    entry repeats), and logs the command."""

    def __init__(self, *procs):
        self.script, self.cmds, self.procs = list(procs) or [None], [], []

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        proc = item if isinstance(item, FakeProc) else FakeProc(item)
        self.procs.append(proc)
        return proc


def lora_of(cmd):
    return cmd[cmd.index("--lora") + 1] if "--lora" in cmd else None


def slot_of(cmd):
    return cmd[cmd.index("--slot-save-path") + 1]


@pytest.fixture
def adapter(tmp_path):
    f = tmp_path / "riya-v1-lora.gguf"
    f.write_bytes(b"GGUF-adapter")
    return f


def make(tmp_path, popen=None, **kw):
    popen = popen or Popen()
    srv = RiyaServer("m.gguf", slot_dir=str(tmp_path / "slots"), popen=popen, healthy=lambda port: True,
                     ready_timeout_s=0, **kw)
    return srv, popen


def test_cmd_passes_lora_and_keys_the_slot_dir(tmp_path, adapter):
    srv, _ = make(tmp_path, lora=str(adapter))
    cmd = srv._cmd(True)
    assert cmd[cmd.index("--lora"):cmd.index("--lora") + 2] == ["--lora", os.path.abspath(adapter)]
    slots = os.path.abspath(tmp_path / "slots")
    key = os.path.basename(slot_of(cmd))
    assert os.path.dirname(slot_of(cmd)) == slots and len(key) == len("lora-") + 12 and key.startswith("lora-")
    int(key[5:], 16)
    base, _ = make(tmp_path)
    assert slot_of(base._cmd(True)) == os.path.join(slots, "base") and "--lora" not in base._cmd(True)
    os.utime(adapter, ns=(1, 1))  # a retrained file at the same path is another variant
    assert brain.slot_key(str(adapter)) != key


def test_game_mode_moves_keep_the_adapter(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, lora=str(adapter))
    posts = []
    monkeypatch.setattr(srv, "_post", lambda path, body: posts.append(path) or {})
    srv.start(True)
    srv.move(False)
    srv.move(True)
    assert len(popen.cmds) == 3
    assert {lora_of(c) for c in popen.cmds} == {os.path.abspath(adapter)}
    assert [c[c.index("-ngl") + 1] for c in popen.cmds] == ["99", "0", "99"]
    assert len({slot_of(c) for c in popen.cmds}) == 1  # the same variant's cache throughout
    assert posts == ["/slots/0?action=save", "/slots/0?action=restore"] * 2
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]


def test_missing_adapter_falls_back_to_base(tmp_path):
    nope = str(tmp_path / "nope.gguf")
    srv, popen = make(tmp_path, lora=nope)
    assert "--lora" not in srv._cmd(True)
    srv.start()
    assert srv.lora is None and srv.wanted == nope and len(popen.cmds) == 1 and "--lora" not in popen.cmds[0]


def test_adapter_deleted_after_construction_still_starts(tmp_path, adapter):
    srv, popen = make(tmp_path, lora=str(adapter))
    adapter.unlink()
    srv.start()
    assert "--lora" not in popen.cmds[0] and srv.lora is None and srv.lora_failed == os.path.abspath(adapter)


def test_broken_adapter_falls_back_to_base(tmp_path, adapter, capsys):
    srv, popen = make(tmp_path, Popen(1, None), lora=str(adapter))
    srv.start()
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[0]) == os.path.abspath(adapter) and lora_of(popen.cmds[1]) is None
    assert srv.lora is None and srv.lora_failed == os.path.abspath(adapter) and srv.wanted == str(adapter)
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]  # the failed one was stopped
    assert slot_of(popen.cmds[1]).endswith("base")
    assert "didn't load; serving the base model" in capsys.readouterr().out
    strict, popen = make(tmp_path, Popen(1, None), lora=str(adapter), fallback_to_base=False)
    with pytest.raises(RuntimeError, match="exited"):
        strict.start()
    assert len(popen.cmds) == 1


def test_a_base_that_fails_is_raised_not_retried(tmp_path):
    srv, popen = make(tmp_path, Popen(1))
    with pytest.raises(RuntimeError, match="exited"):
        srv.start()
    assert len(popen.cmds) == 1


def test_not_ready_times_out(tmp_path):
    srv = RiyaServer("m", slot_dir=str(tmp_path / "s"), popen=Popen(None), healthy=lambda port: False,
                     ready_timeout_s=0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        srv.start()


def test_switch_restarts_without_restoring_across_variants(tmp_path, adapter, monkeypatch, capsys):
    srv, popen = make(tmp_path)
    posts = []
    monkeypatch.setattr(srv, "_post", lambda path, body: posts.append(path) or {})
    srv.start()
    srv.switch(str(adapter))
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[1]) == os.path.abspath(adapter)
    assert slot_of(popen.cmds[0]) != slot_of(popen.cmds[1])
    assert posts == [] and srv.lora == os.path.abspath(adapter) and srv.wanted == str(adapter)
    assert "switched to Riya v1" in capsys.readouterr().out
    srv.switch(str(adapter))
    assert len(popen.cmds) == 2
    srv.switch(None)
    assert len(popen.cmds) == 3 and lora_of(popen.cmds[2]) is None and posts == []
    assert "switched to the base model" in capsys.readouterr().out


def test_step_follows_the_setting_without_retry_loops(tmp_path, adapter):
    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"junk")
    srv, popen = make(tmp_path, Popen(None, 1, None))  # the base, then the broken adapter dies, then the fallback
    srv.start()
    policy = GameModePolicy()
    step = lambda desired: srv.step(policy, desired, free_mb=lambda: 4000, now=lambda: 0.0)  # noqa: E731
    step(lambda: str(broken))
    assert len(popen.cmds) == 3 and srv.lora is None and srv.lora_failed == os.path.abspath(broken)
    for _ in range(3):
        step(lambda: str(broken))
    assert len(popen.cmds) == 3  # tried once, not every tick

    def boom():
        raise sqlite3.OperationalError("database is locked")

    step(boom)
    assert len(popen.cmds) == 3 and srv.lora is None and srv.lora_failed == os.path.abspath(broken)  # no change
    step(lambda: None)  # back to None: she is already on base
    assert len(popen.cmds) == 3 and srv.wanted is None and srv.lora_failed is None

    healthy, popen = make(tmp_path, lora=str(adapter))
    healthy.start()
    step_h = lambda desired: healthy.step(policy, desired, free_mb=lambda: 4000, now=lambda: 0.0)  # noqa: E731
    step_h(lambda: str(adapter))
    assert len(popen.cmds) == 1
    step_h(lambda: None)
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[1]) is None and healthy.lora is None
    step_h(lambda: None)
    assert len(popen.cmds) == 2


def test_a_read_error_never_falls_back_to_base(tmp_path, adapter):
    srv, popen = make(tmp_path, lora=str(adapter))
    srv.start()

    def boom():
        raise sqlite3.DatabaseError("file is not a database")

    srv.step(GameModePolicy(), boom, free_mb=lambda: 4000, now=lambda: 0.0)
    assert srv.lora == os.path.abspath(adapter) and srv.wanted == str(adapter) and len(popen.cmds) == 1


def test_step_still_runs_game_mode(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, lora=str(adapter))
    monkeypatch.setattr(srv, "_post", lambda path, body: {})
    srv.start()
    policy = GameModePolicy()
    srv.step(policy, lambda: str(adapter), free_mb=lambda: 100, now=lambda: 0.0)
    assert len(popen.cmds) == 2 and popen.cmds[1][popen.cmds[1].index("-ngl") + 1] == "0"
    assert lora_of(popen.cmds[1]) == os.path.abspath(adapter)


def test_stop_waits_and_kills(tmp_path):
    srv, _ = make(tmp_path)
    srv.stop()  # no process: a no-op
    srv.proc = FakeProc(wait_raises=[subprocess.TimeoutExpired("x", 1)])
    srv.stop()
    assert srv.proc.calls == ["terminate", "wait", "kill", "wait"]
    srv.proc = FakeProc()
    srv.stop()
    assert srv.proc.calls == ["terminate", "wait"]


def settings_db(path: Path, **patch) -> Path:
    conn = open_db(path)
    save_settings(KV(conn), apply_patch(Settings(), patch))
    conn.close()
    return path


def test_read_local_adapter(tmp_path, adapter):
    db = settings_db(tmp_path / "t.db", local_adapter=str(adapter))
    assert brain.read_local_adapter(db) == str(adapter)
    assert brain.read_local_adapter(settings_db(tmp_path / "none.db")) is None
    missing = tmp_path / "gone" / "x.db"
    assert brain.read_local_adapter(missing) is None and not missing.exists() and not missing.parent.exists()
    before = db.stat().st_mtime_ns
    brain.read_local_adapter(db)
    assert db.stat().st_mtime_ns == before
    junk = tmp_path / "junk.db"
    junk.write_bytes(os.urandom(4096))
    with pytest.raises(sqlite3.DatabaseError):
        brain.read_local_adapter(junk)


def test_read_local_adapter_while_the_core_has_it_open(tmp_path, adapter):
    db = tmp_path / "live.db"
    conn = open_db(db)  # the running core's connection (WAL)
    save_settings(KV(conn), apply_patch(Settings(), {"local_adapter": str(adapter)}))
    try:
        assert brain.read_local_adapter(db) == str(adapter)
        assert conn.execute("select count(*) from kv").fetchone()[0] >= 1  # and the reader changed nothing
    finally:
        conn.close()


def test_read_local_adapter_other_shapes(tmp_path):
    db = tmp_path / "odd.db"
    conn = sqlite3.connect(str(db))
    conn.execute("create table kv(key text primary key, value text)")
    conn.execute("insert into kv values('settings', ?)", (json.dumps({"local_adapter": 5}),))
    conn.commit()
    assert brain.read_local_adapter(db) is None
    conn.execute("update kv set value='not json'")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError):
        brain.read_local_adapter(db)


def test_print_lora(tmp_path, adapter, capsys, monkeypatch):
    def no_server(*a, **k):
        raise AssertionError("--print-lora must not even build a server")

    # RiyaServer's default popen is bound when the class is defined, so patching subprocess.Popen would guard nothing:
    # the server can't be built at all, and nothing can launch.
    monkeypatch.setattr(server, "RiyaServer", no_server)
    monkeypatch.setattr(server, "free_vram_mb", lambda: pytest.fail("no GPU probe"))

    def run(db):
        assert server.main(["--print-lora", "--db", str(db)]) == 0
        return capsys.readouterr().out.strip()

    assert run(settings_db(tmp_path / "a.db", local_adapter=str(adapter))) == str(adapter)
    gone = tmp_path / "gone.gguf"
    assert run(settings_db(tmp_path / "b.db", local_adapter=str(gone))) == ""
    assert run(settings_db(tmp_path / "c.db")) == ""
    assert run(tmp_path / "no" / "such.db") == ""
    junk = tmp_path / "junk.db"
    junk.write_bytes(os.urandom(4096))
    assert run(junk) == ""
    assert not (tmp_path / "no").exists()


def test_main_requires_a_model_unless_printing(capsys):
    with pytest.raises(SystemExit) as e:
        server.main([])
    assert e.value.code == 2 and "--model" in capsys.readouterr().err


def test_main_follows_the_setting_and_falls_back(tmp_path, adapter, monkeypatch):
    """main with --follow-settings: the adapter from the settings DB, watched read-only, with fake processes."""
    db = settings_db(tmp_path / "m.db", local_adapter=str(adapter))
    popen = Popen(1, None)  # the adapter doesn't load; the base does
    seen = {}

    def fake_watch(self, policy, every_s=3.0, desired=None):
        seen["desired"] = desired()
        seen["wanted"] = self.wanted

    monkeypatch.setattr(server.RiyaServer, "_health_ok", staticmethod(lambda port: True))
    monkeypatch.setattr(server.RiyaServer, "watch", fake_watch)
    monkeypatch.setattr(server, "free_vram_mb", lambda: 4000)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    # the default popen is bound at class definition, so inject the fake through the constructor
    real_init = RiyaServer.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **{**kw, "popen": popen})

    monkeypatch.setattr(RiyaServer, "__init__", init)
    assert server.main(["--model", "m.gguf", "--follow-settings", "--db", str(tmp_path / "m.db")]) == 0
    assert seen == {"desired": str(adapter), "wanted": str(adapter)}
    assert len(popen.cmds) == 2 and "--lora" in popen.cmds[0] and "--lora" not in popen.cmds[1]
    assert popen.procs[1].calls[-2:] == ["terminate", "wait"]  # stopped on the way out
    assert db.exists()


def test_served_adapters():
    body = b'[{"id":0,"path":"C:/m/a.gguf","scale":1.0},{"id":1,"path":"b","scale":0.0}]'
    asked = []

    def get(url, timeout):
        asked.append((url, timeout))
        return body

    assert brain.served_adapters(get, "http://127.0.0.1:8091") == ["C:/m/a.gguf"]
    assert asked == [("http://127.0.0.1:8091/lora-adapters", 1.5)]

    def boom(url, timeout):
        raise OSError("down")

    assert brain.served_adapters(boom, "http://x") == []
    assert brain.served_adapters(lambda u, t: b"{}", "http://x") == []
    assert brain.served_adapters(lambda u, t: b"nonsense", "http://x") == []
    assert brain.served_adapters(lambda u, t: b'[{"id":0}]', "http://x") == []
    if os.name == "nt":
        assert brain.same_path("C:/M/A.gguf", "c:\\m\\a.gguf")
    assert not brain.same_path("C:/m/a.gguf", "C:/m/b.gguf")


def test_champion_record(tmp_path, adapter):
    rec = {"decision": "adopt", "targets_base": 8, "targets_candidate": 10}
    out = brain.write_record(adapter, rec)
    assert out == adapter.with_name("riya-v1-lora.gguf.champion.json") == brain.record_path(adapter)
    assert not list(tmp_path.glob("*.tmp"))
    st = adapter.stat()
    assert brain.read_record(adapter) == {**rec, "adapter": str(adapter), "size": st.st_size,
                                          "mtime_ns": st.st_mtime_ns}
    adapter.write_bytes(b"retrained, other bytes")
    assert brain.read_record(adapter) is None
    brain.write_record(adapter, {**rec, "decision": "reject"})
    assert brain.read_record(adapter) is None
    brain.write_record(adapter, rec)
    assert brain.read_record(adapter) is not None
    os.utime(adapter, ns=(5, 5))  # the same bytes but another mtime: not the file that was judged
    assert brain.read_record(adapter) is None
    brain.write_record(adapter, rec)
    brain.record_path(adapter).write_text("{not json", encoding="utf-8")
    assert brain.read_record(adapter) is None
    brain.record_path(adapter).unlink()
    assert brain.read_record(adapter) is None


def test_local_adapter_setting():
    assert Settings().local_adapter is None
    s = apply_patch(Settings(), {"local_adapter": "C:/Users/x/riya-v1-lora.gguf"})
    assert s.local_adapter == "C:/Users/x/riya-v1-lora.gguf"
    assert apply_patch(s, {"local_adapter": None}).local_adapter is None
    assert apply_patch(Settings(), {"local_adapter": "C:/x/A.GGUF"}).local_adapter == "C:/x/A.GGUF"
    for bad in ("", "  ", "riya.gguf", "C:/x/a.bin", 5, True, "C:/x/a\nb.gguf", "C:/x/a\tb.gguf", "C:/x/a\x00b.gguf",
                "C:/x/a\rb.gguf", "C:/" + "a" * (261 - len("C:/") - len(".gguf")) + ".gguf"):
        with pytest.raises(ValueError):
            apply_patch(Settings(), {"local_adapter": bad})
    edge = "C:/" + "a" * (260 - len("C:/") - len(".gguf")) + ".gguf"
    assert len(edge) == 260 and apply_patch(Settings(), {"local_adapter": edge}).local_adapter == edge


def test_paths(tmp_path):
    assert brain.default_db() == tmp_path / "localappdata" / "Riya" / "riya.db"
    assert tmp_path in brain.default_adapter().parents
    assert brain.default_adapter().name == brain.ADAPTER_NAME == "riya-v1-lora.gguf"
    assert brain.ADAPTER_DIR == Path("C:/Users/sreev/models/riya")
    assert brain.DEFAULT_MODEL.endswith("Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf")


def test_start_script_passes_the_adapter_through():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "start_riya.ps1").read_text(encoding="utf-8")
    for needle in ("--print-lora", '"--follow-settings"', '"--lora"', brain.DEFAULT_MODEL):
        assert needle in text
    assert "riya.server" in text and '"--model"' in text  # the no-adapter launch is unchanged


# ---- fix round 1: snapshot, orphans, move through stop(), mid-move fallback, watch wiring ----

class MutatingPopen(Popen):
    """A popen that changes the adapter file while llama-server 'loads' it."""

    def __init__(self, action, *procs):
        super().__init__(*procs)
        self.action = action

    def __call__(self, cmd, **kw):
        proc = super().__call__(cmd, **kw)
        if len(self.cmds) == 1:
            self.action()
        return proc


def test_an_adapter_that_vanishes_while_loading_falls_back_to_base(tmp_path, adapter):
    srv, popen = make(tmp_path, MutatingPopen(adapter.unlink), lora=str(adapter))
    srv.start()  # never a FileNotFoundError
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[1]) is None and srv.lora is None
    assert srv.lora_failed == os.path.abspath(adapter) and slot_of(popen.cmds[1]).endswith("base")
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]


def test_an_adapter_replaced_while_loading_falls_back_to_base(tmp_path, adapter):
    srv, popen = make(tmp_path, MutatingPopen(lambda: os.utime(adapter, ns=(7, 7))), lora=str(adapter))
    srv.start()
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[1]) is None and srv.lora_failed == os.path.abspath(adapter)


def test_the_slot_dir_is_pinned_once_per_launch(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, lora=str(adapter))
    srv.start()
    pinned = slot_of(popen.cmds[0])
    assert srv.slot_dir == pinned and os.path.isdir(pinned)  # the dir the command named was the one created
    stats = []
    real = brain.slot_key
    monkeypatch.setattr(server, "slot_key", lambda p: stats.append(p) or real(p))
    assert srv.slot_dir == pinned and stats == []  # reading it does not stat the adapter again
    adapter.unlink()
    assert srv.slot_dir == pinned


def test_a_stat_failure_never_escapes_as_an_oserror(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, lora=str(adapter))

    def vanished(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(server, "slot_key", lambda p: vanished(p) if p else "base")
    srv.start()
    assert lora_of(popen.cmds[0]) is None and srv.lora is None and srv.lora_failed == os.path.abspath(adapter)
    srv.switch(str(adapter))  # the same through a switch
    assert srv.lora is None and len(popen.cmds) == 2 and srv.lora_failed == os.path.abspath(adapter)


def test_a_launch_that_never_gets_ready_is_stopped(tmp_path):
    popen = Popen(None)
    srv = RiyaServer("m", slot_dir=str(tmp_path / "s"), popen=popen, healthy=lambda port: False, ready_timeout_s=0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        srv.start()
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]  # alive but never ready: no orphan


def test_a_failed_fallback_is_stopped_too(tmp_path, adapter):
    popen = Popen(1, None)
    srv = RiyaServer("m", slot_dir=str(tmp_path / "s"), popen=popen, healthy=lambda port: False, ready_timeout_s=0,
                     lora=str(adapter))
    with pytest.raises(RuntimeError):
        srv.start()
    assert len(popen.cmds) == 2 and popen.procs[1].calls[:2] == ["terminate", "wait"]


def patched_main(monkeypatch, popen, **extra):
    real_init = RiyaServer.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **{**kw, "popen": popen, **extra})

    monkeypatch.setattr(RiyaServer, "__init__", init)
    monkeypatch.setattr(server, "free_vram_mb", lambda: 4000)


def test_main_stops_a_launch_that_never_gets_ready(tmp_path, monkeypatch):
    popen = Popen(None)
    patched_main(monkeypatch, popen, healthy=lambda port: False, ready_timeout_s=0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        server.main(["--model", "m.gguf", "--db", str(tmp_path / "x.db")])
    assert popen.procs[0].calls[:2] == ["terminate", "wait"]


def test_move_stops_through_stop_and_kills_a_hung_server(tmp_path, monkeypatch):
    srv, popen = make(tmp_path)
    monkeypatch.setattr(srv, "_post", lambda path, body: {})
    srv.start()
    popen.procs[0].wait_raises = [subprocess.TimeoutExpired("llama-server", 30)]
    srv.move(False)
    assert popen.procs[0].calls == ["terminate", "wait", "kill", "wait"] and len(popen.cmds) == 2


def test_a_fallback_in_the_middle_of_a_move_does_not_restore(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, Popen(None, 1, None), lora=str(adapter))  # the CPU relaunch can't load the adapter
    posts = []
    monkeypatch.setattr(srv, "_post", lambda path, body: posts.append(path) or {})
    srv.start()
    srv.move(False)
    assert srv.lora is None and srv.lora_failed == os.path.abspath(adapter) and len(popen.cmds) == 3
    assert posts == ["/slots/0?action=save"]  # the base variant's slot dir never gets this conversation
    assert lora_of(popen.cmds[2]) is None and popen.cmds[2][popen.cmds[2].index("-ngl") + 1] == "0"


def test_a_failed_save_never_restores(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path, lora=str(adapter))
    posts = []

    def post(path, body):
        posts.append(path)
        if "save" in path:
            raise OSError("down")
        return {}

    monkeypatch.setattr(srv, "_post", post)
    srv.start()
    srv.move(False)
    assert posts == ["/slots/0?action=save"] and len(popen.cmds) == 2


def test_switch_while_on_cpu_stays_on_cpu(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path)
    monkeypatch.setattr(srv, "_post", lambda path, body: {})
    srv.start(False)
    srv.switch(str(adapter))
    srv.move(True)
    srv.move(False)
    srv.switch(None)
    assert [c[c.index("-ngl") + 1] for c in popen.cmds] == ["0", "0", "99", "0", "0"]
    assert [lora_of(c) is not None for c in popen.cmds] == [False, True, True, True, False]
    assert srv.on_gpu is False


def test_switch_to_a_broken_adapter_after_a_working_one_ends_on_base(tmp_path, adapter, capsys):
    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"junk")
    srv, popen = make(tmp_path, Popen(None, 1, None), lora=str(adapter))
    srv.start()
    assert srv.lora == os.path.abspath(adapter)
    srv.switch(str(broken))
    assert len(popen.cmds) == 3 and lora_of(popen.cmds[1]) == os.path.abspath(broken) and lora_of(popen.cmds[2]) is None
    assert srv.lora is None and srv.lora_failed == os.path.abspath(broken) and srv.wanted == str(broken)  # not adapter
    assert popen.procs[0].calls[:2] == ["terminate", "wait"] and popen.procs[1].calls[:2] == ["terminate", "wait"]
    policy = GameModePolicy()
    for _ in range(2):
        srv.step(policy, lambda: str(broken), free_mb=lambda: 4000, now=lambda: 0.0)
    assert len(popen.cmds) == 3  # and it isn't retried
    assert "switched to the base model" in capsys.readouterr().out


def test_watch_ticks_follow_the_setting(tmp_path, adapter, monkeypatch):
    srv, popen = make(tmp_path)
    srv.start()
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 2:
            srv.proc.exit_code = 1  # the server goes away: watch ends

    monkeypatch.setattr(server.time, "sleep", sleep)
    monkeypatch.setattr(server, "free_vram_mb", lambda: 4000)
    srv.watch(GameModePolicy(), every_s=3.0, desired=lambda: str(adapter))
    assert sleeps == [3.0, 3.0] and len(popen.cmds) == 2 and lora_of(popen.cmds[1]) == os.path.abspath(adapter)


def test_main_watches_the_setting_through_real_ticks(tmp_path, adapter, monkeypatch):
    db = settings_db(tmp_path / "w.db")  # nothing chosen yet
    popen = Popen(None)
    patched_main(monkeypatch, popen, healthy=lambda port: True)
    ticks = []

    def sleep(s):
        ticks.append(s)
        if len(ticks) == 1:  # the user picks Riya v1 in the settings (written by the "core", not by the wrapper)
            settings_db(db, local_adapter=str(adapter))
        else:
            popen.procs[-1].exit_code = 1

    monkeypatch.setattr(server.time, "sleep", sleep)
    assert server.main(["--model", "m.gguf", "--follow-settings", "--db", str(db)]) == 0
    assert ticks == [3.0, 3.0]
    assert len(popen.cmds) == 2 and lora_of(popen.cmds[0]) is None and lora_of(popen.cmds[1]) == os.path.abspath(adapter)
