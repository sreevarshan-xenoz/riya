import json
import os
import time
from datetime import datetime

import pytest

from riya.agents import install_hooks
from riya.core import health, store
from riya.core import secrets as keystore
from riya.core.health import (CONNECTED_S, DETAIL_MAX, FIXES, ROWS, STATUSES, HealthCtx, HealthItem, check_autostart,
                              check_claude_code, check_colab, check_eyes, check_local, check_look, check_memory,
                              check_mic, check_online, check_voice, clip, pack_in_use, run_checks)
from riya.core.secrets import KeyInfo
from riya.core.settings import Settings

CANARY = "AIzaCANARY-0123456789abcdefWXYZ"
NOW = 1_800_000_000.0


def hhmm(at):
    return datetime.fromtimestamp(at).strftime("%H:%M")


def no_http(url, timeout):
    raise AssertionError("no network call expected")


def ctx(tmp_path, **over):
    settings = over.pop("settings", Settings())
    base = dict(now=NOW, settings=settings, local_url="", key=KeyInfo(None, None), last_tests={}, voice_running=True,
                vision_ready=True, receiver_ok=True, hooks_seen_at=None, claude_settings=tmp_path / "claude" / "settings.json",
                repo_root=tmp_path / "repo", db_path="", schema_version=store.SCHEMA_VERSION,
                startup_dir=tmp_path / "startup", http_get=no_http)
    base.update(over)
    return HealthCtx(**base)


def settings(**kw):
    return Settings(**kw)


# ---- the item and the helpers -----------------------------------------------------------------------------------------

def test_health_item_rules():
    item = HealthItem("local", "ok", "x" * 300)
    assert len(item.detail) == DETAIL_MAX and item.detail.endswith("…")
    for bad in (dict(key="nope", status="ok", detail="d"), dict(key="local", status="bad", detail="d"),
                dict(key="local", status="ok", detail="d", fix="rm")):
        with pytest.raises(ValueError):
            HealthItem(**bad)
    with pytest.raises(Exception):
        item.status = "warn"  # frozen
    assert HealthItem("local", "ok", "a\n  b\t c").detail == "a b c"


def test_clip():
    assert clip("  a \n b  ") == "a b"
    assert clip("abcdef", 4) == "abc…"
    assert clip("abcd", 4) == "abcd"


def test_the_constants_and_the_check_table():
    assert ROWS == ("local", "online", "voice", "mic", "look", "eyes", "claude_code", "colab", "memory", "autostart")
    assert STATUSES == ("ok", "warn", "off") and health.CHECK_TIMEOUT_S == 5.0 and CONNECTED_S == 86400
    assert FIXES == ("online_test", "voice_sample", "mic_check", "eyes_test", "hooks_connect", "hooks_disconnect",
                     "autostart")
    assert tuple(health.CHECKS) == ROWS


# ---- local -------------------------------------------------------------------------------------------------------------

def test_local(tmp_path):
    calls = []

    def get(url, timeout):
        calls.append((url, timeout))
        return b'{"data":[{"id":"C:/models/riya-q4.gguf"}]}'
    item = check_local(ctx(tmp_path, local_url="http://127.0.0.1:8090", http_get=get))
    assert (item.key, item.status, item.detail) == ("local", "ok", "llama-server is up · riya-q4.gguf")
    assert calls[0][0] == "http://127.0.0.1:8090/v1/models"
    item = check_local(ctx(tmp_path, local_url="http://127.0.0.1:8090/", http_get=lambda u, t: b'{"data":[]}'))
    assert (item.status, item.detail) == ("ok", "llama-server is up")

    def boom(url, timeout):
        raise OSError("refused")
    item = check_local(ctx(tmp_path, local_url="http://127.0.0.1:8090", http_get=boom))
    assert item.status == "warn" and "http://127.0.0.1:8090" in item.detail
    item = check_local(ctx(tmp_path, local_url=""))  # no_http would raise
    assert (item.status, item.detail) == ("warn", "No local server is configured.")


def test_default_http_get_refuses_other_schemes():
    with pytest.raises(ValueError):
        health.default_http_get("file:///etc/passwd", 1.0)


# ---- online ------------------------------------------------------------------------------------------------------------

def test_online_states(tmp_path):
    mask = keystore.mask(CANARY)
    assert mask.startswith("••••") and mask.endswith(CANARY[-4:])
    app = KeyInfo(mask, "app")
    item = check_online(ctx(tmp_path))
    assert (item.status, item.fix) == ("off", None) and "free Gemini key" in item.detail
    item = check_online(ctx(tmp_path, key=app))
    assert (item.status, item.fix) == ("ok", "online_test")
    assert item.detail == f"Key {mask} saved in Riya · not tested yet"
    t = NOW - 600
    item = check_online(ctx(tmp_path, key=app, last_tests={"online_test": {"at": t, "ok": True, "detail": "ok", "mask": mask}}))
    assert item.status == "ok" and mask in item.detail and f"test worked {hhmm(t)}" in item.detail
    item = check_online(ctx(tmp_path, key=app, last_tests={
        "online_test": {"at": t, "ok": False, "detail": "The key was refused.", "mask": mask}}))
    assert item.status == "warn" and f"last test failed {hhmm(t)}: The key was refused." in item.detail
    item = check_online(ctx(tmp_path, key=app, last_tests={
        "online_test": {"at": t, "ok": True, "detail": "ok", "mask": "••••zzzz"}}))
    assert item.status == "ok" and item.detail.endswith("not tested yet")  # a test of another key doesn't count
    item = check_online(ctx(tmp_path, key=KeyInfo(mask, ".env")))
    assert "set via .env (edit that file to remove it)" in item.detail and item.fix == "online_test"
    item = check_online(ctx(tmp_path, key=KeyInfo(mask, "env")))
    assert "set via the RIYA_ONLINE_API_KEY environment variable" in item.detail
    item = check_online(ctx(tmp_path, key=app, settings=settings(online_enabled=False)))
    assert (item.status, item.detail) == ("off", "Online brain is switched off in Settings.")


# ---- voice, mic, eyes, autostart, colab, memory -----------------------------------------------------------------------

def test_voice_mic_eyes_autostart_colab_memory_states(tmp_path):
    # voice
    item = check_voice(ctx(tmp_path, voice_running=False))
    assert (item.status, item.fix) == ("warn", None) and "text only" in item.detail
    item = check_voice(ctx(tmp_path, settings=settings(mute_voice=True, voice="alba")))
    assert (item.status, item.detail, item.fix) == ("off", "Muted · voice alba", None)
    item = check_voice(ctx(tmp_path, last_tests={"voice_sample": {"at": NOW, "ok": False, "detail": "no audio", "mask": None}}))
    assert (item.status, item.fix) == ("warn", "voice_sample")
    assert item.detail == "Voice alba · the last sample failed: no audio"
    item = check_voice(ctx(tmp_path))
    assert (item.status, item.detail, item.fix) == ("ok", "Voice alba · speaking works", "voice_sample")
    # mic
    item = check_mic(ctx(tmp_path, settings=settings(mute_mic=True)))
    assert (item.status, item.detail, item.fix) == ("off", "Mic muted in Settings.", None)
    item = check_mic(ctx(tmp_path, voice_running=False))
    assert item.status == "warn" and "text only" in item.detail
    item = check_mic(ctx(tmp_path, last_tests={"mic_check": {"at": NOW, "ok": False, "detail": "no device", "mask": None}}))
    assert (item.status, item.detail, item.fix) == ("warn", "The last mic check failed: no device", "mic_check")
    item = check_mic(ctx(tmp_path, last_tests={"mic_check": {"at": NOW, "ok": True, "detail": "Microphone (USB)", "mask": None}}))
    assert (item.status, item.detail, item.fix) == ("ok", "Listening · Microphone (USB)", "mic_check")
    item = check_mic(ctx(tmp_path))
    assert (item.status, item.detail, item.fix) == ("ok", "Listening · not checked yet", "mic_check")
    # eyes
    mask = keystore.mask(CANARY)
    key = KeyInfo(mask, "app")
    item = check_eyes(ctx(tmp_path, settings=settings(eyes_enabled=False)))
    assert (item.status, item.detail) == ("off", "Eyes off.")
    item = check_eyes(ctx(tmp_path, vision_ready=False))
    assert (item.status, item.detail) == ("warn", "Her eyes need the online key.")
    item = check_eyes(ctx(tmp_path, settings=settings(online_enabled=False)))
    assert (item.status, item.detail) == ("warn", "Her eyes need the online brain switched on.")
    item = check_eyes(ctx(tmp_path, key=key))
    assert (item.status, item.detail, item.fix) == ("ok", "On · not tested yet", "eyes_test")
    item = check_eyes(ctx(tmp_path, key=key, settings=settings(dont_look=True),
                          last_tests={"eyes_test": {"at": NOW, "ok": True, "detail": "42", "mask": mask}}))
    assert item.status == "ok" and item.detail == f"On · eye test worked {hhmm(NOW)} · don't-look is on"
    item = check_eyes(ctx(tmp_path, key=key, last_tests={"eyes_test": {"at": NOW, "ok": False, "detail": "no 42", "mask": mask}}))
    assert item.status == "warn" and item.detail == f"On · the last eye test failed {hhmm(NOW)}: no 42"
    item = check_eyes(ctx(tmp_path, key=key, last_tests={"eyes_test": {"at": NOW, "ok": True, "detail": "42", "mask": "••••oth"}}))
    assert item.detail == "On · not tested yet"
    # autostart
    startup = tmp_path / "startup"
    item = check_autostart(ctx(tmp_path))
    assert (item.status, item.detail, item.fix) == ("off", "Doesn't start with Windows.", "autostart")
    item = check_autostart(ctx(tmp_path, settings=settings(autostart=True)))
    assert (item.status, item.detail, item.fix) == ("warn", "Autostart is on but its shortcut is missing.", "autostart")
    startup.mkdir()
    (startup / "riya.cmd").write_text("x")
    item = check_autostart(ctx(tmp_path, settings=settings(autostart=True)))
    assert (item.status, item.detail, item.fix) == ("ok", "Starts with Windows.", "autostart")
    # colab
    item = check_colab(ctx(tmp_path, settings=settings(agent_logs=[])))
    assert (item.status, item.detail) == ("off", "No Colab log folders set.")
    runs, other = tmp_path / "drive" / "runs", tmp_path / "other"
    runs.mkdir(parents=True)
    entry = lambda glob: {"name": glob[-6:], "glob": glob, "done_re": "", "fail_re": "", "lost_re": ""}  # noqa: E731
    item = check_colab(ctx(tmp_path, settings=settings(agent_logs=[entry(f"{runs.as_posix()}/**/*.out")])))
    assert (item.status, item.detail) == ("ok", "Watching 1 folder(s)")
    item = check_colab(ctx(tmp_path, settings=settings(agent_logs=[
        entry(f"{runs.as_posix()}/**/*.out"), entry(f"{other.as_posix()}/*.log")])))
    assert item.status == "warn" and item.detail == f"Folder not found: {other.as_posix()}/"
    # memory
    item = check_memory(ctx(tmp_path))
    assert (item.status, item.detail) == ("ok", f"In memory · schema {store.SCHEMA_VERSION}")
    item = check_memory(ctx(tmp_path, schema_version=store.SCHEMA_VERSION - 1))
    assert item.status == "warn"


def test_memory_names_the_newest_backup(tmp_path):
    db = tmp_path / "riya.db"
    db.write_bytes(b"x" * (3 * 1024 * 1024 // 2))
    (tmp_path / "riya.db-wal").write_bytes(b"x" * (1024 * 1024 // 2))
    item = check_memory(ctx(tmp_path, db_path=str(db)))
    assert item.status == "ok" and item.detail == f"2.0 MB · schema {store.SCHEMA_VERSION} · no backup yet"
    v3, v4 = tmp_path / "riya.db.pre-v3.bak", tmp_path / "riya.db.pre-v4.bak"
    v3.write_bytes(b"a")
    v4.write_bytes(b"b")
    os.utime(v4, (NOW - 5000, NOW - 5000))
    os.utime(v3, (NOW - 100, NOW - 100))  # pre-v3 is the newer one
    (tmp_path / "other.db.pre-v9.bak").write_bytes(b"c")  # another database's backup is not ours
    item = check_memory(ctx(tmp_path, db_path=str(db)))
    assert "backup riya.db.pre-v3.bak" in item.detail and "pre-v4" not in item.detail


# ---- look --------------------------------------------------------------------------------------------------------------

def test_look(tmp_path):
    repo = tmp_path / "repo"
    assert pack_in_use(None, repo) == (None, False)
    item = check_look(ctx(tmp_path))
    assert (item.status, item.detail) == ("warn", "No art pack found.")
    (repo / "body" / "placeholder-pack").mkdir(parents=True)
    (repo / "body" / "placeholder-pack" / "manifest.json").write_text("{}")
    item = check_look(ctx(tmp_path))
    assert item.status == "warn" and item.detail.startswith("Placeholder art · her own pack goes in ")
    assert pack_in_use(None, repo) == (repo / "body" / "placeholder-pack", True)
    (repo / "sprites" / "riya-v1").mkdir(parents=True)
    (repo / "sprites" / "riya-v1" / "manifest.json").write_text("{}")
    item = check_look(ctx(tmp_path))
    assert item.status == "ok" and item.detail.startswith("Her own art · ")
    assert pack_in_use(None, repo) == (repo / "sprites" / "riya-v1", False)
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "manifest.json").write_text("{}")
    assert pack_in_use(str(mine), repo) == (mine, False)  # settings.sprite_pack comes first
    assert pack_in_use("   ", repo) == (repo / "sprites" / "riya-v1", False)


# ---- claude code -------------------------------------------------------------------------------------------------------

def test_claude_code_four_states(tmp_path):
    path = tmp_path / "claude" / "settings.json"
    down = check_claude_code(ctx(tmp_path, receiver_ok=False))
    assert down.status == "warn" and down.detail.startswith("Receiver down") and down.fix is None
    assert check_claude_code(ctx(tmp_path, receiver_ok=None)).detail.startswith("Receiver down")
    item = check_claude_code(ctx(tmp_path))  # no file
    assert (item.status, item.detail, item.fix) == ("off", "Not connected.", "hooks_connect")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(install_hooks.snippet()))
    item = check_claude_code(ctx(tmp_path))
    assert (item.status, item.fix) == ("warn", "hooks_disconnect") and item.detail.startswith("Applied · restart")
    seen = NOW - 3600
    item = check_claude_code(ctx(tmp_path, hooks_seen_at=seen))
    assert (item.status, item.fix) == ("ok", "hooks_disconnect") and item.detail == f"Connected · last event {hhmm(seen)}"
    item = check_claude_code(ctx(tmp_path, hooks_seen_at=NOW - 25 * 3600))
    assert (item.status, item.fix) == ("warn", "hooks_disconnect") and "no event in 24 h" in item.detail
    # events seen but the hooks are not in the file: connected, nothing to disconnect
    other = tmp_path / "other.json"
    other.write_text("{}")
    item = check_claude_code(ctx(tmp_path, claude_settings=other, hooks_seen_at=seen))
    assert (item.status, item.fix) == ("ok", None)
    # an old event and no hooks in the file: simply not connected
    item = check_claude_code(ctx(tmp_path, claude_settings=other, hooks_seen_at=NOW - 25 * 3600))
    assert (item.status, item.fix) == ("off", "hooks_connect")
    bad = "{ not json"
    path.write_text(bad)
    before = path.read_bytes()
    item = check_claude_code(ctx(tmp_path))
    assert item.status == "warn" and item.detail == "Couldn't read Claude Code's settings.json."
    assert path.read_bytes() == before and sorted(p.name for p in path.parent.iterdir()) == ["settings.json"]


# ---- run_checks --------------------------------------------------------------------------------------------------------

def test_run_checks_returns_every_row_in_order(tmp_path):
    items = run_checks(ctx(tmp_path))
    assert len(items) == 10 and [i.key for i in items] == list(ROWS)
    assert all(i.status in STATUSES and i.detail and "\n" not in i.detail and len(i.detail) <= DETAIL_MAX for i in items)


def test_a_hung_check_times_out_and_the_rest_still_report(tmp_path):
    c = ctx(tmp_path)

    def hang(_):
        time.sleep(10)
    def boom(_):
        raise RuntimeError(CANARY)
    t0 = time.monotonic()
    items = run_checks(c, timeout_s=0.2, checks={"local": hang, "look": check_look, "memory": boom})
    assert time.monotonic() - t0 < 1.0
    by = {i.key: i for i in items}
    assert [i.key for i in items] == ["local", "look", "memory"]
    assert (by["local"].status, by["local"].detail) == ("warn", "No answer within 0.2 s.")
    assert by["look"] == check_look(c)
    assert (by["memory"].status, by["memory"].detail) == ("warn", "This check failed (RuntimeError).")
    assert CANARY not in repr(items)


def test_many_hung_checks_share_one_deadline(tmp_path):
    def hang(_):
        time.sleep(10)
    t0 = time.monotonic()
    items = run_checks(ctx(tmp_path), timeout_s=0.3, checks={k: hang for k in ROWS})
    assert time.monotonic() - t0 < 1.0  # not 10 x 0.3 s
    assert all(i.status == "warn" and i.detail == "No answer within 0.3 s." for i in items)
    assert health.CHECK_TIMEOUT_S == 5.0
    assert f"{health.CHECK_TIMEOUT_S:g}" == "5"


def test_checks_run_in_named_daemon_threads(tmp_path):
    import threading
    seen = []

    def probe(_):
        t = threading.current_thread()
        seen.append((t.name, t.daemon))
        return HealthItem("local", "ok", "fine")
    run_checks(ctx(tmp_path), checks={"local": probe})
    assert seen == [("riya-health-local", True)]


def test_a_check_returning_junk_is_a_failed_check(tmp_path):
    items = run_checks(ctx(tmp_path), checks={"local": lambda c: "not an item"})
    assert items[0].status == "warn" and items[0].detail == "This check failed (TypeError)."


# ---- no key in any detail ----------------------------------------------------------------------------------------------

def test_no_row_detail_ever_holds_the_key(tmp_path):
    keystore.remember(CANARY)
    mask = keystore.mask(CANARY)
    leaky = {"at": NOW, "ok": False, "detail": f"HTTP 400 for key={CANARY}", "mask": mask}
    c = ctx(tmp_path, key=KeyInfo(mask, "app"), local_url="http://127.0.0.1:8090",
            http_get=lambda u, t: b'{"data":[{"id":"riya.gguf"}]}',
            last_tests={k: leaky for k in ("online_test", "eyes_test", "mic_check", "voice_sample")})
    items = run_checks(c)
    assert len(items) == 10
    text = json.dumps([[i.key, i.status, i.detail, i.fix] for i in items], ensure_ascii=False)
    assert CANARY not in text and CANARY[:-4] not in text
    assert mask in next(i for i in items if i.key == "online").detail  # only the masked tail shows


# ---- read-only ---------------------------------------------------------------------------------------------------------

def test_checks_write_nothing(tmp_path):
    claude = tmp_path / "claude" / "settings.json"
    claude.parent.mkdir(parents=True)
    claude.write_text(json.dumps(install_hooks.snippet()))
    db = tmp_path / "riya.db"
    db.write_bytes(b"x")
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    stamp = claude.stat().st_mtime_ns
    run_checks(ctx(tmp_path, db_path=str(db), settings=settings(autostart=True)))
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert claude.stat().st_mtime_ns == stamp


def test_default_startup_dir_follows_appdata(tmp_path):
    from riya.core.settings import default_startup_dir
    assert default_startup_dir() == tmp_path / "appdata" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
