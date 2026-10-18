from datetime import datetime

import pytest

from riya.core.clock import FakeClock
from riya.core.initiative import Initiative
from riya.core.memory import Thread
from riya.core.settings import Settings, apply_patch, in_quiet_hours, load_settings, save_settings, set_autostart
from riya.core.store import KV, open_db

NOON = datetime(2026, 10, 19, 12, 0).timestamp()


def ini(settings=None, t=NOON):
    clock = FakeClock(t)
    s = settings or Settings()
    return clock, Initiative(clock, lambda: s)


def test_settings_defaults_patch_and_persist():
    kv = KV(open_db(":memory:"))
    s = load_settings(kv)
    assert s.mode == "buddy" and s.online_consolidation is False and s.may_start_conversations
    save_settings(kv, apply_patch(s, {"mode": "panel", "volume": 0.5}))
    assert load_settings(kv).mode == "panel"


@pytest.mark.parametrize("patch", [{"mode": "fullscreen"}, {"nope": 1}, {"volume": 3}, {"quiet_hours": "25:00-1"}])
def test_patch_rejects_bad_values(patch):
    with pytest.raises(ValueError):
        apply_patch(Settings(), patch)


def test_quiet_hours_cross_midnight():
    assert in_quiet_hours("23:30-08:00", datetime(2026, 10, 19, 1, 0))
    assert not in_quiet_hours("23:30-08:00", datetime(2026, 10, 19, 12, 0))
    assert not in_quiet_hours(None, datetime(2026, 10, 19, 1, 0))


def test_autostart_file(tmp_path):
    set_autostart(True, launcher=tmp_path / "start_riya.ps1", startup_dir=tmp_path)
    assert "start_riya.ps1" in (tmp_path / "riya.cmd").read_text()
    set_autostart(False, launcher=tmp_path / "start_riya.ps1", startup_dir=tmp_path)
    assert not (tmp_path / "riya.cmd").exists()


def test_greets_after_two_hours_only():
    _, i = ini()
    assert i.on_came_back(7199, []) is None
    a = i.on_came_back(7200, [])
    assert a.kind == "greet" and a.spoken


def test_greeting_picks_eligible_thread():
    clock, i = ini()
    fresh = Thread(1, "Ask about the exam", clock.now(), "open", None)
    raised = Thread(2, "Ask about the build", clock.now(), "open", clock.now() - 3600)
    assert i.on_came_back(86400, [raised, fresh]).thread.id == 1


def test_no_greeting_when_disabled():
    _, i = ini(Settings(may_start_conversations=False))
    assert i.on_came_back(86400, []) is None


def test_bid_needs_break_and_reason_and_is_silent():
    _, i = ini()
    assert i.on_break("focused", 3 * 3600, []) is None
    assert i.on_break("break", 600, []) is None
    a = i.on_break("break", 7201, [])
    assert a.kind == "bid" and a.spoken is False


def test_bid_rate_limit():
    clock, i = ini()
    assert i.on_break("break", 7201, [])
    clock.advance(5399)
    assert i.on_break("break", 7201, []) is None
    clock.advance(1)
    assert i.on_break("break", 7201, [])


def test_no_bid_while_gaming_or_quiet_hours():
    _, i = ini()
    assert i.on_break("gaming", 9 * 3600, []) is None
    _, night = ini(t=datetime(2026, 10, 19, 1, 0).timestamp())
    assert night.on_break("break", 9 * 3600, []) is None


def test_load_settings_tolerates_unknown_and_missing_keys():
    kv = KV(open_db(":memory:"))
    kv.set("settings", {"mode": "window", "obsolete_flag": True})
    s = load_settings(kv)
    assert s.mode == "window" and s.voice == "alba" and not hasattr(s, "obsolete_flag")


@pytest.mark.parametrize(
    "patch",
    [
        {"mute_voice": 1},
        {"volume": True},
        {"volume": "0.5"},
        {"buddy_position": [1]},
        {"buddy_position": [1.5, 2]},
        {"buddy_position": [True, 2]},
        {"sprite_pack": 3},
        {"quiet_hours": 5},
        {"voice": ""},
        {"voice": None},
        {"mode": 1},
    ],
)
def test_patch_rejects_wrong_types(patch):
    with pytest.raises(ValueError):
        apply_patch(Settings(), patch)


def test_patch_accepts_valid_values_and_returns_new_object():
    s = Settings()
    n = apply_patch(s, {"buddy_position": [10, 20], "quiet_hours": None, "sprite_pack": "x", "volume": 1, "mute_mic": True})
    assert n is not s and s.buddy_position is None
    assert n.buddy_position == [10, 20] and n.quiet_hours is None and n.volume == 1 and n.mute_mic


def test_autostart_default_dir_and_content(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    launcher = tmp_path / "ü" / "start_riya.ps1"
    set_autostart(True, launcher=launcher)
    f = tmp_path / "Microsoft/Windows/Start Menu/Programs/Startup/riya.cmd"
    assert f.read_text(encoding="utf-8") == f'@start "" powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File "{launcher}"'
    set_autostart(False, launcher=launcher)
    set_autostart(False, launcher=launcher)
    assert not f.exists()


def test_bid_with_due_thread_and_cooldown():
    clock, i = ini()
    fresh = Thread(1, "a", clock.now(), "open", None)
    old = Thread(2, "b", clock.now(), "open", clock.now() - 86400)
    recent = Thread(3, "c", clock.now(), "open", clock.now() - 86399)
    assert i.on_break("break", 600, [recent]) is None
    a = i.on_break("break", 600, [recent, old, fresh])
    assert a.thread.id == 2 and a.spoken is False and i.last_bid_at == clock.now()


def test_greeting_does_not_set_bid_clock_and_ignores_quiet_hours():
    _, i = ini(t=datetime(2026, 10, 19, 1, 0).timestamp())
    assert i.on_came_back(7200, []).kind == "greet"
    assert i.last_bid_at is None


def test_bid_blocked_when_disabled():
    _, i = ini(Settings(may_start_conversations=False))
    assert i.on_break("break", 9 * 3600, []) is None
