"""The v0.5 settings fields and their validation."""
import pytest

from riya.core.hands.tools import TIERS
from riya.core.settings import Settings, apply_patch, load_settings, save_settings
from riya.core.store import KV, open_db

CMD = {"name": "backup", "argv": ["robocopy", "a", "b"], "cwd": None, "timeout_s": 30}


def test_defaults():
    s = Settings()
    assert s.tools_enabled == {n: n not in ("run_command", "write_file") for n in TIERS}
    assert s.tools_enabled["calc"] and not s.tools_enabled["run_command"] and not s.tools_enabled["write_file"]
    assert (s.tools_ask, s.apps, s.commands, s.weather_place, s.hands_online) == ([], [], [], None, True)
    assert Settings().tools_enabled is not Settings().tools_enabled  # not shared


def test_valid_patches_apply():
    s = apply_patch(Settings(), {"tools_ask": ["open_url", "volume"],
                                 "apps": [{"name": "Notepad", "target": "notepad.exe"}],
                                 "commands": [CMD], "weather_place": "Paris", "hands_online": False})
    assert s.tools_ask == ["open_url", "volume"] and s.apps[0]["name"] == "Notepad" and s.commands == [CMD]
    assert s.weather_place == "Paris" and s.hands_online is False
    assert apply_patch(s, {"weather_place": None}).weather_place is None
    assert apply_patch(Settings(), {"commands": [{**CMD, "timeout_s": 60}, {**CMD, "name": "b2", "timeout_s": 1.5}]})


def test_tools_enabled_patch_merges():
    s = apply_patch(Settings(), {"tools_enabled": {"run_command": True, "calc": False}})
    assert s.tools_enabled["run_command"] is True and s.tools_enabled["calc"] is False
    assert s.tools_enabled["note_add"] is True and set(s.tools_enabled) == set(TIERS)


@pytest.mark.parametrize("patch", [
    {"tools_ask": ["calc"]}, {"tools_ask": ["run_command"]}, {"tools_ask": ["write_file"]}, {"tools_ask": ["nope"]},
    {"tools_ask": "open_url"}, {"tools_ask": [1]},
    {"tools_enabled": {"nope": True}}, {"tools_enabled": {"calc": "yes"}}, {"tools_enabled": {"calc": 1}},
    {"tools_enabled": [("calc", True)]},
    {"apps": [{"name": "a", "target": "x"}, {"name": "a", "target": "y"}]},
    {"apps": [{"name": "a", "target": "x"}, {"name": "A", "target": "y"}]},
    {"apps": [{"name": f"a{i}", "target": "x"} for i in range(51)]},
    {"apps": [{"name": "a"}]}, {"apps": [{"name": "", "target": "x"}]}, {"apps": [{"name": "a", "target": 1}]},
    {"apps": [{"name": "a", "target": "x", "extra": 1}]}, {"apps": "x"},
    {"commands": [CMD, dict(CMD)]},
    {"commands": [{**CMD, "name": f"c{i}"} for i in range(21)]},
    {"commands": [{**CMD, "argv": []}]}, {"commands": [{**CMD, "argv": ["x"] * 21}]},
    {"commands": [{**CMD, "argv": "robocopy"}]}, {"commands": [{**CMD, "argv": [1]}]},
    {"commands": [{**CMD, "timeout_s": 0}]}, {"commands": [{**CMD, "timeout_s": 61}]},
    {"commands": [{**CMD, "timeout_s": True}]}, {"commands": [{**CMD, "timeout_s": "5"}]},
    {"commands": [{**CMD, "cwd": 5}]}, {"commands": [{"name": "x", "argv": ["a"]}]},
    {"weather_place": "p" * 81}, {"weather_place": 5}, {"weather_place": "  "},
    {"hands_online": "yes"}, {"hands_online": 1}])
def test_bad_patches_raise(patch):
    with pytest.raises(ValueError):
        apply_patch(Settings(), patch)


def test_twenty_commands_and_fifty_apps_are_fine():
    apply_patch(Settings(), {"commands": [{**CMD, "name": f"c{i}"} for i in range(20)]})
    apply_patch(Settings(), {"apps": [{"name": f"a{i}", "target": "x"} for i in range(50)]})


def test_round_trip_and_old_settings_load(tmp_path):
    kv = KV(open_db(tmp_path / "riya.db"))
    save_settings(kv, apply_patch(Settings(), {"commands": [CMD], "tools_enabled": {"write_file": True}}))
    got = load_settings(kv)
    assert got.commands == [CMD] and got.tools_enabled["write_file"] is True and got.tools_enabled["calc"] is True
    kv.set("settings", {"mode": "panel", "tools_enabled": {"calc": False}})  # a partial map from an older build
    got = load_settings(kv)
    assert got.mode == "panel" and got.tools_enabled["calc"] is False and got.tools_enabled["time_now"] is True
    kv.set("settings", {"mode": "panel"})
    assert load_settings(kv).tools_enabled["run_command"] is False
