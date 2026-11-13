"""Presence on Linux: idle, lock and fullscreen from fake probes (never the real /dev/input, D-Bus or compositor)."""
import json
import os
import struct

import pytest

from riya.core import linux_idle as li


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def runner(outputs: dict):
    """A fake probe runner: the first matching argv prefix (as a string) gives its output; anything else fails."""
    calls = []

    def run(argv, timeout=li.PROBE_TIMEOUT_S):
        calls.append(argv)
        line = " ".join(argv)
        for prefix, out in outputs.items():
            if line.startswith(prefix):
                return out
        return None
    run.calls = calls
    return run


class NoInput:
    available = False

    def idle_seconds(self):
        raise AssertionError("not used without input devices")


def test_the_input_watcher_keeps_only_the_time_of_the_last_event():
    clock = Clock()
    r, w = os.pipe()
    watcher = li.InputWatcher(paths=[], clock=clock)
    watcher.fds = [r]
    clock.t += 30
    assert watcher.idle_seconds() == 30
    os.write(w, b"\0" * li._EVENT_SIZE)  # an EV_SYN report alone is not input
    watcher.drain(r)
    assert watcher.idle_seconds() == 30
    os.write(w, struct.pack("llHHi", 0, 0, 1, 30, 1))  # EV_KEY
    watcher.drain(r)
    assert watcher.idle_seconds() == 0
    clock.t += 5
    assert watcher.idle_seconds() == 5
    assert not hasattr(watcher, "events")  # nothing of the event is kept
    watcher.close()
    os.close(w)


def test_only_keyboards_mice_and_touchpads_are_watched(tmp_path):
    devices = tmp_path / "devices"
    devices.write_text(
        'I: Bus=0011\nN: Name="AT Translated Set 2 keyboard"\nH: Handlers=sysrq kbd leds event3\n\n'
        'I: Bus=0018\nN: Name="Touchpad"\nH: Handlers=event9 mouse1\n\n'
        'I: Bus=0019\nN: Name="Accelerometer"\nH: Handlers=event7 js0\n\n'
        'I: Bus=0019\nN: Name="Lid Switch"\nH: Handlers=event0\n', encoding="utf-8")
    assert li.input_paths(str(devices)) == ["/dev/input/event3", "/dev/input/event9"]
    assert li.input_paths(str(tmp_path / "missing")) == []


def test_unreadable_devices_mean_no_watcher(tmp_path):
    watcher = li.InputWatcher(paths=[str(tmp_path / "missing")])
    assert not watcher.available


def test_idle_falls_back_to_the_session_idle_monitor_then_logind(monkeypatch):
    run = runner({"gdbus call --session --dest org.gnome.Mutter.IdleMonitor": "(uint64 42000,)\n"})
    assert li.LinuxIdleSource(run=run, watcher=NoInput()).idle_seconds() == 42.0
    run = runner({"gdbus call --session --dest org.freedesktop.ScreenSaver": "(uint32 17,)\n"})
    assert li.LinuxIdleSource(run=run, watcher=NoInput()).idle_seconds() == 17.0
    mono = Clock(500.0)
    run = runner({"loginctl show-session": "LockedHint=no\nIdleHint=yes\nIdleSinceHintMonotonic=380000000\n"})
    assert li.LinuxIdleSource(run=run, watcher=NoInput(), monotonic=mono).idle_seconds() == pytest.approx(120.0)
    run = runner({"loginctl show-session": "LockedHint=no\nIdleHint=no\nIdleSinceHintMonotonic=0\n"})
    assert li.LinuxIdleSource(run=run, watcher=NoInput()).idle_seconds() == 0.0
    assert li.LinuxIdleSource(run=runner({}), watcher=NoInput()).idle_seconds() == 0.0  # nothing: never "away"


def test_locked_reads_logind_a_locker_or_the_screensaver():
    assert li.locked(runner({"loginctl show-session": "LockedHint=yes\n"}))
    assert li.locked(runner({"loginctl show-session": "LockedHint=no\n", "pgrep -u": "4242\n"}))
    assert li.locked(runner({"gdbus call --session --dest org.freedesktop.ScreenSaver": "(true,)\n"}))
    assert not li.locked(runner({"loginctl show-session": "LockedHint=no\n"}))
    assert not li.locked(runner({}))


def test_fullscreen_on_hyprland_sway_and_x11(monkeypatch):
    for k in ("HYPRLAND_INSTANCE_SIGNATURE", "SWAYSOCK", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "x")
    assert li.fullscreen(runner({"hyprctl activewindow -j": json.dumps({"class": "game", "fullscreen": 2})}))
    assert not li.fullscreen(runner({"hyprctl activewindow -j": json.dumps({"class": "kitty", "fullscreen": 0})}))
    assert not li.fullscreen(runner({"hyprctl activewindow -j": json.dumps({"fullscreen": 1})}))  # maximized
    assert li.fullscreen(runner({"hyprctl activewindow -j": json.dumps({"fullscreen": 3})}))
    assert li.fullscreen(runner({"hyprctl activewindow -j": json.dumps({"fullscreen": True})}))  # older Hyprland
    assert not li.fullscreen(runner({"hyprctl activewindow -j": "Invalid"}))
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE")
    monkeypatch.setenv("SWAYSOCK", "/run/sway")
    tree = {"nodes": [{"nodes": [{"focused": True, "fullscreen_mode": 1}]}]}
    assert li.fullscreen(runner({"swaymsg -t get_tree": json.dumps(tree)}))
    monkeypatch.delenv("SWAYSOCK")
    monkeypatch.setenv("DISPLAY", ":0")
    run = runner({"xprop -root": "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x3a00007\n",
                  "xprop -id 0x3a00007": "_NET_WM_STATE(ATOM) = _NET_WM_STATE_FULLSCREEN\n"})
    assert li.fullscreen(run)
    monkeypatch.delenv("DISPLAY")
    assert not li.fullscreen(runner({}))


def test_lock_and_fullscreen_probes_are_cached():
    mono = Clock()
    run = runner({"loginctl show-session": "LockedHint=yes\n"})
    src = li.LinuxIdleSource(run=run, watcher=NoInput(), monotonic=mono)
    assert src.locked() and src.locked()
    n = len(run.calls)
    mono.t += li.CACHE_S + 0.1
    assert src.locked() and len(run.calls) > n


def test_make_idle_source_picks_linux_on_linux(monkeypatch):
    from riya.core import presence
    monkeypatch.setattr(presence.sys, "platform", "linux")
    monkeypatch.setattr(li, "InputWatcher", lambda *a, **k: NoInput())
    assert isinstance(presence.make_idle_source(), li.LinuxIdleSource)
    monkeypatch.setattr(presence.sys, "platform", "darwin")
    assert isinstance(presence.make_idle_source(), presence.FakeIdleSource)


def test_an_unplugged_device_is_closed_and_dropped():
    r, w = os.pipe()
    watcher = li.InputWatcher(paths=[])
    watcher.fds, watcher._paths = [r], {r: "/dev/input/event99"}
    os.close(r)  # the device went away
    watcher.drain(r)
    assert watcher.fds == [] and watcher._paths == {}
    os.close(w)


def test_the_fallback_idle_probe_is_cached():
    mono = Clock()
    run = runner({"gdbus call --session --dest org.gnome.Mutter.IdleMonitor": "(uint64 10000,)\n"})
    src = li.LinuxIdleSource(run=run, watcher=NoInput(), monotonic=mono)
    assert src.idle_seconds() == 10.0
    n = len(run.calls)
    mono.t += 1.0
    assert src.idle_seconds() == 11.0 and len(run.calls) == n  # cached, and still counting up
    mono.t += li.CACHE_S
    src.idle_seconds()
    assert len(run.calls) > n
