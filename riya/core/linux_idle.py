"""Presence on Linux: idle time, the session lock and a fullscreen window, from what the desktop already exposes.

- Idle: the time since the last keyboard, mouse or touch event, read from /dev/input (the user must be in the `input`
  group); only the time of an event is kept, never what it was. Without input access it asks the session's idle
  monitor over D-Bus (GNOME's Mutter IdleMonitor, then org.freedesktop.ScreenSaver), then logind's IdleSinceHint.
- Locked: logind's LockedHint, a running screen locker (hyprlock, swaylock, i3lock...), or the ScreenSaver's GetActive.
- Fullscreen: the focused window on Hyprland (hyprctl), sway (swaymsg) or X11 (xprop).

Every probe is a short subprocess or a file read with a timeout; a failing one reads as "not idle, not locked, not
fullscreen" (the safe default: she never thinks they left when she can't tell). Slow probes are cached briefly.
"""
from __future__ import annotations

import glob
import json
import os
import re
import select
import shutil
import struct
import subprocess
import threading
import time
from typing import Callable

PROBE_TIMEOUT_S = 1.0
CACHE_S = 2.0  # lock and fullscreen probes are re-run at most this often
LOCKERS = ("hyprlock", "swaylock", "i3lock", "waylock", "gtklock", "xsecurelock", "slock", "xscreensaver-auth")
_EVENT_SIZE = struct.calcsize("llHHi")  # struct input_event on 64-bit Linux: timeval, type, code, value


def _run(argv: list[str], timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """A probe's stdout, or None when the tool is missing, fails or is slow."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


class InputWatcher:
    """A daemon thread that notes the monotonic time of the latest event on any readable /dev/input/event* device.
    The events themselves are read and dropped: only "something happened, now" is kept."""

    def __init__(self, paths: list[str] | None = None, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.last = clock()
        self.fds: list[int] = []
        for path in paths if paths is not None else sorted(glob.glob("/dev/input/event*")):
            try:
                self.fds.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
            except OSError:
                continue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return bool(self.fds)

    def start(self) -> "InputWatcher":
        if self.fds and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="riya-input-idle", daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set() and self.fds:
            try:
                ready, _, _ = select.select(self.fds, [], [], 1.0)
            except (OSError, ValueError):
                return
            for fd in ready:
                self.drain(fd)

    def drain(self, fd: int) -> None:
        try:
            data = os.read(fd, _EVENT_SIZE * 64)
        except BlockingIOError:
            return
        except OSError:  # unplugged
            self.fds = [f for f in self.fds if f != fd]
            return
        if data:
            self.last = self.clock()

    def idle_seconds(self) -> float:
        return max(0.0, self.clock() - self.last)

    def close(self) -> None:
        self._stop.set()
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []


def _first_number(text: str | None) -> float | None:
    m = re.search(r"(?:uint64|uint32|int64)?\s*(\d+)", text or "")
    return float(m.group(1)) if m else None


def dbus_idle_seconds(run=_run) -> float | None:
    """The session idle monitor's idle time, or None when there is none."""
    ms = _first_number(run(["gdbus", "call", "--session", "--dest", "org.gnome.Mutter.IdleMonitor",
                            "--object-path", "/org/gnome/Mutter/IdleMonitor/Core",
                            "--method", "org.gnome.Mutter.IdleMonitor.GetIdletime"]))
    if ms is not None:
        return ms / 1000.0
    s = _first_number(run(["gdbus", "call", "--session", "--dest", "org.freedesktop.ScreenSaver",
                           "--object-path", "/org/freedesktop/ScreenSaver",
                           "--method", "org.freedesktop.ScreenSaver.GetSessionIdleTime"]))
    return s


def _session_props(run=_run) -> dict[str, str]:
    sid = os.environ.get("XDG_SESSION_ID")
    argv = ["loginctl", "show-session"] + ([sid] if sid else []) + ["-p", "LockedHint", "-p", "IdleHint",
                                                                    "-p", "IdleSinceHintMonotonic"]
    out = run(argv)
    return dict(line.split("=", 1) for line in (out or "").splitlines() if "=" in line)


def logind_idle_seconds(run=_run, monotonic: Callable[[], float] = time.monotonic) -> float | None:
    p = _session_props(run)
    if p.get("IdleHint") != "yes":
        return 0.0 if p else None
    try:
        since_us = int(p.get("IdleSinceHintMonotonic", "0"))
    except ValueError:
        return None
    return max(0.0, monotonic() - since_us / 1e6) if since_us else None


def locked(run=_run) -> bool:
    if _session_props(run).get("LockedHint") == "yes":
        return True
    if run(["pidof", *LOCKERS]):
        return True
    active = run(["gdbus", "call", "--session", "--dest", "org.freedesktop.ScreenSaver",
                  "--object-path", "/org/freedesktop/ScreenSaver", "--method", "org.freedesktop.ScreenSaver.GetActive"])
    return bool(active and "true" in active)


def _hypr_fullscreen(run) -> bool | None:
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return None
    out = run(["hyprctl", "activewindow", "-j"])
    if out is None:
        return None
    try:
        win = json.loads(out)
    except ValueError:
        return None
    if not isinstance(win, dict):
        return False
    fs = win.get("fullscreen")
    return bool(fs) if isinstance(fs, (bool, int)) else False


def _sway_fullscreen(run) -> bool | None:
    if not os.environ.get("SWAYSOCK"):
        return None
    out = run(["swaymsg", "-t", "get_tree"])
    if out is None:
        return None
    try:
        tree = json.loads(out)
    except ValueError:
        return None
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("focused"):
                return bool(node.get("fullscreen_mode"))
            stack += node.get("nodes", []) + node.get("floating_nodes", [])
    return False


def _x11_fullscreen(run) -> bool | None:
    if not os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    active = run(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
    m = re.search(r"window id # (0x[0-9a-fA-F]+)", active or "")
    if not m or int(m.group(1), 16) == 0:
        return False
    state = run(["xprop", "-id", m.group(1), "_NET_WM_STATE"])
    return bool(state and "_NET_WM_STATE_FULLSCREEN" in state)


def fullscreen(run=_run) -> bool:
    for probe in (_hypr_fullscreen, _sway_fullscreen, _x11_fullscreen):
        got = probe(run)
        if got is not None:
            return got
    return False


class LinuxIdleSource:
    """The IdleSource for Linux desktops (X11 or Wayland). See the module docstring for where each reading comes
    from; a reading that can't be had is never "away"."""

    def __init__(self, run=_run, watcher: InputWatcher | None = None, monotonic: Callable[[], float] = time.monotonic):
        self._run, self._mono = run, monotonic
        self.watcher = watcher if watcher is not None else InputWatcher(clock=monotonic)
        if self.watcher.available:
            self.watcher.start()
        self._cache: dict[str, tuple[float, bool]] = {}

    def idle_seconds(self) -> float:
        if self.watcher.available:
            return self.watcher.idle_seconds()
        for probe in (lambda: dbus_idle_seconds(self._run), lambda: logind_idle_seconds(self._run, self._mono)):
            got = probe()
            if got is not None:
                return got
        return 0.0

    def _cached(self, key: str, fn: Callable[[], bool]) -> bool:
        now = self._mono()
        hit = self._cache.get(key)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
        value = bool(fn())
        self._cache[key] = (now, value)
        return value

    def locked(self) -> bool:
        return self._cached("locked", lambda: locked(self._run))

    def fullscreen(self) -> bool:
        return self._cached("fullscreen", lambda: fullscreen(self._run))
