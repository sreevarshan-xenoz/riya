"""Presence on Linux: idle time, the session lock and a fullscreen window, from what the desktop already exposes.

- Idle: the time since the last keyboard, mouse or touchpad event, read from those devices in /dev/input (the user must
  be in the `input` group); only the time of an event is kept, never what it was. Without input access it asks the session's idle
  monitor over D-Bus (GNOME's Mutter IdleMonitor, then org.freedesktop.ScreenSaver), then logind's IdleSinceHint.
- Locked: logind's LockedHint, a running screen locker (hyprlock, swaylock, i3lock...), or the ScreenSaver's GetActive.
- Fullscreen: the focused window on Hyprland (hyprctl), sway (swaymsg) or X11 (xprop).

Every probe is a short subprocess or a file read with a timeout; a failing one reads as "not idle, not locked, not
fullscreen" (the safe default: she never thinks they left when she can't tell). Slow probes are cached briefly.
"""
from __future__ import annotations

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
CACHE_S = 2.0  # lock, fullscreen and fallback idle probes are re-run at most this often
RESCAN_S = 30.0  # the watcher looks for newly plugged keyboards and mice this often
LOCKERS = ("hyprlock", "swaylock", "i3lock", "waylock", "gtklock", "xsecurelock", "slock", "xscreensaver-auth")
_EVENT_SIZE = struct.calcsize("llHHi")  # struct input_event on 64-bit Linux: timeval, type, code, value
_USER_EVENTS = (1, 2, 3)  # EV_KEY, EV_REL, EV_ABS (not EV_SYN or EV_MSC reports)
DEVICES = "/proc/bus/input/devices"


def input_paths(devices: str = DEVICES) -> list[str]:
    """The event devices of keyboards, mice and touchpads (a kbd or mouse handler). Sensors that stream on their own,
    such as an accelerometer (often a js handler) or a lid switch, are left out: they'd never let her see idle."""
    try:
        with open(devices, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    out = []
    for block in text.split("\n\n"):
        m = re.search(r"^H: Handlers=(.*)$", block, re.MULTILINE)
        handlers = m.group(1).split() if m else []
        if any(h == "kbd" or h.startswith("mouse") for h in handlers):
            out += [f"/dev/input/{h}" for h in handlers if re.fullmatch(r"event\d+", h)]
    return sorted(out)


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
        self._paths: dict[int, str] = {}
        self._fixed = paths is not None  # tests pass their own paths; the real watcher rescans for new devices
        self._open(paths if paths is not None else input_paths())
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

    def _open(self, paths: list[str]) -> None:
        known = set(self._paths.values())
        for path in paths:
            if path in known:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            self.fds.append(fd)
            self._paths[fd] = path

    def _drop(self, fd: int) -> None:
        self.fds = [f for f in self.fds if f != fd]
        self._paths.pop(fd, None)
        try:
            os.close(fd)
        except OSError:
            pass

    def _loop(self) -> None:
        rescan = self.clock() + RESCAN_S
        while not self._stop.is_set():
            if not self._fixed and self.clock() >= rescan:  # a keyboard or mouse plugged in later
                rescan = self.clock() + RESCAN_S
                self._open(input_paths())
            if not self.fds:
                self._stop.wait(1.0)
                continue
            try:
                ready, _, _ = select.select(self.fds, [], [], 1.0)
            except (OSError, ValueError):  # a device went away between scans: find it on the next read
                ready = list(self.fds)
            for fd in ready:
                self.drain(fd)

    def drain(self, fd: int) -> None:
        try:
            data = os.read(fd, _EVENT_SIZE * 64)
        except BlockingIOError:
            return
        except OSError:  # unplugged
            self._drop(fd)
            return
        for i in range(0, len(data) - _EVENT_SIZE + 1, _EVENT_SIZE):
            if struct.unpack_from("llHHi", data, i)[2] in _USER_EVENTS:
                self.last = self.clock()
                return

    def idle_seconds(self) -> float:
        return max(0.0, self.clock() - self.last)

    def close(self) -> None:
        self._stop.set()
        for fd in list(self.fds):
            self._drop(fd)


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
    if run(["pgrep", "-u", str(os.getuid()), f"^({'|'.join(LOCKERS)})$"]):  # this user's locker, not another's
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
    if isinstance(fs, bool):  # older Hyprland
        return fs
    return isinstance(fs, int) and fs >= 2  # 0 none, 1 maximized, 2 fullscreen, 3 both


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
        self._idle_cache: tuple[float, float] | None = None

    def idle_seconds(self) -> float:
        if self.watcher.available:
            return self.watcher.idle_seconds()
        now = self._mono()
        hit = self._idle_cache
        if hit and now - hit[0] < CACHE_S:
            return hit[1] + (now - hit[0] if hit[1] > 0 else 0.0)
        got = 0.0
        for probe in (lambda: dbus_idle_seconds(self._run), lambda: logind_idle_seconds(self._run, self._mono)):
            value = probe()
            if value is not None:
                got = value
                break
        self._idle_cache = (now, got)
        return got

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

    def close(self) -> None:
        self.watcher.close()
