"""Presence: notices when the user leaves, takes a break, comes back, talks or games."""
import sys
from dataclasses import dataclass
from typing import Literal, Protocol

from riya.core.store import KV

AWAY_IDLE_S = 900
BREAK_IDLE_S = 120
BREAK_TAIL_S = 60
TALKING_WINDOW_S = 180
_RESUME_IDLE_S = 5
KV_EVERY_S = 30  # last_active_at is rewritten only once it moved this far


class Clock(Protocol):
    def now(self) -> float: ...


class IdleSource(Protocol):
    def idle_seconds(self) -> float: ...
    def locked(self) -> bool: ...
    def fullscreen(self) -> bool: ...


class FakeIdleSource:
    def __init__(self):
        self.idle = 0.0
        self.is_locked = False
        self.is_fullscreen = False

    def idle_seconds(self) -> float:
        return self.idle

    def locked(self) -> bool:
        return self.is_locked

    def fullscreen(self) -> bool:
        return self.is_fullscreen


class WindowsIdleSource:
    """Idle time, session lock and fullscreen foreground window via user32 (ctypes)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._c, self._w = ctypes, wintypes
        # private handles: argtypes on the shared ctypes.windll objects would be overwritten by other modules
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        self._LASTINPUTINFO, self._MONITORINFO = LASTINPUTINFO, MONITORINFO
        u, k = self._user32, self._kernel32
        k.GetTickCount64.restype = ctypes.c_uint64
        u.OpenInputDesktop.restype = wintypes.HANDLE
        u.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        u.CloseDesktop.argtypes = [wintypes.HANDLE]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.MonitorFromWindow.restype = wintypes.HANDLE
        u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]

    def idle_seconds(self) -> float:
        info = self._LASTINPUTINFO(self._c.sizeof(self._LASTINPUTINFO))
        if not self._user32.GetLastInputInfo(self._c.byref(info)):
            return 0.0
        tick = self._kernel32.GetTickCount64() & 0xFFFFFFFF
        return ((tick - info.dwTime) & 0xFFFFFFFF) / 1000.0

    def locked(self) -> bool:
        handle = self._user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
        if not handle:
            return True
        self._user32.CloseDesktop(handle)
        return False

    def fullscreen(self) -> bool:
        u, c = self._user32, self._c
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return False
        name = c.create_unicode_buffer(256)
        u.GetClassNameW(hwnd, name, 256)
        if name.value in ("Progman", "WorkerW"):
            return False
        rect = self._w.RECT()
        mon = self._MONITORINFO()
        mon.cbSize = c.sizeof(self._MONITORINFO)
        if not u.GetWindowRect(hwnd, c.byref(rect)):
            return False
        if not u.GetMonitorInfoW(u.MonitorFromWindow(hwnd, 2), c.byref(mon)):  # MONITOR_DEFAULTTONEAREST
            return False
        m = mon.rcMonitor
        return (rect.left, rect.top, rect.right, rect.bottom) == (m.left, m.top, m.right, m.bottom)


def make_idle_source() -> IdleSource:
    """The real sensors on Windows and Linux; elsewhere a stand-in that always reads as present."""
    if sys.platform == "win32":
        return WindowsIdleSource()
    if sys.platform.startswith("linux"):
        from riya.core.linux_idle import LinuxIdleSource
        return LinuxIdleSource()
    return FakeIdleSource()


@dataclass
class PresenceEvent:
    kind: Literal["came_back", "went_away", "break_started", "break_ended"]
    away_s: float = 0.0


State = Literal["away", "focused", "break", "talking", "gaming"]


class Presence:
    def __init__(self, source: IdleSource, clock: Clock, kv: KV):
        self.source, self.clock, self.kv = source, clock, kv
        now = clock.now()
        self.state: State = "focused"
        self.left_at: float | None = None
        self._last_update = self._work_start = now
        self._last_turn: float | None = None
        self._tail_start: float | None = None
        last_active = kv.get("presence.last_active_at")
        self._saved_active, self._saved_left = last_active, kv.get("presence.left_at")
        if last_active is not None and now - last_active >= AWAY_IDLE_S:
            self.state, self.left_at = "away", last_active

    def note_turn(self) -> None:
        self._last_turn = self.clock.now()

    def work_stretch_s(self) -> float:
        return max(0.0, self.clock.now() - self._work_start)

    def update(self) -> list[PresenceEvent]:
        now, idle, locked = self.clock.now(), self.source.idle_seconds(), self.source.locked()
        events: list[PresenceEvent] = []
        was = self.state
        if now < self._last_update:  # clock jumped back: keep timed states progressing from the new now
            self._tail_start = None if self._tail_start is None else min(self._tail_start, now)
            self._last_turn = None if self._last_turn is None else min(self._last_turn, now)
            self._work_start = min(self._work_start, now)
        if self._last_turn is not None:  # a hands-free conversation is input too
            idle = min(idle, max(0.0, now - self._last_turn))
        slept = now - self._last_update > AWAY_IDLE_S  # the laptop slept: away since the last update
        if was == "away":
            if not locked and idle < _RESUME_IDLE_S:
                left = now if self.left_at is None else self.left_at
                events.append(PresenceEvent("came_back", max(0.0, now - idle - left)))
                self._resume(now)
        elif locked or idle >= AWAY_IDLE_S:
            self.state, self.left_at = "away", now if locked else now - idle
            if slept:
                self.left_at = min(self.left_at, self._last_update)
            events.append(PresenceEvent("went_away"))
        elif slept:
            if idle < _RESUME_IDLE_S:
                events.append(PresenceEvent("came_back", max(0.0, now - idle - self._last_update)))
                self._resume(now)
            else:
                self.state, self.left_at = "away", self._last_update
                events.append(PresenceEvent("went_away"))
        else:
            events += self._active(now, idle, was)
        self._persist(now - idle)
        self._last_update = now
        return events

    def _persist(self, last_active: float) -> None:
        if self.state != "away" and (self._saved_active is None or abs(last_active - self._saved_active) >= KV_EVERY_S):
            self._saved_active = last_active
            self.kv.set("presence.last_active_at", last_active)
        if self.left_at != self._saved_left:
            self._saved_left = self.left_at
            self.kv.set("presence.left_at", self.left_at)

    def _resume(self, now: float) -> None:
        self.left_at, self._tail_start, self._work_start = None, None, now
        self.state = self._base_state(now)

    def _base_state(self, now: float) -> State:
        if self.source.fullscreen():
            return "gaming"
        if self._last_turn is not None and now - self._last_turn <= TALKING_WINDOW_S:
            return "talking"
        return "focused"

    def _active(self, now: float, idle: float, was: State) -> list[PresenceEvent]:
        if self.source.fullscreen():  # a video or a gamepad game sends no keyboard or mouse input: never a break
            self._tail_start, self.state = None, "gaming"
            return [PresenceEvent("break_ended")] if was == "break" else []
        if idle >= BREAK_IDLE_S:
            self._tail_start = None
            self.state = "break"
            return [] if was == "break" else [PresenceEvent("break_started")]
        if was == "break":
            if self._tail_start is None:  # input resumed: the work stretch restarts here (brief's test resumes and waits 61 s in one update)
                self._tail_start = self._work_start = now
            if now - self._tail_start < BREAK_TAIL_S:
                return []
            self._tail_start = None
            self.state = self._base_state(now)
            return [PresenceEvent("break_ended")]
        self.state = self._base_state(now)
        return []
