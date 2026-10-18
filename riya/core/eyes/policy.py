"""Glance policy (pure): when Riya may look at the active window, and when she wants to. Doubt means don't look."""
import re
from datetime import datetime
from functools import lru_cache
from typing import Iterable

from riya.core.eyes.capture import WindowInfo, mad
from riya.core.presence import Clock

DENY_TITLES = ("password|1password|bitwarden|keepass|lastpass|bank|banking|paypal|upi|net ?banking|credit card|"
               "inprivate|incognito|private browsing|windows security|credential|sign in to|login|sign in|sign-in|"
               "log in|log-in")
DENY_PROCESSES = ("keepass.*|1password.*|bitwarden.*|credentialuibroker|dashlane.*|nordpass.*|enpass.*|roboform.*|"
                  "keeper.*|proton ?pass.*")
_LOOKING = ("focused", "talking")
UNUSABLE_BACKOFF_S = (300, 900, 1800, 3600)  # consecutive replies with no usable note: 5, 15, 30, then 60 min
# The desktop, the taskbar and the shell's own surfaces: "no window" (the desktop's rect is the whole monitor, so a
# grab would send every window on it). These classes always; the shell's CoreWindow (Start, search) only in a shell
# process; these processes whatever the class.
SHELL_CLASSES = frozenset(c.casefold() for c in ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"))
SHELL_APPS = frozenset(("searchhost.exe", "startmenuexperiencehost.exe", "shellexperiencehost.exe", "lockapp.exe",
                        "searchapp.exe"))
SHELL_PROCESSES = SHELL_APPS | {"explorer.exe"}
CORE_WINDOW = "windows.ui.core.corewindow"
OVERLAY = 0x80000 | 0x20  # WS_EX_LAYERED | WS_EX_TRANSPARENT: a click-through overlay


@lru_cache(maxsize=256)
def _compile(pattern: str) -> "re.Pattern[str] | None":
    if not isinstance(pattern, str) or not pattern.strip():
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


_TITLES, _PROCESSES = _compile(DENY_TITLES), _compile(DENY_PROCESSES)


def validate_patterns(patterns: Iterable[str]) -> list[str]:
    """The patterns that are not valid regexes (blank ones are fine: they are just ignored)."""
    return [p for p in patterns if not isinstance(p, str) or (p.strip() and _compile(p) is None)]


def _hit(text: str, defaults: "re.Pattern[str]", extra: Iterable[str]) -> bool:
    pats = [defaults, *(_compile(p) for p in extra)]
    return any(r.search(text) for r in pats if r is not None)


def denied(info: WindowInfo, extra_titles: Iterable[str] = (), extra_processes: Iterable[str] = ()) -> bool:
    """True for a window she must not look at; an unreadable process (OpenProcess denied) counts as denied."""
    if not info.process:
        return True
    return _hit(info.title, _TITLES, extra_titles) or _hit(info.process, _PROCESSES, extra_processes)


def is_shell_surface(info: WindowInfo) -> bool:
    cls, proc = info.cls.casefold(), info.process.casefold()
    return cls in SHELL_CLASSES or proc in SHELL_APPS or (cls == CORE_WINDOW and proc in SHELL_PROCESSES)


def window_refusal(info: WindowInfo | None, extra_titles: Iterable[str] = (),
                   extra_processes: Iterable[str] = ()) -> str | None:
    """Why the window itself is never looked at (None: it may be): no window (none, or a shell surface), minimized,
    her own, an unreadable process, a click-through overlay, or denylisted."""
    if info is None or is_shell_surface(info):
        return "no window"
    if info.minimized:
        return "minimized"
    if info.is_riya:
        return "riya window"
    if not info.process:
        return "unknown process"
    if info.ex_style & OVERLAY == OVERLAY:
        return "overlay"
    if denied(info, extra_titles, extra_processes):
        return "denylisted"
    return None


class GlancePolicy:
    def __init__(self, clock: Clock, *, cap: int = 150, min_gap_s: float = 45, settle_s: float = 4,
                 refresh_s: float = 300, mad_threshold: float = 6 / 255):
        self.clock, self.cap, self.min_gap_s = clock, cap, min_gap_s
        self.settle_s, self.refresh_s, self.mad_threshold = settle_s, refresh_s, mad_threshold
        self._prev = clock.now()
        self._day, self._count = self._today(self._prev), 0
        self._last_at: float | None = None
        self._last_key: tuple[int, str] | None = None
        self._last_thumb: bytes | None = None
        self._seen_key: tuple[int, str] | None = None
        self._seen_at = self._prev
        self._until: float | None = None  # backoff ends
        self._unusable = 0  # replies with no usable note in a row

    @staticmethod
    def _today(t: float):
        return datetime.fromtimestamp(t).date()

    def _now(self) -> float:
        """The clock, with timers shifted when it jumped back so elapsed times never shrink or go negative."""
        now = self.clock.now()
        if now < self._prev:
            d = now - self._prev
            self._seen_at += d
            self._last_at = None if self._last_at is None else self._last_at + d
            self._until = None if self._until is None else self._until + d
        self._prev = now
        return now

    def _roll(self, now: float) -> None:
        today = self._today(now)
        if today > self._day:  # a clock jumping back never gives the cap back
            self._day, self._count = today, 0

    def glances_today(self) -> int:
        self._roll(self._now())
        return self._count

    def today(self) -> tuple[str, int]:
        """(local day as ISO date, glances counted on it), for saving across restarts."""
        self._roll(self._now())
        return self._day.isoformat(), self._count

    def restore(self, day, count) -> None:
        """A saved today(): its count carries over only when it was saved today; anything else is ignored."""
        self._roll(self._now())
        if day == self._day.isoformat() and isinstance(count, int) and not isinstance(count, bool) and count > 0:
            self._count = max(self._count, count)

    def allowed(self, *, info: WindowInfo | None, presence: str, eyes_enabled: bool, dont_look: bool, online_ok: bool,
                extra_titles: Iterable[str] = (), extra_processes: Iterable[str] = (), force: bool = False) -> str | None:
        """None when she may look, else why not. `force` (a screen question) skips only min gap and backoff."""
        now = self._now()
        self._roll(now)
        if not eyes_enabled:
            return "eyes off"
        if dont_look:
            return "don't look"
        if presence not in _LOOKING:
            return f"presence {presence}"
        if why := window_refusal(info, extra_titles, extra_processes):
            return why
        if not online_ok:
            return "online model unavailable"
        if self._count >= self.cap:
            return "daily cap"
        if not force and self._until is not None and now < self._until:
            return "backoff"
        if not force and self._last_at is not None and now - self._last_at < self.min_gap_s:
            return "min gap"
        return None

    def wanted(self, info: WindowInfo | None, thumb: bytes | None) -> bool:
        """A new window that has settled, or the same one after refresh_s once its thumbnail changed."""
        now = self._now()
        if info is None:
            self._seen_key = None
            return False
        key = (info.hwnd, info.title)
        if key != self._seen_key:
            self._seen_key, self._seen_at = key, now
        if key != self._last_key:
            return now - self._seen_at >= self.settle_s
        if not thumb or not self._last_thumb or self._last_at is None:  # unknown: don't look
            return False
        return now - self._last_at >= self.refresh_s and mad(thumb, self._last_thumb) > self.mad_threshold

    def record(self, info: WindowInfo, thumb: bytes | None) -> None:
        """A glance happened."""
        now = self._now()
        self._roll(now)
        self._count += 1
        self._last_at, self._last_key, self._last_thumb = now, (info.hwnd, info.title), thumb

    def sent(self) -> None:
        """A vision request went out but brought back no note (or was dropped): it still counts, and the min gap
        runs from it; the window isn't marked as seen."""
        now = self._now()
        self._roll(now)
        self._count += 1
        self._last_at = now

    def unusable(self) -> float:
        """Another reply with no usable note (unparseable or empty): backs off 5, 15, 30, then 60 min; returns it."""
        seconds = UNUSABLE_BACKOFF_S[min(self._unusable, len(UNUSABLE_BACKOFF_S) - 1)]
        self._unusable += 1
        self.backoff(seconds)
        return seconds

    def usable(self) -> None:
        """A parsed note: the next unusable reply backs off 5 min again."""
        self._unusable = 0

    def backoff(self, seconds: float) -> None:
        """No unforced glances for `seconds` (a forced one may try; failing re-arms this); never shortens a longer backoff."""
        now = self._now()
        if seconds > 0:
            self._until = max(self._until or 0.0, now + seconds)
