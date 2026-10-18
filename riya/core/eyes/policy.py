"""Glance policy (pure): when Riya may look at the active window, and when she wants to. Doubt means don't look."""
import re
from datetime import datetime
from functools import lru_cache
from typing import Iterable

from riya.core.eyes.capture import WindowInfo, mad
from riya.core.presence import Clock

DENY_TITLES = ("password|1password|bitwarden|keepass|lastpass|bank|banking|paypal|upi|net ?banking|credit card|"
               "inprivate|incognito|private browsing|windows security|credential|sign in to|login")
DENY_PROCESSES = "keepass.*|1password.*|bitwarden.*|credentialuibroker"
_LOOKING = ("focused", "talking")


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

    def allowed(self, *, info: WindowInfo | None, presence: str, eyes_enabled: bool, dont_look: bool, online_ok: bool,
                extra_titles: Iterable[str] = (), extra_processes: Iterable[str] = (), force: bool = False) -> str | None:
        """None when she may look, else why not. `force` (a screen question) skips only the min gap."""
        now = self._now()
        self._roll(now)
        if not eyes_enabled:
            return "eyes off"
        if dont_look:
            return "don't look"
        if presence not in _LOOKING:
            return f"presence {presence}"
        if info is None:
            return "no window"
        if info.minimized:
            return "minimized"
        if info.is_riya:
            return "riya window"
        if not info.process:
            return "unknown process"
        if denied(info, extra_titles, extra_processes):
            return "denylisted"
        if not online_ok:
            return "online model unavailable"
        if self._count >= self.cap:
            return "daily cap"
        if self._until is not None and now < self._until:
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

    def backoff(self, seconds: float) -> None:
        """No glances (forced or not) for `seconds`; never shortens a longer backoff."""
        now = self._now()
        if seconds > 0:
            self._until = max(self._until or 0.0, now + seconds)
