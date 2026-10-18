"""Initiative: decides when she speaks first. A greeting when the user returns, a silent bid on a break, never more."""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from riya.core.memory import Thread
from riya.core.presence import Clock
from riya.core.settings import Settings, in_quiet_hours

GREET_MIN_AWAY_S = 7200
BID_MIN_GAP_S = 5400
LONG_STRETCH_S = 7200
THREAD_COOLDOWN_S = 86400


@dataclass
class Action:
    kind: Literal["greet", "bid"]
    spoken: bool
    thread: Thread | None
    reason: str


class Initiative:
    def __init__(self, clock: Clock, get_settings: Callable[[], Settings]):
        self.clock, self.get_settings = clock, get_settings
        self.last_bid_at: float | None = None

    def _eligible(self, threads: list[Thread]) -> Thread | None:
        now = self.clock.now()
        return next(
            (t for t in threads if t.last_raised_at is None or now - t.last_raised_at >= THREAD_COOLDOWN_S), None
        )

    def on_came_back(self, away_s: float, due_threads: list[Thread]) -> Action | None:
        if not self.get_settings().may_start_conversations or away_s < GREET_MIN_AWAY_S:
            return None
        return Action("greet", True, self._eligible(due_threads), f"back after {away_s / 3600:.1f} h away")

    def on_break(self, state: str, work_stretch_s: float, due_threads: list[Thread]) -> Action | None:
        s = self.get_settings()
        now = self.clock.now()
        if state != "break" or not s.may_start_conversations:
            return None
        if in_quiet_hours(s.quiet_hours, datetime.fromtimestamp(now)):
            return None
        if self.last_bid_at is not None and now - self.last_bid_at < BID_MIN_GAP_S:
            return None
        thread = self._eligible(due_threads)
        if thread is None and work_stretch_s <= LONG_STRETCH_S:
            return None
        self.last_bid_at = now
        reason = "due thread" if thread else f"worked {work_stretch_s / 3600:.1f} h straight"
        return Action("bid", False, thread, reason)
