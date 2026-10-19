"""Pure reminder delivery rules and templated lines (the engine calls these; nothing here touches a clock or a db).

Times are epoch seconds and the zone is passed in. Reminder text is the user's own data: it is shown back only inside
these templates, control characters stripped, cut to TEXT_MAX and escaped like every quoted value.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from .records import MAX_ACTIVE, Reminder  # noqa: F401  (MAX_ACTIVE is part of this module's surface)

LATE_S = 120  # later than this past due_at, the line says when it was due
REMIND_GAP_S = 20  # at most one spoken reminder line per this many seconds
TEXT_MAX = 80
SHOWN_MAX = 4  # a merged line names at most this many, then "and N more"
LINE_MAX = 300  # a merged line is never longer than this: it names fewer reminders instead


def delivery(state: str, restricted: bool) -> Literal["speak", "bubble", "wait"]:
    """How a due reminder goes out. Away: wait (the came-back greeting carries it). Gaming, or quiet hours / may not
    start conversations (`restricted`): a bubble at once. Otherwise she speaks it. Never "drop"."""
    if state == "away":
        return "wait"
    return "bubble" if state == "gaming" or restricted else "speak"


def gap_ok(last_spoken_at: float | None, now: float) -> bool:
    """True when another spoken reminder line may go out (a clock that stepped back doesn't block it forever)."""
    return last_spoken_at is None or not 0 <= now - last_spoken_at < REMIND_GAP_S


def _local(t: float, tz) -> datetime:
    return datetime.fromtimestamp(t, tz) if tz else datetime.fromtimestamp(t).astimezone()


def fmt_clock(t: float, tz=None) -> str:
    """'5:00 pm'."""
    d = _local(t, tz)
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


def fmt_when(t: float, now: float, tz=None) -> str:
    """'5:00 pm', '5:00 am tomorrow', or '9:00 am on Fri 24 Oct' relative to now's local date."""
    d, n = _local(t, tz), _local(now, tz)
    days = (d.date() - n.date()).days
    if days == 0:
        return fmt_clock(t, tz)
    if days == 1:
        return f"{fmt_clock(t, tz)} tomorrow"
    return f"{fmt_clock(t, tz)} on {d:%a} {d.day} {d:%b}"


def duration_parts(seconds: float) -> tuple[int, str]:
    """(n, unit) for a timer length: 600 -> (10, 'minute'); 5400 -> (90, 'minute'); 45 -> (45, 'second')."""
    s = max(1, round(seconds))
    if s % 3600 == 0:
        return s // 3600, "hour"
    if s % 60 == 0 or s > 3600:
        return round(s / 60), "minute"
    return s, "second"


def _clean(text: str) -> str:
    """One line of the user's text: control characters become spaces, runs collapse, cut to TEXT_MAX, quote-escaped."""
    from .tools import _q  # lazy: tools imports this module
    s = " ".join("".join(" " if ch.isspace() or not ch.isprintable() else ch for ch in str(text)).split())
    if len(s) > TEXT_MAX:
        s = s[:TEXT_MAX - 3] + "..."
    return _q(s, TEXT_MAX)[1:-1]


def _part(r: Reminder, now: float, tz, merged: bool) -> str:
    late = f" (this was due at {fmt_clock(r.due_at, tz)})" if now - r.due_at > LATE_S else ""
    if r.kind == "timer":
        n, unit = duration_parts(r.due_at - r.created_at)
        return (f"your {n}-{unit} timer is done" if merged else f"Your {n}-{unit} timer is done~") + late
    body = _clean(r.text)
    return body + late if merged else f"Hey~ it's time to {body}." + late


def reminder_line(items: list[Reminder], now: float, tz=None) -> str:
    """The templated line for one or more due reminders, merged into one. '' for none."""
    items = sorted(items, key=lambda r: (r.due_at, r.id))
    if not items:
        return ""
    if len(items) == 1:
        return _part(items[0], now, tz, False)
    parts = [_part(r, now, tz, True) for r in items]
    for k in range(min(len(parts), SHOWN_MAX), 0, -1):
        shown = parts[:k] + ([f"and {len(parts) - k} more"] if k < len(parts) else [])
        line = f"Hey~ a few things are due: {'; '.join(shown)}."
        if len(line) <= LINE_MAX:
            break
    return line
