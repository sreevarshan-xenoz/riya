"""Permission policy and the confirmation state machine (pure: time comes from the clock handed in).

Only the engine's user turn (`take_turn`) or a body button carrying the exact pending id (`take_button`) can confirm.
Nothing else here accepts text, and the confirmation wording is a template built from the validated snapshot.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .tools import REGISTRY, TIERS
from .types import Call

if TYPE_CHECKING:
    from riya.core.settings import Settings

CONFIRM_S = 60  # a pending confirmation expires after this
PARK_S = 600  # a parked T2 call (the user is gaming) expires after this
EXPIRED_NOTE_S = 300  # a late "yes" this soon after an expiry gets the timed-out line

_YES = frozenset({"yes", "yeah", "do it", "go ahead", "sure", "ok", "okay"})
_NO = frozenset({"no", "nope", "nah", "don't", "dont", "cancel", "stop", "no thanks"})
_JUNK = re.compile(r"[^\w' ]+")


def decide(call: Call | None, settings: Settings) -> Literal["run", "ask", "refuse"]:
    """What to do with a validated call: None (unknown or invalid) or a disabled tool is refused; T2 always asks."""
    if call is None or call.tool not in TIERS or not settings.tools_enabled.get(call.tool, False):
        return "refuse"
    tier = TIERS[call.tool]
    if tier >= 2:
        return "ask"
    return "ask" if tier == 1 and call.tool in settings.tools_ask else "run"


def yes_no(text: str) -> Literal["yes", "no"] | None:
    """The whole turn is a yes-form or a no-form, ignoring punctuation (not the apostrophe) and a leading "riya"."""
    t = " ".join(_JUNK.sub(" ", str(text).replace("’", "'").lower()).split())
    if t.startswith("riya "):
        t = t[5:]
    return "yes" if t in _YES else "no" if t in _NO else None


@dataclass(frozen=True)
class Pending:
    """A confirmation waiting for an answer: `call` is the validated snapshot a yes runs."""
    id: str
    call: Call
    summary: str
    expires_at: float


def confirm_text(p: Pending) -> str:
    """The deterministic question, built only from the snapshot's args (run_command's argv is the stored one)."""
    tool = REGISTRY.get(p.call.tool)
    s = tool.summary(p.call.args) if tool else p.call.tool
    return f"{s[:1].upper()}{s[1:]}? (yes / no)"


class Confirmations:
    """One pending confirmation at a time, a late-yes note after an expiry, and one parked T2 call."""

    def __init__(self, clock):
        self._clock = clock
        self.pending: Pending | None = None
        self._expired: Pending | None = None  # expired via expire(): a "yes" within EXPIRED_NOTE_S is told so
        self._parked: Pending | None = None

    def _new(self, call: Call, summary: str, ttl: float) -> Pending:
        return Pending(uuid.uuid4().hex, call, summary, self._clock.now() + ttl)

    def ask(self, call: Call, summary: str) -> Pending:
        """Raise a confirmation, silently replacing any pending one."""
        self._expired = None
        self.pending = self._new(call, summary, CONFIRM_S)
        return self.pending

    def take_turn(self, text: str):
        """The user's next turn: yes / no / cancelled (anything else) / expired, or None when nothing applies."""
        now, p = self._clock.now(), self.pending
        if p is None:
            late, self._expired = self._expired, None
            if late and now <= late.expires_at + EXPIRED_NOTE_S and yes_no(text) == "yes":
                return "expired", late
            return None
        self.pending = self._expired = None
        if now >= p.expires_at:
            return "expired", p
        a = yes_no(text)
        return (a, p) if a else ("cancelled", p)

    def take_button(self, id: str, yes: bool):
        """A body button for `id`: stale ids are "replaced" (or "expired" if that confirmation timed out)."""
        p = self.pending
        if p is not None and p.id == id:
            self.pending = None
            return ("expired", p) if self._clock.now() >= p.expires_at else ("yes" if yes else "no", p)
        if self._expired is not None and self._expired.id == id:
            p, self._expired = self._expired, None
            return "expired", p
        return "replaced", None

    def expire(self) -> Pending | None:
        """Drop the pending confirmation if it has timed out (remembering it for a late yes) and return it."""
        p = self.pending
        if p is None or self._clock.now() < p.expires_at:
            return None
        self.pending, self._expired = None, p
        return p

    def park(self, call: Call, summary: str) -> None:
        """Hold a T2 call while the user is gaming; a newer park replaces it."""
        self._parked = self._new(call, summary, PARK_S)

    def unpark(self):
        """Gaming ended: ("raised", pending with a fresh 60 s) or ("expired", parked) after 10 min, else None."""
        k, self._parked = self._parked, None
        if k is None:
            return None
        if self._clock.now() >= k.expires_at:
            return "expired", k
        return "raised", self.ask(k.call, k.summary)
