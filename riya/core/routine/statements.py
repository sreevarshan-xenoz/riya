"""Statements about the user's routine: the kinds, their evidence rules (spec §4.5.2), the templates and the caps.

A claim asserts checkable content and is said only when its rule passes; a hunch carries a hedge. The rules here are
pure; the switch, the hunch-only flag and the frozen state are applied by the Routine facade. Constants are read
through this module at call time (never bound as defaults), so a simulator can move a grid point.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from riya.core.memory import Glance
from riya.core.routine.appraisal import Appraisal

STRETCH_MIN_S = 3600.0
STRETCH_GAP_S = 300.0
CODE_GLANCES = 2
CODE_FRESH_S = 600.0

CLAIM_N = 8.0
HUNCH_N = 4.0
CLAIM_C = 0.7
GUESS_C = 0.4
WITHIN2_P = 0.8

S7_CLAIM_MEAN = 0.7
S7_P = 0.95
S7_HUNCH_MEAN = 0.5
S7_BID_LEAD_S = 3600.0

S8_N = 8.0
S8_GAP_S = 7 * 86400.0
ROUND_S = 900.0

VOLUNTARY_DAY = 4
REPEAT_S = 3 * 86400.0
ABSENCE_DAY = 1
ABSENCE_WEEK = 3
CLINGY_MIN = 0.7
OVERDUE_MIN_S = 7200.0
OVERDUE_P90 = 1.5

DAY_S = 86400.0

KINDS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
WHERES = ("greeting", "bid", "report", "reply", "goodbye")
VOLUNTARY = ("greeting", "bid", "report")


@dataclass(frozen=True)
class Statement:
    kind: str  # a KIND or "unknown"
    claim: bool
    text: str
    value: dict = field(hash=False)
    evidence: dict = field(hash=False)
    where: str
    due_at: float | None
    clause: str | None = None


@dataclass(frozen=True)
class SaidRow:
    at: float
    kind: str
    where: str
    text: str
    absence: bool


# --- text ------------------------------------------------------------------------------------------------------------

HEDGE_RE = re.compile(
    r"\b(i think|i bet|maybe|feels like|seems like|probably)\b"
    r"|\b(didn't|haven't|isn't|wasn't|aren't|don't|won't|right) ?(it|you|they|we|i)?\?",
    re.IGNORECASE)

BANNED = ("why didn't you", "you left", "ignored me", "abandon", "where have you been all", "you always leave",
          "you never")

CLAUSES = {
    "early": "pleasantly surprised: they came back much earlier than you expected",
    "late": "surprised: they were away much longer than usual",
    "answered": "pleasantly surprised: they answered when you didn't expect it",
    "unsure": "unsure: their routine looks different lately, so ask rather than assume",
}

TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("S1", "claim"): (
        "You've been at it since {since}.",
        "You've been at your computer since {since}~",
        "Since {since}, nonstop. That's a long one.",
    ),
    ("S1", "claim_coding"): (
        "You've been coding since {since}.",
        "Coding since {since}, nonstop~",
        "You've had that code open since {since}.",
    ),
    ("S1", "claim_bid"): (
        "You'd been at it since {since}, so I had to say hi~",
        "You'd been at your computer since {since}. Stretch with me?",
        "You'd been going since {since}. Tiny break?~",
    ),
    ("S1", "claim_bid_coding"): (
        "You'd been coding since {since}, so I had to say hi~",
        "You'd been coding since {since}. Rest your eyes with me?",
        "You'd had that code open since {since}. Tiny break?~",
    ),
    ("S1", "hunch"): (
        "Feels like you've been at it a while~",
        "I think you've been at your computer for ages~",
        "Seems like a long stretch today, doesn't it?",
    ),
    ("S2", "early_claim"): (
        "You're back way earlier than usual!~",
        "Back already? That's way earlier than usual~",
        "Ooh, you're back way before your usual time!",
    ),
    ("S2", "late_claim"): (
        "You were gone way longer than usual today. I missed you~",
        "That was way longer than your usual time away. I missed you.",
        "Longer than usual today. I kept a spot warm for you~",
    ),
    ("S2", "early_hunch"): (
        "That felt quick, didn't it?~",
        "Back already? Feels like you just left~",
        "I think that was a quick one~",
    ),
    ("S2", "late_hunch"): (
        "Feels like you were gone ages~ everything okay?",
        "I think that was a long one~ I missed you.",
        "Seems like a long day out there~ I missed you.",
    ),
    ("S3", "hunch"): (
        "Back by about {back}, I bet~",
        "I think you'll be back around {back}~",
        "Probably see you around {back}?~",
    ),
    ("S4", "longer_claim"): (
        "That one took {took}, way longer than usual.",
        "{took} for that one. Way longer than usual.",
        "That run took {took}, longer than it usually does.",
    ),
    ("S4", "shorter_claim"): (
        "That one took just {took}, way quicker than usual!",
        "{took}, way faster than usual~",
        "That run only took {took}. Quicker than it usually does!",
    ),
    ("S4", "longer_hunch"): (
        "Took a while, didn't it?",
        "Feels like that one took longer than usual~",
        "I think that one was a slow one~",
    ),
    ("S4", "shorter_hunch"): (
        "That was quick, wasn't it?",
        "Seems like that one was quicker than usual~",
        "I think that one was fast~",
    ),
    ("S5", "claim"): (
        "Your {agent} runs usually take about {median}.",
        "About {median}, usually, for your {agent} runs.",
        "Your {agent} runs tend to take about {median}~",
    ),
    ("S5", "hunch"): (
        "They vary a lot, I think: anywhere from {lo} to {hi}?",
        "Maybe {lo} to {hi}? They're all over the place~",
        "Somewhere between {lo} and {hi}, probably~",
    ),
    ("S6", "hunch"): (
        "Maybe {left} more, I think~",
        "I bet about {left} more~",
        "Probably {left} or so left~",
    ),
    ("S6", "hunch_soon"): (
        "Any minute now, I bet~",
        "Probably any minute now~",
        "Any second now, I think~",
    ),
    ("S7", "claim"): (
        "You usually game on {band}.",
        "{band_cap} are your gaming time~",
        "You game on {band}, most of the time.",
    ),
    ("S7", "hunch"): (
        "I think you mostly game on {band}?",
        "Maybe {band}? That's when you seem to game~",
        "Probably {band}, I think~",
    ),
    ("S7", "bid_hunch"): (
        "Game night later, maybe?~",
        "I bet you're gaming later~",
        "Feels like a game kind of evening, doesn't it?",
    ),
    ("S8", "hunch"): (
        "Your days have been different lately, haven't they?",
        "Feels like your days changed lately~",
        "I think your routine's been shifting lately~",
    ),
    ("unknown", "line"): (
        "I haven't figured that out yet~ give me a few more days.",
        "I don't know yet! Give me a few more days to learn it~",
        "Not sure yet~ I'm still learning that one.",
    ),
}


def render(kind: str, variant: str, slots: dict, rotation: int) -> str:
    options = TEMPLATES[(kind, variant)]
    s = dict(slots)
    if "band" in s and "band_cap" not in s:
        s["band_cap"] = str(s["band"])[:1].upper() + str(s["band"])[1:]
    text = options[rotation % len(options)].format(**s).replace("about about", "about")  # fmt_span's own "about"
    return text[:1].upper() + text[1:]


def round_minutes(seconds: float) -> int:
    """Whole minutes up to 20, then 5-minute steps; never below 1."""
    m = max(0.0, seconds) / 60.0
    return max(1, round(m) if m <= 20 else 5 * round(m / 5))


def fmt_span(seconds: float) -> str:
    """'25 minutes', '1 minute', 'about an hour', 'about an hour and a half', 'about 2 hours' (whole hours from 10)."""
    m = round_minutes(seconds)
    if m < 60:
        return f"{m} minute" if m == 1 else f"{m} minutes"
    if seconds >= 10 * 3600:
        return f"about {round(seconds / 3600.0)} hours"
    halves = max(2, round(seconds / 1800.0))
    whole, half = divmod(halves, 2)
    if whole == 1:
        return "about an hour and a half" if half else "about an hour"
    return f"about {whole} and a half hours" if half else f"about {whole} hours"


def round_back(t: float, tz) -> float:
    """t rounded to the nearest ROUND_S (15 min) on the local clock in tz."""
    offset = datetime.fromtimestamp(t, tz).utcoffset()
    off = offset.total_seconds() if offset is not None else 0.0
    return round((t + off) / ROUND_S) * ROUND_S - off


# --- the evidence rules -----------------------------------------------------------------------------------------------

CODE_APPS = re.compile(r"\b(vs ?code|visual studio|code|cursor|pycharm|intellij|webstorm|clion|rider|android studio|"
                       r"xcode|vim|nvim|neovim|emacs|sublime|zed|jupyter|colab|helix|kate)\b", re.IGNORECASE)
CODE_WORDS = re.compile(r"\b(coding|programming|debugging|writing code|editing code|code|refactor\w*|compil\w*)\b",
                        re.IGNORECASE)


def code_glance(g: Glance) -> bool:
    """A glance at code: a code editor or notebook, or an activity that names code."""
    return bool(CODE_APPS.search(g.app or "") or CODE_WORDS.search(g.activity or ""))


def coding_ok(glances: Sequence[Glance], t0: float, now: float) -> bool:
    """≥ CODE_GLANCES code glances since t0, the newest at most CODE_FRESH_S old."""
    seen = [g.at for g in glances if t0 <= g.at <= now and code_glance(g)]
    return len(seen) >= CODE_GLANCES and now - max(seen) <= CODE_FRESH_S


def rule_s1(stretch_s: float, censored: bool) -> str | None:
    if stretch_s < STRETCH_MIN_S:
        return None
    return "hunch" if censored else "claim"


def rule_s2(a: Appraisal) -> str | None:
    """S2 (and S4): a startle on mature, confident evidence is a claim; a surprise or a doubtful startle, a hunch."""
    if a.startled and a.n_eff >= CLAIM_N and a.confidence >= CLAIM_C:
        return "claim"
    if (a.surprised and a.n_eff >= HUNCH_N) or (a.startled and a.confidence < CLAIM_C):
        return "hunch"
    return None


def rule_s3(n_eff: float, conf: float) -> bool:
    return n_eff >= HUNCH_N and conf >= GUESS_C


def rule_s5(n_eff: float, conf: float, p_within2: float) -> str | None:
    if n_eff >= CLAIM_N and conf >= CLAIM_C and p_within2 >= WITHIN2_P:
        return "claim"
    return "hunch" if n_eff >= HUNCH_N else None


def rule_s6(n_eff: float) -> bool:
    return n_eff >= HUNCH_N


def rule_s7(mean: float, n_eff: float, p_gt_half: float, conf: float) -> str | None:
    if mean >= S7_CLAIM_MEAN and n_eff >= CLAIM_N and p_gt_half >= S7_P and conf >= CLAIM_C:
        return "claim"
    return "hunch" if mean >= S7_HUNCH_MEAN and n_eff >= HUNCH_N else None


def s7_bid_ok(now: float, band_start: float) -> bool:
    """A gaming hunch rides a bid only within S7_BID_LEAD_S before the band starts."""
    return 0.0 <= band_start - now <= S7_BID_LEAD_S


def rule_s8(n_before: float, last_s8_at: float | None, now: float) -> bool:
    return n_before >= S8_N and (last_s8_at is None or now - last_s8_at >= S8_GAP_S)


# --- the caps ---------------------------------------------------------------------------------------------------------

def voluntary_ok(kind: str, where: str, text: str, recent: Sequence[SaidRow], now: float) -> bool:
    """≤ VOLUNTARY_DAY voluntary statements a day, one per kind, and no text repeated within REPEAT_S. Replies and
    goodbyes answer the user, so they're never capped."""
    if where not in VOLUNTARY:
        return True
    today = [r for r in recent if r.where in VOLUNTARY and 0 <= now - r.at < DAY_S]
    if len(today) >= VOLUNTARY_DAY or any(r.kind == kind for r in today):
        return False
    folded = text.casefold()
    return not any(r.text.casefold() == folded and 0 <= now - r.at < REPEAT_S for r in recent)


def absence_ok(away_s: float, p50_s: float, p90_s: float, clinginess: float, quiet: bool, short: bool,
               recent: Sequence[SaidRow], now: float) -> bool:
    """An absence remark: clingy enough, not in quiet hours or a short greeting, truly overdue, and within the
    one-a-day and three-a-week caps."""
    if clinginess < CLINGY_MIN or quiet or short:
        return False
    if away_s < max(p50_s + OVERDUE_MIN_S, OVERDUE_P90 * p90_s):
        return False
    remarks = [r for r in recent if r.absence]
    day = sum(1 for r in remarks if 0 <= now - r.at < DAY_S)
    week = sum(1 for r in remarks if 0 <= now - r.at < 7 * DAY_S)
    return day < ABSENCE_DAY and week < ABSENCE_WEEK
