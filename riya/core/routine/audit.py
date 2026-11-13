"""The audit: each said statement is scored against what her sensors later saw (spec §4.6), and the honesty numbers
are read from the scores in two readings (MORPHEUS step 5b): the sensor reading, and the user reading, in which a
dispute counts false. The lifetime tally holds integers per kind and nothing about the user.

What each kind's row carries in `value` (the facade writes it; the audit reads it):
- S1: `t0` (the stretch start), `coding` (bool).
- S2, S4: `predictor`, `bucket`, `obs_id`, `x` (the observed value), `direction` (early/late or shorter/longer).
- S3: `back_at` (the guessed return time) and `left_at` (when they left).
- S5: `predictor`, `bucket`, `median_s`, and for a hunch `lo_s` and `hi_s` (the stated interval).
- S6: `predictor`, `bucket`, `left_s` (the guessed remainder).
- S7: `predictor`, `bucket`; a bid hunch also has `band_start` and `band_end`.
- S8: `predictor`, `bucket`, `lo` and `hi` (the pre-change cell's 50% interval).
A row whose value lacks what its kind needs is unverifiable.
"""
import math
from dataclasses import dataclass
from typing import Callable

from riya.core.routine import statements as st
from riya.core.routine.records import TALLY_KEY, RoutineDB, StatementRow

DISPUTE_S = 120.0
HONESTY_WINDOW_S = 30 * 86400.0
FEW = 10
WARN_N = 20
WARN_RATE = 0.9
TARGET = 0.95

DAY_S = 86400.0
S1_EDGE_S = 600.0
COMPARE_S = 14 * DAY_S
EARLY_LATER = 3
COMPARE_MIN = 4
S5_RUNS = 5
S5_MIN = 3
S7_BANDS = 5
S7_HITS = 3
S7_WINDOW_S = 28 * DAY_S
HUNCH_EXPIRE_S = DAY_S
S8_NEXT = 3
S8_EXPIRE_S = 28 * DAY_S
S3_SLACK_S = 1800.0
S3_SLACK_FRAC = 0.25
S6_SLACK_S = 120.0

_ACTIVE = ("back", "break_end", "gaming_end")
_GONE = ("away", "break")


@dataclass(frozen=True)
class Honesty:
    enabled: bool
    claims_enabled: bool
    claims_true: int
    claims_scored: int
    hunches_right: int
    hunches_scored: int
    unverifiable: int
    expired: int


def _num(v) -> float | None:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nums(value: dict, *keys) -> list[float] | None:
    out = [_num(value.get(k)) for k in keys]
    return None if any(v is None for v in out) else out


def _cell(value: dict) -> tuple[str, str] | None:
    p, b = value.get("predictor"), value.get("bucket")
    return (p, b) if isinstance(p, str) and isinstance(b, str) else None


# --- per-kind scoring ------------------------------------------------------------------------------------------------

def _s1(row: StatementRow, db: RoutineDB, now: float, glances) -> str | None:
    nums = _nums(row.value, "t0")
    if nums is None:
        return "unverifiable"
    t0 = nums[0]
    log = db.presence_between(t0 - S1_EDGE_S, row.at)
    # the window ends at the start of a break still in progress when she spoke (the break itself isn't a miss)
    end, strict = row.at, False
    for at, ev in reversed(log):
        if ev == "break_end":
            break
        if ev == "break":
            end, strict = at, True
            break
    if not any(ev in _ACTIVE and abs(at - t0) <= S1_EDGE_S for at, ev in log):
        return "false"
    if any(ev in _GONE and t0 + S1_EDGE_S < at and (at < end if strict else at <= end) for at, ev in log):
        return "false"
    if row.value.get("coding"):
        seen = glances(t0, end)
        if not seen:
            return "unverifiable"
        if not all(st.code_glance(g) for g in seen):
            return "false"
    return "true"


def _s2(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    cell, nums = _cell(row.value), _nums(row.value, "obs_id", "x")
    direction = row.value.get("direction")
    if cell is None or nums is None or direction not in ("early", "late", "shorter", "longer"):
        return "unverifiable"
    obs_id, x = int(nums[0]), nums[1]
    own = [o for o in db.obs(*cell, row.at - COMPARE_S, row.at + COMPARE_S) if o[0] == obs_id]
    own_at = own[0][1] if own else row.at
    others = [o for o in db.obs(*cell, row.at - COMPARE_S, row.at + COMPARE_S) if o[0] != obs_id]
    later = [o for o in others if o[1] > own_at]
    if now < row.at + COMPARE_S and len(later) < EARLY_LATER:
        return None
    if len(others) < COMPARE_MIN:
        return None if now < row.at + COMPARE_S else "unverifiable"
    vals = sorted(v for _, _, v in others)
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    below = direction in ("early", "shorter")
    return "true" if (x < median if below else x > median) else "false"


def _s5(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    cell, nums = _cell(row.value), _nums(row.value, "median_s")
    if cell is None or nums is None:
        return "unverifiable"
    n = nums[0]
    lo, hi = (_nums(row.value, "lo_s", "hi_s") or [n / 2.0, 2.0 * n]) if not row.claim else [n / 2.0, 2.0 * n]
    runs = [v for _, at, v in db.obs(*cell, row.at, row.at + COMPARE_S) if at > row.at][:S5_RUNS]
    if len(runs) < S5_RUNS and now < row.at + COMPARE_S:
        return None
    if len(runs) < S5_MIN:
        return "expired"
    inside = sum(1 for v in runs if lo <= v <= hi)
    return "true" if 2 * inside >= len(runs) else "false"


def _s7(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    cell = _cell(row.value)
    if cell is None:
        return "unverifiable"
    if "band_start" in row.value or "band_end" in row.value:  # the bid hunch: gaming in that band, that day
        nums = _nums(row.value, "band_start", "band_end")
        if nums is None:
            return "unverifiable"
        start, end = nums
        if now < end:
            return None
        return "true" if any(ev == "gaming" for _, ev in db.presence_between(start, end)) else "false"
    bands = [v for _, at, v in db.obs(*cell, row.at, row.at + S7_WINDOW_S) if at > row.at][:S7_BANDS]
    if len(bands) < S7_BANDS and now < row.at + S7_WINDOW_S:
        return None
    if len(bands) < S7_HITS:
        return "expired"
    return "true" if sum(1 for v in bands if v >= 0.5) >= S7_HITS else "false"


def _s3(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    nums = _nums(row.value, "back_at", "left_at")
    if nums is None:
        return "unverifiable"
    back_at, left_at = nums
    backs = [at for at, ev in db.presence_between(row.at, row.at + HUNCH_EXPIRE_S) if ev == "back" and at > row.at]
    if not backs:
        return "expired" if now >= row.at + HUNCH_EXPIRE_S else None
    slack = max(S3_SLACK_S, S3_SLACK_FRAC * max(0.0, back_at - left_at))
    return "true" if abs(backs[0] - back_at) <= slack else "false"


def _s6(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    cell, nums = _cell(row.value), _nums(row.value, "left_s")
    if cell is None or nums is None:
        return "unverifiable"
    g = nums[0]
    ends = [at for _, at, _ in db.obs(*cell, row.at, row.at + HUNCH_EXPIRE_S) if at > row.at]
    if not ends:
        return "expired" if now >= row.at + HUNCH_EXPIRE_S else None
    r = ends[0] - row.at
    lo, hi = min(0.5 * g, g - S6_SLACK_S), max(2.0 * g, g + S6_SLACK_S)
    return "true" if lo <= r <= hi else "false"


def _s8(row: StatementRow, db: RoutineDB, now: float) -> str | None:
    cell, nums = _cell(row.value), _nums(row.value, "lo", "hi")
    if cell is None or nums is None:
        return "unverifiable"
    lo, hi = nums
    nxt = [v for _, at, v in db.obs(*cell, row.at, row.at + S8_EXPIRE_S) if at > row.at][:S8_NEXT]
    if any(lo <= v <= hi for v in nxt):
        return "false"
    if len(nxt) >= S8_NEXT:
        return "true"
    return "expired" if now >= row.at + S8_EXPIRE_S else None


def score(row: StatementRow, db: RoutineDB, now: float, glances: Callable[[float, float], list]) -> str | None:
    """A verdict for one pending row, or None while it is still pending."""
    if row.kind == "S1":
        if not row.claim:
            return "unverifiable"  # the S1 hunch names no time
        return _s1(row, db, now, glances)
    if row.kind in ("S2", "S4"):
        return _s2(row, db, now)
    if row.kind == "S3":
        return _s3(row, db, now)
    if row.kind == "S5":
        return _s5(row, db, now)
    if row.kind == "S6":
        return _s6(row, db, now)
    if row.kind == "S7":
        return _s7(row, db, now)
    if row.kind == "S8":
        return _s8(row, db, now)
    return "unverifiable"


# --- the tally -------------------------------------------------------------------------------------------------------

def tally(db: RoutineDB) -> dict:
    t = db.kv_get(TALLY_KEY, {})
    return t if isinstance(t, dict) else {}


def _entry(t: dict, kind: str) -> dict:
    e = t.get(kind)
    if not isinstance(e, dict):
        e = {}
    for side in ("claim", "hunch"):
        if not isinstance(e.get(side), dict):
            e[side] = {}
    for k in ("said", "disputed"):
        if not isinstance(e.get(k), int):
            e[k] = 0
    t[kind] = e
    return e


def _bump(db: RoutineDB, kind: str, fn) -> None:
    with db.lock:
        t = tally(db)
        fn(_entry(t, kind))
        db.kv_set(TALLY_KEY, t)


def tally_said(db: RoutineDB, row: StatementRow) -> None:
    _bump(db, row.kind, lambda e: e.__setitem__("said", e["said"] + 1))


def tally_scored(db: RoutineDB, row: StatementRow) -> None:
    def add(e):
        side = e["claim" if row.claim else "hunch"]
        n = side.get(row.verdict)
        side[row.verdict] = (n if isinstance(n, int) else 0) + 1
    _bump(db, row.kind, add)


# --- the audit and disputes ------------------------------------------------------------------------------------------

def audit(db: RoutineDB, now: float, *, frozen: bool, glances: Callable[[float, float], list],
          on_scored: Callable[[StatementRow], None] | None = None) -> int:
    """Scores every due pending row (frozen: expires them all) and returns how many it scored. Each row is re-read
    before its verdict is written, so a row deleted meanwhile (a forget) is skipped."""
    done = 0
    for row in db.pending():
        try:
            verdict = "expired" if frozen else score(row, db, now, glances)
        except (ArithmeticError, TypeError, ValueError):  # a hand-edited row never stops the audit
            verdict = "unverifiable"
        if verdict is None:
            continue
        with db.lock:
            fresh = db.statement(row.id)
            if fresh is None or fresh.verdict != "pending" or not db.set_verdict(row.id, verdict, now):
                continue
            scored = db.statement(row.id)
            if scored is None:
                continue
            tally_scored(db, scored)
        done += 1
        if on_scored is not None:
            on_scored(scored)
    return done


def dispute(db: RoutineDB, statement_id: int) -> bool:
    """The user said it was wrong: final for the user reading; the sensor verdict is still computed."""
    with db.lock:
        row = db.statement(statement_id)
        if row is None or row.disputed or not db.set_disputed(statement_id):
            return False
        _bump(db, row.kind, lambda e: e.__setitem__("disputed", e["disputed"] + 1))
    return True


def honesty(db: RoutineDB, now: float, *, enabled: bool, claims_enabled: bool) -> Honesty:
    """The numbers over statements said in the last HONESTY_WINDOW_S. claims_true and claims_scored are the user
    reading: a disputed claim counts as scored and false whatever the sensors said. Unverifiable and expired rows are
    counted, never folded into a rate."""
    rows = [r for r in db.statements_since(now - HONESTY_WINDOW_S) if r.at >= now - HONESTY_WINDOW_S]
    claims = [r for r in rows if r.claim and r.kind != "unknown"]
    hunches = [r for r in rows if not r.claim and r.kind != "unknown"]
    scored = [r for r in claims if r.disputed or r.verdict in ("true", "false")]
    return Honesty(
        enabled=enabled, claims_enabled=claims_enabled,
        claims_true=sum(1 for r in scored if r.verdict == "true" and not r.disputed),
        claims_scored=len(scored),
        hunches_right=sum(1 for r in hunches if r.verdict == "true"),
        hunches_scored=sum(1 for r in hunches if r.verdict in ("true", "false")),
        unverifiable=sum(1 for r in rows if r.verdict == "unverifiable"),
        expired=sum(1 for r in rows if r.verdict == "expired"),
    )
