"""v0.12's routine simulator (spec §4.10, §4.12.4): it drives the real RiyaCore with a fake clock over six personas
and measures her routine predictors, statements and honesty against each persona's own truth.

It never reimplements the speaking gates: greetings, bids, reports and quiet hours run as in live use, and what she
said is read back from her statements table (and the audit's on_scored), never re-decided here. Constants are tuned on
dev seeds 0-19 only (--grid); the held-out run (seeds 100-159) refuses any override and must match the dev record.
Offline and CPU only.

    python -m evaluation.routine_sim --seeds dev --jobs 6 --out scratch_runs/routine_sim/dev.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.hands.effects import FakeEffects
from riya.core.hands.net import FakeHttp
from riya.core.presence import FakeIdleSource
from riya.core.protocol import Say
from riya.core.routine import appraisal as ap
from riya.core.routine import predictors as pr
from riya.core.routine import statements as st
from riya.core.store import KV, open_db

DEV_SEEDS = range(0, 20)
HELDOUT_SEEDS = range(100, 160)
SIM_TZ = timezone(timedelta(hours=5, minutes=30))
SIM_START = datetime(2026, 11, 2, tzinfo=SIM_TZ).timestamp()  # a Monday
DAY = 86400.0
WEEK = 7 * DAY
MIN = 60.0
H = 3600.0
TICK_S = 60.0
QUIET = (23 * 60 + 30, 8 * 60)  # the default quiet hours, 23:30-08:00, for A9's own check
REPO = Path(__file__).resolve().parent.parent
DEV_RECORD = REPO / "evaluation" / "routine_sim_dev.json"
HELDOUT_RECORD = REPO / "evaluation" / "routine_sim_heldout.json"

GRID: dict[str, tuple[float, ...]] = {"SURPRISE_FACTOR": (1.5, 2.0, 3.0), "STARTLE_FACTOR": (4.0, 6.0, 8.0),
                                      "CUSUM_H": (3.0, 3.5, 4.0, 6.0), "CLAIM_C": (0.6, 0.7, 0.8)}
GRID_ORDER: tuple[tuple[str, float], ...] = (("CLAIM_C", 0.8), ("CUSUM_H", 3.0), ("STARTLE_FACTOR", 8.0),
                                             ("SURPRISE_FACTOR", 3.0), ("SURPRISE_FACTOR", 1.5), ("CLAIM_C", 0.6))
_HOMES = {"SURPRISE_FACTOR": ap, "STARTLE_FACTOR": ap, "CUSUM_H": ap, "CLAIM_C": st}

FIVE = ("office", "night_owl", "chaotic", "shift", "vacation")
STATIONARY = ("office", "night_owl", "vacation")
SHIFT_DAY = 14
SHIFT_CELLS = ("P1:afternoon/weekday", "P1:afternoon", "P4:afternoon/weekday", "P4:evening/weekday")


def current_constants() -> dict[str, float]:
    return {k: float(getattr(_HOMES[k], k)) for k in GRID}


def admissible(constants: dict) -> bool:
    """Resolution 10: three startles just over the threshold must raise a change (A1)."""
    c = {**current_constants(), **constants}
    return 3 * (math.log(c["STARTLE_FACTOR"]) - ap.CUSUM_DRIFT) > c["CUSUM_H"]


@contextmanager
def _constants(constants: dict | None):
    old = {k: getattr(_HOMES[k], k) for k in (constants or {})}
    try:
        for k, v in (constants or {}).items():
            setattr(_HOMES[k], k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(_HOMES[k], k, v)


# ---- personas ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Persona:
    name: str
    shift_h: float = 0.0  # every clock time moves by this much (night_owl: 4 h)
    sigma: float = 0.2  # spread of absences (log space)
    run_median_s: float = 480.0
    run_sigma: float = 0.4
    game_weekday_evening: float = 0.75
    game_weekend_afternoon: float = 0.6
    chaotic_gaming: bool = False
    anomalies: bool = True  # 5% of P1 absences and P3 runs x4 or /4
    p_d: float = 0.75  # disputes a false claim with this probability (0.02 for a true one)
    asks: bool = False  # asks the three routine questions, one weekday a week each
    answer_by_band: tuple = (("morning", 0.6), ("afternoon", 0.3), ("evening", 0.8), ("night", 0.5))
    regime: str = "office"  # office, shift, vacation or late: what happens after week 2


PERSONAS: dict[str, Persona] = {
    "office": Persona("office", p_d=0.8, asks=True),
    "night_owl": Persona("night_owl", shift_h=4.0, p_d=0.7, asks=True),
    "chaotic": Persona("chaotic", sigma=1.2, run_sigma=1.2, chaotic_gaming=True, anomalies=False, p_d=0.6),
    "shift": Persona("shift", p_d=0.9, regime="shift"),
    "vacation": Persona("vacation", p_d=0.65, regime="vacation"),
    "late": Persona("late", p_d=0.8, regime="late"),
}


def band_of(t: float) -> tuple[str, str]:
    """(band, daytype) like the facade's: the night band belongs to the date it started."""
    d = datetime.fromtimestamp(t, SIM_TZ)
    h = d.hour
    band = "night" if h >= 22 or h < 5 else "morning" if h < 12 else "afternoon" if h < 17 else "evening"
    day = d.date() - timedelta(days=1) if band == "night" and h < 5 else d.date()
    return band, "weekend" if day.weekday() >= 5 else "weekday"


def in_quiet(t: float) -> bool:
    d = datetime.fromtimestamp(t, SIM_TZ)
    m = d.hour * 60 + d.minute
    return m >= QUIET[0] or m < QUIET[1]


@dataclass
class Segment:
    start: float
    end: float
    kind: str  # "present" or "away"
    gaming: tuple = ()  # (start, end) spans inside a present segment
    work: bool = False
    label: str = ""  # an absence's kind: lunch, evening, night, vacation
    anomaly: bool = False


def _ln(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0.0, sigma))


def _day_plan(p: Persona, day: int, rng: random.Random, sample: bool = False) -> list[tuple]:
    """One day's (start, end, kind, extras) pieces from waking to sleeping, in seconds after the day's midnight; the
    night absence is filled in between days."""
    weekend = day % 7 >= 5
    regime = "base"
    if day >= SHIFT_DAY and p.regime in ("shift", "late"):
        regime = p.regime
    s = p.shift_h * H
    anomaly = lambda: p.anomalies and not sample and rng.random() < 0.05  # noqa: E731
    pieces = []

    def away(start, dur, label):
        a = anomaly()
        if a:
            dur *= rng.choice((4.0, 0.25))
        if regime == "late" and label in ("lunch", "evening"):
            dur *= rng.uniform(3.0, 5.0)
        pieces.append((start, start + dur, "away", {"label": label, "anomaly": a}))
        return start + dur

    def present(start, end, work, games=()):
        if end > start:
            pieces.append((start, end, "present", {"work": work, "gaming": tuple(games)}))
        return max(start, end)

    sig = p.sigma
    if weekend:
        t = 10.5 * H + s + rng.gauss(0, 15 * MIN)
        t = present(t, 13 * H + s + rng.gauss(0, 10 * MIN), False)
        t = away(t, _ln(rng, 60 * MIN, max(sig, 0.25)), "lunch")
        games = []
        if (rng.random() < 0.3) if p.chaotic_gaming else (rng.random() < p.game_weekend_afternoon):
            g0 = 15 * H + s + rng.gauss(0, 10 * MIN)
            games.append((g0, g0 + _ln(rng, 2 * H, 0.2)))
        t = present(t, 19 * H + s + rng.gauss(0, 15 * MIN), False, games)
        t = away(t, _ln(rng, 2 * H, max(sig, 0.3)), "evening")
        sleep = 23.5 * H + s + rng.gauss(0, 20 * MIN)
        present(t, sleep, False)
        return pieces
    start_h, lunch_h, lunch_m = (11.0, 14.0, 120 * MIN) if regime == "shift" else (9.0, 12.5, 45 * MIN)
    t = start_h * H + s + rng.gauss(0, (2 * H) if p.chaotic_gaming else 10 * MIN)
    afternoon_games = []
    t = present(t, lunch_h * H + s + rng.gauss(0, 10 * MIN), True)
    t = away(t, _ln(rng, lunch_m, sig), "lunch")
    if regime == "shift" and rng.random() < p.game_weekday_evening:
        g0 = 16.2 * H + s
        afternoon_games.append((g0, g0 + _ln(rng, 40 * MIN, 0.15)))
    t = present(t, max(t + H, (19.0 if regime == "shift" else 18.0) * H + s + rng.gauss(0, 15 * MIN)), True,
                afternoon_games)
    t = away(t, _ln(rng, 90 * MIN, sig), "evening")
    games = []
    if p.chaotic_gaming:
        if rng.random() < 0.3:
            g0 = t + rng.uniform(0, 2 * H)
            games.append((g0, g0 + _ln(rng, H, 0.4)))
    elif regime != "shift" and rng.random() < p.game_weekday_evening:
        g0 = 20 * H + s + rng.gauss(0, 10 * MIN)
        games.append((g0, g0 + _ln(rng, 90 * MIN, 0.15)))
    sleep = 23 * H + s + rng.gauss(0, 20 * MIN)
    present(t, max(sleep, games[-1][1] + 10 * MIN if games else sleep), False, games)
    return pieces


def timeline(p: Persona, rng: random.Random, weeks: int, sample: bool = False) -> list[Segment]:
    """The persona's true weeks as contiguous segments (present and away), the night absences filled in."""
    segs: list[Segment] = []
    for day in range(weeks * 7):
        base = SIM_START + day * DAY
        for a, b, kind, ex in _day_plan(p, day, rng, sample):
            a, b = base + a, base + b
            if segs and a > segs[-1].end:  # the night: away from the last piece's end to this morning
                segs.append(Segment(segs[-1].end, a, "away", label="night"))
            if segs and a < segs[-1].end:
                a = segs[-1].end
            if b <= a:
                continue
            if kind == "away":
                segs.append(Segment(a, b, "away", label=ex["label"], anomaly=ex["anomaly"]))
            else:
                segs.append(Segment(a, b, "present", gaming=tuple((base + g0, base + g1) for g0, g1 in ex["gaming"]),
                                    work=ex["work"]))
    if p.regime == "vacation":  # week 3: away from Monday 18:00 to Friday 09:00
        v0, v1 = SIM_START + 14 * DAY + 18 * H, SIM_START + 18 * DAY + 9 * H
        kept = [s for s in segs if s.end <= v0 or s.start >= v1]
        segs = sorted(kept + [Segment(v0, v1, "away", label="vacation")], key=lambda s: s.start)
        out = []
        for seg in segs:
            if out and seg.start > out[-1].end:
                out.append(Segment(out[-1].end, seg.start, "away", label="night"))
            out.append(seg)
        segs = out
    merged: list[Segment] = []
    for seg in segs:  # two absences in a row are one absence
        if merged and merged[-1].kind == seg.kind == "away" and merged[-1].end >= seg.start:
            merged[-1] = replace(merged[-1], end=seg.end, anomaly=merged[-1].anomaly or seg.anomaly)
        else:
            merged.append(seg)
    return merged


def _truth_tables(p: Persona) -> list[tuple[float, dict]]:
    """Per regime (from when it starts): the true median of each P1 cell and P3 bucket, and each P4 cell's true gaming
    probability, estimated from 300 sampled days of the persona's own generator (no anomalies)."""
    out = []
    regimes = [(SIM_START, 0)]
    if p.regime in ("shift", "late"):
        regimes.append((SIM_START + SHIFT_DAY * DAY, SHIFT_DAY))
    for start, first_day in regimes:
        rng = random.Random(f"truth:{p.name}:{first_day}")
        absences: dict[str, list[float]] = {}
        gamed: dict[str, list[float]] = {}
        for k in range(300):
            day = first_day + k % 14  # two weeks of the regime, sampled over and over
            pieces = _day_plan(p, day, rng, sample=True)
            base = SIM_START + day * DAY
            bands_gamed = {}
            for a, b, kind, ex in pieces:
                if kind == "away":
                    band, dt = band_of(base + a)
                    for key in (f"{band}/{dt}", band):
                        absences.setdefault(key, []).append(b - a)
                for g0, g1 in ex.get("gaming", ()):
                    if g1 - g0 >= pr.GAMING_SPAN_S:
                        bands_gamed[band_of(base + g0)] = True
            daytype = "weekend" if day % 7 >= 5 else "weekday"
            for band in ("morning", "afternoon", "evening", "night"):
                hit = 1.0 if bands_gamed.get((band, daytype)) else 0.0
                gamed.setdefault(f"{band}/{daytype}", []).append(hit)
                gamed.setdefault(band, []).append(hit)
        runs = [_ln(rng, p.run_median_s, p.run_sigma) for _ in range(2000)]
        table = {("P1", k): statistics.median(v) for k, v in absences.items()}
        table.update({("P3", b): statistics.median(runs) for b in ("claude:riya", "claude:*")})
        table.update({("P4", k): sum(v) / len(v) for k, v in gamed.items()})
        out.append((start, table))
    return out


def _truth_at(truth: dict, at: float) -> dict:
    table = truth["tables"][0][1]
    for start, t in truth["tables"]:
        if at >= start:
            table = t
    return table


def oracle(row: dict, persona: Persona, truth: dict) -> bool:
    """Whether a statement was true by the persona's own truth (Resolution 33), never by her records."""
    kind, v = row["kind"], row.get("value") or {}
    if not row.get("claim"):
        return row.get("sensor") == "true"  # a hunch: its audit rule on the true outcome
    if kind == "S1":
        t0, end = v.get("t0"), truth.get("window_end", {}).get(row.get("id"), row["at"])
        if t0 is None:
            return False
        starts = [s.start for s in truth["segments"] if s.kind == "present"] + truth.get("break_ends", [])
        if not any(abs(s - t0) <= 600 for s in starts):
            return False
        gone = [(s.start, s.end) for s in truth["segments"] if s.kind == "away"] + truth.get("breaks", [])
        return not any(a < end and b > t0 + 600 for a, b in gone)
    table = _truth_at(truth, row["at"])
    key = (v.get("predictor"), v.get("bucket"))
    if kind in ("S2", "S4"):
        med = table.get(key)
        if med is None or not isinstance(v.get("x"), (int, float)):
            return False
        return v["x"] < med if v.get("direction") in ("early", "shorter") else v["x"] > med
    if kind == "S5":
        med, n = table.get(key), v.get("median_s")
        return med is not None and isinstance(n, (int, float)) and n / 2 <= med <= 2 * n
    if kind == "S7":
        p = table.get(key)
        return p is not None and p >= 0.5
    return row.get("sensor") == "true"


# ---- the run -----------------------------------------------------------------------------------------------------------

class SimModel:
    """A local model that answers with one short line that names no time, number or habit."""
    name = "sim"
    LINES = ("Okay~", "Hehe, alright.", "Mm, I hear you~", "That sounds nice.")

    def __init__(self):
        self.n, self.finish_reason = 0, None

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.n += 1
        yield self.LINES[self.n % len(self.LINES)]

    def call_tools(self, messages, tools, timeout_s=8.0, max_tokens=200):
        from riya.core.models import ToolReply
        return ToolReply([], None)


@dataclass
class RunResult:
    persona: str
    seed: int
    variant: str
    statements: list[dict] = field(default_factory=list)
    coverage: list[tuple[str, int, bool]] = field(default_factory=list)
    surprises: list[tuple[str, float, bool]] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    shift_obs_to_change: int | None = None
    absences: list[dict] = field(default_factory=list)
    chances: list[dict] = field(default_factory=list)
    routine_ms: float = 0.0
    ticks: int = 0
    wall_s: float = 0.0
    cells_seen: int = 0


QUESTIONS = ((1, 15, "how long have i been at it?"), (2, 11, "how long do my riya runs take?"),
             (3, 16, "when do i usually game?"))
DISPUTE = "that's wrong"


def _week(at: float) -> int:
    return int((at - SIM_START) // WEEK)


def simulate(persona: str, seed: int, *, weeks: int = 6, evidence: bool = True, clinginess: float | None = None,
             constants: dict | None = None) -> RunResult:
    variant = "plant" if not evidence else "clingy" if clinginess is not None else "full"
    riya_log = logging.getLogger("riya")
    level, real_observe = riya_log.level, pr.observe_lognormal
    riya_log.setLevel(logging.CRITICAL)  # a run logs thousands of expected warnings (her model is a stub)
    try:
        with _constants(constants):
            return _simulate(PERSONAS[persona], seed, weeks, evidence, clinginess, variant)
    finally:
        riya_log.setLevel(level)
        pr.observe_lognormal = real_observe


def _simulate(p: Persona, seed: int, weeks: int, evidence: bool, clinginess: float | None, variant: str) -> RunResult:
    t_wall = time.perf_counter()
    rng = random.Random(f"{p.name}:{seed}")
    segs = timeline(p, rng, weeks)
    truth = {"segments": segs, "tables": _truth_tables(p), "breaks": [], "break_ends": [], "window_end": {}}
    res = RunResult(p.name, seed, variant)

    clock = FakeClock(segs[0].start - 5 * MIN)
    db = open_db(":memory:")
    KV(db).set("settings", {"routine_enabled": True, "onboarded": True})
    src, emitted = FakeIdleSource(), []
    core = RiyaCore(db=db, embedder=HashEmbedder(), clock=clock, idle_source=src, local=SimModel(), online=None,
                    emit=emitted.append, effects=FakeEffects(), http=FakeHttp({}), tz=SIM_TZ,
                    riya_dir=Path("/nonexistent-riya-sim"))
    core.twin_rng, core.game_rng = random.Random(seed), random.Random(seed + 1)
    core.routine.evidence = evidence
    if clinginess is not None:
        core.learn.set_dials({"clinginess": clinginess}, "routine_sim", "explicit", clock.now())
    scored: dict[int, dict] = {}
    core.routine.on_scored = lambda row: scored.__setitem__(row.id, _row(row))

    injected: set[float] = set()
    shift_t = SIM_START + SHIFT_DAY * DAY
    obs_after_shift = {"n": 0}

    # instruments: what the predictors saw, and the cells that entered unsure
    real_observe = pr.observe_lognormal

    def observe(cell, prior, x_s, now):
        c = pr.lognormal_decayed(cell, prior, now)
        pred = pr.predictive(c, prior)
        name = "P1" if prior is pr.P1_PRIOR else "P3"
        q = pred.cdf_log(math.log(x_s))
        step = real_observe(cell, prior, x_s, now)
        if now >= SIM_START + 2 * WEEK:
            res.coverage.append((name, _week(now), 0.1 <= q <= 0.9))
        res.surprises.append((name, step.surprise, x_s in injected))
        return step
    pr.observe_lognormal = observe
    real_unsure = core.routine._unsure

    def unsure(cell, step, now):
        entered = real_unsure(cell, step, now)
        res.changes.append({"cell": cell, "at": now, "change": bool(step.change), "entered": entered})
        if p.regime == "shift" and now >= shift_t and cell in SHIFT_CELLS and res.shift_obs_to_change is None:
            obs_after_shift["n"] += 1
            if step.change:
                res.shift_obs_to_change = obs_after_shift["n"]
        return entered
    core.routine._unsure = unsure
    real_feel = core.feelings.on_appraisal

    def feel(a, cl):
        before = core.feelings.state.concern
        bonus_before = core.feelings._bonus
        real_feel(a, cl)
        if a.topic == "return":
            bonus = core.feelings._bonus[0] if core.feelings._bonus is not bonus_before and core.feelings._bonus else 0.0
            res.absences.append({"at": core.clock.now(), "remark": False, "quiet": in_quiet(core.clock.now()),
                                 "missing_bonus": bonus, "concern_bump": core.feelings.state.concern - before})
    core.feelings.on_appraisal = feel
    real_step = core._routine_step

    def timed_step(now, events):
        t = time.perf_counter()
        try:
            return real_step(now, events)
        finally:
            res.routine_ms += (time.perf_counter() - t) * 1000.0
    core._routine_step = timed_step

    state = {"seen": 0, "said": set(), "last_row": 0, "pending": None}

    def take_speech():
        for m in emitted[state["seen"]:]:
            if isinstance(m, Say) and m.utterance_id not in state["said"]:
                state["said"].add(m.utterance_id)
                if m.kind in ("greet", "bid") or (m.kind == "report" and m.spoken):
                    res.chances.append({"at": clock.now(), "week": _week(clock.now()), "kind": m.kind})
                if m.kind == "greet" and rng.random() < 0.8:
                    state["pending"] = (clock.now() + MIN, "hey riya!")
                elif m.kind == "bid":
                    band = band_of(clock.now())[0]
                    if rng.random() < dict(p.answer_by_band)[band]:
                        state["pending"] = (clock.now() + 2 * MIN, "haha okay, a short break")
        state["seen"] = len(emitted)
        new = [r for r in core.routine.db.statements_since(0) if r.id > state["last_row"]]
        for r in new:
            state["last_row"] = max(state["last_row"], r.id)
            row = _row(r)
            if r.kind == "S1":
                truth["window_end"][r.id] = _s1_end(truth, r.at)
            false = r.claim and not oracle({**row, "id": r.id}, p, truth)
            if rng.random() < (p.p_d if false else 0.02):
                clock.advance(20)
                say(DISPUTE)

    def say(text):
        list(core.reply(text))
        take_speech()

    def tick():
        it = core.tick()
        res.ticks += 1
        if it:
            list(it)
        take_speech()

    runs = _run_schedule(p, segs, rng, injected)
    ri = 0
    asked = set()
    for seg in segs:
        if seg.kind == "away":
            src.is_locked, src.is_fullscreen = True, False
            clock.t = max(clock.t, seg.start)
            tick()
            clock.t = seg.end
            src.is_locked, src.idle = False, 0.0
            tick()
            continue
        breaks = _breaks(seg, rng) if seg.work else []
        for b0, b1 in breaks:
            truth["breaks"].append((b0 + 120, b1))
            truth["break_ends"].append(b1)
        t = max(clock.t + TICK_S, seg.start)
        while t < seg.end:
            clock.t = t
            while ri < len(runs) and runs[ri][0] <= t:
                core.agent_inbox.put(runs[ri][1])
                ri += 1
            in_break = next((b for b in breaks if b[0] <= t < b[1]), None)
            src.idle = (t - in_break[0]) if in_break else 0.0
            src.is_fullscreen = any(g0 <= t < g1 for g0, g1 in seg.gaming)
            tick()
            if state["pending"] and t >= state["pending"][0] and not in_break:
                text, state["pending"] = state["pending"][1], None
                say(text)
            if p.asks and not in_break and not src.is_fullscreen:
                d = datetime.fromtimestamp(t, SIM_TZ)
                for wd, hour, q in QUESTIONS:
                    key = (_week(t), wd)
                    if d.weekday() == wd and d.hour == (hour + p.shift_h) % 24 and key not in asked:
                        asked.add(key)
                        clock.advance(5)
                        say(q)
            t = clock.t + TICK_S
    pr.observe_lognormal = real_observe

    rows = {r.id: _row(r) for r in core.routine.db.statements_since(0)}
    rows.update({k: v for k, v in scored.items() if k not in rows})
    for i, row in sorted(rows.items()):
        row["id"] = i
        row["week"] = _week(row["at"])
        row["oracle"] = oracle(row, p, truth)
        res.statements.append(row)
    late_remarks = [r["at"] for r in res.statements if r["kind"] == "S2" and (r.get("value") or {}).get("direction") == "late"]
    for a in res.absences:
        a["remark"] = any(0 <= x - a["at"] <= 30 * MIN for x in late_remarks)
    res.cells_seen = len({c["cell"] for c in res.changes})
    res.wall_s = time.perf_counter() - t_wall
    return res


def _row(r) -> dict:
    return {"kind": r.kind, "claim": r.claim, "where": r.where, "at": r.at, "sensor": r.verdict,
            "disputed": r.disputed, "value": r.value, "text": r.text}


def _s1_end(truth: dict, said_at: float) -> float:
    for b0, b1 in truth["breaks"]:
        if b0 - 120 <= said_at < b1:
            return b0
    return said_at


def _breaks(seg: Segment, rng: random.Random) -> list[tuple[float, float]]:
    out, t = [], seg.start + rng.uniform(60, 100) * MIN
    while t + 10 * MIN < seg.end:
        out.append((t, t + rng.uniform(4, 7) * MIN))
        t += rng.uniform(100, 170) * MIN
    return out


def _run_schedule(p: Persona, segs: list[Segment], rng: random.Random, injected: set) -> list[tuple[float, dict]]:
    """Claude Code turns in the folder "riya" during work, each a prompt then a Stop after its run time."""
    out, sid = [], "simsession01"
    for seg in segs:
        if seg.kind != "present" or not seg.work:
            continue
        t = seg.start + rng.uniform(10, 40) * MIN
        while t < seg.end - 5 * MIN:
            run_s = _ln(rng, p.run_median_s, p.run_sigma)
            if p.anomalies and rng.random() < 0.05:
                run_s *= rng.choice((4.0, 0.25))
                injected.add(run_s)
            out.append((t, {"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/home/sim/riya",
                            "prompt": "go"}))
            out.append((t + run_s, {"session_id": sid, "hook_event_name": "Stop", "cwd": "/home/sim/riya"}))
            t += run_s + rng.uniform(20, 70) * MIN
    for seg in segs:  # the injected absences are labelled by their length
        if seg.anomaly:
            injected.add(seg.end - seg.start)
    return sorted(out, key=lambda e: e[0])


# ---- criteria ----------------------------------------------------------------------------------------------------------

def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    import bisect
    total = 0.0
    for x in pos:
        lo, hi = bisect.bisect_left(neg_sorted, x), bisect.bisect_right(neg_sorted, x)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def _rate(results: list[RunResult]) -> tuple[float | None, int, int]:
    claims = [s for r in results for s in r.statements if s["claim"]]
    true = sum(1 for s in claims if s["oracle"] and not s["disputed"])
    return (true / len(claims) if claims else None), true, len(claims)


def _c(value, threshold, ok: bool, blocking: bool, note: str = "") -> dict:
    return {"value": value, "threshold": threshold, "pass": bool(ok), "blocking": blocking, "note": note}


def criteria(full: list[RunResult], plant: list[RunResult], clingy: list[RunResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    # A3: 80% interval coverage after week 2, per persona (not chaotic) and predictor
    cov = {}
    for r in full:
        if r.persona in ("chaotic", "late"):
            continue
        for name, _, hit in r.coverage:
            cov.setdefault(f"{r.persona}:{name}", []).append(hit)
    rates = {k: round(sum(v) / len(v), 3) for k, v in cov.items() if v}
    ok = bool(rates) and all(0.72 <= v <= 0.90 for v in rates.values())
    out["A3"] = _c(rates, [0.72, 0.90], ok, True, "chaotic and late are not judged")
    # A4: surprise separates the injected anomalies
    pos = [s for r in full if r.persona != "chaotic" for _, s, a in r.surprises if a]
    neg = [s for r in full if r.persona != "chaotic" for _, s, a in r.surprises if not a]
    auc = _auc(pos, neg)
    out["A4"] = _c(None if auc is None else round(auc, 3), 0.85, auc is not None and auc >= 0.85, True)
    # A5: shift's change within 5 observations on >= 90% of seeds; <= 1 false change per cell per 4 weeks elsewhere
    shifts = [r for r in full if r.persona == "shift"]
    hit = sum(1 for r in shifts if r.shift_obs_to_change is not None and r.shift_obs_to_change <= 5)
    frac = hit / len(shifts) if shifts else None
    false = []
    for r in full:
        if r.persona in STATIONARY:
            weeks = max(1.0, (max((c["at"] for c in r.changes), default=SIM_START) - SIM_START) / WEEK)
            n = sum(1 for c in r.changes if c["change"])
            false.append(n / max(1, r.cells_seen) * 4 / weeks)
    mean_false = statistics.mean(false) if false else None
    ok = frac is not None and frac >= 0.9 and mean_false is not None and mean_false <= 1.0
    out["A5"] = _c({"shift_detected": frac, "false_per_cell_4w": None if mean_false is None else round(mean_false, 3)},
                   {"shift_detected": 0.9, "false_per_cell_4w": 1.0}, ok, True)
    # A6: the user-reading true rate of claims, against the oracle
    two = [r for r in full if r.persona in ("office", "night_owl")]
    five = [r for r in full if r.persona in FIVE]
    r2, t2, n2 = _rate(two)
    r5, t5, n5 = _rate(five)
    unver = sum(1 for r in five for s in r.statements if s["claim"] and s["sensor"] in ("unverifiable", "expired"))
    ok = (r2 is None or r2 >= 0.95) and (r5 is None or r5 >= 0.90)
    out["A6"] = _c({"office_night_owl": r2, "n2": n2, "five": r5, "n5": n5, "unverifiable_or_expired": unver},
                   {"office_night_owl": 0.95, "five": 0.90}, ok, True,
                   "no claims at all counts as passing" if not n5 else "")
    # A7: the evidence-off plant must be caught
    rp, tp, np_ = _rate(plant)
    out["A7"] = _c({"rate": rp, "n": np_}, 0.85, rp is not None and rp <= 0.85, True)
    # A8: claim volume (reported)
    office = [r for r in full if r.persona == "office"]
    week1 = sum(1 for r in office for s in r.statements if s["claim"] and s["week"] == 0
                and s["kind"] in ("S2", "S4", "S5", "S7"))
    per_week = statistics.mean([sum(1 for s in r.statements if s["claim"] and 2 <= s["week"] <= 5) / 4
                                for r in office]) if office else 0.0
    chances = statistics.mean([sum(1 for c in r.chances if 2 <= c["week"] <= 5) / 4 for r in office]) if office else 0
    unreachable = chances < 2
    out["A8"] = _c({"week1_claims": week1, "claims_per_week_3_6": round(per_week, 2),
                    "chances_per_week": round(chances, 2)}, {"week1_claims": 0, "claims_per_week_3_6": 2},
                   week1 == 0 and per_week >= 2, False, "unreachable, reported" if unreachable else "")
    # A9: the clinginess caps, on every seed
    bad = []
    for r in full + clingy:
        remarks = sorted(a["at"] for a in r.absences if a["remark"])
        for i, t in enumerate(remarks):
            if sum(1 for x in remarks if 0 <= t - x < DAY) > 1 or sum(1 for x in remarks if 0 <= t - x < WEEK) > 3:
                bad.append(f"{r.persona}:{r.seed}: volume")
                break
        if r.variant == "clingy" and remarks:
            bad.append(f"{r.persona}:{r.seed}: clingy")
        if any(a["remark"] and a["quiet"] for a in r.absences):
            bad.append(f"{r.persona}:{r.seed}: quiet")
        if any(a["missing_bonus"] > 0.15 + 1e-9 or a["concern_bump"] > 0.15 + 1e-9 for a in r.absences):
            bad.append(f"{r.persona}:{r.seed}: bump")
    out["A9"] = _c({"violations": bad[:20], "n": len(bad)}, 0, not bad, True)
    # A20: the routine step's cost per tick (reported)
    ms = [r.routine_ms / max(1, r.ticks) for r in full]
    mean_ms = statistics.mean(ms) if ms else 0.0
    out["A20"] = _c(round(mean_ms, 3), 2.0, mean_ms <= 2.0, False)
    return out


# ---- the CLI -----------------------------------------------------------------------------------------------------------

RUN_KINDS = ([("full", name) for name in (*FIVE, "late")] + [("plant", name) for name in FIVE]
             + [("clingy", name) for name in ("office", "late")])


def _job(args) -> RunResult:
    kind, name, seed, weeks, constants = args
    if kind == "plant":
        return simulate(name, seed, weeks=weeks, evidence=False, constants=constants)
    if kind == "clingy":
        return simulate(name, seed, weeks=weeks, clinginess=0.6, constants=constants)
    return simulate(name, seed, weeks=weeks, constants=constants)


def run_all(seeds, weeks: int, jobs: int, constants: dict | None = None) -> tuple[dict, dict, float]:
    t = time.perf_counter()
    work = [(k, n, s, weeks, constants or {}) for s in seeds for k, n in RUN_KINDS]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_job, work, chunksize=1)
    else:
        results = [_job(w) for w in work]
    by = {"full": [], "plant": [], "clingy": []}
    for r in results:
        by[r.variant].append(r)
    crit = criteria(by["full"], by["plant"], by["clingy"])
    personas = {}
    for r in by["full"]:
        p = personas.setdefault(r.persona, {"claims": 0, "hunches": 0, "wall_s": 0.0})
        p["claims"] += sum(1 for s in r.statements if s["claim"])
        p["hunches"] += sum(1 for s in r.statements if not s["claim"])
        p["wall_s"] += round(r.wall_s, 1)
    return crit, personas, time.perf_counter() - t


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _set(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in GRID or float(value) not in GRID[name]:
            raise SystemExit(f"--set {pair}: only grid values ({GRID})")
        out[name] = float(value)
    return out


def _grid_points() -> list[dict]:
    base = current_constants()
    singles = [{k: v} for k, v in GRID_ORDER if base[k] != v and admissible({k: v})]
    pairs = [{a[0]: a[1], b[0]: b[1]} for a, b in itertools.combinations(GRID_ORDER, 2)
             if a[0] != b[0] and (base[a[0]] != a[1] or base[b[0]] != b[1]) and admissible({a[0]: a[1], b[0]: b[1]})]
    return singles + pairs


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description="Measure her routine predictors and honesty over simulated personas.")
    ap_.add_argument("--seeds", choices=("dev", "heldout"), required=True)
    ap_.add_argument("--jobs", type=int, default=1)
    ap_.add_argument("--weeks", type=int, default=6)
    ap_.add_argument("--out")
    ap_.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap_.add_argument("--grid", action="store_true")
    a = ap_.parse_args(argv)
    if a.seeds == "heldout":
        if a.set or a.grid:
            print("The held-out run takes no --set and no --grid: constants move on dev seeds only.")
            return 2
        try:
            dev = json.loads(DEV_RECORD.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"No dev record at {DEV_RECORD}: run the dev seeds first.")
            return 2
        if dev.get("constants") != current_constants():
            print("The dev record's constants differ from this code's: re-run the dev seeds first.")
            return 2
    seeds = DEV_SEEDS if a.seeds == "dev" else HELDOUT_SEEDS
    points = _grid_points() if a.grid else [_set(a.set)]
    for constants in points:
        crit, personas, wall = run_all(seeds, a.weeks, max(1, a.jobs), constants)
        failed = [k for k, v in crit.items() if v["blocking"] and not v["pass"]]
        for k, v in crit.items():
            print(f"{k:4} {'PASS' if v['pass'] else 'FAIL'}{'' if v['blocking'] else ' (reported)'}  "
                  f"{json.dumps(v['value'])}  vs {json.dumps(v['threshold'])}  {v['note']}")
        print(f"constants {json.dumps({**current_constants(), **constants})}  wall {wall:.0f} s")
        print("VERDICT: PASS" if not failed else f"VERDICT: FAIL ({', '.join(failed)})")
        record = {"version": 1, "seeds": a.seeds, "seed_list": list(seeds),
                  "constants": {**current_constants(), **constants}, "criteria": crit, "blocking_pass": not failed,
                  "personas": personas, "wall_s": round(wall, 1), "git": _git()}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if a.grid and not failed:
            print(f"first passing grid point: {json.dumps(constants)}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
