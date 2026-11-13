"""The ported MORPHEUS appraisal: surprise, a robust expected-loss scale, startle, the CUSUM change test with
softening, and confidence.

Ported from the user's own research, MORPHEUS `morpheus_core.py` (iter 22, tightening step 2: `Appraisal`,
`Config.surprise_factor`, `Config.startle_factor`, `robust_scale`), with the CUSUM novelty test of iter 23, "newborn
skills regain doubt" of iter 32, and "the tracker learns the present" of step 4b. The LAB_NOTES sections behind the
formulas: step 3 (confidence is load-bearing; surprise is the compute and attention signal), step 5a (confidence-weighted
self-prediction), step 5b (act on a hunch, claim only on evidence). The user confirmed MORPHEUS is their own work and may
be vendored here.

Pure Python. Constants are read through this module at call time (never bound as defaults), so a simulator can move a
grid point.
"""
import math
from dataclasses import dataclass

SURPRISE_FACTOR = 2.0
STARTLE_FACTOR = 8.0  # ruling C1 (2026-11-13): 6 failed A2 (189/200 quiet streams); 8 is in the grid and keeps A1
CUSUM_H = 3.5
CUSUM_DRIFT = 0.5
SURPRISE_CLIP = 3.0

MATURE_N = 5.0
SOFT_N = 3.0
STARTLED_MAX = 3
MATURITY_K = 4.0

FLOOR = {"lognormal": 0.25, "beta": 0.1}
SCALE0 = {"lognormal": 0.25, "beta": 0.5}


@dataclass(frozen=True)
class AppraisalState:
    scale: float
    cusum: float
    startled: tuple[float, ...]


@dataclass(frozen=True)
class Appraisal:
    topic: str  # "return", "answer", "run", "gaming"
    predictor: str  # "P1".."P4"
    bucket: str
    surprise: float
    s: float
    direction: str
    surprised: bool
    startled: bool
    confidence: float
    change: bool
    n_eff: float
    value: float  # seconds, or 1.0/0.0


def surprise(loss: float, scale: float, floor: float) -> tuple[float, float]:
    """MORPHEUS's surprise: the log ratio of the expected loss to the robust scale, clipped, plus that ratio."""
    ratio = loss / (scale + floor) if scale + floor > 0 else math.inf
    s = math.log(ratio + 1e-12) if math.isfinite(ratio) else SURPRISE_CLIP
    return max(-SURPRISE_CLIP, min(SURPRISE_CLIP, s)), ratio


def next_scale(scale: float, loss: float, n_eff: float) -> float:
    """The robust scale tracks the median loss: 10% steps while evidence is young, 5% after 20 observations."""
    rate = 0.1 if n_eff < 20 else 0.05
    return scale * (1.0 + rate * (1.0 if loss > scale else -1.0))


def next_cusum(cusum: float, surprise: float) -> float:
    """One-sided CUSUM: drift up on surprising observations, never below zero."""
    return max(0.0, cusum + surprise - CUSUM_DRIFT)


def confidence(n_eff: float, cusum: float) -> float:
    """Maturity (n / (n + K)) times no-change (1 - min(1, cusum / H)), in [0, 1]."""
    n = max(0.0, n_eff)
    maturity = n / (n + MATURITY_K) if n + MATURITY_K > 0 else 0.0
    calm = 1.0 - min(1.0, max(0.0, cusum / CUSUM_H)) if CUSUM_H > 0 else 0.0
    return max(0.0, min(1.0, maturity * calm))


def strength(surprise: float) -> float:
    """0..1, reaching 1 at the startle threshold."""
    return max(0.0, min(1.0, surprise / math.log(STARTLE_FACTOR)))


def is_surprised(ratio: float) -> bool:
    return ratio > SURPRISE_FACTOR


def is_startled(ratio: float, n_eff: float) -> bool:
    return ratio > STARTLE_FACTOR and n_eff >= MATURE_N


def fresh(kind: str) -> AppraisalState:
    """A cell's appraisal state before anything was learned."""
    return AppraisalState(scale=SCALE0[kind], cusum=0.0, startled=())


def of(step, *, topic: str, predictor: str, bucket: str, value: float, direction: str) -> Appraisal:
    """An Appraisal from a predictor Step (the predictor doesn't know its own name or bucket).

    step.startled is the ring after the observation; a startle here is an observation that was neither learned nor a
    change.
    """
    this_startle = (not step.learned) and (not step.change) and bool(step.startled)
    return Appraisal(topic=topic, predictor=predictor, bucket=bucket, surprise=step.surprise, s=step.s,
                     direction=direction, surprised=step.surprised, startled=this_startle,
                     confidence=step.confidence_before, change=step.change, n_eff=step.n_eff_before, value=value)
