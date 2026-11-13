"""Builders for routine tests: mature cells written as if their values had been observed, and a Sensed with defaults."""
import math

from riya.core.routine import Sensed
from riya.core.routine import appraisal as ap
from riya.core.routine.predictors import P4_PRIOR, BetaCell, LogNormalCell
from riya.core.routine.records import RoutineDB


def mature_lognormal(db: RoutineDB, predictor: str, bucket: str, median_s: float, *, n: int = 12, sigma: float = 0.3,
                     now: float) -> None:
    mu = math.log(median_s)
    db.put_cell(predictor, bucket, LogNormalCell(float(n), n * mu, n * (mu * mu + sigma * sigma),
                                                 ap.AppraisalState(0.25, 0.0, ()), now))


def mature_beta(db: RoutineDB, predictor: str, bucket: str, p: float, *, days: int, now: float) -> None:
    a0, b0 = P4_PRIOR
    db.put_cell(predictor, bucket, BetaCell(a0 + p * days, b0 + (1 - p) * days, ap.AppraisalState(0.5, 0.0, ()), now))


def sensed(**kw) -> Sensed:
    now = kw.pop("now", 0.0)
    base = dict(now=now, presence="focused", stretch_start=now - 1800.0, stretch_censored=False, glances=(),
                clinginess=0.8, quiet=False)
    return Sensed(**{**base, **kw})
