"""The four routine predictors: P1 (when they come back), P3 (how long a run takes) and P4 (when they game) as
Normal-Gamma cells on log seconds and Beta cells on a yes/no outcome; P2 reads the twin's own Beta cells.

Pure Python plus numpy. The evidence decays with the twin's 30-day half-life, and a corrupt cell reads as its prior
(the twin's rule). Constants are read through the module at call time so a simulator can move a grid point.
"""
import math
from dataclasses import dataclass

import numpy as np

from riya.core.learn import twin
from riya.core.routine import appraisal as ap

MIN_ABSENCE_S = 900.0
MIN_RUN_S = 30.0
GAMING_SPAN_S = 600.0
BAND_COVER = 0.5
BACKOFF_N = 4.0


@dataclass(frozen=True)
class LogNormalPrior:
    median_s: float
    sigma0: float = 1.2

    @property
    def mu0(self) -> float:
        return math.log(self.median_s)

    @property
    def kappa0(self) -> float:
        return 1.0

    @property
    def alpha0(self) -> float:
        return 2.0

    @property
    def beta0(self) -> float:
        return self.alpha0 * self.sigma0 ** 2


P1_PRIOR = LogNormalPrior(7200.0)
P3_PRIORS = {"claude": LogNormalPrior(180.0), "colab": LogNormalPrior(3600.0)}
P4_PRIOR = (1.0, 3.0)


@dataclass(frozen=True)
class LogNormalCell:
    n: float
    s1: float
    s2: float
    state: ap.AppraisalState
    updated_at: float


@dataclass(frozen=True)
class BetaCell:
    alpha: float
    beta: float
    state: ap.AppraisalState
    updated_at: float


@dataclass(frozen=True)
class Predictive:
    loc: float
    scale: float
    dof: float

    def median_s(self) -> float:
        return math.exp(self.loc)

    def quantile_s(self, q: float) -> float:
        return math.exp(self.loc + self.scale * t_ppf(q, self.dof))

    def cdf_log(self, y: float) -> float:
        return t_cdf((y - self.loc) / self.scale, self.dof)

    def p_within(self, factor: float = 2.0) -> float:
        return 2.0 * self.cdf_log(self.loc + math.log(factor)) - 1.0


@dataclass(frozen=True)
class Step:
    cell: LogNormalCell | BetaCell | None
    surprise: float
    ratio: float
    s: float
    surprised: bool
    startled: tuple[float, ...]
    change: bool
    learned: bool
    n_eff_before: float
    confidence_before: float
    sign: float


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _decay_factor(updated_at: float, now: float) -> float:
    return 0.5 ** (max(0.0, now - updated_at) / twin.HALF_LIFE_S)


def _corrupt(*values) -> bool:
    return any(not _finite(v) for v in values)


# --- P1/P3: the Normal-Gamma cell on y = ln seconds ----------------------------------------------------------------

def lognormal_decayed(cell: LogNormalCell | None, prior: LogNormalPrior, now: float) -> LogNormalCell:
    """The cell as of now, evidence above the prior faded; a missing or corrupt cell reads as the prior."""
    if (cell is None or _corrupt(cell.n, cell.s1, cell.s2, cell.state.scale, cell.state.cusum, cell.updated_at)
            or cell.n < 0 or any(not _finite(v) for v in cell.state.startled)):
        return LogNormalCell(0.0, 0.0, 0.0, ap.fresh("lognormal"), now)
    f = _decay_factor(cell.updated_at, now)
    return LogNormalCell(cell.n * f, cell.s1 * f, cell.s2 * f, cell.state, now)


def predictive(cell: LogNormalCell, prior: LogNormalPrior) -> Predictive:
    """The Student-t predictive of the Normal-Gamma posterior from the stored sufficient statistics."""
    n = cell.n
    kappa = prior.kappa0 + n
    mu = (prior.kappa0 * prior.mu0 + cell.s1) / kappa
    alpha = prior.alpha0 + n / 2.0
    scatter = max(0.0, cell.s2 - (cell.s1 * cell.s1 / n if n > 0 else 0.0))
    shift = prior.kappa0 * n * (cell.s1 / n - prior.mu0) ** 2 / (2.0 * kappa) if n > 0 else 0.0
    beta = prior.beta0 + 0.5 * scatter + shift
    scale = math.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
    return Predictive(loc=mu, scale=scale, dof=2.0 * alpha)


def _soften(n: float, s1: float, s2: float) -> tuple[float, float, float]:
    f = min(1.0, ap.SOFT_N / n) if n > 0 else 1.0
    return n * f, s1 * f, s2 * f


def _learn(n: float, s1: float, s2: float, y: float) -> tuple[float, float, float]:
    return n + 1.0, s1 + y, s2 + y * y


def observe_lognormal(cell: LogNormalCell | None, prior: LogNormalPrior, x_s: float, now: float) -> Step:
    """One observation of x_s seconds, in Resolution 9's order; the returned cell is the new state."""
    c = lognormal_decayed(cell, prior, now)
    pred = predictive(c, prior)
    y = math.log(x_s)
    z = (y - pred.loc) / pred.scale
    loss = 0.5 * z * z
    n_before = c.n
    conf = ap.confidence(n_before, c.state.cusum)
    sur, ratio = ap.surprise(loss, c.state.scale, ap.FLOOR["lognormal"])
    surprised = ap.is_surprised(ratio)
    cusum = ap.next_cusum(c.state.cusum, sur)
    change = cusum > ap.CUSUM_H
    if change:  # soften (the mean is kept), learn the buffered startled values and then this one
        n, s1, s2 = _soften(c.n, c.s1, c.s2)
        for buffered in c.state.startled:
            n, s1, s2 = _learn(n, s1, s2, buffered)
        n, s1, s2 = _learn(n, s1, s2, y)
        state = ap.AppraisalState(ap.next_scale(c.state.scale, loss, n_before), 0.0, ())
        learned = True
    elif ap.is_startled(ratio, n_before):  # the ring holds it; nothing is learned
        n, s1, s2 = c.n, c.s1, c.s2
        state = ap.AppraisalState(c.state.scale, cusum, (c.state.startled + (y,))[-ap.STARTLED_MAX:])
        learned = False
    else:
        n, s1, s2 = _learn(c.n, c.s1, c.s2, y)
        state = ap.AppraisalState(ap.next_scale(c.state.scale, loss, n_before), cusum, c.state.startled)
        learned = True
    new_cell = LogNormalCell(n, s1, s2, state, now)
    return Step(cell=new_cell, surprise=sur, ratio=ratio, s=ap.strength(sur), surprised=surprised,
                startled=state.startled, change=change, learned=learned, n_eff_before=n_before,
                confidence_before=conf, sign=z)


# --- P2/P4: the Beta cell -------------------------------------------------------------------------------------------

def beta_decayed(cell: BetaCell | None, prior: tuple[float, float], now: float) -> BetaCell:
    """The cell as of now, evidence above the prior faded; a missing or corrupt cell reads as the prior."""
    a0, b0 = prior
    if (cell is None or _corrupt(cell.alpha, cell.beta, cell.state.scale, cell.state.cusum, cell.updated_at)
            or cell.alpha <= 0 or cell.beta <= 0 or any(not _finite(v) for v in cell.state.startled)):
        return BetaCell(a0, b0, ap.fresh("beta"), now)
    f = _decay_factor(cell.updated_at, now)
    return BetaCell(a0 + (cell.alpha - a0) * f, b0 + (cell.beta - b0) * f, cell.state, now)


def beta_evidence(cell: BetaCell, prior: tuple[float, float]) -> float:
    return max(0.0, cell.alpha + cell.beta - (prior[0] + prior[1]))


def _learn_beta(alpha: float, beta: float, outcome: bool) -> tuple[float, float]:
    return (alpha + 1.0, beta) if outcome else (alpha, beta + 1.0)


def observe_beta(cell: BetaCell | None, prior: tuple[float, float], outcome: bool, now: float) -> Step:
    """One yes/no observation, in Resolution 9's order; the returned cell is the new state."""
    c = beta_decayed(cell, prior, now)
    p = c.alpha / (c.alpha + c.beta)
    p = min(1.0 - 1e-12, max(1e-12, p))
    loss = -math.log(p) if outcome else -math.log(1.0 - p)
    sign = 1.0 if outcome else -1.0
    n_before = beta_evidence(c, prior)
    conf = ap.confidence(n_before, c.state.cusum)
    sur, ratio = ap.surprise(loss, c.state.scale, ap.FLOOR["beta"])
    surprised = ap.is_surprised(ratio)
    cusum = ap.next_cusum(c.state.cusum, sur)
    change = cusum > ap.CUSUM_H
    if change:  # soften the evidence (the mean is kept), learn the buffered outcomes and then this one
        f = min(1.0, ap.SOFT_N / n_before) if n_before > 0 else 1.0
        alpha, beta = prior[0] + (c.alpha - prior[0]) * f, prior[1] + (c.beta - prior[1]) * f
        for buffered in c.state.startled:
            alpha, beta = _learn_beta(alpha, beta, buffered > 0)
        alpha, beta = _learn_beta(alpha, beta, outcome)
        state = ap.AppraisalState(ap.next_scale(c.state.scale, loss, n_before), 0.0, ())
        learned = True
    elif ap.is_startled(ratio, n_before):
        alpha, beta = c.alpha, c.beta
        state = ap.AppraisalState(c.state.scale, cusum, (c.state.startled + (sign,))[-ap.STARTLED_MAX:])
        learned = False
    else:
        alpha, beta = _learn_beta(c.alpha, c.beta, outcome)
        state = ap.AppraisalState(ap.next_scale(c.state.scale, loss, n_before), cusum, c.state.startled)
        learned = True
    new_cell = BetaCell(alpha, beta, state, now)
    return Step(cell=new_cell, surprise=sur, ratio=ratio, s=ap.strength(sur), surprised=surprised,
                startled=state.startled, change=change, learned=learned, n_eff_before=n_before,
                confidence_before=conf, sign=sign)


def appraise_p2(state: ap.AppraisalState | None, p_engage: float, engaged: bool, n_eff: float) -> tuple[ap.AppraisalState, Step]:
    """P2: the twin owns the statistics, so only the appraisal state moves."""
    st = state or ap.fresh("beta")
    p = min(1.0 - 1e-12, max(1e-12, p_engage))
    loss = -math.log(p) if engaged else -math.log(1.0 - p)
    sign = 1.0 if engaged else -1.0
    conf = ap.confidence(n_eff, st.cusum)
    sur, ratio = ap.surprise(loss, st.scale, ap.FLOOR["beta"])
    surprised = ap.is_surprised(ratio)
    cusum = ap.next_cusum(st.cusum, sur)
    change = cusum > ap.CUSUM_H
    if change:
        state = ap.AppraisalState(ap.next_scale(st.scale, loss, n_eff), 0.0, ())
        learned = True
    elif ap.is_startled(ratio, n_eff):
        state = ap.AppraisalState(st.scale, cusum, (st.startled + (sign,))[-ap.STARTLED_MAX:])
        learned = False
    else:
        state = ap.AppraisalState(ap.next_scale(st.scale, loss, n_eff), cusum, st.startled)
        learned = True
    step = Step(cell=None, surprise=sur, ratio=ratio, s=ap.strength(sur), surprised=surprised,
                startled=state.startled, change=change, learned=learned, n_eff_before=n_eff,
                confidence_before=conf, sign=sign)
    return state, step


# --- numeric helpers (numpy + math.lgamma only) ---------------------------------------------------------------------

def _simpson(y: np.ndarray, h: float) -> float:
    w = np.ones(len(y))
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return float(np.dot(w, y) * h / 3.0)


def t_cdf(x: float, dof: float) -> float:
    """The Student-t CDF by Simpson integration of the density on [0, |x|]."""
    if x == 0:
        return 0.5
    if math.isinf(x):
        return 1.0 if x > 0 else 0.0
    sign = 1.0 if x > 0 else -1.0
    a = abs(x)
    n = 2048
    t = np.linspace(0.0, a, n + 1)
    log_c = math.lgamma((dof + 1.0) / 2.0) - math.lgamma(dof / 2.0) - 0.5 * math.log(dof * math.pi)
    pdf = np.exp(log_c - (dof + 1.0) / 2.0 * np.log1p(t * t / dof))
    return min(1.0, max(0.0, 0.5 + sign * _simpson(pdf, a / n)))


def t_ppf(q: float, dof: float) -> float:
    """The Student-t quantile by bisection on t_cdf."""
    if q <= 0.0:
        return -math.inf
    if q >= 1.0:
        return math.inf
    if q == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, dof) > q:
        lo *= 2.0
    while t_cdf(hi, dof) < q:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, dof) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def beta_sf(a: float, b: float, x: float = 0.5) -> float:
    """P(X > x) for Beta(a, b), by Simpson integration of the density on [x, 1]."""
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    n = 4096
    t = np.linspace(x, 1.0, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pdf = (a - 1.0) * np.log(t) + np.where(t < 1.0, (b - 1.0) * np.log1p(-t), 0.0)
    pdf = np.exp(log_pdf - log_b)
    if b < 1.0:  # the density is infinite at 1: integrate up to the last grid point only
        pdf[-1] = pdf[-2]
    elif b > 1.0:
        pdf[-1] = 0.0
    return max(0.0, min(1.0, _simpson(pdf, (1.0 - x) / n)))


def direction_for(topic: str, sign: float) -> str:
    """P1: early/late; P3: shorter/longer; P2: answered/ignored; P4: gamed/didnt."""
    if topic == "return":
        return "early" if sign < 0 else "late"
    if topic == "run":
        return "shorter" if sign < 0 else "longer"
    if topic == "answer":
        return "answered" if sign > 0 else "ignored"
    return "gamed" if sign > 0 else "didnt"


# numpy is imported for the vector helpers the simulator and the statements use; keep the dependency explicit.
def within_band(p: Predictive, factor: float = 2.0) -> float:
    """The predictive's mass within [median / factor, median * factor]."""
    return float(np.clip(p.p_within(factor), 0.0, 1.0))
