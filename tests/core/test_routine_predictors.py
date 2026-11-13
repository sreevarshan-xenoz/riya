"""The routine predictors: Normal-Gamma and Beta cells, decay, startle and change (v0.12 §4.2, §4.3, A1, A2)."""
import math

import numpy as np
import pytest

from riya.core.routine import appraisal as ap
from riya.core.routine import predictors as pr

DAY = 86400.0


def _cell(values, prior=pr.P1_PRIOR, now=0.0):
    """A cell that learned these log values, all at the same instant."""
    cell = None
    for y in values:
        cell = pr.observe_lognormal(cell, prior, math.exp(y), now).cell
    return cell


def _mature(n=20):
    return _cell([pr.P1_PRIOR.mu0] * n)


def _far_y(cell, ratio=ap.STARTLE_FACTOR * 1.03):
    """The log value whose loss puts the appraisal ratio just over the startle threshold."""
    pred = pr.predictive(cell, pr.P1_PRIOR)
    loss = ratio * (cell.state.scale + ap.FLOOR["lognormal"])
    return pred.loc + math.sqrt(2.0 * loss) * pred.scale


def test_lognormal_update_matches_the_closed_form():
    p = pr.P1_PRIOR
    ys = [math.log(v) for v in (6000.0, 7000.0, 9000.0, 5000.0, 8000.0)]
    cell = _cell(ys)
    n = len(ys)
    ybar = sum(ys) / n
    kappa = p.kappa0 + n
    mu = (p.kappa0 * p.mu0 + n * ybar) / kappa
    alpha = p.alpha0 + n / 2
    beta = p.beta0 + 0.5 * sum((y - ybar) ** 2 for y in ys) + p.kappa0 * n * (ybar - p.mu0) ** 2 / (2 * kappa)
    assert cell.n == pytest.approx(n)
    assert cell.s1 == pytest.approx(sum(ys))
    assert cell.s2 == pytest.approx(sum(y * y for y in ys))
    pred = pr.predictive(cell, p)
    assert pred.loc == pytest.approx(mu)
    assert pred.dof == pytest.approx(2 * alpha)
    assert pred.scale == pytest.approx(math.sqrt(beta * (kappa + 1) / (alpha * kappa)))


def test_beta_update_matches_the_closed_form():
    cell = None
    outcomes = [True, False, True, True, False]
    for o in outcomes:
        cell = pr.observe_beta(cell, pr.P4_PRIOR, o, 0.0).cell
    assert cell.alpha == pytest.approx(pr.P4_PRIOR[0] + sum(outcomes))
    assert cell.beta == pytest.approx(pr.P4_PRIOR[1] + len(outcomes) - sum(outcomes))


def test_decay_halves_the_evidence_in_30_days():
    cell = pr.LogNormalCell(10.0, 80.0, 650.0, ap.fresh("lognormal"), 0.0)
    assert pr.lognormal_decayed(cell, pr.P1_PRIOR, 30 * DAY).n == pytest.approx(5.0)
    b = pr.BetaCell(1.0 + 6.0, 3.0 + 4.0, ap.fresh("beta"), 0.0)
    d = pr.beta_decayed(b, pr.P4_PRIOR, 30 * DAY)
    assert pr.beta_evidence(d, pr.P4_PRIOR) == pytest.approx(5.0)
    assert (d.alpha, d.beta) == pytest.approx((4.0, 5.0))


def test_corrupt_cells_read_as_the_prior():
    st = ap.fresh("lognormal")
    for bad in (pr.LogNormalCell(math.nan, 1.0, 1.0, st, 0.0),
                pr.LogNormalCell(-1.0, 1.0, 1.0, st, 0.0),
                pr.LogNormalCell(3.0, 1.0, 1.0, st, math.inf)):
        c = pr.lognormal_decayed(bad, pr.P1_PRIOR, 5.0)
        assert (c.n, c.s1, c.s2, c.state, c.updated_at) == (0.0, 0.0, 0.0, ap.fresh("lognormal"), 5.0)
    bst = ap.fresh("beta")
    for bad in (pr.BetaCell(0.0, 3.0, bst, 0.0), pr.BetaCell(2.0, math.nan, bst, 0.0)):
        assert pr.beta_decayed(bad, pr.P4_PRIOR, 5.0) == pr.BetaCell(1.0, 3.0, ap.fresh("beta"), 5.0)


def _cell_at(n):
    mu = pr.P1_PRIOR.mu0
    return pr.LogNormalCell(float(n), n * mu, n * (mu * mu + 0.25), ap.fresh("lognormal"), 0.0)


def test_startle_needs_a_mature_cell_and_is_not_learned():
    young = _cell_at(4)
    pred = pr.predictive(young, pr.P1_PRIOR)
    y = pred.loc + 8 * pred.scale
    step = pr.observe_lognormal(young, pr.P1_PRIOR, math.exp(y), 0.0)
    assert step.learned and not step.startled and step.cell.n == pytest.approx(5.0)

    mature = _cell_at(20)
    pred = pr.predictive(mature, pr.P1_PRIOR)
    y = pred.loc + 8 * pred.scale
    step = pr.observe_lognormal(mature, pr.P1_PRIOR, math.exp(y), 0.0)
    assert not step.learned and not step.change
    assert step.cell.n == pytest.approx(20.0)
    assert step.startled == pytest.approx((y,))


def test_three_startles_soften_learn_the_buffer_and_reset():
    cell = _mature()
    assert cell.state.cusum == 0.0
    mean_before = cell.s1 / cell.n
    steps = []
    for i in range(3):
        y = _far_y(cell)
        step = pr.observe_lognormal(cell, pr.P1_PRIOR, math.exp(y), float(i + 1))
        steps.append(step)
        cell = step.cell
    assert [s.learned for s in steps[:2]] == [False, False]
    assert all(s.startled for s in steps[:2]) and not any(s.change for s in steps[:2])
    last = steps[2]
    assert last.change and last.learned
    assert cell.n <= ap.SOFT_N + 3
    assert cell.s1 / cell.n > mean_before
    assert cell.state.cusum == 0
    assert cell.state.startled == ()


def test_the_startled_buffer_is_a_ring_of_three():
    cell = _mature()
    far = []
    for i in range(4):
        y = _far_y(cell)
        step = pr.observe_lognormal(cell, pr.P1_PRIOR, math.exp(y), float(2 * i))
        assert not step.learned and not step.change
        far.append(y)
        cell = step.cell
        calm = pr.predictive(cell, pr.P1_PRIOR).loc  # a value right at the median drains the cusum
        step = pr.observe_lognormal(cell, pr.P1_PRIOR, math.exp(calm), float(2 * i + 1))
        assert step.learned and not step.change
        cell = step.cell
    assert cell.state.startled == pytest.approx(tuple(far[-3:]))


def test_surprise_and_confidence_stay_in_bounds():
    rng = np.random.default_rng(7)
    ys = rng.normal(math.log(3600), 0.5, 500)
    ys[::37] += rng.choice([-6.0, 6.0], size=len(ys[::37]))
    cell, b = None, None
    for i, y in enumerate(ys):
        step = pr.observe_lognormal(cell, pr.P1_PRIOR, math.exp(y), i * 3600.0)
        cell = step.cell
        bstep = pr.observe_beta(b, pr.P4_PRIOR, bool(y > math.log(3600)), i * 3600.0)
        b = bstep.cell
        for s in (step, bstep):
            assert -3.0 <= s.surprise <= 3.0
            assert 0.0 <= s.confidence_before <= 1.0
            assert 0.0 <= s.s <= 1.0


def test_a_stationary_stream_rarely_changes():
    quiet = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        cell, changes = None, 0
        for i, y in enumerate(rng.normal(math.log(3600), 0.5, 500)):
            step = pr.observe_lognormal(cell, pr.P1_PRIOR, math.exp(y), i * 3 * 3600.0)
            changes += step.change
            cell = step.cell
        quiet += changes <= 1
    assert quiet >= 190  # if this fails, stop and report: the constants move only by a ruling


def test_predictive_quantiles_and_within_two():
    pred = pr.predictive(pr.lognormal_decayed(None, pr.P1_PRIOR, 0.0), pr.P1_PRIOR)
    assert pred.median_s() == pytest.approx(7200.0, rel=1e-6)
    assert pred.quantile_s(0.5) == pred.median_s()
    assert pred.p_within(2) == pytest.approx(2 * pr.t_cdf(math.log(2) / pred.scale, pred.dof) - 1)
    assert pred.quantile_s(0.1) < pred.median_s() < pred.quantile_s(0.9)


def test_t_and_beta_helpers():
    for x in (-3.0, -0.5, 0.0, 1.0, 4.0):
        assert pr.t_cdf(x, 1) == pytest.approx(0.5 + math.atan(x) / math.pi, abs=1e-6)
        assert pr.t_cdf(x, 2) == pytest.approx(0.5 + x / (2 * math.sqrt(2 + x * x)), abs=1e-6)
    assert pr.t_cdf(1.959964, 1e6) == pytest.approx(0.975, abs=1e-3)
    for d in (1, 4, 30):
        for x in (-2.0, -0.3, 0.7, 2.5):
            assert pr.t_ppf(pr.t_cdf(x, d), d) == pytest.approx(x, abs=1e-6)
    assert pr.beta_sf(1, 1) == pytest.approx(0.5, abs=1e-6)
    assert pr.beta_sf(9, 1) == pytest.approx(1 - 0.5 ** 9, abs=1e-6)
    assert pr.beta_sf(3, 3) == pytest.approx(0.5, abs=1e-6)
    assert pr.beta_sf(1, 2) == pytest.approx(0.25, abs=1e-6)


def test_p2_moves_only_the_appraisal_state():
    state, step = pr.appraise_p2(None, 0.2, True, 0.0)
    assert step.ratio == pytest.approx(-math.log(0.2) / (0.5 + 0.1))
    assert step.sign > 0
    assert step.cell is None
    loss = -math.log(0.2)
    assert state.scale == pytest.approx(ap.next_scale(ap.SCALE0["beta"], loss, 0.0))
    assert state.cusum == pytest.approx(ap.next_cusum(0.0, step.surprise))


def test_p2_startles_then_changes_like_the_cells():
    state = None  # a mature bucket that almost never engages, then three engagements just over the startle line
    p = math.exp(-ap.STARTLE_FACTOR * 1.03 * (ap.SCALE0["beta"] + ap.FLOOR["beta"]))
    steps = []
    for _ in range(3):
        state, step = pr.appraise_p2(state, p, True, 20.0)
        steps.append(step)
    assert [s.learned for s in steps[:2]] == [False, False]
    assert steps[2].change and steps[2].learned
    assert state.cusum == 0.0 and state.startled == ()
    loss = -math.log(p)
    assert state.scale == pytest.approx(ap.next_scale(ap.SCALE0["beta"], loss, 20.0))
