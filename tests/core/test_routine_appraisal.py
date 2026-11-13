"""The ported MORPHEUS appraisal (v0.12 §4.3, A1)."""
import math

import pytest

from riya.core.routine import appraisal as ap


def test_the_constants_are_the_spec_s():
    assert (ap.SURPRISE_FACTOR, ap.STARTLE_FACTOR, ap.CUSUM_H, ap.CUSUM_DRIFT, ap.SURPRISE_CLIP) == (2.0, 8.0, 3.5, 0.5, 3.0)
    assert (ap.MATURE_N, ap.SOFT_N, ap.STARTLED_MAX, ap.MATURITY_K) == (5.0, 3.0, 3, 4.0)
    assert ap.FLOOR == {"lognormal": 0.25, "beta": 0.1}
    assert ap.SCALE0 == {"lognormal": 0.25, "beta": 0.5}


def test_surprise_matches_the_ported_formula_and_is_clipped():
    assert ap.surprise(1.0, 0.25, 0.25) == pytest.approx((math.log(2), 2.0))
    assert ap.surprise(0.0, 0.25, 0.25)[0] == -3.0
    assert ap.surprise(1e9, 0.25, 0.25)[0] == 3.0


def test_the_scale_tracks_the_median_like_morpheus():
    assert ap.next_scale(0.25, 1.0, 5) == pytest.approx(0.275)
    assert ap.next_scale(0.25, 0.25, 5) == pytest.approx(0.225)  # equal goes down, as in MORPHEUS
    assert ap.next_scale(0.25, 1.0, 25) == pytest.approx(0.2625)


def test_confidence_is_maturity_times_no_change():
    assert ap.confidence(4, 0) == 0.5
    assert ap.confidence(8, 1.75) == pytest.approx(8 / 12 * 0.5)
    assert ap.confidence(8, ap.CUSUM_H) == 0
    for n in (0, 1, 5, 50):
        for c in (0, 1, 3.5, 10):
            assert 0.0 <= ap.confidence(n, c) <= 1.0


def test_strength_is_one_at_the_startle_threshold():
    assert ap.strength(math.log(ap.STARTLE_FACTOR)) == pytest.approx(1.0)
    assert ap.strength(10.0) == 1.0
    assert ap.strength(-1.0) == 0.0
    assert ap.strength(math.log(ap.STARTLE_FACTOR) / 2) == pytest.approx(0.5)


def test_three_startles_just_over_the_threshold_raise_a_change():
    step = math.log(ap.STARTLE_FACTOR) - ap.CUSUM_DRIFT
    assert 3 * step > ap.CUSUM_H > 2 * step
    s = math.log(ap.STARTLE_FACTOR + 1e-4)
    c1 = ap.next_cusum(0.0, s)
    c2 = ap.next_cusum(c1, s)
    c3 = ap.next_cusum(c2, s)
    assert c1 <= ap.CUSUM_H and c2 <= ap.CUSUM_H and c3 > ap.CUSUM_H
