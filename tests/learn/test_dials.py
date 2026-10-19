import pytest

from riya.core.learn.dials import BANDS, DELTA, DIALS, clamp


def test_bands_cover_every_dial():
    assert set(BANDS) == set(DIALS) and DELTA == 0.15


@pytest.mark.parametrize("dial", DIALS)
def test_band_edges_are_kept(dial):
    lo, hi = BANDS[dial]
    assert clamp(dial, lo) == lo and clamp(dial, hi) == hi
    assert clamp(dial, lo - 1) == lo and clamp(dial, hi + 1) == hi


def test_clamp_pulls_values_into_the_band():
    assert clamp("teasing", 0.05) == 0.1
    assert clamp("emoji", 0.7) == 0.6


def test_clamp_rounds_float_noise():
    assert clamp("teasing", 0.5 - 0.15 - 0.15) == 0.2


def test_unknown_dial_raises():
    with pytest.raises(ValueError):
        clamp("bogus", 0.5)
