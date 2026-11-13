"""Surprise and confidence move her feelings, inside her bands (v0.12 §4.4, Resolutions 16-18, P07, A9)."""
import copy
from dataclasses import asdict

import pytest

from riya.core.clock import FakeClock
from riya.core.feelings import CONCERN_MAX, MISSING_BONUS_MAX, Feelings
from riya.core.routine.appraisal import Appraisal

V011 = ("[Riya's state: feeling neutral | valence +0.10 arousal 0.30 warmth 0.60 mood +0.10 energy 0.80 "
        "curiosity 0.50]")


def _a(topic="return", direction="early", *, s=1.0, surprise=2.0, surprised=True, startled=False, n=8.0, change=False,
       value=3600.0):
    return Appraisal(topic=topic, predictor="P1", bucket="b", surprise=surprise, s=s, direction=direction,
                     surprised=surprised, startled=startled, confidence=0.7, change=change, n_eff=n, value=value)


def _f(clinginess=0.8):
    return Feelings(FakeClock(), {"clinginess": clinginess})


def _state(f):
    d = asdict(f.state)
    d.pop("user_mood")
    return d


ROWS = [
    ("early", _a(s=0.5), {"valence": 0.1, "arousal": 0.125, "curiosity": 0.05}),
    ("late", _a(direction="late", s=0.5), {"concern": 0.075, "arousal": 0.05}),
    ("late, young cell", _a(direction="late", n=3.9), {}),
    ("as expected", _a(direction="late", surprise=-1.0, surprised=False, s=0.0, n=5.0), {"warmth": 0.05}),
    ("as expected, young", _a(direction="late", surprise=-1.0, surprised=False, s=0.0, n=4.9), {}),
    ("answered, unexpected", _a("answer", "answered", s=0.4, value=0.2), {"valence": 0.08, "warmth": 0.04}),
    ("answered, expected", _a("answer", "answered", s=0.4, value=0.6), {}),
    ("ignored", _a("answer", "ignored", value=0.8), {}),
    ("run longer", _a("run", "longer", s=0.5), {"curiosity": 0.05}),
    ("run shorter", _a("run", "shorter", s=0.5), {"valence": 0.05}),
    ("run as usual", _a("run", "longer", s=0.0, surprised=False), {}),
    ("gaming unusual", _a("gaming", "gamed", s=0.5), {"curiosity": 0.05}),
    ("gaming expected", _a("gaming", "gamed", s=0.0, surprised=False), {}),
    ("a change", _a("run", "longer", s=0.0, surprised=False, change=True), {"curiosity": 0.15}),
]


@pytest.mark.parametrize("name,a,moves", ROWS, ids=[r[0] for r in ROWS])
def test_each_appraisal_moves_her_as_the_table_says(name, a, moves):
    f = _f()
    before = _state(f)
    f.on_appraisal(a, 0.8)
    after = _state(f)
    for k in before:
        assert after[k] == pytest.approx(before[k] + moves.get(k, 0.0)), k


def test_the_absence_bumps_are_capped():
    plain, bumped = _f(1.0), _f(1.0)
    concern0 = bumped.state.concern
    bumped.on_appraisal(_a(direction="late", s=1.0), 1.0)
    assert bumped.state.concern - concern0 <= CONCERN_MAX + 1e-9
    plain.on_return(3 * 3600)
    bumped.on_return(3 * 3600)
    assert 0 < bumped.state.missing - plain.state.missing <= MISSING_BONUS_MAX + 1e-9


def test_the_bonus_waits_for_on_return_and_goes_stale():
    f = _f()
    f.on_appraisal(_a(direction="late", s=1.0), 0.8)
    assert f.state.missing == 0.0
    f.clock.advance(1800)
    plain = _f()
    plain.on_return(3 * 3600)
    f.on_return(3 * 3600)
    assert f.state.missing == pytest.approx(plain.state.missing + 0.15)
    m = f.state.missing
    f.on_return(3 * 3600)
    assert f.state.missing == pytest.approx(m)  # used once

    g = _f()
    g.on_appraisal(_a(direction="late", s=1.0), 0.8)
    g.clock.advance(1801)
    g.on_return(3 * 3600)
    assert g.state.missing == pytest.approx(plain.state.missing)


def test_a_startle_face_shows_before_the_return_face():
    f = _f()
    f.on_appraisal(_a(s=1.0, startled=True), 0.8)
    f.on_return(3 * 3600)
    assert f.expression() == "surprised"
    f.clock.advance(5.9)
    assert f.expression() == "surprised"
    f.clock.advance(0.2)
    face = f.expression()
    assert face in ("missed_you", "happy")
    f.clock.advance(19.8)  # 25.9 s since the startle
    assert f.expression() == face
    f.clock.advance(0.2)
    derived = Feelings(f.clock, {"clinginess": 0.8})
    derived.state = f.state
    assert f.expression() == derived.expression()  # the overrides are over: derived from her state again


def test_a_return_with_no_startle_is_v011s():
    f, g = _f(), _f()
    g.on_return(3 * 3600)
    f.on_return(3 * 3600)
    assert _state(f) == _state(g) and f.expression() == g.expression() and f._after is None


def test_the_tag_without_a_clause_is_v011s():
    f = Feelings(FakeClock(), {"clinginess": 0.8})
    assert f.prompt_tag() == f.prompt_tag(None) == V011


def test_the_tag_carries_one_clause():
    f = Feelings(FakeClock(), {"clinginess": 0.8})
    c = "unsure: their routine looks different lately, so ask rather than assume"
    assert f.prompt_tag(c) == V011[:-1] + " | " + c + "]"


def test_the_dials_are_never_touched():
    traits = {"clinginess": 0.8, "sweetness": 0.5, "teasing": 0.4}
    snapshot = copy.deepcopy(traits)
    f = Feelings(FakeClock(), traits)
    for _, a, _ in ROWS:
        f.on_appraisal(a, traits["clinginess"])
    f.on_return(3 * 3600)
    assert traits == snapshot
