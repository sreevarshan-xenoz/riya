"""The twin's cells take a configured prior (v0.12 P4's Beta(1, 3)); the default is unchanged."""
from riya.core.learn.twin import Cell, decayed, evidence, mean, updated


def test_the_default_prior_is_unchanged():
    assert decayed(None, 5.0) == Cell(2.0, 2.0, 5.0)


def test_a_configured_prior():
    assert decayed(None, 5.0, prior=(1.0, 3.0)) == Cell(1.0, 3.0, 5.0)
    assert mean(None, 0.0, prior=(1.0, 3.0)) == 0.25
    assert evidence(updated(None, True, 0.0, prior=(1.0, 3.0)), 0.0, prior=(1.0, 3.0)) == 1.0
