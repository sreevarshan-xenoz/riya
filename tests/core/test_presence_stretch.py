"""The observed work stretch (v0.12 Resolution 20, Review Focus 1)."""
from riya.core.clock import FakeClock
from riya.core.presence import BREAK_TAIL_S, STRETCH_GAP_S, FakeIdleSource, Presence
from riya.core.store import KV, open_db


def make():
    clock, src = FakeClock(), FakeIdleSource()
    return clock, src, Presence(src, clock, KV(open_db(":memory:")))


def _away_and_back(clock, src, p):
    clock.advance(60); src.idle = 1000
    p.update()
    assert p.state == "away"
    clock.advance(3600); src.idle = 0
    p.update()


def test_the_stretch_starts_at_an_observed_return_or_break_end():
    clock, src, p = make()
    p.update()
    _away_and_back(clock, src, p)
    assert p.stretch_start() == clock.t and p.censored is False
    clock.advance(60); src.idle = 0
    p.update()
    clock.advance(200); src.idle = 200
    p.update()
    assert p.state == "break"
    src.idle = 0
    p.update()
    tail = clock.t
    clock.advance(BREAK_TAIL_S + 1)
    p.update()
    assert p.state == "focused" and p.stretch_start() == tail and p.censored is False


def test_the_first_stretch_after_start_up_is_censored():
    clock, src, p = make()
    assert p.censored is True and p.stretch_start() == clock.t
    for _ in range(10):
        clock.advance(60)
        p.update()
    assert p.censored is True


def test_a_tick_gap_over_five_minutes_censors_the_stretch():
    clock, src, p = make()
    p.update()
    _away_and_back(clock, src, p)
    clock.advance(STRETCH_GAP_S); src.idle = 0
    p.update()
    assert p.censored is False
    clock.advance(STRETCH_GAP_S + 1)
    p.update()
    assert p.censored is True and p.state == "focused"


def test_a_clock_jump_back_censors_the_stretch():
    clock, src, p = make()
    p.update()
    _away_and_back(clock, src, p)
    assert p.censored is False
    clock.advance(-7200)
    p.update()
    assert p.censored is True
    assert p.work_stretch_s() >= 0 and p.stretch_start() <= clock.t
