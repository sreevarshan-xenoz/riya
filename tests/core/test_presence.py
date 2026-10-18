from riya.core.clock import FakeClock
from riya.core.presence import FakeIdleSource, Presence
from riya.core.store import KV, open_db


def make(clock=None, src=None, kv=None):
    clock, src = clock or FakeClock(), src or FakeIdleSource()
    return clock, src, Presence(src, clock, kv or KV(open_db(":memory:")))


def kinds(events):
    return [e.kind for e in events]


def test_focused_break_away_and_back():
    clock, src, p = make()
    assert p.update() == [] and p.state == "focused"
    clock.advance(150); src.idle = 150
    assert kinds(p.update()) == ["break_started"] and p.state == "break"
    clock.advance(800); src.idle = 950
    assert kinds(p.update()) == ["went_away"] and p.state == "away"
    clock.advance(3 * 3600); src.idle = 0
    ev = p.update()
    assert kinds(ev) == ["came_back"] and abs(ev[0].away_s - (3 * 3600 + 950)) < 2


def test_break_tail_then_focus():
    clock, src, p = make()
    p.update()
    clock.advance(200); src.idle = 200
    p.update()
    src.idle = 0
    p.update()
    assert p.state == "break"
    clock.advance(61)
    assert kinds(p.update()) == ["break_ended"] and p.state == "focused"


def test_talking_and_gaming():
    clock, src, p = make()
    p.note_turn(); p.update()
    assert p.state == "talking"
    clock.advance(181); p.update()
    assert p.state == "focused"
    src.is_fullscreen = True; p.update()
    assert p.state == "gaming"


def test_locked_is_away_immediately():
    clock, src, p = make()
    p.update()
    src.is_locked = True
    assert kinds(p.update()) == ["went_away"]


def test_restart_after_long_gap_counts_as_back():
    clock, src = FakeClock(), FakeIdleSource()
    kv = KV(open_db(":memory:"))
    _, _, p = make(clock, src, kv)
    p.update()
    clock.advance(2 * 86400)
    _, _, p2 = make(clock, src, kv)
    assert p2.state == "away"
    ev = p2.update()
    assert kinds(ev) == ["came_back"] and ev[0].away_s >= 2 * 86400


def test_sleep_gap_counts_as_away():
    clock, src, p = make()
    p.update()
    clock.advance(8 * 3600)  # laptop slept, no updates; input idle reads 0 on wake
    ev = p.update()
    assert kinds(ev) == ["came_back"] and abs(ev[0].away_s - 8 * 3600) < 2


def test_clock_going_backwards_never_negative():
    clock, src, p = make()
    p.update()
    src.idle = 950; p.update()
    clock.t -= 7200; src.idle = 0
    ev = p.update()
    assert all(e.away_s >= 0 for e in ev)


def test_work_stretch_resets_on_break():
    clock, src, p = make()
    for _ in range(18):
        clock.advance(600); p.update()
    assert p.work_stretch_s() >= 3 * 3600 - 1
    clock.advance(150); src.idle = 150; p.update()
    src.idle = 0; clock.advance(61); p.update()
    assert p.work_stretch_s() < 120


def test_came_back_state_is_normal_and_work_stretch_resets():
    clock, src, p = make()
    p.update()
    clock.advance(1000); src.idle = 1000
    p.update()
    clock.advance(3600); src.idle = 0
    assert kinds(p.update()) == ["came_back"] and p.state == "focused"
    assert p.work_stretch_s() == 0


def test_persists_last_active_and_left_at():
    clock, src = FakeClock(), FakeIdleSource()
    kv = KV(open_db(":memory:"))
    _, _, p = make(clock, src, kv)
    src.idle = 30
    p.update()
    assert kv.get("presence.last_active_at") == clock.t - 30
    clock.advance(10); src.idle = 1000
    p.update()
    assert p.left_at == clock.t - 1000 and kv.get("presence.left_at") == p.left_at
