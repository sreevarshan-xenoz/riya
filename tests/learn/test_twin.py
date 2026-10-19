import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from riya.core.clock import FakeClock
from riya.core.learn.twin import HALF_LIFE_S, Cell, MemoryCells, Twin, bucket
from tests.fakes import FixedRng

B = "evening/break/long"
UTC = timezone.utc


def twin(cells=None):
    clock = FakeClock()
    return Twin(cells or MemoryCells(), clock), clock


def test_fresh_bucket_is_the_prior():
    t, _ = twin()
    assert t.p_engage("bid", B) == 0.5
    assert t.chance("bid", B) == 1.0
    assert t.greet_style(B) == "normal"


def test_records_count_into_the_cell():
    cells = MemoryCells()
    t, clock = twin(cells)
    for engaged in (True, True, False):
        t.record("bid", B, engaged)
    assert cells.get_cell("bid", B) == Cell(4.0, 3.0, clock.now())
    assert t.p_engage("bid", B) == pytest.approx(4 / 7)


def test_decay_on_read_and_update():
    cells = MemoryCells()
    t, clock = twin(cells)
    t0 = clock.now()
    cells.put_cell("bid", B, Cell(6, 2, t0))
    clock.advance(HALF_LIFE_S)
    assert t.p_engage("bid", B) == pytest.approx(4 / 6)
    assert cells.get_cell("bid", B) == Cell(6, 2, t0)  # reads never write
    t.record("bid", B, True)
    assert cells.get_cell("bid", B) == Cell(5, 2, clock.now())


def test_floor_and_chance():
    cells = MemoryCells()
    t, clock = twin(cells)
    cells.put_cell("bid", B, Cell(2, 200, clock.now()))
    assert t.chance("bid", B) == 0.15
    assert t.should("bid", B, FixedRng(0.149)) is True
    assert t.should("bid", B, FixedRng(0.151)) is False
    rng = random.Random(5)
    fired = sum(t.should("bid", B, rng) for _ in range(2000))
    assert 0.12 <= fired / 2000 <= 0.18
    cells.put_cell("bid", B, Cell(3, 9, clock.now()))  # p 0.25
    assert t.chance("bid", B) == pytest.approx(0.5)
    cells.put_cell("bid", B, Cell(5, 5, clock.now()))
    assert t.chance("bid", B) == 1.0
    cells.put_cell("bid", B, Cell(9, 2, clock.now()))
    assert t.chance("bid", B) == 1.0


def test_greeting_style():
    cells = MemoryCells()
    t, clock = twin(cells)
    cells.put_cell("greeting", B, Cell(2, 4, clock.now()))
    assert t.greet_style(B) == "short"
    cells.put_cell("greeting", B, Cell(4, 6, clock.now()))
    assert t.greet_style(B) == "normal"


def test_unknown_kind_is_an_error():
    t, _ = twin()
    with pytest.raises(ValueError):
        t.record("nope", B, True)


def _at(h, m=0, tz=UTC):
    return datetime(2026, 5, 4, h, m, tzinfo=tz).timestamp()


def test_bucket_bands_and_stretch():
    cases = [((4, 59), "night"), ((5, 0), "morning"), ((11, 59), "morning"), ((12, 0), "afternoon"),
             ((16, 59), "afternoon"), ((17, 0), "evening"), ((21, 59), "evening"), ((22, 0), "night"),
             ((0, 0), "night")]
    for (h, m), band in cases:
        assert bucket(_at(h, m), UTC, "here", 0) == f"{band}/here/short"
    assert bucket(_at(12), UTC, "here", 7200) == "afternoon/here/short"
    assert bucket(_at(12), UTC, "here", 7201) == "afternoon/here/long"
    assert bucket(_at(18), UTC, "break", 9000) == "evening/break/long"


def test_buckets_across_dst():
    tz = ZoneInfo("America/New_York")
    spring = datetime(2026, 3, 8, 1, 30, tzinfo=tz).timestamp() + 3600  # 03:30 EDT
    assert bucket(spring, tz, "here", 0).startswith("night/")
    assert bucket(datetime(2026, 3, 8, 5, 0, tzinfo=tz).timestamp(), tz, "here", 0).startswith("morning/")
    assert bucket(datetime(2026, 11, 1, 4, 59, tzinfo=tz).timestamp(), tz, "here", 0).startswith("night/")
    assert bucket(datetime(2026, 11, 1, 5, 0, tzinfo=tz).timestamp(), tz, "here", 0).startswith("morning/")
