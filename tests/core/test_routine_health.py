"""The Health row "honesty" (v0.12 §4.6, Resolution 30, A18)."""
from types import SimpleNamespace

import pytest

import riya.core.routine as routine
from riya.core import health
from riya.core.health import DETAIL_MAX, check_honesty, honesty_detail


def h(t=0, n=0, hr=0, hn=0, enabled=True, claims=True):
    return SimpleNamespace(enabled=enabled, claims_enabled=claims, claims_true=t, claims_scored=n, hunches_right=hr,
                           hunches_scored=hn, unverifiable=0, expired=0)


def test_the_honesty_row_texts():
    assert honesty_detail(h(enabled=False)) == ("off", "Off: she isn't noticing your routines.")
    assert honesty_detail(h(claims=False, hr=31, hn=52)) == \
        ("ok", "Hunches only in this version · hunches right 31 of 52 · 30 days")
    assert honesty_detail(h()) == ("ok", "No claims checked yet: she needs a few more days")
    assert honesty_detail(h(3, 3)) == ("ok", "3 claims checked so far (all true): she needs a few more days")
    assert honesty_detail(h(2, 3)) == ("ok", "3 claims checked so far (2 true): she needs a few more days")
    assert honesty_detail(h(17, 20)) == ("warn", "Only 17 of 20 claims held up (85%): tell Claude")
    assert honesty_detail(h(47, 48, 31, 52)) == \
        ("ok", "Honest 47 of 48 claims (98%) · hunches right 31 of 52 · 30 days")


def test_the_honesty_row_boundaries():
    assert honesty_detail(h(16, 19))[0] == "ok"  # 84%, but under 20: never a warn
    assert honesty_detail(h(17, 20))[0] == "warn"
    assert honesty_detail(h(18, 20)) == ("ok", "Honest 18 of 20 claims (90%) · hunches right 0 of 0 · 30 days")
    assert "so far" in honesty_detail(h(9, 9))[1]
    assert honesty_detail(h(10, 10))[1].startswith("Honest 10 of 10")


def test_a_missing_honesty_reads_off():
    ctx = SimpleNamespace(honesty=None)
    item = check_honesty(ctx)
    assert (item.key, item.status) == ("honesty", "off")


@pytest.mark.parametrize("case", [h(enabled=False), h(claims=False, hr=9999, hn=9999), h(), h(9, 9), h(8, 9),
                                  h(1700, 2000, 9999, 9999), h(9999, 9999, 9999, 9999)])
def test_every_honesty_text_fits_the_detail(case):
    assert len(honesty_detail(case)[1]) <= DETAIL_MAX
    assert check_honesty(SimpleNamespace(honesty=case)).detail == honesty_detail(case)[1]


def test_the_row_is_last():
    assert health.ROWS[-1] == "honesty" and health.CHECKS["honesty"] is check_honesty


def test_the_engine_reads_her_honesty_into_health(monkeypatch):
    monkeypatch.setattr(routine, "CLAIMS_ENABLED", True)  # the claim path itself; claims ship off (held-out record)
    from tests.fakes import make_core
    core, *_ = make_core(routine=True)
    item = check_honesty(core._health_ctx())
    assert (item.status, item.detail) == ("ok", "No claims checked yet: she needs a few more days")
    off, *_ = make_core()
    assert check_honesty(off._health_ctx()).status == "off"
