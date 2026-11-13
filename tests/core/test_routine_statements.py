"""Statements: the evidence rules at every threshold, the templates, the caps and the rounding (v0.12 §4.4, §4.5, A10)."""
import math
from datetime import datetime, timedelta, timezone

import pytest

from riya.core import shaping
from riya.core.memory import Glance
from riya.core.routine import predictors as pr
from riya.core.routine import statements as st
from riya.core.routine.appraisal import Appraisal

TZ = timezone(timedelta(hours=5, minutes=30))
DAY = 86400.0
SLOTS = {"since": "9:05 am", "back": "6:00 pm", "took": "25 minutes", "agent": "riya", "median": "10 minutes",
         "lo": "2 minutes", "hi": "30 minutes", "left": "6 minutes", "band": "weekday evenings"}


def _all_rendered():
    for (kind, variant), options in st.TEMPLATES.items():
        for i in range(len(options)):
            yield kind, variant, st.render(kind, variant, SLOTS, i)


def _a(*, surprised=True, startled=False, n=8.0, c=0.7):
    return Appraisal(topic="return", predictor="P1", bucket="b", surprise=2.0, s=1.0, direction="late",
                     surprised=surprised, startled=startled, confidence=c, change=False, n_eff=n, value=1.0)


def _g(at, app="Visual Studio Code", activity="editing a python file"):
    return Glance(id=0, at=at, app=app, title="", activity=activity, details="", notable=None)


def test_every_hunch_has_a_hedge_and_no_claim_does():
    for kind, variant, text in _all_rendered():
        if "hunch" in variant or kind in ("S3", "S6", "S8"):
            assert st.HEDGE_RE.search(text), (kind, variant, text)
        if "claim" in variant:
            assert not st.HEDGE_RE.search(text), (kind, variant, text)


def test_no_template_holds_a_banned_phrase_or_an_emoji():
    assert set(st.BANNED) >= {"why didn't you", "you left", "ignored me", "abandon", "where have you been all"}
    for kind, variant, text in _all_rendered():
        folded = text.casefold()
        assert not any(b in folded for b in st.BANNED), text
        assert shaping.count_emoji(text) == 0, text


def test_each_kind_and_variant_has_three_or_four_templates():
    want = {"S1": {"claim", "claim_coding", "claim_bid", "claim_bid_coding", "hunch"},
            "S2": {"early_claim", "late_claim", "early_hunch", "late_hunch"}, "S3": {"hunch"},
            "S4": {"longer_claim", "shorter_claim", "longer_hunch", "shorter_hunch"}, "S5": {"claim", "hunch"},
            "S6": {"hunch", "hunch_soon"}, "S7": {"claim", "hunch", "bid_hunch"}, "S8": {"hunch"},
            "unknown": {"line"}}
    assert set(st.TEMPLATES) == {(k, v) for k, vs in want.items() for v in vs}
    for options in st.TEMPLATES.values():
        assert 3 <= len(options) <= 4


def test_claim_wording_follows_the_sensors():
    for kind, variant, text in _all_rendered():
        assert "working" not in text.casefold()
        if "coding" in text.casefold():
            assert variant in ("claim_coding", "claim_bid_coding"), text


def test_s1_at_every_threshold():
    assert st.rule_s1(3599, False) is None
    assert st.rule_s1(3600, False) == "claim"
    assert st.rule_s1(3600, True) == "hunch"
    now, t0 = 10_000.0, 5_000.0
    assert not st.coding_ok([_g(now - 10)], t0, now)
    assert st.coding_ok([_g(now - 900), _g(now - 10)], t0, now)
    assert st.coding_ok([_g(now - 900), _g(now - 600)], t0, now)
    assert not st.coding_ok([_g(now - 900), _g(now - 601)], t0, now)
    assert not st.coding_ok([_g(t0 - 1), _g(now - 10)], t0, now)  # a glance before the stretch doesn't count
    assert not st.coding_ok([_g(now - 900, app="Firefox", activity="watching a video"), _g(now - 10)], t0, now)


def test_s2_and_s4_at_every_threshold():
    assert st.rule_s2(_a(startled=True, n=8.0, c=0.7)) == "claim"
    assert st.rule_s2(_a(startled=True, n=7.99, c=0.7)) == "hunch"
    assert st.rule_s2(_a(startled=True, n=8.0, c=0.69)) == "hunch"
    assert st.rule_s2(_a(n=4.0)) == "hunch"
    assert st.rule_s2(_a(n=3.99)) is None
    assert st.rule_s2(_a(surprised=False, n=50.0, c=0.9)) is None


def test_s3_s5_s6_s7_s8_at_every_threshold():
    assert st.rule_s3(4.0, 0.4) and not st.rule_s3(3.99, 0.4) and not st.rule_s3(4.0, 0.39)

    assert st.rule_s5(8.0, 0.7, 0.8) == "claim"
    assert st.rule_s5(7.99, 0.7, 0.8) == "hunch"
    assert st.rule_s5(8.0, 0.69, 0.8) == "hunch"
    assert st.rule_s5(8.0, 0.7, 0.79) == "hunch"
    assert st.rule_s5(4.0, 0.0, 0.0) == "hunch"
    assert st.rule_s5(3.99, 0.9, 0.9) is None

    assert st.rule_s6(4.0) and not st.rule_s6(3.99)

    # P(p > 0.5) at the 0.95 boundary: Beta(6, 1) gives 1 - 0.5**6 ≈ 0.984; Beta(4, 1) gives 0.9375
    hi, lo = pr.beta_sf(6, 1), pr.beta_sf(4, 1)
    assert hi >= st.S7_P > lo
    assert st.rule_s7(0.7, 8.0, hi, 0.7) == "claim"
    assert st.rule_s7(0.7, 8.0, lo, 0.7) == "hunch"
    assert st.rule_s7(0.69, 8.0, hi, 0.7) == "hunch"
    assert st.rule_s7(0.7, 7.99, hi, 0.7) == "hunch"
    assert st.rule_s7(0.7, 8.0, hi, 0.69) == "hunch"
    assert st.rule_s7(0.5, 4.0, 0.0, 0.0) == "hunch"
    assert st.rule_s7(0.49, 4.0, 0.0, 0.0) is None
    assert st.rule_s7(0.5, 3.99, 0.0, 0.0) is None
    assert st.s7_bid_ok(1000.0, 1000.0 + 3600) and not st.s7_bid_ok(1000.0, 1000.0 + 3601)
    assert not st.s7_bid_ok(1000.0, 999.0)  # the band already began

    assert st.rule_s8(8.0, None, 0.0) and not st.rule_s8(7.99, None, 0.0)
    assert not st.rule_s8(8.0, 0.0, 7 * DAY - 1) and st.rule_s8(8.0, 0.0, 7 * DAY)


def test_absence_remark_caps():
    now, p50, p90 = 100 * DAY, 3600.0, 10_000.0
    need = max(p50 + 7200, 1.5 * p90)
    ok = dict(away_s=need, p50_s=p50, p90_s=p90, clinginess=0.7, quiet=False, short=False, recent=[], now=now)
    assert st.absence_ok(**ok)
    assert not st.absence_ok(**{**ok, "clinginess": 0.69})
    assert not st.absence_ok(**{**ok, "quiet": True})
    assert not st.absence_ok(**{**ok, "short": True})
    assert not st.absence_ok(**{**ok, "away_s": need - 1})

    def remark(ago):
        return st.SaidRow(at=now - ago, kind="S2", where="greeting", text="x", absence=True)
    assert not st.absence_ok(**{**ok, "recent": [remark(3600)]})
    assert st.absence_ok(**{**ok, "recent": [remark(DAY + 1)]})
    week = [remark(1.5 * DAY), remark(3 * DAY), remark(5 * DAY)]
    assert not st.absence_ok(**{**ok, "recent": week})
    assert st.absence_ok(**{**ok, "recent": week[:2]})


def test_volume_caps():
    now = 100 * DAY

    def row(ago, kind, where="bid", text=None):
        return st.SaidRow(at=now - ago, kind=kind, where=where, text=text or f"{kind}-{ago}", absence=False)
    four = [row(100, "S1"), row(200, "S2", "greeting"), row(300, "S4", "report"), row(400, "S8")]
    assert not st.voluntary_ok("S7", "bid", "new", four, now)
    assert st.voluntary_ok("S7", "bid", "new", four[:3], now)
    assert not st.voluntary_ok("S1", "bid", "new", [row(100, "S1")], now)
    assert st.voluntary_ok("S1", "bid", "new", [row(DAY + 1, "S1")], now)
    same = "You've been at it since 9:05 am."
    assert not st.voluntary_ok("S1", "bid", same, [row(3 * DAY - 1, "S8", text=same)], now)
    assert st.voluntary_ok("S1", "bid", same, [row(3 * DAY + 1, "S8", text=same)], now)
    assert st.voluntary_ok("S1", "reply", same, four + [row(10, "S1", text=same)], now)


def test_values_are_rounded():
    t = datetime(2026, 11, 13, 17, 53, tzinfo=TZ).timestamp()
    back = datetime.fromtimestamp(st.round_back(t, TZ), TZ)
    assert (back.hour, back.minute, back.second) == (18, 0, 0)
    t2 = datetime(2026, 11, 13, 17, 37, tzinfo=TZ).timestamp()
    back2 = datetime.fromtimestamp(st.round_back(t2, TZ), TZ)
    assert (back2.hour, back2.minute) == (17, 30)
    assert st.round_minutes(1490) == 25
    assert st.round_minutes(1730) == 30
    assert st.round_minutes(10 * 60 + 20) == 10
    assert st.fmt_span(25 * 60) == "25 minutes"
    assert st.fmt_span(60) == "1 minute"
    assert st.fmt_span(2 * 3600) == "about 2 hours"
    assert st.fmt_span(3600) == "about an hour"
    soon = st.render("S6", "hunch_soon", {}, 0)
    assert st.HEDGE_RE.search(soon) and not any(ch.isdigit() for ch in soon)


def test_the_clauses_are_the_spec_s():
    assert st.CLAUSES == {
        "early": "pleasantly surprised: they came back much earlier than you expected",
        "late": "surprised: they were away much longer than usual",
        "answered": "pleasantly surprised: they answered when you didn't expect it",
        "unsure": "unsure: their routine looks different lately, so ask rather than assume",
    }


def test_render_rotates_and_fills_slots():
    texts = {st.render("S1", "claim", SLOTS, i) for i in range(3)}
    assert len(texts) == 3 and all("9:05 am" in t for t in texts)
    assert st.render("S1", "claim", SLOTS, 3) == st.render("S1", "claim", SLOTS, 0)
    assert math.isfinite(len(st.render("S7", "claim", SLOTS, 1)))
