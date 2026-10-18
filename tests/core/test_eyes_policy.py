import time
from datetime import datetime

import pytest

from riya.core.clock import FakeClock
from riya.core.eyes.capture import WindowInfo
from riya.core.eyes.policy import DENY_PROCESSES, DENY_TITLES, GlancePolicy, denied, validate_patterns

MON = (0, 0, 1920, 1080)
FLAT = bytes([100]) * (64 * 36)


def info(**kw):
    base = dict(hwnd=1, title="main.py - Riya", process="code.exe", rect=(0, 0, 800, 600), monitor=MON,
                minimized=False, is_riya=False)
    return WindowInfo(**{**base, **kw})


def ok(**kw):
    base = dict(info=info(), presence="focused", eyes_enabled=True, dont_look=False, online_ok=True)
    return {**base, **kw}


def make(**kw):
    clock = FakeClock()
    return GlancePolicy(clock, **kw), clock


def shifted(delta):  # a thumbnail whose MAD against FLAT is delta/255
    return bytes([100 + delta]) * (64 * 36)


# ---- denylist ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Bitwarden - Vault", "KeePass Database", "1Password", "LastPass: login", "HDFC NetBanking", "Net Banking",
    "State Bank of India", "PayPal: Send money", "Pay with UPI", "Add a credit card", "New InPrivate window",
    "Gmail - Google Chrome (Incognito)", "Private Browsing - Firefox", "Windows Security", "Credential Manager",
    "Sign in to your account", "Login - Example", "PASSWORD reset", "Tab - Microsoft Edge [InPrivate]",
])
def test_default_titles_denied(title):
    assert denied(info(title=title), (), ())


@pytest.mark.parametrize("process", ["keepass.exe", "KeePassXC.exe", "1Password.exe", "Bitwarden.exe",
                                     "CredentialUIBroker.exe"])
def test_default_processes_denied(process):
    assert denied(info(process=process), (), ())


@pytest.mark.parametrize("title,process", [("main.py - Riya", "code.exe"), ("Untitled - Notepad", "notepad.exe"),
                                           ("", "explorer.exe"), ("Docs", "chrome.exe")])
def test_ordinary_windows_not_denied(title, process):
    assert not denied(info(title=title, process=process), (), ())


def test_defaults_are_the_spec_patterns():
    assert DENY_TITLES.startswith("password|1password|bitwarden") and DENY_TITLES.endswith("sign in to|login")
    assert DENY_PROCESSES == "keepass.*|1password.*|bitwarden.*|credentialuibroker"


def test_extra_patterns_case_insensitive_on_own_field():
    assert denied(info(title="My Secret Notes"), ["secret"], ())
    assert denied(info(process="Vault.EXE"), (), [r"vault\.exe"])
    assert not denied(info(title="vault"), (), ["vault"])  # a process pattern doesn't look at the title
    assert not denied(info(process="vault.exe"), ["vault"], ())


def test_invalid_extra_regex_is_skipped_never_raises():
    assert not denied(info(), ["(unclosed", "["], ["*bad"])
    assert denied(info(title="secret"), ["(unclosed", "secret"], ())
    assert validate_patterns(["ok", "(unclosed", r"\d+", "["]) == ["(unclosed", "["]
    assert validate_patterns([]) == []


def test_blank_extra_pattern_does_not_match_everything():
    assert not denied(info(), ["", "  "], [""])
    assert validate_patterns(["", "  "]) == []


def test_empty_process_is_denied():
    assert denied(info(process=""), (), ())


def test_unicode_and_emoji_titles():
    assert not denied(info(title="こんにちは 🌸 – メモ帳"), (), ())
    assert denied(info(title="パスワード 🔑 password"), (), ())
    assert denied(info(title="Привет 🙂 секрет"), ["секрет"], ())  # extra pattern in another script
    assert denied(info(title="ÜBERWEISUNG BANK 🏦"), (), ())
    assert not denied(info(title="lone \ud800 surrogate"), (), ())


# ---- allowed: every reason ---------------------------------------------------------------------------------------

def test_allowed_when_everything_holds():
    p, _ = make()
    assert p.allowed(**ok()) is None
    assert p.allowed(**ok(presence="talking")) is None


@pytest.mark.parametrize("kw,reason", [
    (dict(eyes_enabled=False), "eyes off"),
    (dict(dont_look=True), "don't look"),
    (dict(presence="away"), "presence away"),
    (dict(presence="break"), "presence break"),
    (dict(presence="gaming"), "presence gaming"),
    (dict(info=None), "no window"),
    (dict(info=info(minimized=True)), "minimized"),
    (dict(info=info(is_riya=True)), "riya window"),
    (dict(info=info(process="")), "unknown process"),
    (dict(info=info(title="Bitwarden")), "denylisted"),
    (dict(info=info(process="keepassxc.exe")), "denylisted"),
    (dict(extra_titles=["main"]), "denylisted"),
    (dict(extra_processes=["code"]), "denylisted"),
    (dict(online_ok=False), "online model unavailable"),
])
def test_each_deny_reason(kw, reason):
    p, _ = make()
    assert p.allowed(**ok(**kw)) == reason


def test_daily_cap_reason():
    p, _ = make(cap=2)
    p.record(info(), FLAT)
    p.record(info(), FLAT)
    assert p.allowed(**ok(), force=True) == "daily cap"


def test_min_gap_reason_and_release():
    p, clock = make()
    p.record(info(), FLAT)
    clock.advance(44.9)
    assert p.allowed(**ok()) == "min gap"
    clock.advance(0.2)
    assert p.allowed(**ok()) is None


def test_custom_min_gap():
    p, clock = make(min_gap_s=10)
    p.record(info(), FLAT)
    clock.advance(10)
    assert p.allowed(**ok()) is None


def test_force_bypasses_min_gap_and_backoff_only():
    p, _ = make(cap=1)
    p.record(info(), FLAT)
    assert p.allowed(**ok()) == "daily cap"
    p2, _ = make()
    p2.record(info(), FLAT)
    assert p2.allowed(**ok()) == "min gap"
    assert p2.allowed(**ok(), force=True) is None
    for kw, reason in [(dict(dont_look=True), "don't look"), (dict(eyes_enabled=False), "eyes off"),
                       (dict(info=info(title="password")), "denylisted"), (dict(info=info(is_riya=True)), "riya window"),
                       (dict(presence="gaming"), "presence gaming"), (dict(online_ok=False), "online model unavailable"),
                       (dict(info=info(minimized=True)), "minimized"), (dict(info=info(process="")), "unknown process")]:
        assert p2.allowed(**ok(**kw), force=True) == reason
    p2.backoff(1800)  # under an active backoff, forced still lets every other reason win
    assert p2.allowed(**ok(), force=True) is None
    for kw, reason in [(dict(dont_look=True), "don't look"), (dict(eyes_enabled=False), "eyes off"),
                       (dict(info=info(title="password")), "denylisted"), (dict(info=info(is_riya=True)), "riya window"),
                       (dict(presence="away"), "presence away"), (dict(online_ok=False), "online model unavailable"),
                       (dict(info=info(minimized=True)), "minimized"), (dict(info=info(process="")), "unknown process"),
                       (dict(info=None), "no window")]:
        assert p2.allowed(**ok(**kw), force=True) == reason
    p3, _ = make(cap=1)
    p3.record(info(), FLAT)
    p3.backoff(1800)
    assert p3.allowed(**ok(), force=True) == "daily cap"


def test_privacy_checks_come_before_cap_and_gap():
    p, _ = make(cap=1)
    p.record(info(), FLAT)
    assert p.allowed(**ok(dont_look=True)) == "don't look"
    assert p.allowed(**ok(info=info(title="Bitwarden"))) == "denylisted"


# ---- wanted: settle ----------------------------------------------------------------------------------------------

def test_new_window_wanted_after_settle():
    p, clock = make()
    a = info()
    assert not p.wanted(a, None)
    clock.advance(3.9)
    assert not p.wanted(a, None)
    clock.advance(0.2)
    assert p.wanted(a, None)


def test_window_change_restarts_settle():
    p, clock = make()
    a, b = info(hwnd=1), info(hwnd=2)
    p.wanted(a, None)
    clock.advance(3)
    assert not p.wanted(b, None)
    clock.advance(3)
    assert not p.wanted(b, None)  # only 3 s since b appeared
    clock.advance(1.1)
    assert p.wanted(b, None)


def test_title_change_counts_as_change():
    p, clock = make()
    p.wanted(info(title="a"), None)
    clock.advance(10)
    assert p.wanted(info(title="a"), None)
    assert not p.wanted(info(title="b"), None)
    clock.advance(4)
    assert p.wanted(info(title="b"), None)


def test_settle_custom_and_none_window():
    p, clock = make(settle_s=1)
    p.wanted(info(), None)
    clock.advance(1)
    assert p.wanted(info(), None)
    assert not p.wanted(None, None)
    clock.advance(5)
    assert not p.wanted(None, None)
    assert not p.wanted(info(), None)  # reappearing restarts the settle timer


def test_glanced_window_not_wanted_again_by_settle():
    p, clock = make()
    a = info()
    p.wanted(a, None)
    clock.advance(5)
    assert p.wanted(a, None)
    p.record(a, FLAT)
    assert not p.wanted(a, FLAT)
    clock.advance(10)
    assert not p.wanted(a, FLAT)


def test_back_to_an_earlier_window_is_a_change_again():
    p, clock = make()
    a, b = info(hwnd=1), info(hwnd=2)
    p.wanted(a, None)
    clock.advance(5)
    p.record(a, FLAT)
    p.wanted(b, None)
    clock.advance(5)
    assert p.wanted(b, None)
    p.record(b, FLAT)
    p.wanted(a, None)
    clock.advance(5)
    assert p.wanted(a, None)  # b was the last glance, so a is a change again


# ---- wanted: refresh ---------------------------------------------------------------------------------------------

def settled_and_glanced(**kw):
    p, clock = make(**kw)
    a = info()
    p.wanted(a, None)
    clock.advance(5)
    p.record(a, FLAT)
    return p, clock, a


def test_refresh_needs_time_and_change():
    p, clock, a = settled_and_glanced()
    clock.advance(299)
    assert not p.wanted(a, shifted(40))  # changed, too early
    clock.advance(1)
    assert p.wanted(a, shifted(40))
    assert not p.wanted(a, FLAT)  # old enough, but unchanged


def test_refresh_mad_threshold_is_strict_six_over_255():
    p, clock, a = settled_and_glanced()
    clock.advance(400)
    assert not p.wanted(a, shifted(6))  # exactly 6/255 is not above it
    assert p.wanted(a, shifted(7))


def test_refresh_thumb_none_is_unknown_not_wanted():
    p, clock, a = settled_and_glanced()
    clock.advance(1000)
    assert not p.wanted(a, None)


def test_refresh_empty_thumb_is_unknown_too():
    p, clock, a = settled_and_glanced()
    clock.advance(1000)
    assert not p.wanted(a, b"")  # a failed grab: don't retry every tick


def test_refresh_incomparable_thumb_counts_as_changed():
    p, clock, a = settled_and_glanced()
    clock.advance(400)
    assert p.wanted(a, b"short")  # mad() says 1.0 for different sizes


def test_record_without_thumb_then_refresh_by_time():
    p, clock = make()
    a = info()
    p.record(a, None)
    clock.advance(400)
    assert not p.wanted(a, FLAT)  # nothing to compare against: unknown, not wanted


def test_refresh_clock_restarts_at_each_glance():
    p, clock, a = settled_and_glanced()
    clock.advance(301)
    assert p.wanted(a, shifted(40))
    p.record(a, shifted(40))
    clock.advance(100)
    assert not p.wanted(a, FLAT)


def test_custom_refresh_and_threshold():
    p, clock, a = settled_and_glanced(refresh_s=60, mad_threshold=0.5)
    clock.advance(61)
    assert not p.wanted(a, shifted(100))  # 100/255 < 0.5
    assert p.wanted(a, shifted(150))


# ---- cap and midnight --------------------------------------------------------------------------------------------

def test_glances_today_counts_records():
    p, _ = make()
    assert p.glances_today() == 0
    p.record(info(), FLAT)
    p.record(info(), FLAT)
    assert p.glances_today() == 2


def local_ts(y, m, d, h, mi=0, s=0):
    return datetime(y, m, d, h, mi, s).timestamp()


def test_cap_resets_at_local_midnight():
    clock = FakeClock(local_ts(2026, 10, 18, 23, 59, 0))
    p = GlancePolicy(clock, cap=2, min_gap_s=0)
    p.record(info(), FLAT)
    p.record(info(), FLAT)
    assert p.allowed(**ok()) == "daily cap"
    clock.advance(30)
    assert p.glances_today() == 2
    assert p.allowed(**ok()) == "daily cap"
    clock.advance(31)  # 00:00:01
    assert p.glances_today() == 0
    assert p.allowed(**ok()) is None


def test_midnight_uses_local_time_not_utc(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("needs tzset")
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        clock = FakeClock(local_ts(2026, 10, 18, 23, 59, 50))
        p = GlancePolicy(clock, cap=1, min_gap_s=0)
        p.record(info(), FLAT)
        clock.advance(20)
        assert p.glances_today() == 0
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_clock_jump_back_across_midnight_keeps_count():
    clock = FakeClock(local_ts(2026, 10, 19, 0, 30))
    p = GlancePolicy(clock, cap=1, min_gap_s=0)
    p.record(info(), FLAT)
    clock.t = local_ts(2026, 10, 18, 22)
    assert p.glances_today() == 1
    assert p.allowed(**ok()) == "daily cap"


# ---- backoff -----------------------------------------------------------------------------------------------------

def test_backoff_blocks_then_releases_and_forced_gets_through():
    p, clock = make()
    p.backoff(1800)
    clock.advance(1799)
    assert p.allowed(**ok()) == "backoff"
    assert p.allowed(**ok(), force=True) is None
    clock.advance(2)
    assert p.allowed(**ok()) is None


def test_forced_attempt_during_backoff_then_failure_rearms_it():
    p, clock = make()
    p.backoff(1800)
    clock.advance(100)
    assert p.allowed(**ok(), force=True) is None
    p.backoff(1800)  # the forced attempt failed
    assert p.allowed(**ok()) == "backoff"
    clock.advance(1700)
    assert p.allowed(**ok()) == "backoff"  # re-armed from the later failure
    clock.advance(201)
    assert p.allowed(**ok()) is None


def test_backoff_never_shortens():
    p, clock = make()
    p.backoff(1800)
    p.backoff(60)
    clock.advance(100)
    assert p.allowed(**ok()) == "backoff"


def test_backoff_nonpositive_is_noop():
    p, _ = make()
    p.backoff(0)
    p.backoff(-5)
    assert p.allowed(**ok()) is None


def test_backoff_comes_after_privacy_reasons():
    p, _ = make()
    p.backoff(100)
    assert p.allowed(**ok(dont_look=True)) == "don't look"


# ---- clock going backwards ---------------------------------------------------------------------------------------

def test_clock_back_does_not_unlock_min_gap_early():
    p, clock = make()
    p.record(info(), FLAT)
    clock.advance(30)
    assert p.allowed(**ok()) == "min gap"
    clock.advance(-1000)
    assert p.allowed(**ok()) == "min gap"
    clock.advance(14)  # 44 s of real progress since the glance
    assert p.allowed(**ok()) == "min gap"
    clock.advance(1.1)
    assert p.allowed(**ok()) is None


def test_clock_back_keeps_backoff_remaining():
    p, clock = make()
    p.backoff(100)
    clock.advance(40)
    assert p.allowed(**ok()) == "backoff"
    clock.advance(-5000)
    assert p.allowed(**ok()) == "backoff"
    clock.advance(59)
    assert p.allowed(**ok()) == "backoff"
    clock.advance(2)
    assert p.allowed(**ok()) is None


def test_clock_back_does_not_settle_early_or_go_negative():
    p, clock = make()
    a = info()
    p.wanted(a, None)
    clock.advance(3)
    p.wanted(a, None)
    clock.advance(-1000)
    assert not p.wanted(a, None)
    clock.advance(0.9)
    assert not p.wanted(a, None)
    clock.advance(0.2)
    assert p.wanted(a, None)


def test_clock_back_does_not_refresh_early():
    p, clock, a = settled_and_glanced()
    clock.advance(200)
    assert not p.wanted(a, shifted(40))
    clock.advance(-5000)
    assert not p.wanted(a, shifted(40))
    clock.advance(99)
    assert not p.wanted(a, shifted(40))
    clock.advance(2)
    assert p.wanted(a, shifted(40))


def test_today_count_can_be_saved_and_restored_the_same_day():
    p, clock = make()
    p.record(info(), None)
    p.record(info(hwnd=2), None)
    day, n = p.today()
    assert day == datetime.fromtimestamp(clock.now()).date().isoformat() and n == 2
    again = GlancePolicy(clock)
    again.restore(day, n)
    assert again.glances_today() == 2


@pytest.mark.parametrize("day,count", [("1999-01-01", 5), (None, 5), ("x", 5), ("today", "5"), ("today", -3),
                                       ("today", True)])
def test_restore_ignores_another_day_or_junk(day, count):
    p, clock = make()
    today = datetime.fromtimestamp(clock.now()).date().isoformat()
    p.restore(today if day == "today" else day, count)
    assert p.glances_today() == 0


def test_restored_count_still_resets_at_midnight():
    p, clock = make(cap=3)
    p.restore(p.today()[0], 3)
    assert p.allowed(**ok()) == "daily cap"
    clock.advance(86400)
    assert p.glances_today() == 0
