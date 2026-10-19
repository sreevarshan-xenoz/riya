"""HandsDB: reminders, notes and the action log on schema v3."""
import sqlite3

import pytest

from riya.core.hands.records import MAX_ACTIVE, OUTCOMES, HandsDB
from riya.core.store import open_db

DAY = 86400


@pytest.fixture
def h():
    return HandsDB(open_db(":memory:"))


def test_add_active_and_order(h):
    b = h.add_reminder("reminder", 200.0, "b", 1.0)
    a = h.add_reminder("timer", 100.0, "timer", 1.0)
    assert [r.id for r in h.active()] == [a, b]
    r = h.active()[0]
    assert (r.kind, r.due_at, r.text, r.created_at) == ("timer", 100.0, "timer", 1.0)


def test_cap_of_fifty_and_cancel_frees_a_slot(h):
    ids = [h.add_reminder("reminder", 1000.0 + i, f"r{i}", 0.0) for i in range(MAX_ACTIVE)]
    assert None not in ids and h.add_reminder("reminder", 5.0, "one too many", 0.0) is None
    assert len(h.active()) == MAX_ACTIVE
    assert h.cancel(ids[3], 1.0) is True
    assert h.add_reminder("reminder", 5.0, "fits now", 0.0) is not None


def test_fired_ones_do_not_count_against_the_cap(h):
    ids = [h.add_reminder("reminder", 10.0, f"r{i}", 0.0) for i in range(MAX_ACTIVE)]
    h.mark_fired(ids[:5], 20.0)
    assert h.add_reminder("reminder", 30.0, "x", 0.0) is not None


def test_due_skips_fired_and_cancelled_and_never_loses_a_late_one(h):
    a = h.add_reminder("reminder", 100.0, "a", 0.0)
    b = h.add_reminder("reminder", 110.0, "b", 0.0)
    c = h.add_reminder("reminder", 120.0, "c", 0.0)
    d = h.add_reminder("reminder", 10_000.0, "later", 0.0)
    assert [r.id for r in h.due(99.0)] == []
    assert [r.id for r in h.due(130.0)] == [a, b, c]
    assert [r.id for r in h.due(130.0 + 30 * DAY)] == [a, b, c, d]      # however late, still returned until fired
    h.mark_fired([a], 131.0)
    h.cancel(b, 131.0)
    assert [r.id for r in h.due(1e9)] == [c, d]


def test_mark_fired_is_idempotent_and_ignores_cancelled(h):
    a = h.add_reminder("reminder", 100.0, "a", 0.0)
    b = h.add_reminder("reminder", 100.0, "b", 0.0)
    h.cancel(b, 5.0)
    h.mark_fired([a, b, 999], 150.0)
    h.mark_fired([a, b], 900.0)
    h.mark_fired([], 900.0)
    rows = {r["id"]: (r["fired_at"], r["cancelled_at"]) for r in h.conn.execute("select * from reminders")}
    assert rows == {a: (150.0, None), b: (None, 5.0)}


def test_cancel_returns_false_for_unknown_fired_or_cancelled(h):
    a = h.add_reminder("reminder", 100.0, "a", 0.0)
    assert h.cancel(999, 1.0) is False
    assert h.cancel(a, 1.0) is True and h.cancel(a, 2.0) is False
    b = h.add_reminder("reminder", 100.0, "b", 0.0)
    h.mark_fired([b], 3.0)
    assert h.cancel(b, 4.0) is False


def test_find_active(h):
    h.add_reminder("reminder", 300.0, "Call Mom later", 0.0)
    soon = h.add_reminder("reminder", 200.0, "call mom now", 0.0)
    h.add_reminder("reminder", 100.0, "water plants", 0.0)
    assert h.find_active("MOM").id == soon                         # the soonest match, case-insensitive
    assert h.find_active("nothing") is None and h.find_active("  ") is None
    h.cancel(soon, 1.0)
    assert h.find_active("mom").text == "Call Mom later"           # cancelled ones are never found


def test_find_active_timer_prefers_a_real_timer_newest_first(h):
    h.add_reminder("timer", 900.0, "timer", 0.0)
    t2 = h.add_reminder("timer", 500.0, "timer", 5.0)
    h.add_reminder("reminder", 100.0, "water plants", 0.0)
    assert h.find_active("timer").id == t2                         # the newest-created timer
    h.cancel(t2, 6.0)
    assert h.find_active("timer").due_at == 900.0


def test_find_active_timer_beats_a_reminder_that_mentions_one(h):
    h.add_reminder("reminder", 100.0, "check the timer on oven", 0.0)       # due sooner, and the text contains "timer"
    t = h.add_reminder("timer", 900.0, "timer", 0.0)
    assert h.find_active("timer").id == t
    assert h.find_active("oven").text == "check the timer on oven"
    h.cancel(t, 1.0)
    assert h.find_active("timer").text == "check the timer on oven"         # no real timer left: the text match


def test_like_wildcards_in_the_search_are_literal(h):
    h.add_reminder("reminder", 1.0, "call mom", 0.0)
    assert h.find_active("%") is None and h.find_active("c_ll") is None


def test_notes(h):
    n1, n2 = h.add_note("buy milk", 1.0), h.add_note("call dad", 2.0)
    assert [n.text for n in h.notes()] == ["call dad", "buy milk"]
    assert h.delete_note(n1) is True and h.delete_note(n1) is False
    assert [n.id for n in h.notes()] == [n2]


def test_log_action_outcomes(h):
    assert OUTCOMES == ("ran", "failed", "refused", "declined", "expired", "cancelled", "replaced")
    for o in OUTCOMES:
        h.log_action(1.0, "open_url", 1, "open x", o)
    assert len(h.actions()) == len(OUTCOMES)
    for bad in ("ok", "", "RAN", None):
        with pytest.raises(ValueError):
            h.log_action(1.0, "t", 0, "s", bad)
    assert len(h.actions()) == len(OUTCOMES)


def test_actions_newest_first_limit_and_clear(h):
    for i in range(5):
        h.log_action(float(i), "calc", 0, f"s{i}", "ran")
    assert [a.summary for a in h.actions()] == ["s4", "s3", "s2", "s1", "s0"]
    assert [a.summary for a in h.actions(limit=2)] == ["s4", "s3"]
    a = h.actions(1)[0]
    assert (a.at, a.tool, a.tier, a.outcome) == (4.0, "calc", 0, "ran")
    h.clear_actions()
    assert h.actions() == []


def test_purge_thirty_days(h):
    now = 100 * DAY
    h.log_action(now - 29 * DAY, "calc", 0, "recent", "ran")
    h.log_action(now - 31 * DAY, "calc", 0, "old", "ran")
    h.add_note("ancient note", now - 400 * DAY)
    old_fired = h.add_reminder("reminder", 1.0, "old fired", 0.0)
    old_cancelled = h.add_reminder("reminder", 1.0, "old cancelled", 0.0)
    recent = h.add_reminder("reminder", 1.0, "recent fired", 0.0)
    overdue_active = h.add_reminder("reminder", 1.0, "never fired, long overdue", 0.0)
    h.mark_fired([old_fired], now - 31 * DAY)
    h.cancel(old_cancelled, now - 40 * DAY)
    h.mark_fired([recent], now - 29 * DAY)
    h.purge(now)
    assert [a.summary for a in h.actions()] == ["recent"]
    assert [n.text for n in h.notes()] == ["ancient note"]                     # notes are never purged
    ids = {r["id"] for r in h.conn.execute("select id from reminders")}
    assert ids == {recent, overdue_active}                                    # an active reminder is never purged
    assert [r.id for r in h.due(now)] == [overdue_active]


def test_persistence_across_reopen(tmp_path):
    p = tmp_path / "riya.db"
    h = HandsDB(open_db(p))
    rid = h.add_reminder("reminder", 500.0, "call mom", 1.0)
    h.add_note("buy milk", 2.0)
    h.log_action(3.0, "calc", 0, "s", "ran")
    h.conn.close()
    h2 = HandsDB(open_db(p))
    assert [(r.id, r.text) for r in h2.active()] == [(rid, "call mom")]
    assert [n.text for n in h2.notes()] == ["buy milk"] and len(h2.actions()) == 1


def test_kind_is_constrained(h):
    with pytest.raises(sqlite3.IntegrityError):
        h.add_reminder("alarm", 1.0, "x", 0.0)
