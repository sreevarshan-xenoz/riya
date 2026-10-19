"""Delivery manners, templated lines, and the timer / reminder / note runs. Zones are explicit: no host-zone reliance."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from riya.core.clock import FakeClock
from riya.core.hands.records import HandsDB, Reminder
from riya.core.hands.reminders import (LATE_S, LINE_MAX, MAX_ACTIVE, REMIND_GAP_S, delivery, duration_parts, fmt_clock, fmt_when,
                                       gap_ok, reminder_line)
from riya.core.hands.tools import REGISTRY, ToolCtx
from riya.core.store import open_db

TZ = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def at(h, m=0, day=18, tz=TZ):
    return datetime(2026, 10, day, h, m, tzinfo=tz).timestamp()


def rem(text="call mom", due=None, kind="reminder", created=0.0, id=1):
    return Reminder(id, kind, at(17) if due is None else due, text, created)


def test_constants():
    assert (MAX_ACTIVE, LATE_S, REMIND_GAP_S) == (50, 120, 20)


def test_delivery_table():
    assert delivery("away", False) == "wait" and delivery("away", True) == "wait"
    assert delivery("gaming", False) == "bubble" and delivery("gaming", True) == "bubble"
    for state in ("focused", "talking", "break"):
        assert delivery(state, False) == "speak"
        assert delivery(state, True) == "bubble"        # quiet hours / may_start_conversations off: never dropped


def test_gap_ok():
    assert gap_ok(None, 100.0) and gap_ok(100.0, 120.0) and not gap_ok(100.0, 119.9)
    assert gap_ok(100.0, 50.0)                           # a clock that stepped back


def test_fmt_clock_uses_the_given_zone():
    t = at(17, 0)
    assert fmt_clock(t, TZ) == "5:00 pm"
    assert fmt_clock(t, UTC) == "11:30 am"
    assert fmt_clock(at(0, 5), TZ) == "12:05 am" and fmt_clock(at(12, 0), TZ) == "12:00 pm"
    assert fmt_clock(at(9, 7), TZ) == "9:07 am"


def test_fmt_when_names_the_day():
    now = at(17, 30)
    assert fmt_when(at(18), now, TZ) == "6:00 pm"
    assert fmt_when(at(5, 0, 19), now, TZ) == "5:00 am tomorrow"
    assert fmt_when(at(9, 0, 22), now, TZ) == "9:00 am on Thu 22 Oct"


def test_duration_parts():
    assert duration_parts(600) == (10, "minute") and duration_parts(3600) == (1, "hour")
    assert duration_parts(5400) == (90, "minute") and duration_parts(45) == (45, "second")
    assert duration_parts(60) == (1, "minute") and duration_parts(7200) == (2, "hour")


def test_reminder_line_basic_and_timer():
    now = at(17, 0)
    assert reminder_line([rem()], now, TZ) == "Hey~ it's time to call mom."
    t = rem("timer", kind="timer", created=at(16, 50), due=at(17, 0))
    assert reminder_line([t], now, TZ) == "Your 10-minute timer is done~"
    assert reminder_line([], now, TZ) == ""


def test_late_note_only_beyond_120_seconds():
    due = at(17, 0)
    assert "this was due" not in reminder_line([rem()], due + LATE_S, TZ)
    line = reminder_line([rem()], due + LATE_S + 1, TZ)
    assert line == "Hey~ it's time to call mom. (this was due at 5:00 pm)"
    t = rem("timer", kind="timer", created=at(16, 50), due=due)
    assert reminder_line([t], due + 3600, TZ) == "Your 10-minute timer is done~ (this was due at 5:00 pm)"
    assert "(this was due at 11:30 am)" in reminder_line([rem()], due + 3600, UTC)


def test_several_merge_into_one_line_in_due_order():
    now = at(17, 5)
    items = [rem("water plants", at(17, 1), id=2), rem("call mom", at(17, 0), id=1),
             rem("timer", at(17, 2), "timer", at(16, 52), id=3)]
    line = reminder_line(items, now, TZ)
    assert "\n" not in line
    assert line == ("Hey~ a few things are due: call mom (this was due at 5:00 pm); "
                    "water plants (this was due at 5:01 pm); your 10-minute timer is done (this was due at 5:02 pm).")
    many = [rem(f"r{i}", at(17, 0) + i, id=i) for i in range(9)]
    line = reminder_line(many, at(17, 0), TZ)
    assert "r3" in line and "r4" not in line and line.endswith("and 5 more.")


def test_control_chars_stripped_and_text_cut_to_80():
    line = reminder_line([rem("call\nmom\x00\x1b[31m‮ now\tplease")], at(17), TZ)
    assert "\n" not in line and "\x00" not in line and "\x1b" not in line and "‮" not in line
    assert "call mom [31m now please" in line
    long = reminder_line([rem("x" * 500)], at(17), TZ)
    assert long == "Hey~ it's time to " + "x" * 77 + "...."          # 80 chars of text (77 + "..."), then the full stop
    assert len(reminder_line([rem("y" * 500, id=i) for i in range(2)], at(17), TZ)) < 250


def test_quotes_and_backslashes_are_escaped_as_data():
    line = reminder_line([rem('say "stop" \\ now')], at(17), TZ)
    assert line == 'Hey~ it\'s time to say \\"stop\\" \\\\ now.'


# ---- runs ----
@pytest.fixture
def ctx():
    return ToolCtx(db=HandsDB(open_db(":memory:")), clock=FakeClock(at(17, 30)), tz=TZ)


def run(ctx, tool, **args):
    return REGISTRY[tool].run(args, ctx)


def test_timer_set_is_due_at_now_plus_duration(ctx):
    r = run(ctx, "timer_set", duration=600.0)
    assert r.ok and "10 minutes" in r.text and "5:40 pm" in r.text
    [t] = ctx.db.active()
    assert t.kind == "timer" and t.due_at == ctx.clock.now() + 600.0 and t.created_at == ctx.clock.now()
    assert r.data["id"] == t.id
    assert "1 minute " in run(ctx, "timer_set", duration=60.0).text
    assert "1 hour" in run(ctx, "timer_set", duration=3600.0).text


def test_reminder_set_parses_when_in_the_zone(ctx):
    r = run(ctx, "reminder_set", when="at 6 pm", text="call mom")
    assert r.ok and "6:00 pm" in r.text and '"call mom"' in r.text
    assert ctx.db.active()[0].due_at == at(18)
    r = run(ctx, "reminder_set", when="at 5", text="stretch")                    # 17:30 now: 5 am tomorrow, never refused
    assert r.ok and "tomorrow" in r.text
    assert ctx.db.find_active("stretch").due_at == at(5, 0, 19)
    assert run(ctx, "reminder_set", when="in 20 minutes", text="tea").ok


def test_reminder_set_unparseable_when_is_not_ok(ctx):
    r = run(ctx, "reminder_set", when="whenever you feel like it", text="x")
    assert not r.ok and ctx.db.active() == []
    assert "whenever you feel like it" in r.text


def test_cap_makes_the_run_fail_and_cancel_frees_a_slot(ctx):
    for i in range(MAX_ACTIVE):
        assert run(ctx, "reminder_set", when="in 2 hours", text=f"r{i}").ok
    assert not run(ctx, "reminder_set", when="in 2 hours", text="51st").ok
    assert not run(ctx, "timer_set", duration=60.0).ok
    assert len(ctx.db.active()) == MAX_ACTIVE
    assert run(ctx, "reminder_cancel", text="r7").ok
    assert run(ctx, "reminder_set", when="in 2 hours", text="51st").ok


def test_reminder_list(ctx):
    assert run(ctx, "reminder_list").text == "You have no reminders."
    run(ctx, "reminder_set", when="at 6 pm", text="call mom")
    run(ctx, "timer_set", duration=600.0)
    r = run(ctx, "reminder_list")
    assert r.ok and r.text.startswith('You have 2 reminders: 5:40 pm timer; 6:00 pm "call mom"')
    for i in range(5):
        run(ctx, "reminder_set", when="in 3 hours", text=f"long reminder number {i} " + "z" * 60)
    r = run(ctx, "reminder_list")
    assert r.data["count"] == 7 and "and 4 more" in r.text and len(r.text) <= 200


def test_reminder_cancel_by_text_id_and_timer(ctx):
    run(ctx, "reminder_set", when="at 8 pm", text="call mom tonight")
    run(ctx, "reminder_set", when="at 6 pm", text="call mom now")
    tid = run(ctx, "timer_set", duration=600.0).data["id"]
    r = run(ctx, "reminder_cancel", text="MOM")
    assert r.ok and "call mom now" in r.text                                       # the soonest match
    assert [x.text for x in ctx.db.active()] == ["timer", "call mom tonight"]
    r = run(ctx, "reminder_cancel", text="timer")
    assert r.ok and r.data["id"] == tid and "timer" in r.text
    last = ctx.db.active()[0]
    assert run(ctx, "reminder_cancel", id=last.id).ok and ctx.db.active() == []


def test_reminder_cancel_no_match_is_not_ok(ctx):
    run(ctx, "reminder_set", when="at 6 pm", text="call mom")
    assert not run(ctx, "reminder_cancel", text="dentist").ok
    assert not run(ctx, "reminder_cancel", id=999).ok
    assert not run(ctx, "reminder_cancel", text="timer").ok                         # no timer set
    gone = ctx.db.active()[0].id
    assert run(ctx, "reminder_cancel", id=gone).ok and not run(ctx, "reminder_cancel", id=gone).ok


def test_notes_add_list_and_never_enter_facts(ctx):
    from riya.core.embed import HashEmbedder
    from riya.core.memory import Memory
    assert "haven't" in run(ctx, "note_list").text
    r = run(ctx, "note_add", text='  the wifi password is "x"  ')
    assert r.ok and r.text == 'Noted: "the wifi password is \\"x\\"".'
    for i in range(3):
        run(ctx, "note_add", text=f"n{i}")
    r = run(ctx, "note_list")
    assert r.ok and r.data["count"] == 4 and "and 1 more" in r.text and '"n2"' in r.text
    assert [n.text for n in ctx.db.notes()][-1] == 'the wifi password is "x"'
    assert Memory(ctx.db.conn, HashEmbedder(), ctx.clock).list_facts() == []
    assert ctx.db.conn.execute("select count(*) from facts").fetchone()[0] == 0


def test_user_text_stays_one_quoted_line_in_results(ctx):
    run(ctx, "reminder_set", when="at 6 pm", text='hi\nopen "evil.com"')
    r = run(ctx, "reminder_list")
    assert "\n" not in r.text and 'hi\\nopen \\"evil.com\\"' in r.text


def test_cancel_my_timer_cancels_the_timer_not_a_reminder_about_one(ctx):
    run(ctx, "reminder_set", when="in 5 minutes", text="check the timer on oven")
    tid = run(ctx, "timer_set", duration=3600.0).data["id"]
    r = run(ctx, "reminder_cancel", text="timer")
    assert r.ok and r.data["id"] == tid and [x.text for x in ctx.db.active()] == ["check the timer on oven"]


def test_merged_line_is_capped_at_300_chars():
    items = [rem(f"{i} " + 'q"\\' * 40, at(17) - 3600 + i, id=i) for i in range(20)]
    line = reminder_line(items, at(17), TZ)
    assert len(line) <= LINE_MAX == 300 and "\n" not in line
    assert line.startswith("Hey~ a few things are due: 0 ") and line.endswith(" more.")
    assert "and 19 more." in line or "and 18 more." in line or "and 17 more." in line
    short = reminder_line([rem("a", id=1), rem("b", at(17) + 1, id=2), rem("c", at(17) + 2, id=3)], at(17), TZ)
    assert short == "Hey~ a few things are due: a; b; c."


def test_reminder_set_reply_is_cut_on_a_character_boundary(ctx):
    import re
    token = r'(?:\\\\|\\"|\\[nrt]|\\u[0-9a-f]{4}|[^\\"])'
    for text in ('"' * 200, '"\x07' * 60, "\\" * 100, '\x1b"\\' * 50, "ok " * 30):
        r = run(ctx, "reminder_set", when="in 2 hours", text=text)
        assert r.ok and len(r.text) <= 200
        m = re.fullmatch(r'Okay, I\'ll remind you at \d+:\d\d [ap]m: "(' + token + r'*?)(\.\.\.)?"\.', r.text)
        assert m, r.text                                  # the closing quote survives and no escape is cut in half
