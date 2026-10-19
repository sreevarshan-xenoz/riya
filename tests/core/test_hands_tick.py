"""v0.5 Task 8: her hands on the tick path: reminders and timers with manners (spoken, bubble, on return), queued
command and button results, parked T2 calls raised after a game, confirmation expiry, the reminders prompt line, the
30-day purge, hello and the runner's wiring. Offline: FakeEffects, FakeHttp, fake models and a fake clock."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from riya.core import __main__ as runner
from riya.core.clock import FakeClock
from riya.core.engine import AFTER_GAME, MODEL_DOWN, PURGE_EVERY_S, Utterance
from riya.core.hands.policy import confirm_text
from riya.core.hands.reminders import fmt_clock
from riya.core.hands.tools import ToolResult
from riya.core.models import ModelUnavailable
from riya.core.presence import FakeIdleSource
from riya.core.prompt import MEMORY_END, TOOL_NOTE, reminders_block
from riya.core.protocol import (ActionsMsg, ConfirmDone, ConfirmRequest, ErrorMsg, RemindersMsg, Say, SayEnd,
                                SetSettings)
from tests.fakes import make_core

NOON = 1_800_000_000.0 + 4 * 3600  # 2027-01-15 12:00 UTC
BACKUP = {"name": "backup", "argv": ["C:/tools/backup.exe", "--all"], "cwd": None, "timeout_s": 60}
T2 = {"commands": [BACKUP], "tools_enabled": {"run_command": True, "write_file": True}}


def tick_core(local_replies=("Done~",), tmp_path=None, tz=timezone.utc, start=NOON, **settings):
    clock = FakeClock(start)
    core, emitted, src, local = make_core(clock, local_replies=local_replies, tz=tz, tmp_path=tmp_path)
    if settings:
        core._set_settings(settings)
    ticks(core, clock, 1)
    emitted.clear()
    return core, emitted, src, local, clock


def ticks(core, clock, seconds, step=1, flags=None):
    out = []
    for _ in range(int(seconds / step)):
        clock.advance(step)
        if (it := core.tick()) is not None:
            out.append(list(it))
            if flags is not None:
                flags.append(it.spoken)
    return out


def of(emitted, cls):
    return [m for m in emitted if isinstance(m, cls)]


def reminder_says(emitted, text):
    return [m for m in of(emitted, Say) if text in m.text]


def logged(core):
    return [(a.tool, a.tier, a.outcome) for a in reversed(core.hands.actions())]


def situation(local, i=-1):
    return next(p for p in local.calls[i][0]["content"].split("\n\n") if p.startswith("[Situation]"))


def add(core, after_s, text="call mom", kind="reminder"):
    now = core.clock.now()
    return core.hands.add_reminder(kind, now + after_s, text, now)


def away(core, src, clock):
    src.idle = 950
    ticks(core, clock, 1)
    assert core.presence.state == "away"


def come_back(src, clock, after_s=3 * 3600):
    clock.advance(after_s)
    src.idle = 0


# ---- due while focused --------------------------------------------------------------------------------------------
def test_due_while_focused_is_spoken_fired_and_sent():
    core, emitted, _, local, clock = tick_core()
    add(core, 60)
    assert ticks(core, clock, 59) == []
    flags = []
    assert ticks(core, clock, 1, flags=flags) == [["Hey~ it's time to call mom."]]
    assert flags == [True]
    [say] = of(emitted, Say)
    assert say.spoken is True and say.kind == "report"
    assert any(isinstance(m, SayEnd) and m.utterance_id == say.utterance_id for m in emitted)
    assert core.hands.active() == [] and of(emitted, RemindersMsg)[-1].items == []
    assert local.calls == []                                     # a template: the model never words it
    assert core.memory.unconsolidated() == []
    assert ticks(core, clock, 120) == []                         # fired once


def test_three_due_at_once_merge_and_the_next_spoken_one_waits_20_s():
    core, emitted, _, _, clock = tick_core()
    for t in ("call mom", "buy milk", "stretch"):
        add(core, 10, t)
    add(core, 15, "water the plants")
    assert ticks(core, clock, 9) == []
    assert ticks(core, clock, 1) == [["Hey~ a few things are due: call mom; buy milk; stretch."]]
    assert ticks(core, clock, 19) == []                          # due at +15, but the last spoken one was at +10
    assert ticks(core, clock, 1) == [["Hey~ it's time to water the plants."]]
    assert core.hands.active() == []
    assert len(reminder_says(emitted, "water the plants")) == 1


@pytest.mark.parametrize("how", ["quiet", "no_initiative", "gaming"])
def test_reminder_in_quiet_hours_or_gaming_is_a_bubble(how):
    patch = {"quiet": {"quiet_hours": "11:00-13:00"}, "no_initiative": {"may_start_conversations": False},
             "gaming": {}}[how]
    core, emitted, src, local, clock = tick_core(**patch)
    if how == "gaming":
        src.is_fullscreen = True
        ticks(core, clock, 1)
        assert core.presence.state == "gaming"
    add(core, 5)
    assert ticks(core, clock, 10) == []                          # no utterance: never spoken, never the mic
    [say] = reminder_says(emitted, "call mom")
    assert say.spoken is False and say.text == "Hey~ it's time to call mom."
    assert core.hands.active() == []                             # fired, never dropped
    assert local.calls == []
    assert of(emitted, RemindersMsg)[-1].items == []


# ---- away and the came-back greeting ------------------------------------------------------------------------------
def test_away_nothing_fires_then_the_greeting_carries_it_with_the_note():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    rid = add(core, 300)
    due = core.hands.active()[0].due_at
    assert ticks(core, clock, 900, step=10) == [] and [r.id for r in core.hands.active()] == [rid]
    assert not reminder_says(emitted, "call mom")
    come_back(src, clock)
    line = f"Hey~ it's time to call mom. (this was due at {fmt_clock(due, timezone.utc)})"
    assert ticks(core, clock, 1) == [["Welcome back!", line]]
    assert core.hands.active() == [] and of(emitted, RemindersMsg)[-1].items == []
    assert ticks(core, clock, 120) == [] and len(reminder_says(emitted, "call mom")) == 1


def test_the_greeting_tail_carries_the_reminder_even_when_the_model_fails():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)
    local.fail = ModelUnavailable("down")
    come_back(src, clock)
    [[line]] = ticks(core, clock, 1)
    assert line.startswith("Hey~ it's time to call mom. (this was due at ")
    assert core.hands.active() == []
    assert ticks(core, clock, 60) == [] and len(reminder_says(emitted, "call mom")) == 1


def test_restricted_on_return_is_a_bubble_after_the_greeting():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)
    core.handle(SetSettings(patch={"quiet_hours": "15:00-16:00"}))
    come_back(src, clock)
    assert ticks(core, clock, 1) == [["Welcome back!"]]          # the greeting itself is v0.1's
    assert core.hands.active() != []
    assert ticks(core, clock, 1) == []
    [say] = reminder_says(emitted, "call mom")
    assert say.spoken is False and "(this was due at " in say.text and core.hands.active() == []


def test_no_initiative_on_return_is_a_bubble_and_no_greeting():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",), may_start_conversations=False)
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)
    come_back(src, clock)
    assert ticks(core, clock, 1) == []
    [say] = reminder_says(emitted, "call mom")
    assert say.spoken is False and core.hands.active() == [] and local.calls == []


@pytest.mark.parametrize("broken", ["_greeted", "_finish_turn"])
def test_a_greeting_bookkeeping_failure_never_repeats_its_reminder_tail(broken):
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)

    def boom(*a, **kw):
        raise RuntimeError("bookkeeping failed")
    setattr(core, broken, boom)
    come_back(src, clock)
    [greeting] = ticks(core, clock, 1)
    assert greeting[-1].startswith("Hey~ it's time to call mom.")
    assert core.hands.active() == []                             # fired before the bookkeeping
    assert ticks(core, clock, 120) == [] and len(reminder_says(emitted, "call mom")) == 1


def test_the_greeting_tail_counts_for_the_reminder_spacing():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)
    come_back(src, clock)
    now = clock.now()
    core.hands.add_reminder("reminder", now + 2, "buy milk", now)
    assert ticks(core, clock, 1)[0][-1].startswith("Hey~ it's time to call mom.")
    assert ticks(core, clock, 19) == []                          # due a second later, but 20 s apart
    assert ticks(core, clock, 1) == [["Hey~ it's time to buy milk."]]


def test_reminder_due_on_greeting_tick():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    ticks(core, clock, 60, step=10)
    come_back(src, clock)
    now = clock.now() + 1                                        # the tick below runs at exactly this time
    core.hands.add_reminder("reminder", now, "call mom", now - 600)
    assert ticks(core, clock, 1) == [["Welcome back!", "Hey~ it's time to call mom."]]
    assert core.hands.active() == []
    assert ticks(core, clock, 120) == [] and len(reminder_says(emitted, "call mom")) == 1
    assert core._reminding == set()


# ---- never lost ---------------------------------------------------------------------------------------------------
def test_a_reminder_utterance_closed_before_its_first_step_is_offered_again():
    core, emitted, _, _, clock = tick_core()
    add(core, 1)
    clock.advance(1)
    u = core.tick()
    assert isinstance(u, Utterance) and core._reminding
    u.close()
    assert core._reminding == set() and len(core.hands.active()) == 1
    assert not reminder_says(emitted, "call mom")
    assert ticks(core, clock, 1) == [["Hey~ it's time to call mom."]]
    assert core.hands.active() == []


def test_a_claimed_reminder_is_not_offered_twice_while_its_utterance_waits():
    core, emitted, _, _, clock = tick_core()
    add(core, 1)
    clock.advance(1)
    u = core.tick()
    assert ticks(core, clock, 5) == []                           # still claimed: not offered again meanwhile
    assert list(u) == ["Hey~ it's time to call mom."]
    assert core._reminding == set() and core.hands.active() == []


def test_a_greeting_closed_before_its_first_step_leaves_its_reminders_for_the_next_tick():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    away(core, src, clock)
    add(core, 300)
    ticks(core, clock, 600, step=10)
    come_back(src, clock)
    clock.advance(1)
    u = core.tick()
    assert isinstance(u, Utterance) and core._reminding
    u.close()
    assert core._reminding == set() and len(core.hands.active()) == 1
    [[line]] = ticks(core, clock, 1)
    assert line.startswith("Hey~ it's time to call mom. (this was due at ")
    assert core.hands.active() == []


def test_utterance_close_calls_on_close_once():
    calls = []
    u = Utterance((x for x in ["a"]), True, on_close=lambda: calls.append(1))
    u.close()
    u.close()
    assert calls == [1]


def test_sleep_past_due_fires_once_with_note():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!",))
    add(core, 1200)
    due = core.hands.active()[0].due_at
    clock.advance(3 * 3600)                                      # the laptop slept across the due time
    out = ticks(core, clock, 1)
    line = f"Hey~ it's time to call mom. (this was due at {fmt_clock(due, timezone.utc)})"
    assert out == [["Welcome back!", line]]
    assert ticks(core, clock, 300) == [] and len(reminder_says(emitted, "call mom")) == 1
    assert core.hands.active() == []


def test_an_overdue_reminder_fires_once_after_a_restart(tmp_path):
    core, *_ = tick_core(tmp_path=tmp_path)
    add(core, 60)
    due = core.hands.active()[0].due_at
    clock = FakeClock(NOON + 3600)                               # restarted an hour later, on the same db
    core2, emitted, src, local = make_core(clock, tmp_path=tmp_path, tz=timezone.utc)
    line = f"Hey~ it's time to call mom. (this was due at {fmt_clock(due, timezone.utc)})"
    assert ticks(core2, clock, 30) == [[line]]
    assert core2.hands.active() == [] and len(reminder_says(emitted, "call mom")) == 1
    clock3 = FakeClock(NOON + 7200)
    core3, emitted3, _, _ = make_core(clock3, tmp_path=tmp_path, tz=timezone.utc)
    assert ticks(core3, clock3, 30) == [] and not reminder_says(emitted3, "call mom")


def test_dst_reminder_fires_at_local_9():
    ny = ZoneInfo("America/New_York")
    start = datetime(2026, 3, 7, 20, 0, tzinfo=ny).timestamp()
    core, emitted, src, local, clock = tick_core(tz=ny, start=start - 1, quiet_hours=None)
    assert clock.now() == start
    list(core.reply("remind me tomorrow at 9 to x"))
    [r] = core.hands.active()
    due = datetime(2026, 3, 8, 9, 0, tzinfo=ny).timestamp()
    assert r.due_at == due and due - start == 12 * 3600          # 13 wall hours, 12 elapsed: spring forward
    emitted.clear()
    assert ticks(core, clock, 42600, step=600) == []
    assert ticks(core, clock, 540, step=540) == []               # 08:59 EDT
    assert datetime.fromtimestamp(clock.now(), ny).strftime("%H:%M") == "08:59"
    assert core.hands.active() != []
    assert ticks(core, clock, 60, step=60) == [["Hey~ it's time to x."]]   # 09:00 EDT
    assert core.hands.active() == []


# ---- command and button results ------------------------------------------------------------------------------------


def queue_result(core, ok=True, text="\"backup\" finished: 12 files copied"):
    core._results.put(("run \"backup\" (C:/tools/backup.exe --all)", ToolResult(ok, text)))


def test_a_command_result_while_focused_is_spoken_through_the_model_quoted():
    core, emitted, _, local, clock = tick_core(local_replies=("Your backup is done~",), **T2)
    core.effects.run_result = (0, "z" * 500 + " copied 12 files")
    list(core.reply("run backup"))
    list(core.reply("yes"))
    core._command_thread.join(5)
    emitted.clear()
    n = len(local.calls)
    assert ticks(core, clock, 1) == [["Your backup is done~"]]
    assert len(local.calls) == n + 1
    s = situation(local)
    assert TOOL_NOTE in s
    quote = s.split('the result: "', 1)[1].split('"', 1)[0]
    assert "copied 12 files" in quote and len(quote) <= 200
    assert all(m.spoken for m in of(emitted, Say))
    assert core.turns[-1]["content"].startswith("(You told them what your tools found:")
    assert ticks(core, clock, 60) == []                          # once


def test_a_command_result_while_gaming_is_the_template_bubble():
    core, emitted, src, local, clock = tick_core(**T2)
    src.is_fullscreen = True
    ticks(core, clock, 1)
    queue_result(core)
    assert ticks(core, clock, 5) == [] and local.calls == []
    [say] = of(emitted, Say)
    assert say.spoken is False and say.text == "\"backup\" finished: 12 files copied"
    assert TOOL_NOTE in core.turns[-1]["content"] and core.turns[-1]["content"].startswith("(You told them")


@pytest.mark.parametrize("patch", [{"quiet_hours": "11:00-13:00"}, {"may_start_conversations": False}])
def test_a_restricted_command_result_is_the_result_bubble_and_no_model(patch):
    core, emitted, src, local, clock = tick_core(**T2, **patch)
    queue_result(core)
    assert ticks(core, clock, 5) == [] and local.calls == []
    [say] = of(emitted, Say)
    assert say.spoken is False and say.text == "\"backup\" finished: 12 files copied"


def test_a_command_result_while_away_waits_for_after_the_greeting():
    core, emitted, src, local, clock = tick_core(local_replies=("Welcome back!", "Backup's done~"), **T2)
    away(core, src, clock)
    queue_result(core)
    assert ticks(core, clock, 300, step=10) == [] and not of(emitted, Say) and local.calls == []
    come_back(src, clock)
    assert ticks(core, clock, 1) == [["Welcome back!"]]
    assert ticks(core, clock, 1) == [["Backup's done~"]]
    assert TOOL_NOTE in situation(local)


def test_a_command_result_the_model_fails_on_is_the_result_text():
    core, emitted, _, local, clock = tick_core(**T2)
    local.fail = ModelUnavailable("down")
    queue_result(core)
    assert ticks(core, clock, 1) == [["\"backup\" finished: 12 files copied"]]
    assert not [m for m in of(emitted, ErrorMsg) if m.message == MODEL_DOWN]


def test_results_share_the_report_gap():
    core, emitted, _, local, clock = tick_core(local_replies=("One~", "Two~"), **T2)
    queue_result(core)
    queue_result(core, text="second")
    assert ticks(core, clock, 1) == [["One~"]]
    assert ticks(core, clock, 19) == []
    assert ticks(core, clock, 1) == [["Two~"]]


def test_a_button_result_is_delivered_by_the_tick(tmp_path):
    from riya.core.protocol import Confirm
    core, emitted, _, local, clock = tick_core(local_replies=("Saved it~",), tmp_path=tmp_path, **T2)
    list(core.reply("write a file called notes.txt saying hello there"))
    core.handle(Confirm(id=core.confirms.pending.id, yes=True))
    assert ticks(core, clock, 1) == [["Saved it~"]]
    assert core._results.empty()


# ---- parked calls and expiry --------------------------------------------------------------------------------------
def gaming_park(core, src, clock):
    src.is_fullscreen = True
    ticks(core, clock, 1)
    assert core.presence.state == "gaming"
    assert list(core.reply("run backup")) == [AFTER_GAME]


def test_a_parked_call_is_raised_when_the_game_ends():
    core, emitted, src, local, clock = tick_core(**T2)
    gaming_park(core, src, clock)
    assert ticks(core, clock, 60, step=10) == [] and not of(emitted, ConfirmRequest)
    src.is_fullscreen = False
    flags = []
    out = ticks(core, clock, 1, flags=flags)
    p = core.confirms.pending
    assert p is not None and out == [[confirm_text(p)]] and flags == [True]
    [req] = of(emitted, ConfirmRequest)
    assert req.id == p.id and req.expires_at == clock.now() + 60
    assert local.calls == []                                     # a template, and never the model during the game


@pytest.mark.parametrize("patch", [{"quiet_hours": "11:00-13:00"}, {"may_start_conversations": False}])
def test_a_parked_call_raised_while_restricted_is_a_bubble_in_the_tick(patch):
    core, emitted, src, local, clock = tick_core(**T2)
    gaming_park(core, src, clock)
    core.handle(SetSettings(patch=patch))
    src.is_fullscreen = False
    clock.advance(1)
    assert core.tick() is None                                   # no Utterance: nothing spoken, the mic stays shut
    p = core.confirms.pending
    assert p is not None
    assert [m.spoken for m in of(emitted, Say) if m.text == confirm_text(p)] == [False]
    assert [m.id for m in of(emitted, ConfirmRequest)] == [p.id]
    assert local.calls == []


def test_a_raised_parked_call_is_shown_before_it_can_be_answered():
    """A yes racing the raise (waiting on the core lock) is only taken once the question is out."""
    import threading
    import time
    core, emitted, src, local, clock = tick_core(**T2)
    gaming_park(core, src, clock)
    src.is_fullscreen = False
    unpark, answers = core.confirms.unpark, []

    def racing_unpark():
        got = unpark()
        t = threading.Thread(target=lambda: answers.append(list(core.reply("yes"))))
        t.start()
        time.sleep(0.1)                                           # the turn is now waiting for the core lock
        racing_unpark.thread = t
        return got
    core.confirms.unpark = racing_unpark
    clock.advance(1)
    u = core.tick()
    racing_unpark.thread.join(5)
    order = [type(m).__name__ for m in emitted if isinstance(m, (ConfirmRequest, ConfirmDone))]
    assert order[0] == "ConfirmRequest"                          # asked before any answer was taken
    assert u is not None
    u.close()


def test_a_raised_parked_call_is_out_even_if_its_utterance_is_never_started():
    core, emitted, src, local, clock = tick_core(**T2)
    gaming_park(core, src, clock)
    src.is_fullscreen = False
    clock.advance(1)
    u = core.tick()
    p = core.confirms.pending
    assert [m.id for m in of(emitted, ConfirmRequest)] == [p.id]
    assert [m.spoken for m in of(emitted, Say) if m.text == confirm_text(p)] == [True]
    u.close()                                                    # closed before its first step: it was still shown
    list(core.reply("yes"))
    core._command_thread.join(5)
    assert core.effects.calls == [("run", BACKUP["argv"], None, 60)]


def test_a_call_parked_over_10_minutes_is_dropped_and_logged_expired():
    core, emitted, src, local, clock = tick_core(**T2)
    gaming_park(core, src, clock)
    assert ticks(core, clock, 660, step=30) == []
    src.is_fullscreen = False
    assert ticks(core, clock, 5) == []
    assert not of(emitted, ConfirmRequest) and core.confirms.pending is None
    assert logged(core)[-1] == ("run_command", 2, "expired")
    assert core.effects.calls == []


def test_a_pending_confirmation_expires_on_the_tick():
    core, emitted, src, local, clock = tick_core(**T2)
    list(core.reply("run backup"))
    pid = core.confirms.pending.id
    assert ticks(core, clock, 59) == [] and not of(emitted, ConfirmDone)
    assert ticks(core, clock, 1) == []
    assert [(m.id, m.outcome) for m in of(emitted, ConfirmDone)] == [(pid, "expired")]
    assert logged(core) == [("run_command", 2, "expired")]
    assert ticks(core, clock, 10) == [] and len(of(emitted, ConfirmDone)) == 1


# ---- the prompt line, the purge, hello, the runner -----------------------------------------------------------------
def test_reminders_block_lists_at_most_three():
    from riya.core.hands.records import Reminder
    now = NOON
    items = [Reminder(i, "reminder", now + 3600 * (i + 1), f"thing {i}", now) for i in range(4)]
    items.append(Reminder(9, "timer", now + 600, "timer", now))
    block = reminders_block(sorted(items, key=lambda r: r.due_at), now, timezone.utc)
    assert block == ('### Your reminders: your 10-minute timer at 12:10 pm; "thing 0" at 1:00 pm; '
                     '"thing 1" at 2:00 pm')
    assert reminders_block([], now, timezone.utc) is None
    odd = [Reminder(1, "reminder", now + 86400, 'say "hi"\nnow', now)]
    assert reminders_block(odd, now, timezone.utc) == '### Your reminders: "say \\"hi\\" now" at 12:00 pm tomorrow'


def test_the_prompt_lists_the_next_three_reminders_after_the_memory_block():
    core, emitted, src, local, clock = tick_core()
    for i in range(4):
        add(core, 3600 * (i + 1), f"thing {i}")
    list(core.reply("hey there"))
    system = local.calls[-1][0]["content"]
    assert system.index("### Your reminders:") > system.index(MEMORY_END)
    line = next(p for p in system.split("\n\n") if p.startswith("### Your reminders:"))
    assert '"thing 2"' in line and '"thing 3"' not in line


def test_the_daily_purge_drops_a_31_day_old_action():
    core, emitted, src, local, clock = tick_core()
    core.hands.log_action(clock.now() - 31 * 86400, "calc", 0, "old", "ran")
    core.hands.log_action(clock.now(), "calc", 0, "new", "ran")
    ticks(core, clock, 1)
    assert len(core.hands.actions()) == 2                        # the purge runs daily, not every tick
    clock.advance(PURGE_EVERY_S)
    ticks(core, clock, 1)
    assert [a.summary for a in core.hands.actions()] == ["new"]


def test_hello_carries_the_reminders_the_actions_and_the_pending_confirm():
    core, emitted, src, local, clock = tick_core(**T2)
    add(core, 600)
    core.hands.log_action(clock.now(), "calc", 0, "calc 1+1", "ran")
    hello = core.hello_messages()
    [rem] = of(hello, RemindersMsg)
    [acts] = of(hello, ActionsMsg)
    assert [i.text for i in rem.items] == ["call mom"] and [a.summary for a in acts.items] == ["calc 1+1"]
    assert not of(hello, ConfirmRequest)
    list(core.reply("run backup"))
    p = core.confirms.pending
    [req] = of(core.hello_messages(), ConfirmRequest)
    assert (req.id, req.summary, req.expires_at) == (p.id, p.summary, p.expires_at)
    assert [m.type for m in core.hello_messages()][:3] == ["hello", "settings", "state"]


def test_the_runner_hands_the_core_real_effects_and_http(tmp_path, monkeypatch):
    import argparse

    class StubEffects:
        pass

    class StubHttp:
        pass
    monkeypatch.setattr(runner, "WindowsEffects", StubEffects)
    monkeypatch.setattr(runner, "UrlHttp", StubHttp)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(db=str(tmp_path / "d" / "riya.db"), local_url="http://127.0.0.1:1", hash_embed=True,
                              legacy_memory=str(tmp_path / "none.json"))
    core = runner.build_core(args, lambda m: None)
    assert isinstance(core.effects, StubEffects) and isinstance(core.http, StubHttp)
