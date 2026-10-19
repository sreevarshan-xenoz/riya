"""v0.6 Task 6: the decision twin in the engine. It gates break bids and game offers once per opportunity, engagement is
recorded per kind, the greeting style follows greeting engagement, barge-in is timed in the core, the verbosity drift,
and the event log. Urgent reports, reminders, timers, confirmations and command results never consult it."""
import random
from datetime import timedelta, timezone

import pytest

from riya.agents.hub import AgentEvent
from riya.core import engine
from riya.core.clock import FakeClock
from riya.core.engine import Utterance
from riya.core.learn.twin import Cell, bucket
from riya.core.prompt import greeting_situation
from riya.core.protocol import Say, SetSettings
from tests.fakes import FixedRng, make_core

NIGHT = timezone(timedelta(hours=14))  # the fake clock's 08:00 UTC is 22:00 here: every test stays in the night band
SHORT_TEXT = "Just a quick, warm hello in one short sentence."
NORMAL_TEXT = "Keep it to one or two short sentences."
BACKUP = {"name": "backup", "argv": ["C:/tools/backup.exe", "--all"], "cwd": None, "timeout_s": 60}
BANDS = ("night", "morning", "afternoon", "evening")
STATES = ("away", "focused", "break", "talking", "gaming")


def ticks(core, clock, seconds, step=1):
    spoken = []
    for _ in range(int(seconds / step)):
        clock.advance(step)
        it = core.tick()
        if it:
            spoken.append(list(it))
    return spoken


def bids(emitted):
    return [m for m in emitted if isinstance(m, Say) and m.kind == "bid"]


def cells(core, kind):
    return {b: c for k, b, c in core.learn.cells() if k == kind}


def events(core, kind=None, outcome=None):
    return [e for e in core.learn.events(10_000) if kind in (None, e.kind) and outcome in (None, e.outcome)]


def near(cell, alpha, beta):
    return abs(cell.alpha - alpha) < 0.05 and abs(cell.beta - beta) < 0.05  # hours of decay over a 30-day half-life


def situation(local, i=-1):
    return next(p for p in local.calls[i][0]["content"].split("\n\n") if p.startswith("[Situation]"))


def night_core(replies=("Water break?",), tmp_path=None, **kw):
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=replies, tz=NIGHT, tmp_path=tmp_path, **kw)
    core.handle(SetSettings(patch={"quiet_hours": None}))
    return core, emitted, src, local, clock


def breaking(replies=("Water break?",), busy=False, learning=True):
    """A long stretch, then a break: exactly one bid (a game offer while an agent is busy)."""
    core, emitted, src, local, clock = night_core(replies)
    if not learning:
        core.handle(SetSettings(patch={"learning_enabled": False}))
    ticks(core, clock, 2 * 3600 + 600, step=300)
    if busy:
        core.agent_inbox.put({"hook_event_name": "UserPromptSubmit", "session_id": "abc123def", "cwd": "E:/riya"})
    src.idle = 150
    ticks(core, clock, 150, step=150)
    assert core.presence.state == "break" and len(bids(emitted)) == 1
    return core, emitted, src, local, clock


def away_and_back(core, clock, src, hours=3):
    """Away for hours, then back: the tick's greeting Utterance (not yet said)."""
    src.idle = 950
    ticks(core, clock, 1000, step=100)                      # their last turn counts as input too
    assert core.presence.state == "away"
    clock.advance(hours * 3600)
    src.idle = 0
    return core.tick()


def came_back_bucket(clock):
    return bucket(clock.now(), NIGHT, "focused", 0.0)


def spy_twin(core):
    seen = []
    for name in ("should", "greet_style", "p_engage", "chance", "record"):
        real = getattr(core.twin, name)
        setattr(core.twin, name, lambda *a, _n=name, _r=real: (seen.append((_n, *a[:2])), _r(*a))[1])
    return seen


def barge(core, clock, after=0.5, text="hello there"):
    r = core.reply(text)
    next(r)
    clock.advance(after)
    r.close()


def replies(core, clock, barged, n=10):
    """n replies, the last barged of them barged in on within 0.5 s."""
    for i in range(n):
        if i >= n - barged:
            barge(core, clock)
        else:
            list(core.reply("hello there"))


# --- the gate ------------------------------------------------------------------------------------------------------

def test_ignored_bids_make_a_bucket_rarer_but_never_below_the_floor(monkeypatch):
    core, emitted, _, _, clock = night_core()
    core.presence.state = "break"
    monkeypatch.setattr(core.presence, "work_stretch_s", lambda: 3 * 3600.0)
    b, now = core._bucket(), clock.now()
    assert b == "night/break/long"
    core.learn.put_cell("bid", b, Cell(2, 200, now))  # far past the 23 ignores that first reach the floor
    core.twin_rng = random.Random(11)
    fired = 0
    for _ in range(300):
        core.initiative.last_bid_at = None
        before = len(bids(emitted))
        core._maybe_bid()
        assert core.initiative.last_bid_at == now          # the opportunity is used up either way
        fired += len(bids(emitted)) - before
        assert core.learn.get_cell("bid", b) == Cell(2, 200, now)
    assert 0.10 <= fired / 300 <= 0.20
    suppressed = events(core, outcome="suppressed")
    assert len(suppressed) == 300 - fired
    assert all(e.kind == "bid" and e.context == {"bucket": b} for e in suppressed)


def test_the_gate_rolls_once_per_opportunity_and_a_suppressed_bid_says_nothing(monkeypatch):
    core, emitted, _, local, clock = night_core()
    core.presence.state = "break"
    monkeypatch.setattr(core.presence, "work_stretch_s", lambda: 3 * 3600.0)
    core.learn.put_cell("bid", core._bucket(), Cell(2, 200, clock.now()))
    rolls = []
    core.twin_rng = type("Rng", (), {"random": lambda self: rolls.append(1) or 0.5})()
    core._maybe_bid()
    assert rolls == [1] and bids(emitted) == [] and local.calls == []
    assert core.pending_bid_at is None and core._twin_open is None
    core._maybe_bid()                                       # the cooldown was consumed: no second opportunity
    assert rolls == [1]


# --- engagement ----------------------------------------------------------------------------------------------------

def test_an_ignored_bid_is_recorded_and_an_answered_one_engages():
    core, emitted, src, _, clock = breaking()
    assert core._twin_open[:2] == ("bid", "night/break/long")
    ticks(core, clock, 300, step=10)
    [(b, c)] = cells(core, "bid").items()
    assert b == "night/break/long" and near(c, 2, 3) and core._twin_open is None
    assert [e.context for e in events(core, "bid", "ignored")] == [{"bucket": b}]
    for _ in range(100):                                    # the next bid, BID_MIN_GAP_S after the first
        if len(bids(emitted)) == 2:
            break
        ticks(core, clock, 60, step=60)
    assert len(bids(emitted)) == 2
    clock.advance(120)
    src.idle = 0
    list(core.reply("yeah, a break sounds right"))
    [(b2, c)] = cells(core, "bid").items()
    assert b2 == b and near(c, 3, 3) and core._twin_open is None
    assert len(events(core, "bid", "engaged")) == 1


def test_a_game_offer_counts_only_when_accepted():
    core, emitted, src, _, clock = breaking(replies=("Water break?", "Yay, let's go!"), busy=True)
    assert core._twin_open[0] == "game_offer"
    src.idle = 0
    list(core.reply("sure"))
    assert core.game is not None
    [c] = cells(core, "game_offer").values()
    assert near(c, 3, 2) and cells(core, "bid") == {}
    list(core.reply("stop"))
    assert core.game is None
    core.initiative.last_bid_at = None
    core.presence.state = "break"
    core._maybe_bid()                                       # a second offer, in the same bucket
    assert len(bids(emitted)) == 2 and core._twin_open[0] == "game_offer"
    clock.advance(30)
    list(core.reply("nah, not now"))
    assert core.game is None
    [c] = cells(core, "game_offer").values()
    assert near(c, 3, 3) and cells(core, "bid") == {}
    assert [e.outcome for e in events(core, "game_offer")] == ["ignored", "engaged"]


def test_an_unanswered_game_offer_is_ignored_after_bid_ignored_s():
    core, emitted, src, _, clock = breaking(replies=("Water break?",), busy=True)
    ticks(core, clock, 300, step=10)
    [c] = cells(core, "game_offer").values()
    assert near(c, 2, 3) and cells(core, "bid") == {}


def report_core():
    core, emitted, src, local, clock = night_core(replies=("Hey you~", "Your agent is done. All tests pass."))
    ticks(core, clock, 1)
    list(core.reply("hey there"))
    ticks(core, clock, 1)
    assert core.presence.state == "talking"
    return core, emitted, src, local, clock


def report_now(core, clock, kind="finished"):
    clock.advance(30)                                       # past REPORT_GAP_S
    core._reports["a1"] = (AgentEvent(kind, "a1", "riya a1", "All done."), clock.now())
    u = core.tick()
    assert isinstance(u, Utterance) and u.spoken
    return u


def test_a_finished_report_heard_out_engages_and_a_barge_in_does_not():
    core, emitted, _, _, clock = report_core()
    u = report_now(core, clock)
    assert next(u) == "Your agent is done."
    clock.advance(3)
    assert list(u) == []
    [c] = cells(core, "report").values()
    assert near(c, 3, 2)
    u = report_now(core, clock)
    next(u)
    clock.advance(1)
    u.close()
    [c] = cells(core, "report").values()
    assert near(c, 3, 3)
    assert [e.outcome for e in events(core, "report")] == ["ignored", "engaged"]
    assert [e.context for e in events(core, "barge")] == [{"kind": "report"}]
    u = report_now(core, clock, "failed")                  # urgent: shown and said, never recorded
    next(u)
    clock.advance(1)
    u.close()
    [c] = cells(core, "report").values()
    assert near(c, 3, 3) and len(events(core, "report")) == 2


# --- urgent paths --------------------------------------------------------------------------------------------------

def deaf_twin(core, clock):
    """Every cell of every kind at p ~ 0 and an rng that suppresses anything the twin is asked about."""
    now = clock.now()
    for kind in ("bid", "game_offer", "report", "greeting"):
        for band in BANDS:
            for state in STATES:
                for stretch in ("short", "long"):
                    core.learn.put_cell(kind, f"{band}/{state}/{stretch}", Cell(2, 10_000, now))
    core.twin_rng = FixedRng(0.999)
    return spy_twin(core)


@pytest.mark.parametrize("kind", sorted(engine.URGENT))
def test_urgent_reports_and_reminders_never_consult_the_twin(kind):
    core, emitted, src, local, clock = night_core(replies=("Your agent needs you. Go look now.",))
    ticks(core, clock, 1)
    seen = deaf_twin(core, clock)
    core._reports["a1"] = (AgentEvent(kind, "a1", "riya a1", "it needs you"), clock.now())
    u = core.tick()
    assert isinstance(u, Utterance) and u.spoken
    assert list(u) == ["Your agent needs you."]
    now = clock.now()
    core.hands.add_reminder("reminder", now + 30, "call mom", now)
    spoken = ticks(core, clock, 40)
    assert len(spoken) == 1 and "call mom" in spoken[0][0]
    assert core.hands.due(clock.now()) == []
    assert seen == []
    assert events(core, outcome="suppressed") == [] and len(events(core, "report")) == 0


def test_timers_confirmations_and_command_results_never_consult_the_twin():
    core, emitted, src, local, clock = night_core(replies=("Okay~", "Backup's done, all good."))
    core._set_settings({"commands": [BACKUP], "tools_enabled": {"run_command": True}})
    ticks(core, clock, 1)
    seen = deaf_twin(core, clock)
    out = list(core.reply("set a timer for 1 minute"))
    assert out and core.hands.active()
    spoken = ticks(core, clock, 70)
    assert any("timer" in line.lower() for said in spoken for line in said)
    out = list(core.reply("run backup"))
    assert core.confirms.pending is not None and out and "backup" in out[0]
    list(core.reply("yes"))
    core._command_thread.join(5)
    spoken = ticks(core, clock, 30)
    assert spoken                                           # the command's result was said
    assert seen == [] and events(core, outcome="suppressed") == []


# --- greetings -----------------------------------------------------------------------------------------------------

def test_greeting_situation_short_text():
    normal = greeting_situation(3 * 3600, 0.0, None)
    short = greeting_situation(3 * 3600, 0.0, None, short=True)
    assert normal.endswith(NORMAL_TEXT) and short.endswith(SHORT_TEXT) and NORMAL_TEXT not in short
    assert short.replace(SHORT_TEXT, NORMAL_TEXT) == normal


def test_greeting_style_follows_greeting_engagement():
    core, emitted, src, local, clock = night_core(replies=("Welcome back! I missed you. How was it?",))
    ticks(core, clock, 1)
    src.idle = 950
    ticks(core, clock, 1)
    clock.advance(3 * 3600)
    b = came_back_bucket(clock)
    core.learn.put_cell("greeting", b, Cell(2, 6, clock.now()))
    src.idle = 0
    u = core.tick()
    assert list(u) == ["Welcome back!"]                     # at most one sentence
    assert SHORT_TEXT in situation(local) and NORMAL_TEXT not in situation(local)
    assert core._greet_open[0] == b and core._greet_bucket is None and core._greet_short is False
    clock.advance(60)
    list(core.reply("it was good, thanks"))
    assert near(cells(core, "greeting")[b], 3, 6) and core._greet_open is None
    u = away_and_back(core, clock, src)
    b2 = core._greet_bucket
    assert b2 == came_back_bucket(clock) == b
    list(u)
    ticks(core, clock, 121)
    assert near(cells(core, "greeting")[b], 3, 7) and core._greet_open is None
    assert [e.outcome for e in events(core, "greeting")] == ["ignored", "engaged"]


def test_a_fresh_greeting_bucket_gets_the_normal_greeting():
    core, emitted, src, local, clock = night_core(replies=("Welcome back! I missed you. How was it?",))
    ticks(core, clock, 1)
    assert len(list(away_and_back(core, clock, src))) > 1
    assert NORMAL_TEXT in situation(local) and SHORT_TEXT not in situation(local)


@pytest.mark.parametrize("news", ["report", "reminder"])
def test_a_greeting_carrying_news_is_never_short(news):
    core, emitted, src, local, clock = night_core(replies=("Welcome back! Your agent failed, go look. How was it?",))
    ticks(core, clock, 1)
    src.idle = 950
    ticks(core, clock, 1)
    if news == "report":
        core._reports["a1"] = (AgentEvent("failed", "a1", "riya a1", "boom"), clock.now())
    else:
        core.hands.add_reminder("reminder", clock.now() + 60, "call mom", clock.now())
    clock.advance(3 * 3600)
    core.learn.put_cell("greeting", came_back_bucket(clock), Cell(2, 6, clock.now()))
    src.idle = 0
    said = list(core.tick())
    assert NORMAL_TEXT in situation(local) and SHORT_TEXT not in situation(local)
    assert len(said) > 1
    if news == "report":
        assert "riya a1" in situation(local)
    else:
        assert any("call mom" in s for s in said)


def test_speak_first_without_came_back_uses_the_current_bucket():
    core, emitted, src, local, clock = night_core(replies=("Hi again!",))
    list(core.speak_first(engine.Action("greet", True, None, "back")))
    assert core._greet_open[0] == core._bucket()
    clock.advance(30)
    list(core.reply("hi"))
    assert near(cells(core, "greeting")[core._bucket()], 3, 2)


# --- barge-in and the drift ----------------------------------------------------------------------------------------

def test_barge_in_within_2s_is_measured_in_the_core():
    core, emitted, _, local, clock = night_core(replies=("One here. Two here. Three here.",))
    barge(core, clock, 1.5)
    assert [e.context for e in events(core, "barge", "barged")] == [{"kind": "reply"}]
    barge(core, clock, 2.5)
    assert len(events(core, "barge")) == 1
    r = core.reply("hello there")                           # closed before its first sentence
    r.close()
    assert len(events(core, "barge")) == 1
    real = local.stream

    def slow(*a, **k):                                      # five seconds before the first word
        clock.advance(5)
        yield from real(*a, **k)
    local.stream = slow
    barge(core, clock, 1.5)                                 # timed from her first sentence, not the turn's start
    assert len(events(core, "barge")) == 2


def test_a_typed_turn_while_she_streams_is_a_barge_in():
    """Typed turns still queue (the reply runs to its end), but one within 2 s of her first sentence is measured."""
    core, _, _, _, clock = night_core(replies=("One here. Two here. Three here.",))
    core.note_typed_turn()                                  # nothing streaming: nothing to barge in on
    r = core.reply("hello there")
    core.note_typed_turn()                                  # not started, no first sentence yet
    assert next(r) == "One here."
    clock.advance(1)
    core.note_typed_turn()
    assert list(r) == ["Two here.", "Three here."]          # queued, not cut off
    assert [e.context for e in events(core, "barge", "barged")] == [{"kind": "reply"}]
    assert list(core._barges) == [True] and core._last_barged is True
    r = core.reply("hello there")
    next(r)
    clock.advance(3)
    core.note_typed_turn()
    list(r)
    assert len(events(core, "barge")) == 1 and list(core._barges) == [True, False]
    list(core.reply("hello there"))                         # no stale flag carries over
    assert len(events(core, "barge")) == 1 and list(core._barges) == [True, False, False]


def test_typed_barge_ins_feed_the_drift():
    core, _, _, _, clock = night_core(replies=("One here. Two here.",))
    for i in range(10):
        r = core.reply("hello there")
        next(r)
        clock.advance(0.5)
        if i >= 5:
            core.note_typed_turn()
        list(r)
    assert core.learn.dials()["verbosity"] == 0.3 and core.learn.history()[0].source == "drift"


def test_a_typed_turn_during_a_report_means_it_was_not_heard_out():
    core, _, _, _, clock = night_core(replies=("One here. Two here.",))
    seen = []
    it = core._speak("report", user_text=None, query="", situation=None, spoken=True,
                     after=lambda outcome: seen.append(core._last_barged))
    next(it)
    clock.advance(1)
    core.note_typed_turn()
    list(it)
    assert seen == [True] and [e.context for e in events(core, "barge")] == [{"kind": "report"}]


def test_drift_is_daily_and_stops_at_the_floor(tmp_path):
    (tmp_path / "a").mkdir()
    core, _, _, _, clock = night_core(replies=("One here. Two here.",), tmp_path=tmp_path / "a")
    verbosity = lambda c: c.learn.dials()["verbosity"]  # noqa: E731
    assert verbosity(core) == 0.35
    replies(core, clock, 5)
    assert verbosity(core) == 0.3
    [h] = core.learn.history()
    assert h.source == "drift" and h.reason == "they cut in within 2 s on 5 of her last 10 replies"
    replies(core, clock, 5)
    assert verbosity(core) == 0.3                           # at most once a day
    clock.advance(86400)
    replies(core, clock, 5)
    assert verbosity(core) == 0.25
    for _ in range(6):
        clock.advance(86400)
        replies(core, clock, 6)
    assert verbosity(core) == 0.15 and min(h.new for h in core.learn.history(100)) == 0.15
    (tmp_path / "b").mkdir()
    core, _, _, _, clock = night_core(replies=("One here. Two here.",), tmp_path=tmp_path / "b")
    replies(core, clock, 4)
    assert verbosity(core) == 0.35 and core.learn.history() == []
    replies(core, clock, 1, n=1)                            # the window slides: 5 of the last 10 now
    assert verbosity(core) == 0.3
    again, *_ = make_core(clock, local_replies=("One here. Two here.",), tz=NIGHT, tmp_path=tmp_path / "b")
    clock.advance(3600)
    replies(again, clock, 10)
    assert verbosity(again) == 0.3                          # a restart the same day reads drift_at
    clock.advance(86400)
    replies(again, clock, 5)
    assert verbosity(again) == 0.25


def test_only_plain_replies_count_toward_the_drift():
    core, _, _, _, clock = night_core(replies=("One here. Two here.",))
    calls = []
    real = core._note_reply
    core._note_reply = lambda barged: (calls.append(barged), real(barged))
    list(core.reply("hello there"))
    list(core.speak_first(engine.Action("greet", True, None, "back")))
    list(core.reply("let's play 20 questions"))
    assert calls == [False]


# --- learning off, mood, purge, reset ------------------------------------------------------------------------------

def test_learning_off_freezes_everything():
    core, emitted, src, local, clock = breaking(replies=("Water break? Two here.",), learning=False)
    assert core.settings.learning_enabled is False
    assert core._twin_open is None                          # never opened while off
    ticks(core, clock, 300, step=10)
    src.idle = 0
    ticks(core, clock, 1)
    replies(core, clock, 5)
    list(away_and_back(core, clock, src))
    assert core._greet_open is None
    ticks(core, clock, 121)
    list(core.reply("stop teasing me"))
    assert core.learn.cells() == [] and core.learn.events() == [] and core.learn.history() == []
    assert core.learn.dials() == core.learn.defaults and core.learn.notes() == []
    assert core.learn.drift_at() is None


def test_learning_off_still_consults_the_frozen_twin(monkeypatch):
    core, emitted, _, _, clock = night_core()
    core.handle(SetSettings(patch={"learning_enabled": False}))
    core.presence.state = "break"
    monkeypatch.setattr(core.presence, "work_stretch_s", lambda: 3 * 3600.0)
    core.learn.put_cell("bid", core._bucket(), Cell(2, 200, clock.now()))
    seen = spy_twin(core)
    core.twin_rng = FixedRng(0.5)
    core._maybe_bid()
    assert [s for s in seen if s[0] == "should"] == [("should", "bid", core._bucket())] and bids(emitted) == []
    assert events(core) == []                               # suppressed, but nothing logged while off
    core.twin_rng, core.initiative.last_bid_at = FixedRng(0.0), None
    core._maybe_bid()
    assert len(bids(emitted)) == 1 and core._twin_open is None  # it went out, but nothing waits to be recorded


def test_turning_learning_off_drops_what_was_open():
    core, emitted, src, _, clock = breaking()
    assert core._twin_open is not None
    core.handle(SetSettings(patch={"learning_enabled": False}))
    ticks(core, clock, 10, step=10)
    assert core._twin_open is None
    core.handle(SetSettings(patch={"learning_enabled": True}))
    ticks(core, clock, 400, step=10)
    assert core.learn.cells() == [] and events(core) == []


def test_mood_after_her_line_is_logged():
    core, *_ = night_core(replies=("Hey you~",))
    list(core.reply("ugh I feel terrible today"))           # her line isn't last yet: no mood event
    assert events(core, "mood") == []
    list(core.reply("hello there"))
    list(core.reply("ugh I feel terrible today, everything is awful"))
    down, flat = events(core, "mood")                       # the newest first
    assert flat.outcome == "flat" and flat.context == {"delta": 0.0}   # "hello there" moved nothing
    assert down.outcome == "down" and down.context["delta"] <= -engine.MOOD_MOVE


def test_events_purged_after_30_days():
    core, _, _, _, clock = night_core()
    now = clock.now()
    core.learn.log_event(now - 31 * 86400, "bid", "ignored", {"bucket": "x"})
    core.learn.log_event(now - 29 * 86400, "bid", "engaged", {"bucket": "x"})
    core._maybe_purge(now)
    assert [e.outcome for e in core.learn.events()] == ["engaged"]


def test_reset_clears_the_open_opportunities_and_the_barge_window():
    core, emitted, src, _, clock = breaking()
    core._greet_open = ("night/focused/short", clock.now())
    core._barges.extend([True, False])
    core.learn.set_drift_at(clock.now())
    core._reset_learning()
    assert core._twin_open is None and core._greet_open is None and len(core._barges) == 0
    assert core.learn.drift_at() is None  # the drift clock is learned state too
