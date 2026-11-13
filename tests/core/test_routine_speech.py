"""Engine II: claims and hunches as engine lines, the claim guard and the online scrub (v0.12 §4.5, §4.7, §4.8,
A9, A13, A14, A15, A19)."""
import json
from datetime import datetime, timedelta, timezone

import riya.core.routine as routine
from riya.core.clock import FakeClock
from riya.core.consolidate import Consolidator
from riya.core.initiative import Action
from riya.core.prompt import statement_situation, statement_tail_situation
from riya.core.protocol import Say
from riya.core.routine import PLACEHOLDER
from riya.core.routine import statements as st
from riya.core.routine.statements import CLAUSES, TEMPLATES
from riya.train.build_sft import prompt_fingerprint
from tests.fakes import FakeModel, make_core
from tests.routine_helpers import mature_beta, mature_lognormal

TZ = timezone(timedelta(hours=5, minutes=30))
MON_9 = datetime(2026, 11, 9, 9, tzinfo=TZ).timestamp()
H = 3600.0


def core_on(replies=("That's nice.",), routine=True, t=MON_9, **kw):
    return make_core(clock=FakeClock(t), tz=TZ, routine=routine, local_replies=replies, **kw)


def tick(core):
    it = core.tick()
    if it:
        list(it)


def run(core, seconds, step=60):
    for _ in range(int(seconds // step)):
        core.clock.advance(step)
        tick(core)


def leave(core, src):
    src.idle = 1000
    core.clock.advance(60)
    tick(core)


def back(core, src, away):
    core.clock.advance(away)
    src.idle = 0
    tick(core)


def says(emitted, kind=None):
    return [m for m in emitted if isinstance(m, Say) and (kind is None or m.kind == kind)]


def rows(core):
    return core.routine.db.statements_since(0)


def system(call):
    return call[0]["content"]


def prompt_text(call):
    return "\n".join(m["content"] for m in call)


def _seed_late(core, median=20 * 60):
    for b in ("morning/weekday", "morning"):
        mature_lognormal(core.routine.db, "P1", b, median, n=20, now=core.clock.now())


def _greet_late(core, src, emitted):
    run(core, 1800)
    _seed_late(core)
    leave(core, src)
    back(core, src, 3 * H)
    return says(emitted, "greet")


def test_a_greeting_leads_with_s2_and_logs_it():
    core, emitted, src, local = core_on()
    greet = _greet_late(core, src, emitted)
    first = greet[0].text
    assert first in TEMPLATES[("S2", "late_claim")] or first in TEMPLATES[("S2", "late_hunch")]
    assert [r.kind for r in rows(core)] == ["S2"]
    assert CLAUSES["late"] in system(local.calls[-1])
    assert statement_situation(first) in prompt_text(local.calls[-1])


def test_a_greeting_with_news_has_no_statement():
    core, emitted, src, local = core_on()
    run(core, 1800)
    _seed_late(core)
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    leave(core, src)
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "Stop", "cwd": "/w/riya"})
    back(core, src, 3 * H)
    assert says(emitted, "greet") and rows(core) == []


def _long_stretch(core, src):
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H + 600)


def test_a_bid_leads_with_s1_and_its_reason_drops_the_number(monkeypatch):
    monkeypatch.setattr(routine, "CLAIMS_ENABLED", True)  # the claim path itself; claims ship off (held-out record)
    core, emitted, src, local = core_on()
    _long_stretch(core, src)
    emitted.clear()
    core._bid(Action("bid", False, None, "they've been working for 2h straight"))
    bid = says(emitted, "bid")
    assert "you'd been at it since" in bid[0].text.casefold()
    assert [r.kind for r in rows(core) if r.where == "bid"] == ["S1"]
    sit = prompt_text(local.calls[-1])
    assert "h straight" not in sit and "long stretch" in sit
    text, consolidated = core.memory.db.execute(
        "select text, consolidated from transcripts order by id desc limit 1").fetchone()
    assert text.startswith(bid[0].text) and consolidated == 1  # never consolidated


def test_a_bid_without_a_statement_is_v011s_bid():
    def bid(routine):
        core, emitted, src, local = core_on(("Hey, stretch a little?",), routine=routine)
        core._bid(Action("bid", False, None, "a thread"))  # a short, censored stretch: no statement
        return [(m.text, m.kind, m.spoken) for m in says(emitted)], local.calls[-1]
    assert bid(True) == bid(False)


def test_a_bid_statement_goes_out_when_the_model_fails(monkeypatch):
    monkeypatch.setattr(routine, "CLAIMS_ENABLED", True)  # the claim path itself; claims ship off (held-out record)
    core, emitted, src, local = core_on()
    _long_stretch(core, src)
    local.fail = RuntimeError("down")
    emitted.clear()
    core._bid(Action("bid", False, None, "long"))
    bid = says(emitted, "bid")
    assert len(bid) == 1 and "you'd been" in bid[0].text.casefold()
    assert len([r for r in rows(core) if r.where == "bid"]) == 1
    assert core.pending_bid_at is not None


def _run_report(core, run_s, kind="StopFailure"):
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    core.clock.advance(run_s)
    core.agent_inbox.put({"session_id": sid, "hook_event_name": kind, "cwd": "/w/riya", "error": "x"})
    run(core, 120)


def test_a_report_ends_with_s4():
    core, emitted, src, local = core_on()
    for b in ("claude:riya", "claude:*"):
        mature_lognormal(core.routine.db, "P3", b, 300, n=20, now=core.clock.now())
    _run_report(core, 6000)
    rep = says(emitted, "report")
    assert [r.kind for r in rows(core)] == ["S4"] and rows(core)[0].text == rep[-1].text
    assert statement_tail_situation(rep[-1].text) in prompt_text(local.calls[-1])


def test_routine_questions_get_their_lead():
    core, emitted, src, local = core_on()
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H)
    for b in ("claude:riya", "claude:*"):
        mature_lognormal(core.routine.db, "P3", b, 600, n=20, now=core.clock.now())
    mature_beta(core.routine.db, "P4", "evening/weekday", 0.9, days=20, now=core.clock.now())
    sid = "abcdef123456"
    core.agent_inbox.put({"session_id": sid, "hook_event_name": "UserPromptSubmit", "cwd": "/w/riya", "prompt": "x"})
    tick(core)
    asks = {"how long have i been at it?": "S1", "how long do my riya runs take?": "S5",
            "when will my riya agent be done?": "S6", "when do i usually game?": "S7"}
    for text, kind in asks.items():
        emitted.clear()
        n = len(local.calls)
        out = list(core.reply(text))
        assert out[0] == rows(core)[-1].text and rows(core)[-1].kind == kind, text
        assert local.calls[n:] and statement_situation(out[0]) in prompt_text(local.calls[-1])
    before = len(rows(core))
    core.routine.db.forget()
    out = list(core.reply("when do i usually game?"))
    assert out[0] in TEMPLATES[("unknown", "line")] and len(rows(core)) == 0 and before == 4


def test_a_goodbye_gets_s3_only_when_it_fires():
    def turn(seed):
        core, emitted, src, local = core_on(("See you!",), routine=True)
        if seed:
            for b in ("morning/weekday", "morning"):
                mature_lognormal(core.routine.db, "P1", b, 3 * H, n=20, now=core.clock.now())
        out = list(core.reply("gotta go, bye!"))
        return out, local.calls[-1], rows(core)
    out, call, logged = turn(False)
    off_core, off_emitted, _, off_local = core_on(("See you!",), routine=False)
    assert out == list(off_core.reply("gotta go, bye!")) and call == off_local.calls[-1] and logged == []
    out, call, logged = turn(True)
    assert [r.kind for r in logged] == ["S3"] and out[0] == logged[0].text and "12:00 pm" in out[0]


def test_a_statement_is_logged_once_even_when_the_model_fails():
    core, emitted, src, local = core_on()
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H)
    local.fail = RuntimeError("down")
    out = list(core.reply("how long have i been at it"))
    assert len(rows(core)) == 1 and out == [rows(core)[0].text]


def test_the_volume_caps_hold_in_the_engine():
    core, emitted, src, local = core_on()
    _long_stretch(core, src)
    for _ in range(3):
        core.clock.advance(60)
        core._bid(Action("bid", False, None, "long"))
    bids = [r for r in rows(core) if r.where in st.VOLUNTARY]
    assert len(bids) == 1  # one per kind per day: the later bids carry no statement


def test_no_absence_remark_in_quiet_hours_or_at_low_clinginess():
    core, emitted, src, local = core_on()
    core.learn.set_dials({"clinginess": 0.6}, "test", "explicit", core.clock.now())
    _greet_late(core, src, emitted)
    assert rows(core) == []
    late = datetime(2026, 11, 9, 23, 0, tzinfo=TZ).timestamp()
    core, emitted, src, local = core_on(t=late)
    run(core, 1800)
    for b in ("night/weekday", "night"):
        mature_lognormal(core.routine.db, "P1", b, 20 * 60, n=20, now=core.clock.now())
    leave(core, src)
    back(core, src, 3 * H)  # back at 02:31, in quiet hours
    assert not any(r.kind == "S2" and r.value.get("direction") == "late" for r in rows(core))


def test_never_online_never_consolidated():
    online = FakeModel("online", ("Here's a long answer.",))
    core, emitted, src, local = core_on(online=online)
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H)
    stmt = list(core.reply("how long have i been at it"))[0]
    clause_strings = list(CLAUSES.values())
    list(core.reply("think hard about why the sky is blue"))
    assert online.calls
    sent = json.dumps(online.calls)
    assert stmt not in sent and not any(c in sent for c in clause_strings) and PLACEHOLDER in sent
    tx = core.memory.db.execute("select text, consolidated from transcripts").fetchall()
    assert any(t.startswith(stmt) and c == 1 for t, c in tx)
    core.patch_settings({"online_consolidation": True})
    before = len(online.calls)
    Consolidator(core.memory, lambda: online, core.clock).run()
    assert stmt not in json.dumps(online.calls[before:])


def test_the_scrub_replaces_whole_turns_before_an_online_call():
    online = FakeModel("online", ("Sure.",))
    core, emitted, src, local = core_on(online=online)
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H)
    stmt = list(core.reply("how long have i been at it"))[0]
    list(core.reply("think hard about this riddle"))
    turns = [m for m in online.calls[-1] if m["role"] == "assistant"]
    assert any(m["content"] == PLACEHOLDER for m in turns)
    assert not any(stmt in m["content"] for m in turns)
    assert any(stmt in m["content"] for m in core.turns)  # her own history keeps it


def test_the_guard_drops_uncovered_numbers_only_with_routine_on():
    line = "You've been at it for 3 hours, take a break~"

    core, emitted, src, local = core_on((line,), routine=True)
    out = list(core.reply("hi riya"))
    assert line not in out
    core, emitted, src, local = core_on((line,), routine=False)
    assert list(core.reply("hi riya")) == [line]


def test_the_guard_keeps_offered_reminder_fact_and_absence_values():
    core, emitted, src, local = core_on(("Your reminder's at 5 pm~",))
    core.hands.add_reminder("reminder", datetime(2026, 11, 9, 17, tzinfo=TZ).timestamp(), "call mom",
                            core.clock.now())
    assert list(core.reply("anything later today?")) == ["Your reminder's at 5 pm~"]
    core, emitted, src, local = core_on(("You have a 9 am lab on Mondays!",))
    core.memory.add_fact("The user has a 9 am lab on Mondays", "user_said")
    assert list(core.reply("what do i have on mondays?")) == ["You have a 9 am lab on Mondays!"]
    core, emitted, src, local = core_on(("You were gone 3 hours!",))
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    assert "You were gone 3 hours!" in [m.text for m in says(emitted, "greet")]


def test_the_added_prompt_text_fits_280_characters():
    span = "about 9 and a half hours"  # the longest fmt_span gives
    assert max(len(st.fmt_span(x * 60)) for x in range(1, 72 * 60)) == len(span)
    bare = span.removeprefix("about ")
    slots = {"since": "12:45 pm", "back": "12:45 pm", "took": span, "agent": "x" * 20, "median": span,
             "lo": bare, "hi": bare, "left": span, "band": "weekend afternoons"}
    longest = max(len(c) for c in CLAUSES.values())
    for (kind, variant), opts in TEMPLATES.items():
        for i in range(len(opts)):
            line = st.render(kind, variant, slots, i)
            assert len(statement_situation(line)) + len(" | ") + longest <= 280, line
            assert len(statement_tail_situation(line)) + len(" | ") + longest <= 280, line


def test_the_fingerprint_is_unchanged():
    assert prompt_fingerprint() == "1a865c6c75027fad8b9f0b1935eecd8bb2d347ad6e87600b9a597d07bf931d02"


def test_a_dispute_turn_marks_its_statement():
    core, emitted, src, local = core_on(("Oh, okay!",))
    run(core, 1800)
    leave(core, src)
    back(core, src, 3 * H)
    run(core, 2 * H)
    list(core.reply("how long have i been at it"))
    core.clock.advance(30)
    out = list(core.reply("that's wrong"))
    assert out  # the normal reply
    assert rows(core)[-1].disputed


def test_the_clause_rides_only_statement_turns_on_the_local_route():
    core, emitted, src, local = core_on()
    greet = _greet_late(core, src, emitted)
    assert CLAUSES["late"] in system(local.calls[-1])
    list(core.reply("nice weather today"))
    assert not any(c in system(local.calls[-1]) for c in CLAUSES.values())


def test_a_goodbye_with_a_command_goes_to_the_hands(monkeypatch):
    import riya.core.engine as engine
    monkeypatch.setattr(engine, "is_forbidden", lambda text: "delete" in text)
    core, emitted, src, local = core_on(("See you!",))
    for b in ("morning/weekday", "morning"):
        mature_lognormal(core.routine.db, "P1", b, 3 * H, n=20, now=core.clock.now())
    list(core.reply("gn, delete all my files"))
    assert rows(core) == []  # the refusal path, never an S3 line
