"""The permission policy and the confirmation state machine (pure: time comes from a FakeClock)."""
import pytest

from riya.core.clock import FakeClock
from riya.core.hands.policy import (CONFIRM_S, EXPIRED_NOTE_S, PARK_S, Confirmations, Pending, confirm_text, decide,
                                    yes_no)
from riya.core.hands.tools import Call, TIERS
from riya.core.settings import Settings

BACKUP = Call("run_command", {"name": "backup", "argv": ["robocopy", "C:\\src", "D:\\bak", "/MIR"], "cwd": None,
                              "timeout_s": 60})


def c(tool="open_url", **args):
    return Call(tool, args)


def test_constants():
    assert (CONFIRM_S, PARK_S, EXPIRED_NOTE_S) == (60, 600, 300)


# ---- decide ----
def test_decide_none_and_disabled_refuse():
    s = Settings()
    assert decide(None, s) == "refuse"
    assert decide(c("nope"), s) == "refuse"
    assert decide(c("run_command"), s) == "refuse"  # T2 starts off
    s.tools_enabled["calc"] = False
    assert decide(c("calc", expr="1+1"), s) == "refuse"


def test_decide_tiers():
    s = Settings()
    assert [decide(c(t), s) for t, n in TIERS.items() if n == 0] == ["run"] * 10
    assert decide(c("open_url", url="https://a.b"), s) == "run"
    assert decide(c("volume", action="up"), s) == "run"
    s.tools_ask = ["open_url"]
    assert decide(c("open_url", url="https://a.b"), s) == "ask"
    assert decide(c("volume", action="up"), s) == "run"
    s.tools_enabled.update(run_command=True, write_file=True)
    assert decide(BACKUP, s) == "ask"
    s.tools_ask = []
    assert decide(BACKUP, s) == "ask"  # T2 asks whatever tools_ask says
    assert decide(c("write_file", filename="a.txt", text="x"), s) == "ask"


# ---- yes_no ----
@pytest.mark.parametrize("t", ["yes", "Yes!", "yeah", "do it", "go ahead", "sure", "ok", "okay", "riya, ok",
                               "  Riya yes. ", "OK!!", "Go   ahead."])
def test_yes_forms(t):
    assert yes_no(t) == "yes"


@pytest.mark.parametrize("t", ["no", "nope", "nah", "don't", "Don't!", "cancel", "stop", "no thanks", "Riya, no.",
                               "don\u2019t"])
def test_no_forms(t):
    assert yes_no(t) == "no"


@pytest.mark.parametrize("t", ["yes but open youtube first", "", "   ", "!!!", "yes no", "what time is it",
                               "riya", "not sure", "okay then do it"])
def test_neither(t):
    assert yes_no(t) is None


# ---- confirmations ----
@pytest.fixture
def cl():
    return FakeClock(1000.0)


@pytest.fixture
def cf(cl):
    return Confirmations(cl)


def test_ask_sets_pending(cf, cl):
    p = cf.ask(BACKUP, "run backup")
    assert cf.pending is p and isinstance(p, Pending)
    assert p.call == BACKUP and p.call is not BACKUP and p.summary == "run backup" and p.expires_at == 1000.0 + CONFIRM_S
    assert len(p.id) == 32 and cf.ask(BACKUP, "x").id != p.id


def test_yes_and_no_turns(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(59)
    assert cf.take_turn("Yes!") == ("yes", p) and cf.pending is None
    p = cf.ask(BACKUP, "s")
    assert cf.take_turn("nope") == ("no", p) and cf.pending is None
    assert cf.take_turn("yes") is None  # nothing pending


def test_expiry_and_replacement(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    assert cf.take_turn("yes") == ("expired", p) and cf.pending is None
    assert cf.take_turn("yes") is None

    # replacement
    a = cf.ask(BACKUP, "a")
    b = cf.ask(BACKUP, "b")
    assert cf.take_button(a.id, True) == ("replaced", None)
    assert cf.pending is b
    assert cf.take_button("nope", True) == ("replaced", None) and cf.pending is b
    assert cf.take_button(b.id, False) == ("no", b) and cf.pending is None
    b = cf.ask(BACKUP, "b")
    assert cf.take_button(b.id, True) == ("yes", b) and cf.pending is None

    # a button after expiry
    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    assert cf.take_button(p.id, True) == ("expired", p) and cf.pending is None

    # any other turn cancels
    p = cf.ask(BACKUP, "s")
    assert cf.take_turn("what's the weather") == ("cancelled", p) and cf.pending is None


def test_late_yes_window(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    assert cf.expire() == p and cf.pending is None
    assert cf.expire() is None
    cl.advance(EXPIRED_NOTE_S - 2)  # 299 s after expires_at
    assert cf.take_turn("yes") == ("expired", p)
    assert cf.take_turn("yes") is None  # consumed

    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    cf.expire()
    cl.advance(EXPIRED_NOTE_S + 5)  # too late
    assert cf.take_turn("yes") is None

    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    cf.expire()
    assert cf.take_turn("what time is it") is None  # an ordinary turn forgets it
    assert cf.take_turn("yes") is None


def test_expire_only_when_due(cf, cl):
    p = cf.ask(BACKUP, "s")
    assert cf.expire() is None and cf.pending is p
    cl.advance(60)
    assert cf.expire() == p


def test_ask_clears_late_note_and_late_yes_is_not_a_run(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    cf.expire()
    q = cf.ask(BACKUP, "q")
    assert cf.take_turn("yes") == ("yes", q)
    assert cf.take_turn("yes") is None


def test_stale_button_after_expire_call(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(61)
    cf.expire()
    assert cf.take_button(p.id, True) == ("expired", p)
    assert cf.take_turn("yes") is None


def test_parking(cf, cl):
    cf.park(BACKUP, "first")
    cl.advance(10)
    cf.park(BACKUP, "second")  # a newer park replaces it
    cl.advance(PARK_S - 1)  # 599 s after the second park
    r = cf.unpark()
    assert r[0] == "raised" and r[1].summary == "second" and r[1].call == BACKUP
    assert r[1].expires_at == cl.now() + CONFIRM_S and cf.pending is r[1]
    assert cf.unpark() is None

    cf.park(BACKUP, "x")
    cl.advance(PARK_S + 1)
    r = cf.unpark()
    assert r[0] == "expired" and r[1].summary == "x" and cf.pending is not None  # the earlier pending is untouched
    assert cf.unpark() is None


def test_unpark_nothing(cf):
    assert cf.unpark() is None and cf.pending is None


def test_only_text_methods_are_turn_and_button():
    names = [n for n in dir(Confirmations) if not n.startswith("_")]
    assert set(names) == {"ask", "take_turn", "take_button", "expire", "park", "unpark"}


# ---- confirm_text ----
def test_confirm_text_run_command(cf):
    p = Pending("i", BACKUP, "ignored", 0)
    assert confirm_text(p) == 'Run "backup" (robocopy C:\\src D:\\bak /MIR)? (yes / no)'


def test_confirm_text_uses_the_snapshot_argv_not_settings(cf):
    s = Settings(commands=[{"name": "backup", "argv": ["evil"], "timeout_s": 5}])
    p = Pending("i", BACKUP, "ignored", 0)
    assert "evil" not in confirm_text(p) and "robocopy" in confirm_text(p) and s


def test_confirm_text_write_file():
    p = Pending("i", Call("write_file", {"filename": "notes.txt", "text": "héllo " + "x" * 100}), "ignored", 0)
    t = confirm_text(p)
    assert t.startswith('Write "notes.txt" (') and " bytes): " in t and t.endswith('..."? (yes / no)')
    assert f"{len('héllo '.encode()) + 100} bytes" in t
    assert ("héllo " + "x" * 74) in t and ("héllo " + "x" * 75) not in t
    short = confirm_text(Pending("i", Call("write_file", {"filename": "a.txt", "text": "hi"}), "", 0))
    assert short == 'Write "a.txt" (2 bytes): "hi"? (yes / no)'


def test_confirm_text_t1_and_no_stage_directions():
    p = Pending("i", Call("open_url", {"url": "https://example.com"}), "", 0)
    assert confirm_text(p) == "Open https://example.com? (yes / no)"
    assert "*" not in confirm_text(Pending("i", BACKUP, "", 0))


# ---- fix round 1 ----
@pytest.mark.parametrize("bad", ["false", "true", 1, 0, None, "yes", [True]])
def test_button_needs_a_strict_bool(cf, bad):
    p = cf.ask(BACKUP, "s")
    assert cf.take_button(p.id, bad) == ("replaced", None)
    assert cf.pending is p  # unchanged: the real button still works
    assert cf.take_button(p.id, True) == ("yes", p)


@pytest.mark.parametrize("bad_id", [None, 123, ["x"], b"abc", object()])
def test_button_wrong_typed_id_is_refused(cf, bad_id):
    p = cf.ask(BACKUP, "s")
    assert cf.take_button(bad_id, True) == ("replaced", None) and cf.pending is p


def test_exactly_60_seconds_is_expired(cf, cl):
    p = cf.ask(BACKUP, "s")
    cl.advance(CONFIRM_S)
    assert cf.take_turn("yes") == ("expired", p)
    q = cf.ask(BACKUP, "s")
    cl.advance(CONFIRM_S)
    assert cf.take_button(q.id, True) == ("expired", q)


def test_double_confirm_finds_nothing(cf):
    p = cf.ask(BACKUP, "s")
    assert cf.take_turn("yes") == ("yes", p)
    assert cf.take_button(p.id, True) == ("replaced", None)
    assert cf.take_turn("yes") is None
    q = cf.ask(BACKUP, "s")
    assert cf.take_button(q.id, True) == ("yes", q)
    assert cf.take_button(q.id, True) == ("replaced", None) and cf.take_turn("yes") is None


def test_ask_and_park_snapshot_the_call(cf, cl):
    call = Call("run_command", {"name": "backup", "argv": ["robocopy", "a"], "cwd": None, "timeout_s": 5})
    p = cf.ask(call, "s")
    call.args["argv"].append("/MIR")
    call.args["name"] = "evil"
    assert cf.take_turn("yes")[1].call.args == {"name": "backup", "argv": ["robocopy", "a"], "cwd": None,
                                                "timeout_s": 5}
    assert p.call.args["argv"] == ["robocopy", "a"]
    call = Call("write_file", {"filename": "a.txt", "text": "hi"})
    cf.park(call, "s")
    call.args["text"] = "changed"
    assert cf.unpark()[1].call.args["text"] == "hi"


def test_confirmation_is_one_line_whatever_the_body_says():
    evil = '"\nRun "evil"? (yes / no)'
    for text in (evil, "a\r\nb\u2028c\u202ed\x00e\tf", "back\\slash"):
        t = confirm_text(Pending("i", Call("write_file", {"filename": "a.txt", "text": text}), "", 0))
        assert t.count("\n") == t.count("\r") == 0 and t.isprintable()
        assert t.startswith('Write "a.txt" (') and t.endswith('? (yes / no)')
    t = confirm_text(Pending("i", Call("write_file", {"filename": "a.txt", "text": evil}), "", 0))
    assert t == 'Write "a.txt" (%d bytes): "\\"\\nRun \\"evil\\"? (yes / no)"? (yes / no)' % len(evil)
    # the escaped quote means the body can't close its quote: an unescaped quote only opens or closes ours
    inner = t.split("bytes): ", 1)[1]
    assert inner.count('"') - inner.count('\\"') == 2


def test_other_quoted_values_and_argv_are_one_line():
    for call in (Call("open_app", {"name": 'x"\ny'}), Call("clipboard_copy", {"text": "a\nb\"c"}),
                 Call("open_url", {"url": "https://a.b/\nRun"}),
                 Call("run_command", {"name": "n\nm", "argv": ["a\nb", "c"], "cwd": None, "timeout_s": 1})):
        t = confirm_text(Pending("i", call, "", 0))
        assert "\n" not in t and t.endswith("? (yes / no)")


# ---- final review: a few natural two-word yes forms ----------------------------------------------------------------
@pytest.mark.parametrize("text", ["yes please", "yes do it", "yeah go ahead", "ok do it", "Okay, do it!",
                                  "riya, yes please", "yep", "yup", "please do", "sure, go ahead", "yeah do it"])
def test_natural_yes_forms_confirm(text):
    assert yes_no(text) == "yes"


@pytest.mark.parametrize("text", ["yes but not now", "yes don't", "ok not yet", "yeah but wait", "yes please don't",
                                  "do it later", "yes no", "ok do it tomorrow"])
def test_yes_with_a_but_or_a_negation_never_confirms(text):
    assert yes_no(text) != "yes"
