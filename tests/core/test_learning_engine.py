"""The dials in effect: the style line in both personas, the shaping scales, the emoji dial and the live feelings."""
from riya.core.prompt import STYLE_LINE, style_line
from riya.core.shaping import count_emoji
from tests.fakes import FakeModel, make_core

TEASE_LOW = "Keep teasing very light."


def system(model, i=-1) -> str:
    return model.calls[i][0]["content"]


def said(emitted) -> list[str]:
    return [m.text for m in emitted if type(m).__name__ == "Say"]


def test_defaults_change_nothing():
    core, _, _, local = make_core()
    list(core.reply("hello there"))
    assert STYLE_LINE not in system(local)
    assert core._persona("local") == core.personas["local"] and core._persona("online") == core.personas["online"]


def test_dials_shape_the_persona_for_both_routes():
    online = FakeModel("online", ["Recursion is a function calling itself. Neat, right?"])
    core, _, _, local = make_core(online=online)
    core.learn.nudge("teasing", -0.15, "test", "explicit", core.clock.now())
    list(core.reply("hello there"))
    assert STYLE_LINE in system(local) and TEASE_LOW in system(local)
    list(core.reply("explain recursion to me"))
    assert online.calls and TEASE_LOW in system(online) and STYLE_LINE in system(online)


def test_only_the_style_line_goes_online():
    online = FakeModel("online", ["Recursion is a function calling itself."])
    core, _, _, _ = make_core(online=online)
    core.learn.set_dials({"teasing": 0.3, "verbosity": 0.2}, "test", "explicit", core.clock.now())
    list(core.reply("explain recursion to me"))
    line = style_line(core.learn.dials())
    persona = core.personas["online"] + chr(10) * 2 + line + chr(10) * 2
    assert system(online).startswith(persona)
    assert "How they like you" not in system(online)


def test_low_verbosity_caps_small_talk_at_two():
    core, emitted, _, local = make_core(local_replies=["One. Two. Three. Four."])
    core.learn.set_dials({"verbosity": 0.2}, "test", "explicit", core.clock.now())
    assert len(list(core.reply("hello there"))) == 2
    assert "at most 2 sentences" in system(local)
    core2, _, _, local2 = make_core(local_replies=["One. Two. Three. Four."])
    core2.learn.set_dials({"verbosity": 0.5}, "test", "explicit", core2.clock.now())
    assert len(list(core2.reply("hello there"))) == 4
    assert "at most 4 sentences" in system(local2)


def test_default_verbosity_keeps_three():
    core, _, _, local = make_core(local_replies=["One. Two. Three. Four."])
    assert len(list(core.reply("hello there"))) == 3 and "at most 3 sentences" in system(local)


def test_emoji_dial_below_015_drops_every_emoji():
    core, emitted, _, _ = make_core(local_replies=["Hi \U0001F60A there."])
    core.learn.set_dials({"emoji": 0.1}, "test", "explicit", core.clock.now())
    list(core.reply("hello there"))
    assert not any(count_emoji(t) for t in said(emitted))
    assert core.turns[-1]["content"] == "Hi there."
    core2, emitted2, _, _ = make_core(local_replies=["Hi \U0001F60A there."])
    list(core2.reply("hello there"))
    assert sum(count_emoji(t) for t in said(emitted2)) == 1 and "\U0001F60A" in core2.turns[-1]["content"]


def test_emoji_dial_applies_to_bids():
    core, emitted, _, _ = make_core(local_replies=["Water break? \U0001F4A7"])
    core.learn.set_dials({"emoji": 0.1}, "test", "explicit", core.clock.now())
    from riya.core.initiative import Action
    core._bid(Action("bid", False, None, "a break"))
    assert said(emitted) and not any(count_emoji(t) for t in said(emitted))


def test_clinginess_dial_feeds_missing_you():
    core, _, _, _ = make_core()
    core.learn.set_dials({"clinginess": 0.4}, "test", "explicit", core.clock.now())
    assert core.feelings.traits["clinginess"] == 0.4
    core.feelings.on_return(30 * 3600)
    low = core.feelings.state.missing
    core2, _, _, _ = make_core()
    core2.feelings.on_return(30 * 3600)
    assert low < core2.feelings.state.missing
