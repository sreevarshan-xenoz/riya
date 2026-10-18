from riya.core.clock import FakeClock
from riya.core.feelings import EXPRESSIONS, MAX_TURN_DELTA, Feelings, missing_curve

TRAITS = {"sweetness": 0.85, "teasing": 0.5, "clinginess": 0.8, "verbosity": 0.35}


def tick_for(f, clock, seconds, present=True):
    for _ in range(int(seconds)):
        clock.advance(1)
        f.tick(present=present)


def test_missing_curve_targets():
    assert missing_curve(0) == 0
    assert abs(missing_curve(2) - 0.10) < 0.02
    assert abs(missing_curve(24) - 0.50) < 0.02
    assert abs(missing_curve(48) - 0.85) < 0.02
    vals = [missing_curve(h) for h in (0, 1, 2, 6, 12, 24, 36, 48, 96, 720)]
    assert vals == sorted(vals) and vals[-1] <= 1.0


def test_clinginess_scales_missing():
    f = Feelings(FakeClock(), {**TRAITS, "clinginess": 0.4})
    f.on_return(away_s=48 * 3600)
    assert abs(f.state.missing - 0.425) < 0.03


def test_return_burst_then_fades_while_together():
    clock = FakeClock()
    f = Feelings(clock, TRAITS)
    v0 = f.state.valence
    f.on_return(away_s=2 * 86400)
    assert f.state.missing > 0.8 and f.state.valence > v0 + 0.2
    assert f.expression() == "missed_you"
    tick_for(f, clock, 120)
    assert f.state.missing > 0.4
    tick_for(f, clock, 18 * 60)
    assert f.state.missing < 0.15


def test_short_absence_is_light():
    f = Feelings(FakeClock(), TRAITS)
    f.on_return(away_s=3 * 3600)
    assert f.state.missing < 0.2 and f.expression() != "missed_you"


def test_change_per_turn_is_bounded():
    f = Feelings(FakeClock(), TRAITS)
    prev = f.state.valence
    for text in ["I love you, you're amazing!!!", "you are useless and stupid", "thanks so much, great!!",
                 "I hate this, awful"] * 3:
        f.on_user_text(text)
        assert abs(f.state.valence - prev) <= MAX_TURN_DELTA + 1e-9
        prev = f.state.valence


def test_user_mood_estimate_is_separate():
    f = Feelings(FakeClock(), TRAITS)
    before = f.state.valence
    f.estimate_user("today was awful, I failed and I'm so tired")
    assert f.state.user_mood.valence < 0 and 0 < f.state.user_mood.confidence <= 1
    assert f.state.valence == before


def test_persistence_round_trip():
    clock = FakeClock()
    f = Feelings(clock, TRAITS)
    f.on_return(away_s=86400)
    g = Feelings.from_dict(f.to_dict(), clock, TRAITS)
    assert g.state == f.state


def test_prompt_tag_and_expressions():
    f = Feelings(FakeClock(), TRAITS)
    f.on_return(away_s=2 * 86400)
    assert f.prompt_tag().startswith("[Riya's state: ") and "missed them" in f.prompt_tag()
    f.on_event("bid_ignored")
    assert f.expression() == "pouty" and f.expression() in EXPRESSIONS


def test_expression_names():
    assert EXPRESSIONS == ("neutral", "happy", "excited", "missed_you", "pouty", "flustered", "teasing", "worried",
                           "sad", "sleepy", "curious", "thinking", "surprised", "covering_eyes")


def test_expression_table_priority():
    f = Feelings(FakeClock(), TRAITS)
    assert f.expression() == "neutral"
    f.state.curiosity = 0.7
    assert f.expression() == "curious"
    f.state.valence = 0.35
    assert f.expression() == "happy"
    f.state.warmth, f.state.arousal = 0.9, 0.55
    assert f.expression() == "flustered"
    f.state.valence, f.state.arousal = 0.5, 0.7
    assert f.expression() == "excited"
    f.state.valence = -0.5
    assert f.expression() == "sad"
    f.state.concern = 0.6
    assert f.expression() == "worried"
    f.state.surprise = 0.7
    assert f.expression() == "surprised"
    f.state.energy = 0.2
    assert f.expression() == "sleepy"


def test_overrides_expire_and_toggle():
    clock = FakeClock()
    f = Feelings(clock, TRAITS)
    f.on_event("bid_ignored")
    clock.advance(29)
    assert f.expression() == "pouty"
    clock.advance(2)
    assert f.expression() == "neutral"
    f.on_event("teased")
    assert f.expression() == "teasing"
    clock.advance(11)
    assert f.expression() == "neutral"
    f.on_event("thinking")
    clock.advance(3600)
    assert f.expression() == "thinking"
    f.on_event("thinking_done")
    assert f.expression() == "neutral"


def test_negative_user_raises_concern_and_confidence_formula():
    f = Feelings(FakeClock(), TRAITS)
    f.on_user_text("I'm so sad and tired and lonely")
    assert f.state.concern > 0 and f.state.user_mood.valence < 0
    assert abs(f.state.user_mood.confidence - 0.6) < 1e-9
    f.estimate_user("a b c d e f g h")
    assert f.state.user_mood.confidence <= 0.8


def test_affect_decays_toward_mood_and_restore_decays_elapsed_time():
    clock = FakeClock()
    f = Feelings(clock, TRAITS)
    f.on_return(away_s=2 * 86400)
    dumped = f.to_dict()
    clock.advance(3600)
    g = Feelings.from_dict(dumped, clock, TRAITS)
    g.tick(present=True)
    assert g.state.missing < 0.001
    assert abs(g.state.valence - g.state.mood) < 0.01
