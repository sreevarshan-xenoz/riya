"""v0.4 Task 5b: the game lifecycle in the engine: start, the secret guard, win, stop, the 15-turn and idle ends, and
bids offering a game while an agent is busy. Offline: fake models and a fake clock."""
import random

from riya.core import games
from riya.core.clock import FakeClock
from riya.core.initiative import Action
from riya.core.protocol import Say, SetSettings
from tests.fakes import make_core


def run_ticks(core, clock, seconds, step=1):
    for _ in range(int(seconds / step)):
        clock.advance(step)
        it = core.tick()
        if it:
            list(it)


def situation(local, i=-1):
    return next(p for p in local.calls[i][0]["content"].split("\n\n") if p.startswith("[Situation]"))


def said(emitted):
    return [m.text for m in emitted if isinstance(m, Say)]


def transcript(core):
    return [(r["role"], r["text"]) for r in core.memory.db.execute("select role, text from transcripts order by id")]


def gaming(kind_text="let's play 20 questions", replies=("Okay, got one~",)):
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=replies)
    core.game_rng = random.Random(3)
    out = list(core.reply(kind_text))
    return core, emitted, src, local, clock, out


def test_lets_play_20_questions_starts_a_game():
    core, emitted, _, local, _, out = gaming()
    g = core.game
    assert g is not None and g.kind == "twenty_questions" and g.secret
    s = situation(local)
    assert "20 questions" in s and g.secret in s and "one short" in s
    assert local.max_tokens[-1] == 110                     # the small budget
    assert out == ["Okay, got one~"]
    assert not any(g.secret in t for _, t in transcript(core)) and "[Situation]" not in str(transcript(core))


def test_the_secret_never_reaches_transcripts_or_say_text_while_live():
    core, emitted, _, local, _, _ = gaming()
    secret = core.game.secret
    local.replies = [f"Hehe, it's {games._a(secret)}! Yes."]
    for q in ("is it alive?", "is it big?", "can I eat it?"):
        out = list(core.reply(q))
        assert len(out) == 1 and out[0] in games.TEASES      # one sentence, and the guard swapped it
    assert not any(secret in t.lower() for t in said(emitted))
    assert not any(secret in t.lower() for _, t in transcript(core))
    assert not any(secret in t["content"].lower() for t in core.turns)
    assert core.game is not None and core.game.questions == 3


def test_a_correct_guess_wins_and_ends():
    core, emitted, _, local, _, _ = gaming(replies=("Okay, got one~", "Yes, you got it!"))
    secret = core.game.secret
    out = list(core.reply(f"is it {games._a(secret)}?"))
    assert core.game is None
    assert "They win" in situation(local) and out == ["Yes, you got it!"]
    list(core.reply("is it alive?"))                       # no game anymore: a normal reply
    assert "[Situation]" not in local.calls[-1][0]["content"]


def test_stop_ends_the_game():
    core, emitted, _, local, _, _ = gaming()
    secret = core.game.secret
    list(core.reply("stop"))
    assert core.game is None
    s = situation(local)
    assert "stop the game" in s and secret not in s


def test_giving_up_reveals_the_secret():
    core, emitted, _, local, _, _ = gaming(replies=("Okay, got one~", "Aww, okay."))
    secret = core.game.secret
    out = list(core.reply("i give up"))
    assert core.game is None and out == ["Aww, okay.", f"It was {games._a(secret)}!"]
    assert ("assistant", f"Aww, okay. It was {games._a(secret)}!") in transcript(core)


def test_the_15th_turn_ends_the_game():
    core, emitted, _, local, _, _ = gaming()
    for i in range(games.MAX_TURNS):
        assert core.game is not None
        list(core.reply("is it alive?"))
    assert core.game is None and "game is over" in situation(local)
    calls = len(local.calls)
    list(core.reply("is it alive?"))
    assert len(local.calls) == calls + 1 and "[Situation]" not in local.calls[-1][0]["content"]


def test_an_idle_game_ends_after_10_minutes():
    core, emitted, _, local, clock, _ = gaming()
    run_ticks(core, clock, 540, step=60)
    assert core.game is not None
    list(core.reply("is it alive?"))
    run_ticks(core, clock, 599, step=1)
    assert core.game is not None
    before = said(emitted)
    run_ticks(core, clock, 1)
    assert core.game is None and said(emitted) == before    # silently


def test_word_chain_her_word_is_always_said():
    core, emitted, _, local, _, out = gaming("let's play word chain", replies=("Ooh, fun!", "Nice one~"))
    first = core.game.word
    assert out == ["Ooh, fun!", f"I'll start: {first}!"]
    need = first[-1]
    word = next(w for w in games.WORDS[need] if w != first)
    out = list(core.reply(word))
    mine = core.game.word
    assert out == ["Nice one~", f"My word: {mine}!"] and mine != word
    assert f'"{mine}"' in situation(local)


def test_word_chain_invalid_word_is_judged_by_the_engine():
    core, emitted, _, local, _, _ = gaming("let's play word chain", replies=("Ooh, fun!", "Hmm, nope~"))
    before = core.game.word
    out = list(core.reply("zzzzz"))
    assert out == ["Hmm, nope~"] and core.game.word == before and "isn't a real word" in situation(local)


def test_riddles_ask_the_riddle_after_her_line():
    core, emitted, _, local, _, out = gaming("can we play riddles", replies=("Riddle time!",))
    q = games.RIDDLES[core.game.riddle].question
    assert out == ["Riddle time!", q]
    assert not any(games.mentions(situation(local), a) for a in games.RIDDLES[core.game.riddle].answers)


def test_a_game_turn_never_saves_a_remember_fact():
    core, *_ = gaming("let's play word chain")
    list(core.reply("remember that I like trains"))
    assert core.memory.list_facts() == []


# --- bids offer a game while an agent is busy -----------------------------------------------------------------------

def on_break(busy, replies=("Water break?", "Yay, let's go!")):
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=replies)
    core.game_rng = random.Random(1)
    core.handle(SetSettings(patch={"quiet_hours": None}))
    run_ticks(core, clock, 2 * 3600 + 600, step=300)
    if busy:
        core.agent_inbox.put({"hook_event_name": "UserPromptSubmit", "session_id": "abc123def", "cwd": "E:/riya"})
    src.idle = 150
    run_ticks(core, clock, 150, step=150)
    assert core.presence.state == "break"
    bids = [m for m in emitted if isinstance(m, Say) and m.kind == "bid"]
    assert len(bids) == 1
    return core, emitted, src, local, clock, bids[0]


def test_a_bid_offers_a_game_while_an_agent_is_busy_and_sure_accepts():
    core, emitted, src, local, clock, bid = on_break(busy=True)
    kind = core._offer[0]
    assert kind in games.GAMES
    assert "busy" in situation(local) and games.NAMES[kind] in situation(local)
    assert bid.text == f"Water break? {games.offer_line(kind)}"   # the model didn't name it: the engine did
    src.idle = 0
    list(core.reply("sure"))
    assert core.game is not None and core.game.kind == kind and core._offer is None
    assert "You're playing" in situation(local)


def test_a_bid_offers_no_game_when_no_agent_is_busy():
    core, emitted, src, local, clock, bid = on_break(busy=False)
    assert bid.text == "Water break?" and core._offer is None and "busy" not in situation(local)
    src.idle = 0
    list(core.reply("sure"))
    assert core.game is None


def test_an_offer_expires():
    core, emitted, src, local, clock, bid = on_break(busy=True)
    clock.advance(301)
    src.idle = 0
    list(core.reply("sure"))
    assert core.game is None


def test_no_game_offer_while_a_game_is_on():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=("Water break?",))
    core.agent_inbox.put({"hook_event_name": "UserPromptSubmit", "session_id": "abc123def", "cwd": "E:/riya"})
    run_ticks(core, clock, 1)
    core.game = games.new_game("riddles", random.Random(0))
    core._bid(Action("bid", False, None, "worked 2.5 h straight"))
    assert core._offer is None and "busy" not in situation(local)
    core.game = None                                       # the same bid without a game does offer one
    core._bid(Action("bid", False, None, "worked 2.5 h straight"))
    assert core._offer is not None and "busy" in situation(local)
