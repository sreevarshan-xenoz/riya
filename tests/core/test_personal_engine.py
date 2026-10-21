"""v0.11 Task 7: the engine's side of a personal request (§4.4, §4.7.8, §4.7.9). Only a plain reply on the local route
asks memory for the personal fill, and only then, with a fact in the block, does the situation gain the one
personal_situation() line, after any situation the turn already has; games, tools, remember-that turns, screen
questions (even ones she can't look for), bids, greetings, reports and every online-routed request (even one that falls
back to local) get neither. On those paths a strong fact still reaches the prompt by ordinary retrieval, and the line
still never does."""
from riya.agents.hub import AgentEvent
from riya.core.eyes.see import is_screen_question
from riya.core.initiative import Action
from riya.core.memory import Fact, personal_request
from riya.core.prompt import PERSONAL_SITUATION, _fact_line, personal_situation
from tests.embed_stubs import TableEmbedder
from tests.fakes import FakeModel, make_core

FILM = "pick a film for me tonight"  # a personal request, small talk on the local route
FACTS = ("The user loves old noir films", "The user adores slow quiet dramas", "The user hates jump scares")
STRONG = "The user always watches with a big bowl of popcorn"  # at 0.9: eligible without the fill
SCREEN = "what's on my screen? pick a film for me from it"  # a screen question and a personal request


def _core(sims: dict[str, float], **kw):
    """make_core, its memory on a TableEmbedder holding sims' texts as facts."""
    core, emitted, src, local = make_core(**kw)
    core.memory.embedder = TableEmbedder(sims)
    for text in sims:
        core.memory.add_fact(text, source="user_said")
    return core, emitted, src, local


def _system(model, i: int = 0) -> str:
    return model.calls[i][0]["content"]


def _recording(core) -> list[bool]:
    """Wraps core.memory.retrieve so each call records the personal flag it got (False when it passed none)."""
    seen, real = [], core.memory.retrieve

    def retrieve(query, *a, personal=False, **kw):
        seen.append(personal)
        return real(query, *a, personal=personal, **kw)

    core.memory.retrieve = retrieve
    return seen


def test_a_personal_request_gets_the_fill_and_the_situation():
    assert personal_request(FILM)
    core, _, _, local = _core(dict.fromkeys(FACTS, 0.5))
    assert list(core.reply(FILM))
    system = _system(local)
    for fact in FACTS:
        assert fact in system, fact
    assert PERSONAL_SITUATION in system and system.count(PERSONAL_SITUATION) == 1
    assert "### Reply rules: List" not in system  # a personal turn is not an about-me turn


def test_an_online_routed_personal_request_gets_no_fill_and_no_situation():
    text = "plan a quiet evening for me"  # router.THINKING matches "plan": it goes online (§4.7.9, a known limit)
    assert personal_request(text)
    online = FakeModel("online", ["Sure."])
    core, _, _, local = _core({**dict.fromkeys(FACTS, 0.5), STRONG: 0.9}, online=online)
    assert core.memory.retrieve(text, personal=True).filled  # on the local route these would fill
    seen = _recording(core)
    assert list(core.reply(text))
    assert seen == [False]
    system = _system(online)
    for fact in FACTS:
        assert fact not in system, fact
    assert STRONG in system  # ordinary retrieval, as before v0.11
    assert PERSONAL_SITUATION not in system and local.calls == []


def test_an_online_request_that_falls_back_to_local_gets_no_fill_and_no_situation():
    """The prompt is built once for the online route; the local fallback streams those same messages."""
    text = "plan a quiet evening for me"
    assert personal_request(text)
    online = FakeModel("online", ["x"], fail=RuntimeError("down"))
    core, _, _, local = _core({**dict.fromkeys(FACTS, 0.5), STRONG: 0.9}, online=online)
    seen = _recording(core)
    assert list(core.reply(text))
    assert seen == [False] and online.calls and local.calls
    system = _system(local)
    for fact in FACTS:
        assert fact not in system, fact
    assert STRONG in system and PERSONAL_SITUATION not in system


def test_the_personal_situation_needs_a_fact():
    assert personal_request(FILM)
    core, _, _, local = _core(dict.fromkeys(FACTS, 0.40))  # below the floor: nothing fills
    assert list(core.reply(FILM))
    system = _system(local)
    assert PERSONAL_SITUATION not in system and not any(f in system for f in FACTS)
    core, _, _, local = _core({})  # an empty memory
    assert list(core.reply(FILM))
    assert PERSONAL_SITUATION not in _system(local)


def test_a_fact_alone_never_adds_the_situation():
    """A plain reply that isn't a personal request keeps its facts and gets no situation line."""
    text = "i'm bored"
    assert not personal_request(text)
    core, _, _, local = _core({FACTS[0]: 0.9})  # strong: in the block on its own
    assert list(core.reply(text))
    system = _system(local)
    assert FACTS[0] in system and PERSONAL_SITUATION not in system


def test_only_a_plain_local_reply_fills():
    assert personal_request(FILM)
    core, _, _, local = _core({**dict.fromkeys(FACTS, 0.5), STRONG: 0.9})
    seen = _recording(core)
    list(core.reply(FILM))
    assert seen == [True]
    seen.clear()
    list(core.speak_first(Action("greet", True, None, "came back")))
    assert seen == [False]
    seen.clear()
    core._bid(Action("bid", False, None, FILM))
    assert seen == [False]
    seen.clear()
    list(core._speak("reply", user_text=FILM, query=FILM, situation=None, spoken=False, shape="tool"))
    assert seen == [False]
    assert STRONG in _system(local, -1) and PERSONAL_SITUATION not in _system(local, -1)
    seen.clear()
    list(core._report([AgentEvent("finished", "agent-0", "tests-1", "all 4 tests passed")]))
    assert seen == [False]
    seen.clear()
    list(core.reply("remember that i love pickles"))  # a remember-that turn is small: it just reacts
    assert seen == [False]


def test_a_game_turn_never_fills():
    assert personal_request(FILM)
    core, _, _, local = _core(dict.fromkeys(FACTS, 0.5), local_replies=("Okay, got one~",))
    seen = _recording(core)
    list(core.reply("let's play 20 questions"))
    assert core.game is not None and seen == [False]
    seen.clear()
    list(core.reply("is it alive?"))
    assert seen == [False] and PERSONAL_SITUATION not in _system(local, -1)


def test_a_screen_question_never_fills():
    from tests.core.test_eyes_engine import eyes_core

    assert personal_request(SCREEN) and is_screen_question(SCREEN)
    vision = FakeModel("vision", ["Looks like a film list."])
    core, *_ = eyes_core(vision=vision)
    sims = {**dict.fromkeys(FACTS, 0.5), STRONG: 0.9}
    core.memory.embedder = TableEmbedder(sims)
    for fact in sims:
        core.memory.add_fact(fact, source="user_said")
    seen = _recording(core)
    assert list(core.reply(SCREEN))
    assert vision.calls and seen == [False]
    system = vision.calls[-1][0]["content"]
    assert STRONG in system and PERSONAL_SITUATION not in system


def test_a_screen_question_she_cant_look_for_never_fills():
    """No window in front of her: the turn stays local with the can't-see reason, and still gets no fill and no line
    (the Task 7 ruling: any screen question, whether or not she can look)."""
    assert personal_request(SCREEN) and is_screen_question(SCREEN)
    core, _, _, local = _core({**dict.fromkeys(FACTS, 0.5), STRONG: 0.9})
    seen = _recording(core)
    assert list(core.reply(SCREEN))
    assert seen == [False]
    system = _system(local, -1)
    assert "you can't see right now" in system and STRONG in system
    assert PERSONAL_SITUATION not in system and not any(f in system for f in FACTS)


def test_the_line_follows_the_situation_a_turn_already_has():
    """Chat mid-game: the game's note stays, and the line comes after it."""
    assert personal_request(FILM)
    core, _, _, local = _core(dict.fromkeys(FACTS, 0.5), local_replies=("Okay, got one~",))
    list(core.reply("let's play 20 questions"))
    assert core.game is not None
    seen = _recording(core)
    assert list(core.reply(FILM))
    assert seen == [True] and core.game is not None
    system = _system(local, -1)
    note = system.find("You're still in the middle of")
    assert 0 <= note < system.find(PERSONAL_SITUATION)


def test_one_strong_fact_without_a_fill_still_gets_the_line():
    """§4.4: the line needs a fact in the block, not a filled one."""
    assert personal_request(FILM)
    core, _, _, local = _core({STRONG: 0.9})
    r = core.memory.retrieve(FILM, personal=True)
    assert [f.text for f in r.facts] == [STRONG] and r.filled == ()
    assert list(core.reply(FILM))
    system = _system(local)
    assert STRONG in system and PERSONAL_SITUATION in system


def test_the_added_text_fits():
    """A12: the fill's three longest bank facts plus the situation line add at most 600 characters, and the line is
    at most 160 characters (its "≤ 40 tokens")."""
    from evaluation.persona_eval import load_probes

    texts = {t for c in load_probes()["conversations"] for t in c["facts"]}
    lines = sorted((len(_fact_line(Fact(0, t, None, "user_said", 0.9, 0.0, None, "active", None))) for t in texts),
                   reverse=True)
    assert len(personal_situation()) <= 160
    assert sum(lines[:3]) + len(personal_situation()) + 1 <= 600


def test_personal_situation_reads_the_constant_at_call_time(monkeypatch):
    from riya.core import prompt

    assert personal_situation() == PERSONAL_SITUATION
    monkeypatch.setattr(prompt, "PERSONAL_SITUATION", "changed")
    assert personal_situation() == "changed"
