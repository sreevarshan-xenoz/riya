import json

import pytest

from riya.core.clock import FakeClock
from riya.core.consolidate import Consolidator, Proposals, parse_proposals, remember_intent, third_person
from riya.core.embed import HashEmbedder
from riya.core.memory import Memory
from riya.core.models import ModelUnavailable
from riya.core.store import open_db
from tests.fakes import FakeModel

PROPOSAL = {"facts": [{"text": "The user has a calculus exam on Thursday", "key": "user.exam", "confidence": 0.9,
                       "source": "user_said"}],
            "episode_summary": "They talked about the user's calculus exam.", "user_mood": "nervous",
            "threads": [{"text": "Ask how the calculus exam went", "due_in_hours": 20}]}


def setup():
    clock = FakeClock()
    return clock, Memory(open_db(":memory:"), HashEmbedder(), clock)


def test_parse_proposals_tolerates_prose():
    p = parse_proposals("Sure! " + json.dumps(PROPOSAL) + " hope that helps")
    assert p.facts[0]["key"] == "user.exam" and p.threads[0]["due_in_hours"] == 20


def test_parse_proposals_bad_json_is_empty():
    assert parse_proposals("nope {") == Proposals.empty()


def test_due_after_ten_quiet_minutes():
    clock, mem = setup()
    c = Consolidator(mem, lambda: FakeModel(), clock)
    assert not c.due(clock.now())
    mem.append_transcript("user", "hi")
    t = clock.now()
    clock.advance(599)
    assert not c.due(t)
    clock.advance(1)
    assert c.due(t)


def test_run_applies_everything():
    clock, mem = setup()
    mem.append_transcript("user", "I have a calculus exam on Thursday, I'm nervous")
    mem.append_transcript("assistant", "You'll do great.")
    Consolidator(mem, lambda: FakeModel(replies=[json.dumps(PROPOSAL)]), clock).run()
    assert mem.recent_episodes(1)[0].summary.startswith("They talked")
    assert mem.list_facts()[0].source == "user_said"
    clock.advance(20 * 3600)
    assert mem.due_threads()[0].text == "Ask how the calculus exam went"
    assert mem.unconsolidated() == []


def test_run_does_not_resurrect_forgotten_facts():
    clock, mem = setup()
    f = mem.add_fact("The user has a calculus exam on Thursday", source="user_said", key="user.exam")
    mem.delete_fact(f.id)
    mem.append_transcript("user", "exam talk")
    Consolidator(mem, lambda: FakeModel(replies=[json.dumps(PROPOSAL)]), clock).run()
    assert mem.list_facts() == []


def test_run_sends_transcript_lines_as_role_text():
    clock, mem = setup()
    mem.append_transcript("user", "hello there")
    mem.append_transcript("assistant", "hi!")
    model = FakeModel(replies=[json.dumps(PROPOSAL)])
    Consolidator(mem, lambda: model, clock).run()
    system, user = model.calls[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"] == "user: hello there\nassistant: hi!"


def test_unknown_fact_source_becomes_inferred():
    clock, mem = setup()
    mem.append_transcript("user", "talk")
    odd = {**PROPOSAL, "facts": [{**PROPOSAL["facts"][0], "source": "looked_up"}]}
    Consolidator(mem, lambda: FakeModel(replies=[json.dumps(odd)]), clock).run()
    assert mem.list_facts()[0].source == "inferred"


def test_run_survives_unavailable_model_and_retries_later():
    clock, mem = setup()
    mem.append_transcript("user", "I have an exam")
    out = Consolidator(mem, lambda: FakeModel(fail=ModelUnavailable("down")), clock).run()
    assert out == Proposals.empty() and len(mem.unconsolidated()) == 1
    out = Consolidator(mem, lambda: FakeModel(replies=["not json at all"]), clock).run()
    assert out == Proposals.empty() and len(mem.unconsolidated()) == 1
    assert mem.recent_episodes(1) == []
    Consolidator(mem, lambda: FakeModel(replies=[json.dumps(PROPOSAL)]), clock).run()
    assert mem.unconsolidated() == [] and len(mem.recent_episodes(1)) == 1


def test_run_with_nothing_to_consolidate_does_not_call_the_model():
    clock, mem = setup()
    model = FakeModel()
    assert Consolidator(mem, lambda: model, clock).run() == Proposals.empty()
    assert model.calls == []


def test_remember_intent():
    assert remember_intent("Riya, remember that my exam is on Friday.") == "my exam is on Friday"
    assert remember_intent("remember I hate mornings") == "I hate mornings"
    assert remember_intent("also remember that i hate cilantro") == "i hate cilantro"
    assert remember_intent("Riya, please also remember I like tea") == "I like tea"
    assert remember_intent("do you remember me?") is None
    assert remember_intent("remember that I like tea?") is None


# --- final review ---------------------------------------------------------------------------------------------------

def test_threads_are_due_from_when_the_conversation_happened():
    clock, mem = setup()
    mem.append_transcript("user", "I have a calculus exam on Thursday")
    said_at = clock.now()
    clock.advance(2 * 86400)                       # consolidation only runs two days later
    Consolidator(mem, lambda: FakeModel(replies=[json.dumps(PROPOSAL)]), clock).run()
    assert mem.due_threads()[0].due_at == said_at + 20 * 3600


def run_chunks(mem, clock):
    model = FakeModel(replies=[json.dumps(PROPOSAL)])
    c = Consolidator(mem, lambda: model, clock)
    sizes = []
    while mem.unconsolidated():
        c.run()
        sizes.append(len(model.calls[-1][1]["content"].splitlines()))
    return sizes


def test_consolidates_at_most_60_rows_per_run_oldest_first():
    clock, mem = setup()
    for i in range(70):
        mem.append_transcript("user", f"line {i}")
        clock.advance(10)
    model = FakeModel(replies=[json.dumps(PROPOSAL)])
    Consolidator(mem, lambda: model, clock).run()
    sent = model.calls[0][1]["content"].splitlines()
    assert len(sent) == 60 and sent[0] == "user: line 0" and len(mem.unconsolidated()) == 10
    clock2, mem2 = setup()
    for i in range(70):
        mem2.append_transcript("user", f"line {i}")
    assert run_chunks(mem2, clock2) == [60, 10]


def test_consolidation_chunks_split_at_long_gaps_and_8000_characters():
    clock, mem = setup()
    mem.append_transcript("user", "morning chat")
    mem.append_transcript("assistant", "hi")
    clock.advance(31 * 60)
    mem.append_transcript("user", "evening chat")
    assert run_chunks(mem, clock) == [2, 1]
    clock, mem = setup()
    for _ in range(5):
        mem.append_transcript("user", "x" * 3000)
    assert run_chunks(mem, clock) == [2, 2, 1]


def test_a_chunk_that_fails_three_times_becomes_a_minimal_episode():
    from datetime import datetime
    clock, mem = setup()
    mem.append_transcript("user", "something the model chokes on")
    c = Consolidator(mem, lambda: FakeModel(replies=["not json"]), clock)
    c.run()
    c.run()
    assert mem.recent_episodes(5) == [] and len(mem.unconsolidated()) == 1
    c.run()
    (ep,) = mem.recent_episodes(5)
    day = datetime.fromtimestamp(clock.now()).strftime("%Y-%m-%d")
    assert ep.summary == f"Conversation on {day}: summary unavailable" and mem.unconsolidated() == []


def test_an_outage_never_counts_toward_the_failure_cap():
    clock, mem = setup()
    mem.append_transcript("user", "something to remember")
    c = Consolidator(mem, lambda: FakeModel(fail=ModelUnavailable("down")), clock)
    for _ in range(6):
        c.run()
    assert mem.recent_episodes(5) == [] and len(mem.unconsolidated()) == 1
    c.model_getter = lambda: FakeModel(replies=[json.dumps(PROPOSAL)])
    c.run()
    assert mem.unconsolidated() == [] and mem.recent_episodes(5)[0].summary == PROPOSAL["episode_summary"]


def test_an_outage_between_unusable_replies_neither_counts_nor_resets():
    clock, mem = setup()
    mem.append_transcript("user", "something the model chokes on")
    c = Consolidator(mem, lambda: FakeModel(replies=["not json"]), clock)
    c.run(); c.run()
    c.model_getter = lambda: FakeModel(fail=ModelUnavailable("down"))
    c.run(); c.run()
    assert mem.recent_episodes(5) == [] and len(mem.unconsolidated()) == 1
    c.model_getter = lambda: FakeModel(replies=["not json"])
    c.run()  # the third unusable reply
    assert mem.unconsolidated() == [] and mem.recent_episodes(5)[0].outcome == "unconsolidated"


def test_a_success_resets_the_failure_count():
    clock, mem = setup()
    mem.append_transcript("user", "first")
    c = Consolidator(mem, lambda: FakeModel(replies=["not json"]), clock)
    c.run(); c.run()
    c.model_getter = lambda: FakeModel(replies=[json.dumps(PROPOSAL)])
    c.run()
    mem.append_transcript("user", "second")
    c.model_getter = lambda: FakeModel(replies=["not json"])
    c.run()
    assert len(mem.unconsolidated()) == 1


@pytest.mark.parametrize("said, want", [
    ("my favourite colour is teal", "their favourite colour is teal"),
    ("I hate cilantro", "they hate cilantro"),
    ("I'm allergic to cats and the dog is mine", "they're allergic to cats and the dog is theirs"),
    ("I've got a cat, I'll call her Mimi, I'd like you to call me Sree",
     "they've got a cat, they'll call her Mimi, they'd like you to call them Sree"),
    ("MY exam is on Friday, I promised myself", "Their exam is on Friday, they promised themselves"),
    ("I\u2019m vegetarian", "they're vegetarian"),
    ("mymy and Mike and time stay", "mymy and Mike and time stay")])
def test_third_person_rewrites_first_person_words_only(said, want):
    assert third_person(said) == want


@pytest.mark.parametrize("said, want", [
    ("World War I and II", "World War I and II"),
    ("Chapter I is my favourite", "Chapter I is their favourite"),
    ("for i in range(3)", "for i in range(3)"),
    ("i.e. my laptop", "i.e. their laptop"),
    ("I/O is my job", "I/O is their job"),
    ("me@x.com or my-app.me", "me@x.com or my-app.me"),
    ("ME/CFS", "ME/CFS"),
    ("my_var and me2", "my_var and me2"),
    ("I work in a mine", "they work in a mine"),
    ("the gold mine is mine.", "the gold mine is theirs."),
    ('my friend said "I love you"', 'their friend said "I love you"'),
    ("my mum said \u201cI miss you\u201d and \u2018my bad\u2019",
     "their mum said \u201cI miss you\u201d and \u2018my bad\u2019"),
    ("my motto is 'I never quit'", "their motto is 'I never quit'"),
    ("my favourite colour is teal", "their favourite colour is teal"),
    ("I'm allergic to nuts", "they're allergic to nuts"),
    ("i hate mornings", "they hate mornings"),
    ("remind me that I have a test", "remind them that they have a test")])
def test_third_person_leaves_numerals_code_addresses_and_quotes_alone(said, want):
    assert third_person(said) == want
