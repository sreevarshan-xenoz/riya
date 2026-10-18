import json
import threading
from datetime import datetime, timedelta

import pytest

from riya.core.clock import FakeClock
from riya.core.initiative import Action
from riya.core.models import ModelUnavailable
from riya.core.protocol import (ErrorMsg, ExpressionMsg, GetMemories, MemoriesMsg, Say, SayEnd, SetSettings,
                                StateMsg)
from riya.core.prompt import NO_MEMORY_LINE
from tests.fakes import make_core

CONSOLIDATION = json.dumps({
    "facts": [{"text": "The user has a calculus exam on Thursday", "key": "user.exam", "confidence": 0.9,
               "source": "user_said"}],
    "episode_summary": "The user is nervous about the calculus exam.", "user_mood": "nervous",
    "threads": [{"text": "Ask how the calculus exam went", "due_in_hours": 20}]})


def run_ticks(core, clock, seconds, step=1):
    spoken = []
    for _ in range(int(seconds / step)):
        clock.advance(step)
        it = core.tick()
        if it:
            spoken.append(list(it))
    return spoken


def test_reply_streams_sentences_and_records_transcript():
    core, emitted, _, _ = make_core(local_replies=["Hey you. I was waiting~"])
    assert list(core.reply("hi riya")) == ["Hey you.", "I was waiting~"]
    assert [t["role"] for t in core.memory.unconsolidated()] == ["user", "assistant"]
    assert [m.kind for m in emitted if isinstance(m, Say)] == ["reply", "reply"]


def test_remember_intent_saves_fact_immediately():
    core, *_ = make_core()
    list(core.reply("remember that I hate mornings"))
    assert any("I hate mornings" in f.text for f in core.memory.list_facts())


def coming_home(tmp_path, absence="no_tick"):
    """The spec's scenario: she learns about the exam, they leave for 2 days, she greets them about it. Returns the
    greeting, the prompt it was generated from, her missing-you at that moment, and the emitted messages."""
    greeting = "You're back!! I missed you. How did the calculus exam go?"
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=[
        "Aww, you'll do great.", CONSOLIDATION, greeting], tmp_path=tmp_path)
    list(core.reply("I have my calculus exam on Thursday and I'm nervous"))
    run_ticks(core, clock, 660, step=60)          # 11 quiet minutes -> consolidation
    assert core.memory.list_facts()[0].key == "user.exam"
    src.idle = 950
    run_ticks(core, clock, 60, step=60)           # goes away
    if absence == "ticking":
        run_ticks(core, clock, 2 * 86400, step=3600)
    elif absence == "restart":
        clock.advance(86400)
        core, emitted, src, local = make_core(clock, local_replies=[greeting], tmp_path=tmp_path)
        src.idle = 950
        clock.advance(86400)
    else:
        clock.advance(2 * 86400)
    missing_then = []
    stream = local.stream
    local.stream = lambda messages, *a, **kw: missing_then.append(core.feelings.state.missing) or stream(
        messages, *a, **kw)
    src.idle = 0
    spoken = run_ticks(core, clock, 1)
    return core, spoken, local.calls[-1][0]["content"], missing_then, emitted


@pytest.mark.parametrize("absence", ["no_tick", "ticking", "restart"])
def test_coming_home_scenario(tmp_path, absence):
    core, spoken, prompt, missing_then, emitted = coming_home(tmp_path, absence)
    assert spoken and "missed you" in " ".join(spoken[0])
    assert "2 days" in prompt and "calculus exam" in prompt
    assert "You missed them" in prompt and missing_then and missing_then[-1] >= 0.4
    assert any(isinstance(m, ExpressionMsg) and m.name == "missed_you" for m in emitted)
    assert any(isinstance(m, Say) and m.kind == "greet" and m.spoken for m in emitted)
    assert core.memory.due_threads() == []        # thread followed up, not raised twice


def test_no_greeting_after_short_absence():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock)
    src.idle = 950
    run_ticks(core, clock, 1)
    clock.advance(3600)
    src.idle = 0
    assert run_ticks(core, clock, 1) == []


def test_bid_is_a_silent_bubble_and_ignoring_it_makes_her_pout():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock, local_replies=["Hey... water break? you've been at it forever~"])
    core.handle(SetSettings(patch={"quiet_hours": None}))   # the default quiet hours depend on the local timezone
    hooked = []
    core.on_spoke_first = hooked.append
    run_ticks(core, clock, 2 * 3600 + 600, step=300)   # long focused stretch
    src.idle = 150
    assert run_ticks(core, clock, 150, step=150) == []  # nothing spoken aloud
    bids = [m for m in emitted if isinstance(m, Say) and m.kind == "bid"]
    assert len(bids) == 1 and bids[0].spoken is False and hooked == ["bid"]
    src.idle = 0
    run_ticks(core, clock, 301, step=1)
    assert core.feelings.expression() == "pouty"


def test_concurrent_replies_do_not_interleave():
    core, *_ = make_core(local_replies=["One. Two. Three."])
    threads = [threading.Thread(target=lambda: list(core.reply("hey"))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [t["role"] for t in core.memory.unconsolidated()] == ["user", "assistant", "user", "assistant"]


def test_feelings_survive_restart(tmp_path):
    clock = FakeClock()
    core, _, src, _ = make_core(clock, tmp_path=tmp_path)
    core.feelings.on_return(2 * 86400)
    run_ticks(core, clock, 31)
    again, *_ = make_core(clock, tmp_path=tmp_path)
    assert abs(again.feelings.state.missing - core.feelings.state.missing) < 0.05


def test_handle_memory_and_settings_messages():
    core, emitted, _, _ = make_core()
    f = core.memory.add_fact("The user likes biryani", source="user_said")
    core.handle(GetMemories())
    assert isinstance(emitted[-1], MemoriesMsg) and emitted[-1].facts[0]["id"] == f.id
    core.handle(SetSettings(patch={"mode": "fullscreen"}))
    assert isinstance(emitted[-1], ErrorMsg)


# --- controller rulings ---------------------------------------------------------------------------------------------

def away_then_back_into_a_game(core, clock, src, hours=3):
    src.idle = 950
    run_ticks(core, clock, 1)
    clock.advance(hours * 3600)
    src.idle, src.is_fullscreen = 0, True


def test_greeting_waits_until_the_game_ends():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock, local_replies=["Welcome back! I missed you."])
    away_then_back_into_a_game(core, clock, src, hours=30)
    assert run_ticks(core, clock, 600, step=60) == []
    assert not any(isinstance(m, Say) for m in emitted)
    src.is_fullscreen = False
    spoken = run_ticks(core, clock, 1)
    assert spoken == [["Welcome back!", "I missed you."]]
    assert any(isinstance(m, ExpressionMsg) and m.name == "missed_you" for m in emitted)


def test_greeting_is_dropped_after_half_an_hour_of_gaming():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock, local_replies=["Welcome back!"])
    away_then_back_into_a_game(core, clock, src)
    assert run_ticks(core, clock, 1860, step=60) == []
    src.is_fullscreen = False
    assert run_ticks(core, clock, 60) == []
    assert not any(isinstance(m, Say) for m in emitted)


def test_model_down_during_reply_reports_error_and_records_no_answer():
    core, emitted, _, local = make_core()
    local.fail = ModelUnavailable("llama-server is down")
    assert list(core.reply("hi riya")) == []
    assert any(isinstance(m, ErrorMsg) and m.message == "Riya's local model isn't responding" for m in emitted)
    assert sum(isinstance(m, SayEnd) for m in emitted) == 1
    assert [t["role"] for t in core.memory.unconsolidated()] == ["user"]
    local.fail = None
    assert list(core.reply("are you back?")) == ["Okay."]   # the lock was released


def test_model_dropping_mid_reply_keeps_what_was_said():
    core, emitted, _, local = make_core(local_replies=["First bit. Second bit. Third bit."])
    local.fail_after = 3
    assert list(core.reply("hi")) == ["First bit."]
    says = [m for m in emitted if isinstance(m, Say)]
    ends = [m for m in emitted if isinstance(m, SayEnd)]
    assert len(ends) == 1 and ends[0].utterance_id == says[0].utterance_id
    assert any(isinstance(m, ErrorMsg) for m in emitted)
    assert [t["role"] for t in core.memory.unconsolidated()] == ["user"]


def test_greeting_with_model_down_does_not_raise():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock)
    local.fail = RuntimeError("boom")
    src.idle = 950
    run_ticks(core, clock, 1)
    clock.advance(3 * 3600)
    src.idle = 0
    assert run_ticks(core, clock, 1) == [[]]
    assert any(isinstance(m, ErrorMsg) for m in emitted)


def test_online_route_shows_thinking_until_the_first_sentence():
    from tests.fakes import FakeModel
    online = FakeModel("online", ["Recursion is a function calling itself. Neat, right?"])
    core, emitted, _, _ = make_core(online=online)
    list(core.reply("explain recursion to me"))
    seq = [m.name if isinstance(m, ExpressionMsg) else "SAY" for m in emitted if isinstance(m, (ExpressionMsg, Say))]
    first_say = seq.index("SAY")
    assert seq[0] == "thinking" and first_say == 2 and seq[1] != "thinking"   # thinking ends at the first sentence
    assert core.feelings.expression() != "thinking"


def test_turns_keep_the_last_24_messages():
    core, *_ = make_core()
    for i in range(15):
        list(core.reply(f"message {i}"))
    assert len(core.turns) == 24 and core.turns[-2]["content"] == "message 14"


# --- fix round 1 ----------------------------------------------------------------------------------------------------

def test_no_bid_during_quiet_hours():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=["Water break?"])
    bid_at = datetime.fromtimestamp(clock.now() + 2 * 3600 + 600 + 150)
    window = f"{bid_at - timedelta(hours=1):%H:%M}-{bid_at + timedelta(hours=1):%H:%M}"
    core.handle(SetSettings(patch={"quiet_hours": window}))
    run_ticks(core, clock, 2 * 3600 + 600, step=300)
    src.idle = 150
    run_ticks(core, clock, 150, step=150)
    assert core.presence.state == "break"
    assert not any(isinstance(m, Say) for m in emitted) and local.calls == []


def test_greeting_that_fails_leaves_the_thread_open():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock)
    thread = core.memory.add_thread("Ask how the calculus exam went", due_at=clock.now())
    src.idle = 950
    run_ticks(core, clock, 1)
    clock.advance(3 * 3600)
    src.idle = 0
    local.fail = ModelUnavailable("llama-server is down")
    assert run_ticks(core, clock, 1) == [[]]
    assert [t.id for t in core.memory.due_threads()] == [thread.id]
    assert core.memory.due_threads()[0].last_raised_at is None


def test_failing_consolidation_does_not_starve_persistence():
    clock = FakeClock()
    core, emitted, _, _ = make_core(clock)
    list(core.reply("hi riya"))

    def boom():
        raise RuntimeError("bad proposals")
    core.consolidator.run = boom
    run_ticks(core, clock, 600, step=60)          # the last tick is the consolidation tick
    assert core.kv.get("feelings")["saved_at"] == clock.now()
    assert sum(isinstance(m, StateMsg) for m in emitted) == 10


def test_state_messages_keep_coming_while_talking():
    clock = FakeClock()
    core, emitted, _, _ = make_core(clock)
    for second in range(1, 121):
        clock.advance(1)
        if second % 10 == 0:
            list(core.reply("still here"))
        core.tick()
    assert sum(isinstance(m, StateMsg) for m in emitted) == 4


def test_barge_in_records_what_was_said_and_releases_the_lock():
    core, emitted, _, _ = make_core(local_replies=["One. Two. Three."])
    gen = core.reply("count for me")
    assert next(gen) == "One."
    gen.close()
    assert [(t["role"], t["text"]) for t in core.memory.unconsolidated()] == [
        ("user", "count for me"), ("assistant", "One.")]
    assert core.turns == [{"role": "user", "content": "count for me"}, {"role": "assistant", "content": "One."}]
    assert sum(isinstance(m, SayEnd) for m in emitted) == 1
    assert not core._lock.locked()


# --- final review ---------------------------------------------------------------------------------------------------

def long_stretch(core, clock):
    core.handle(SetSettings(patch={"quiet_hours": None}))
    run_ticks(core, clock, 2 * 3600 + 600, step=300)


def test_fullscreen_without_input_never_bids_or_opens_the_mic():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=["Water break?"])
    hooked = []
    core.on_spoke_first = hooked.append
    long_stretch(core, clock)
    src.is_fullscreen, src.idle = True, 150
    run_ticks(core, clock, 150, step=150)
    assert core.presence.state == "gaming"
    assert not any(isinstance(m, Say) for m in emitted) and hooked == [] and local.calls == []


def test_a_bid_never_opens_the_mic_while_gaming():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock, local_replies=["psst"])
    hooked = []
    core.on_spoke_first = hooked.append
    src.is_fullscreen = True
    run_ticks(core, clock, 1)
    core._bid(Action("bid", False, None, "test"))
    assert hooked == []


def test_hands_free_conversation_is_neither_a_break_nor_away():
    clock = FakeClock()
    core, emitted, src, _ = make_core(clock, local_replies=["Mm-hm."])
    long_stretch(core, clock)
    states = set()
    for _ in range(20):
        list(core.reply("and then what happened"))
        for _ in range(6):
            clock.advance(10); src.idle += 10
            core.tick()
            states.add(core.presence.state)
    assert states <= {"talking", "focused"}
    assert not any(isinstance(m, Say) and m.kind == "bid" for m in emitted)


def test_what_do_you_remember_about_me_lists_her_memories():
    core, _, _, local = make_core()
    core.memory.add_fact("The user likes biryani", source="user_said")
    core.memory.add_episode(core.clock.now() - 3600, core.clock.now() - 600, "Talked about weekend plans")
    list(core.reply("what do you remember about me?"))
    system = local.calls[-1][0]["content"]
    assert "The user likes biryani" in system and "Talked about weekend plans" in system
    assert NO_MEMORY_LINE not in system


def test_settings_writes_wait_for_a_memory_transaction_on_another_thread():
    core, *_ = make_core()
    held, wrote = threading.Event(), threading.Event()

    def hold():
        with core.memory.lock:
            held.set()
            wrote.wait(0.3)
    t = threading.Thread(target=hold)
    t.start()
    held.wait(1)
    writer = threading.Thread(target=lambda: (core.handle(SetSettings(patch={"volume": 0.5})), wrote.set()))
    writer.start()
    assert not wrote.wait(0.1)          # blocked behind the memory's lock
    t.join(); writer.join(1)
    assert wrote.is_set() and core.settings.volume == 0.5


# --- final review: follow-up threads --------------------------------------------------------------------------------

def follow_ups(messages):
    return [line for line in messages[0]["content"].splitlines() if line.startswith("- Follow up:")]


def test_a_reply_carries_at_most_two_relevant_threads():
    core, _, _, local = make_core()
    for text in ("Ask how the calculus exam went", "Ask whether the calculus exam results are out",
                 "Ask if the calculus exam hall was cold", "Ask about the trip to Goa"):
        core.memory.add_thread(text, due_at=core.clock.now() - 60)
    list(core.reply("ugh, the calculus exam"))
    assert len(follow_ups(local.calls[-1])) == 2
    list(core.reply("what should I cook tonight?"))
    assert follow_ups(local.calls[-1]) == []


def test_a_greeting_carries_only_its_own_thread():
    clock = FakeClock()
    core, _, src, local = make_core(clock, local_replies=["Welcome back! How was the exam?"])
    for text in ("Ask whether the calculus exam results are out", "Ask how the calculus exam went"):
        core.memory.add_thread(text, due_at=clock.now() - 60)
    src.idle = 950
    run_ticks(core, clock, 1)
    clock.advance(3 * 3600)
    src.idle = 0
    run_ticks(core, clock, 1)
    assert len(follow_ups(local.calls[-1])) == 1


def bid_about_a_thread(clock):
    core, emitted, src, local = make_core(clock, local_replies=["So... how did the exam go?", "Nice."])
    core.handle(SetSettings(patch={"quiet_hours": None}))
    core.memory.add_thread("Ask how the calculus exam went", due_at=clock.now())
    core.memory.add_thread("Ask whether the calculus exam results are out", due_at=clock.now())
    run_ticks(core, clock, 600, step=300)
    src.idle = 150
    run_ticks(core, clock, 150, step=150)
    assert any(isinstance(m, Say) and m.kind == "bid" for m in emitted)
    assert len(follow_ups(local.calls[-1])) == 1
    (raised,) = core.memory.db.execute("select id from threads where last_raised_at is not null").fetchall()
    return core, src, core.memory.due_threads()[[t.id for t in core.memory.due_threads()].index(raised[0])]


def test_answering_a_bid_follows_its_thread_up():
    clock = FakeClock()
    core, src, thread = bid_about_a_thread(clock)
    clock.advance(120); src.idle = 0
    list(core.reply("it went okay actually"))
    assert core.memory.db.execute("select status from threads where id=?", (thread.id,)).fetchone()[0] == "followed_up"


def test_a_late_answer_leaves_the_bid_thread_open():
    clock = FakeClock()
    core, src, thread = bid_about_a_thread(clock)
    run_ticks(core, clock, 301, step=1)
    list(core.reply("sorry, was away"))
    assert core.memory.db.execute("select status from threads where id=?", (thread.id,)).fetchone()[0] == "open"


def test_the_daily_purge_resolves_week_old_threads():
    clock = FakeClock()
    core, *_ = make_core(clock)
    stale = core.memory.add_thread("Ask about the old trip", due_at=clock.now() - 8 * 86400)
    run_ticks(core, clock, 1)
    assert core.memory.db.execute("select status from threads where id=?", (stale.id,)).fetchone()[0] == "resolved"


# --- final review: consolidation timing -----------------------------------------------------------------------------

def test_lid_closed_right_after_the_exam_talk_still_greets_about_the_exam():
    clock = FakeClock()
    core, emitted, src, local = make_core(clock, local_replies=[
        "Aww, you'll do great.", CONSOLIDATION, "You're back! How did the calculus exam go?"])
    list(core.reply("I have my calculus exam on Thursday and I'm nervous"))
    said_at = clock.now()
    run_ticks(core, clock, 300, step=60)          # 5 minutes, then the lid closes: no consolidation yet
    assert core.memory.unconsolidated()
    clock.advance(2 * 86400)
    spoken = run_ticks(core, clock, 1)
    (thread,) = core.memory.db.execute("select due_at, status from threads").fetchall()
    assert thread["due_at"] == said_at + 20 * 3600 and thread["status"] == "followed_up"
    assert spoken and "calculus exam" in local.calls[-1][0]["content"].split("[Situation]")[1]


def test_startup_takes_the_last_turn_time_from_the_transcript(tmp_path):
    clock = FakeClock()
    core, *_ = make_core(clock, tmp_path=tmp_path)
    list(core.reply("hi"))
    said_at = clock.now()
    clock.advance(300)
    again, *_ = make_core(clock, tmp_path=tmp_path)
    assert again.last_turn_at == said_at


def test_no_consolidation_while_gaming():
    clock = FakeClock()
    core, _, src, local = make_core(clock, local_replies=["Okay.", CONSOLIDATION])
    list(core.reply("hi"))
    src.is_fullscreen = True
    run_ticks(core, clock, 900, step=60)
    assert len(local.calls) == 1 and core.memory.unconsolidated()
    src.is_fullscreen = False
    run_ticks(core, clock, 60, step=60)
    assert len(local.calls) == 2 and not core.memory.unconsolidated()


# --- final review: episodes and threads are visible and deletable ---------------------------------------------------

def test_memories_list_episodes_and_open_threads_and_both_can_be_removed():
    from riya.core.protocol import EpisodeDelete, ThreadResolve
    core, emitted, _, _ = make_core()
    now = core.clock.now()
    ep = core.memory.add_episode(now - 600, now - 60, "Talked about the calculus exam")
    other = core.memory.add_episode(now - 60, now, "Played a word game")
    thread = core.memory.add_thread("Ask how the calculus exam went", due_at=now + 3600)
    done = core.memory.add_thread("Ask about the trip", due_at=now)
    core.memory.mark_thread(done.id, status="followed_up")
    core.handle(GetMemories())
    m = emitted[-1].model_dump()
    assert [e["id"] for e in m["episodes"]] == [other.id, ep.id]
    assert m["episodes"][1] == {"id": ep.id, "started_at": now - 600, "ended_at": now - 60,
                             "summary": "Talked about the calculus exam"}
    assert m["threads"] == [{"id": thread.id, "text": "Ask how the calculus exam went", "due_at": now + 3600}]
    core.handle(EpisodeDelete(id=ep.id))
    assert [e.id for e in emitted[-1].episodes] == [other.id]
    assert core.memory.db.execute("select count(*) from episodes where id=?", (ep.id,)).fetchone()[0] == 0
    core.handle(ThreadResolve(id=thread.id))
    assert emitted[-1].threads == []
    assert core.memory.db.execute("select status from threads where id=?", (thread.id,)).fetchone()[0] == "resolved"


# --- v0.2 reply shaping ---------------------------------------------------------------------------------------------

def spy(local):
    """Record each stream call's max_tokens and how many tokens were pulled from the model."""
    seen = {"max_tokens": [], "pulled": 0}
    stream = local.stream

    def wrapped(messages, max_tokens=300, temperature=0.8):
        seen["max_tokens"].append(max_tokens)
        for tok in stream(messages, max_tokens, temperature):
            seen["pulled"] += 1
            yield tok
    local.stream = wrapped
    return seen


RAMBLE = "One here. Two here. Three here. Four here. Five here. Six here. Seven here. Eight here. Nine here. Ten here. "


def test_small_talk_is_capped_at_three_sentences_and_the_model_stream_is_closed():
    core, emitted, _, local = make_core(local_replies=[RAMBLE])
    seen = spy(local)
    said = list(core.reply("hey riya"))
    assert said == ["One here.", "Two here.", "Three here."]
    assert seen["max_tokens"] == [110] and seen["pulled"] == 6      # 3 sentences of 2 tokens, nothing more
    assert [m.text for m in emitted if isinstance(m, Say)] == said
    assert core.memory.unconsolidated()[-1]["text"] == "One here. Two here. Three here."
    assert core.turns[-1] == {"role": "assistant", "content": "One here. Two here. Three here."}


def test_explain_and_about_me_turns_get_their_own_budget_and_cap():
    core, _, _, local = make_core(local_replies=[RAMBLE])
    seen = spy(local)
    assert len(list(core.reply("explain how a rainbow forms"))) == 8
    assert len(list(core.reply("what do you remember about me?"))) == 6
    assert seen["max_tokens"] == [380, 160]


def test_a_short_reply_is_recorded_as_streamed():
    core, *_ = make_core(local_replies=["Hey you.\n\nI was waiting~"])
    list(core.reply("hi riya"))
    assert core.memory.unconsolidated()[-1]["text"] == "Hey you.\n\nI was waiting~"


def test_barge_in_on_a_capped_reply_records_only_what_was_said():
    core, *_ = make_core(local_replies=[RAMBLE])
    it = core.reply("hey riya")
    assert next(it) == "One here."
    it.close()
    assert core.memory.unconsolidated()[-1]["text"] == "One here."


def test_only_the_first_emoji_survives_in_the_bubble_and_the_transcript():
    core, emitted, _, _ = make_core(local_replies=["Hi \U0001F600 there. Bye \U0001F44D now. Ok \U0001F525 sure."])
    said = list(core.reply("hey riya"))
    assert said == ["Hi \U0001F600 there.", "Bye now.", "Ok sure."]
    assert [m.text for m in emitted if isinstance(m, Say)] == said
    assert core.memory.unconsolidated()[-1]["text"] == "Hi \U0001F600 there. Bye now. Ok sure."


def test_an_emoji_only_sentence_is_dropped_and_does_not_count_toward_the_cap():
    core, *_ = make_core(local_replies=["A \U0001F600 one. \U0001F44D! B two. C three. D four."])
    assert list(core.reply("hey riya")) == ["A \U0001F600 one.", "B two.", "C three."]


def test_greeting_uses_its_own_budget_and_cap():
    core, emitted, src, local = make_core(local_replies=[RAMBLE])
    seen = spy(local)
    spoken = list(core.speak_first(Action("greet", True, None, "back")))
    assert spoken == ["One here.", "Two here.", "Three here."] and seen["max_tokens"] == [90]
    assert [m.kind for m in emitted if isinstance(m, Say)] == ["greet"] * 3


def test_bid_keeps_its_40_token_budget_and_one_emoji():
    core, emitted, _, local = make_core(local_replies=["Water \U0001F4A7 break? \U0001F600"])
    seen = spy(local)
    core._bid(Action("bid", False, None, "test"))
    assert seen["max_tokens"] == [40]
    assert [m.text for m in emitted if isinstance(m, Say)] == ["Water \U0001F4A7 break?"]


BLOCK = "```py\n# do it.\n    x = f(1)\n```"


def test_a_fenced_block_keeps_its_indentation_byte_for_byte():
    reply = f"Try this. {BLOCK} Done."
    core, emitted, *_ = make_core(local_replies=[reply])
    said = list(core.reply("hey riya"))
    assert said == ["Try this.", f"{BLOCK} Done."]
    assert [m.text for m in emitted if isinstance(m, Say)] == said
    assert core.memory.unconsolidated()[-1]["text"] == reply


def test_a_fenced_block_survives_a_cap_and_counts_as_one_sentence():
    core, *_ = make_core(local_replies=[f"One. {BLOCK} Two. Three. Four. Five."])
    said = list(core.reply("hey riya"))
    assert said == ["One.", f"{BLOCK} Two.", "Three."]
    recorded = core.memory.unconsolidated()[-1]["text"]
    assert recorded == " ".join(said) and BLOCK in recorded


def test_the_engine_lock_is_free_after_a_cap_close_and_after_a_barge_in():
    core, *_ = make_core(local_replies=[RAMBLE])

    def free() -> bool:
        got = core._lock.acquire(blocking=False)
        if got:
            core._lock.release()
        return got
    list(core.reply("hey riya"))                      # capped
    assert free()
    it = core.reply("hey riya again")
    next(it)
    it.close()                                        # barge-in
    assert free()
