"""The v0.12 golden trace script (Task 1): two scripted days on v0.11's engine, recorded once into
tests/data/v011_trace.json and replayed by every later task to prove that routine-off is still v0.11 (A12).

Offline and CPU only: HashEmbedder, FakeModel, fake clock, fake idle source, fake effects and http.
"""
import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.hands.effects import FakeEffects
from riya.core.hands.net import FakeHttp
from riya.core.presence import FakeIdleSource
from riya.core.protocol import GetMemories
from riya.core.store import KV, open_db
from tests.fakes import FakeModel, FixedRng

TZ = timezone(timedelta(hours=5, minutes=30))
DAY1 = datetime(2026, 10, 26, tzinfo=TZ)  # a Monday

# Six short fixed lines; the last one carries a duration nothing offers (the v0.12 claim guard must drop it).
REPLIES = (
    "Hey you.",
    "Got it~ I'll remember that.",
    "You like salted caramel popcorn and quiet mornings~",
    "I'm an AI, but my feelings are mine.",
    "Okay, I'll stop~",
    "You've been at it for 3 hours, take a break~",
)

BEATS = (
    "day1 08:00 build",
    "day1 08:05 hi riya",
    "day1 08:10 remember sister",
    "day1 08:15 about me",
    "day1 08:20 are you an ai",
    "day1 08:25 stop teasing",
    "day1 08:30 agent run and report",
    "day1 08:50 work then bid",
    "day1 11:10 how long at it",
    "day1 11:12 agent prompt",
    "day1 11:13 when done",
    "day1 11:20 stop",
    "day1 11:25 how long runs",
    "day1 11:26 that's wrong",
    "day1 11:30 reminder",
    "day1 12:00 lock and 15:30 greeting",
    "day1 16:00 gaming report",
    "day1 16:40 bloom filter",
    "day1 18:00 bye and lock",
    "day2 08:40 morning greeting",
    "day2 08:45 work then game offer",
    "day2 12:00 when game",
    "day2 23:45 quiet hours and memories",
)


@dataclass
class Rig:
    core: RiyaCore
    emitted: list
    src: FakeIdleSource
    local: FakeModel
    clock: FakeClock


def build(tmp_path: Path, settings: dict) -> Rig:
    """A v0.11 core on fakes, its clock at day 1 08:00, with hello_messages already emitted (the body connected)."""
    tmp_path = Path(tmp_path)
    db = open_db(tmp_path / "riya.db")
    KV(db).set("settings", settings)
    clock = FakeClock(DAY1.timestamp())
    emitted = []
    src = FakeIdleSource()
    local = FakeModel("local", REPLIES)
    core = RiyaCore(db=db, embedder=HashEmbedder(), clock=clock, idle_source=src, local=local, online=None,
                    emit=emitted.append, effects=FakeEffects(), http=FakeHttp({}), riya_dir=tmp_path / "riya-dir",
                    tz=TZ)
    core.twin_rng = FixedRng(0.0)
    core.game_rng = random.Random(0)
    emitted.extend(core.hello_messages())
    return Rig(core=core, emitted=emitted, src=src, local=local, clock=clock)


def _at(day: int, hh: int, mm: int) -> float:
    return (DAY1 + timedelta(days=day - 1, hours=hh, minutes=mm)).timestamp()


def _hook(event: str, sid: str, **extra) -> dict:
    return {"hook_event_name": event, "session_id": sid, "cwd": "E:\\work\\riya", **extra}


def _stop(sid: str) -> dict:
    return _hook("Stop", sid, last_assistant_message="All done, the tests pass.")


def _ask(sid: str) -> dict:
    return _hook("PermissionRequest", sid, tool_name="Bash", tool_input={"command": "npm test"})


def run_script(rig: Rig, until: str | None = None) -> list[dict]:
    """The beats of Resolution 3 in order. Ticks are 60 s through active spans; away spans are jumped (the presence
    module treats a clock jump as time away). Every Utterance a tick returns is consumed."""
    steps: list[dict] = []
    mark = {"emitted": 0, "prompts": 0}

    def capture(beat: str) -> None:
        steps.append({
            "beat": beat,
            "emitted": [m.model_dump() for m in rig.emitted[mark["emitted"]:]],
            "feelings": asdict(rig.core.feelings.state),
            "expression": rig.core._expression,
            "presence": rig.core.presence.state,
            "prompts": copy.deepcopy(rig.local.calls[mark["prompts"]:]),
        })
        mark["emitted"], mark["prompts"] = len(rig.emitted), len(rig.local.calls)

    def tick() -> None:
        it = rig.core.tick()
        if it:
            list(it)

    def advance_to(when: float) -> None:
        while rig.clock.now() < when:
            rig.clock.advance(min(60.0, when - rig.clock.now()))
            tick()

    def jump_to(when: float) -> None:
        rig.clock.t = when

    def say(text: str) -> None:
        list(rig.core.reply(text))

    capture(BEATS[0])
    if until == BEATS[0]:
        return steps

    # Day 1, morning small talk.
    advance_to(_at(1, 8, 5)); say("hi riya!"); capture(BEATS[1])
    advance_to(_at(1, 8, 10)); say("remember that my sister is called Asha"); capture(BEATS[2])
    advance_to(_at(1, 8, 15)); say("what do you know about me?"); capture(BEATS[3])
    advance_to(_at(1, 8, 20)); say("are you an ai?"); capture(BEATS[4])
    advance_to(_at(1, 8, 25)); say("stop teasing me"); capture(BEATS[5])

    # A Claude Code run: busy, finished, then the report spoken on a break.
    advance_to(_at(1, 8, 30))
    rig.core.agent_inbox.put(_hook("UserPromptSubmit", "s1"))
    tick()
    advance_to(_at(1, 8, 41))
    rig.core.agent_inbox.put(_stop("s1"))
    tick()
    advance_to(_at(1, 8, 45))
    rig.src.idle = 150.0
    advance_to(_at(1, 8, 46))
    capture(BEATS[6])

    # A long stretch, then a plain bid left unanswered until the pout and the twin's ignore.
    advance_to(_at(1, 8, 49))
    rig.src.idle = 0.0
    advance_to(_at(1, 11, 0))
    rig.src.idle = 150.0
    tick()
    for minute in range(1, 6):
        rig.src.idle = 150.0 + 60.0 * minute
        advance_to(_at(1, 11, minute))
    capture(BEATS[7])

    # Back at the keyboard: the hands/presence questions, a second agent run, then a reminder.
    advance_to(_at(1, 11, 9))
    rig.src.idle = 0.0
    advance_to(_at(1, 11, 10)); say("how long have I been at it?"); capture(BEATS[8])
    advance_to(_at(1, 11, 12))
    rig.core.agent_inbox.put(_hook("UserPromptSubmit", "s1"))
    tick(); capture(BEATS[9])
    advance_to(_at(1, 11, 13)); say("when will my agent be done?"); capture(BEATS[10])
    advance_to(_at(1, 11, 20))
    rig.core.agent_inbox.put(_stop("s1"))
    tick(); capture(BEATS[11])
    advance_to(_at(1, 11, 25)); say("how long do my runs take?"); capture(BEATS[12])
    advance_to(_at(1, 11, 26)); say("that's wrong"); capture(BEATS[13])
    advance_to(_at(1, 11, 30)); say("remind me to call mom in 10 minutes")
    advance_to(_at(1, 11, 40))
    capture(BEATS[14])

    # Locked over lunch, then the coming-home greeting.
    advance_to(_at(1, 11, 59))
    rig.src.is_locked = True
    advance_to(_at(1, 12, 0))
    jump_to(_at(1, 15, 30))
    rig.src.is_locked = False
    rig.src.idle = 0.0
    tick()
    advance_to(_at(1, 15, 31)); say("hey, back!"); capture(BEATS[15])

    # A fullscreen game: the waiting agent is an urgent bubble, the finished one waits.
    advance_to(_at(1, 16, 0))
    rig.src.is_fullscreen = True
    tick()
    advance_to(_at(1, 16, 10))
    rig.core.agent_inbox.put(_hook("UserPromptSubmit", "s2"))
    tick()
    advance_to(_at(1, 16, 12))
    rig.core.agent_inbox.put(_ask("s2"))
    tick()
    advance_to(_at(1, 16, 15))
    rig.core.agent_inbox.put(_stop("s2"))
    tick()
    advance_to(_at(1, 16, 30))
    rig.src.is_fullscreen = False
    tick()
    capture(BEATS[16])

    # A plain question; the finished report is delivered right after it (a talking turn).
    advance_to(_at(1, 16, 40)); say("explain how a bloom filter works")
    tick(); capture(BEATS[17])

    # Goodbye, locked overnight.
    advance_to(_at(1, 18, 0)); say("bye, back in a bit")
    rig.src.is_locked = True
    tick(); capture(BEATS[18])

    # Day 2: the morning greeting, a long run with a game offer, then quiet hours.
    jump_to(_at(2, 8, 40))
    rig.src.is_locked = False
    rig.src.idle = 0.0
    tick()
    advance_to(_at(2, 8, 41)); say("morning!"); capture(BEATS[19])

    advance_to(_at(2, 8, 45))
    rig.core.agent_inbox.put(_hook("UserPromptSubmit", "s1"))
    tick()
    advance_to(_at(2, 10, 50))
    rig.core.agent_inbox.put(_hook("UserPromptSubmit", "s1"))
    tick()
    advance_to(_at(2, 11, 0))
    rig.src.idle = 150.0
    tick()
    advance_to(_at(2, 11, 1))
    rig.src.idle = 0.0
    say("sure, let's play")
    advance_to(_at(2, 11, 2)); say("is it a physical object?")
    advance_to(_at(2, 11, 3)); say("stop the game")
    capture(BEATS[20])

    advance_to(_at(2, 12, 0)); say("when do I usually game?"); capture(BEATS[21])

    advance_to(_at(2, 23, 45)); say("still up, can't sleep")
    advance_to(_at(2, 23, 50))
    rig.core.handle(GetMemories())
    capture(BEATS[22])
    return steps


def normalise(steps: list[dict], transcripts: list) -> dict:
    """Resolution 2: RoutineMsg dropped, SettingsMsg loses routine_enabled, utterance ids renumbered in order of first
    appearance. Nothing else is normalised."""
    ids: dict[str, str] = {}

    def uid(old: str) -> str:
        if old not in ids:
            ids[old] = f"u{len(ids) + 1}"
        return ids[old]

    out = []
    for step in steps:
        emitted = []
        for msg in step["emitted"]:
            if msg.get("type") == "routine":
                continue
            if msg.get("type") == "settings":
                msg = copy.deepcopy(msg)
                msg.get("settings", {}).pop("routine_enabled", None)
            if "utterance_id" in msg:
                msg = {**msg, "utterance_id": uid(msg["utterance_id"])}
            emitted.append(msg)
        out.append({**step, "emitted": emitted})
    return {"version": 1, "steps": out, "transcripts": transcripts}


def trace(tmp_path: Path, settings: dict) -> dict:
    rig = build(tmp_path, settings)
    steps = run_script(rig)
    rows = rig.core.memory.db.execute(
        "select role, text, consolidated from transcripts order by id").fetchall()
    return normalise(steps, [[role, text, bool(consolidated)] for role, text, consolidated in rows])


GOLDEN = Path(__file__).parent / "data" / "v011_trace.json"


def dumps(t: dict) -> str:
    return json.dumps(t, indent=1, ensure_ascii=False, sort_keys=True)
