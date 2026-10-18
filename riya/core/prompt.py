"""Builds the chat messages for a turn: persona, feelings, retrieved memories and the situation."""
from riya.core.memory import Fact, Retrieved, Thread

NO_MEMORY_LINE = "You have no stored memory about this. If asked, say you don't remember instead of guessing."
HONESTY_LINE = ("Only mention memories listed here. Never invent memories or facts about them. "
                "If what they ask about isn't listed, say you don't remember.")
SPEAK_FIRST_TURN = "(no message yet: you speak first)"

_SOURCES = {"user_said": "user said", "inferred": "you inferred", "looked_up": "looked up"}


def _fact_line(f: Fact) -> str:
    tags = _SOURCES.get(f.source, f.source) + (", CONFLICTING" if f.status == "conflict" else "")
    return f"- {f.text} ({tags})"


def _when(ended_at: float, now: float) -> str:
    days = max(0, int((now - ended_at) // 86400))
    return "today" if days == 0 else "1 day ago" if days == 1 else f"{days} days ago"


def _memory_block(r: Retrieved, now: float) -> str:
    lines = [_fact_line(f) for f in r.facts]
    lines += [f"- {_when(e.ended_at, now)}: {e.summary}" for e in r.episodes]
    lines += [f"- Follow up: {t.text}" for t in r.threads]
    return "Memories:\n" + "\n".join(lines) if lines else NO_MEMORY_LINE


def build_messages(*, persona: str, feelings_tag: str, retrieved: Retrieved, turns: list[dict],
                   user_text: str | None, situation: str | None, now: float) -> list[dict]:
    parts = [persona, feelings_tag, _memory_block(retrieved, now), HONESTY_LINE]
    if situation:
        parts.append("[Situation] " + situation)
    last = {"role": "user", "content": SPEAK_FIRST_TURN if user_text is None else user_text}
    return [{"role": "system", "content": "\n\n".join(parts)}, *turns, last]


def _duration(away_s: float) -> str:
    if away_s < 2 * 86400:
        h = max(1, round(away_s / 3600))
        return f"{h} hour" + ("" if h == 1 else "s")
    d = int(away_s // 86400)
    return f"{d} day" + ("" if d == 1 else "s")


def greeting_situation(away_s: float, missing: float, thread: Thread | None) -> str:
    s = f"The user just came back after being away for {_duration(away_s)}."
    if missing >= 0.4:
        s += " You missed them and can say so."
    s += (f" Greet them warmly and ask about this: {thread.text}." if thread
          else " Greet them warmly and ask how their day went.")
    return s + " Keep it to one or two short sentences."


def bid_situation(reason: str, thread: Thread | None) -> str:
    s = f"The user is on a break and you want their attention ({reason})."
    if thread:
        s += f" You could bring up: {thread.text}."
    return s + " Say one short, playful line. No lecture."
