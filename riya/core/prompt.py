"""Builds the chat messages for a turn: persona (plus the style examples), feelings, a delimited memory block, the
honesty line, the situation and a reply-rules line (v0.2 spec section 4.2)."""
from riya.core.memory import Fact, Retrieved, Thread
from riya.core.shaping import CAPS

NO_MEMORY_LINE = "(nothing stored about this yet)"
NO_ABOUT_ME_LINE = "Nothing yet. You don't know much about them so far: say so, and don't guess."
HONESTY_LINE = ("Only mention memories listed here. Never invent memories or facts about them. "
                "If what they ask about isn't listed, say you don't remember.")
SPEAK_FIRST_TURN = "(no message yet: you speak first)"
MEMORY_START = "### What you know about them (the only true things you know):"
MEMORY_END = "### End"
AFTER_ABOUT_ME = "Recent conversations (context only, not facts to list):"
REPLY_RULES = {"small": f"Reply in at most {CAPS['small']} sentences.",
               "explain": f"Explain clearly in at most {CAPS['explain']} sentences. No ~ in explanations.",
               "about_me": "List what's in the memory block, nothing else."}
# the last line of the system message is the one the small model reliably obeys, so the hard rules ride on it
HARD_RULES = ("You're an AI and say so if asked. Only the memory block is true about them; if something isn't there, "
              "say you don't know.")

_SOURCES = {"user_said": "user said", "inferred": "you inferred", "looked_up": "looked up"}


def _fact_line(f: Fact) -> str:
    tags = _SOURCES.get(f.source, f.source) + (", CONFLICTING" if f.status == "conflict" else "")
    return f"- {f.text} ({tags})"


def _when(ended_at: float, now: float) -> str:
    days = max(0, int((now - ended_at) // 86400))
    return "today" if days == 0 else "1 day ago" if days == 1 else f"{days} days ago"


def reply_rule(kind: str | None) -> str | None:
    """The `### Reply rules:` line for a reply's turn kind (small, explain, about_me): its length rule plus the hard
    rules. None for anything else."""
    return f"### Reply rules: {REPLY_RULES[kind]} {HARD_RULES}" if kind in REPLY_RULES else None


def _memory_block(r: Retrieved, now: float, about_me: bool = False) -> str:
    """Everything she knows, between MEMORY_START and MEMORY_END. An about-me turn keeps only the facts inside, so
    "list what's in the memory block" lists facts; its episodes and follow-ups come after the block."""
    facts = [_fact_line(f) for f in r.facts]
    rest = [f"- {_when(e.ended_at, now)}: {e.summary}" for e in r.episodes]
    rest += [f"- Follow up: {t.text}" for t in r.threads]
    inside = (facts or [NO_ABOUT_ME_LINE]) if about_me else (facts + rest or [NO_MEMORY_LINE])
    block = "\n".join([MEMORY_START, *inside, MEMORY_END])
    return block + "\n\n" + "\n".join([AFTER_ABOUT_ME, *rest]) if about_me and rest else block


def build_messages(*, persona: str, feelings_tag: str, retrieved: Retrieved, turns: list[dict],
                   user_text: str | None, situation: str | None, now: float, examples: str = "",
                   kind: str | None = None) -> list[dict]:
    """examples (the made-up style exchanges) sit right after the persona, never inside the memory block; kind is
    the reply's turn kind (shaping.turn_kind), which picks the reply-rules line and the facts-only about-me block."""
    parts = [persona, *([examples] if examples else []), feelings_tag,
             _memory_block(retrieved, now, about_me=kind == "about_me"), HONESTY_LINE]
    if situation:
        parts.append("[Situation] " + situation)
    if rule := reply_rule(kind):
        parts.append(rule)
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


def remember_situation(fact: str) -> str:
    """A remember-that turn: the fact is already saved, so she only has to confirm it."""
    return f'You just saved this to memory: "{fact}". Confirm in one short sentence that you\'ll remember it.'


def bid_situation(reason: str, thread: Thread | None) -> str:
    s = f"The user is on a break and you want their attention ({reason})."
    if thread:
        s += f" You could bring up: {thread.text}."
    return s + " Say one short, playful line. No lecture."
