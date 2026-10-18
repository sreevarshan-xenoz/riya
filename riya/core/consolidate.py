"""Background memory consolidation: turn quiet-time transcripts into facts, an episode and follow-up threads."""
import json
import re
from dataclasses import dataclass
from typing import Callable

from riya.core.memory import Memory
from riya.core.models import ChatModel, ModelUnavailable, complete

CONSOLIDATE_PROMPT = (
    "You read a transcript between the user and Riya, their companion, and extract what is worth remembering. "
    "Return ONLY a JSON object, with no other text, with these keys:\n"
    '"facts": a list of {"text", "key", "confidence", "source"}. Write each text in the third person ("The user ..."). '
    'key is a short dotted label such as "user.exam" or null. confidence is 0 to 1. source is "user_said" when the '
    'user stated it plainly, otherwise "inferred".\n'
    '"episode_summary": at most 60 words on what happened in this conversation.\n'
    '"user_mood": one or two words for how the user seemed, or null.\n'
    '"threads": a list of {"text", "due_in_hours"} for things Riya should follow up on later, such as asking how '
    "an event went.\n"
    "Use empty lists and an empty summary when there is nothing worth keeping.")
REMEMBER_RE = re.compile(r"^\s*(?:riya[,!\s]+)?(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*[.!]*$", re.IGNORECASE)


@dataclass
class Proposals:
    facts: list[dict]
    episode_summary: str
    user_mood: str | None
    threads: list[dict]

    @classmethod
    def empty(cls) -> "Proposals":
        return cls([], "", None, [])


def _parse(text: str) -> Proposals | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    facts, threads = data.get("facts"), data.get("threads")
    summary, mood = data.get("episode_summary"), data.get("user_mood")
    return Proposals([f for f in facts if isinstance(f, dict)] if isinstance(facts, list) else [],
                     summary.strip() if isinstance(summary, str) else "",
                     mood.strip() or None if isinstance(mood, str) else None,
                     [t for t in threads if isinstance(t, dict)] if isinstance(threads, list) else [])


def parse_proposals(text: str) -> Proposals:
    return _parse(text) or Proposals.empty()


def remember_intent(text: str) -> str | None:
    if text.rstrip().endswith("?"):
        return None
    m = REMEMBER_RE.match(text)
    if not m or m.group(1).lower() == "that":
        return None
    return m.group(1)


def _number(value, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


class Consolidator:
    def __init__(self, memory: Memory, model_getter: Callable[[], ChatModel], clock, quiet_s: float = 600):
        self.memory, self.model_getter, self.clock, self.quiet_s = memory, model_getter, clock, quiet_s

    def due(self, last_turn_at: float) -> bool:
        return bool(self.memory.unconsolidated()) and self.clock.now() - last_turn_at >= self.quiet_s

    def run(self) -> Proposals:
        rows = self.memory.unconsolidated()
        if not rows:
            return Proposals.empty()
        messages = [{"role": "system", "content": CONSOLIDATE_PROMPT},
                    {"role": "user", "content": "\n".join(f"{r['role']}: {r['text']}" for r in rows)}]
        try:
            proposals = _parse(complete(self.model_getter(), messages, max_tokens=600))
        except ModelUnavailable:
            return Proposals.empty()
        if proposals is None:  # unusable reply: leave the transcripts so a later run retries
            return Proposals.empty()
        mem = self.memory
        episode_id = None
        if proposals.episode_summary:
            episode_id = mem.add_episode(rows[0]["at"], rows[-1]["at"], proposals.episode_summary,
                                         user_mood=proposals.user_mood).id
        for f in proposals.facts:
            text = f.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = f.get("key")
            source = f.get("source") if f.get("source") in ("user_said", "inferred") else "inferred"
            mem.add_fact(text.strip(), source, confidence=min(max(_number(f.get("confidence"), 0.7), 0.0), 1.0),
                         key=key if isinstance(key, str) and key else None, episode_id=episode_id,
                         check_forgotten=True)
        now = self.clock.now()
        for t in proposals.threads:
            text = t.get("text")
            if isinstance(text, str) and text.strip():
                mem.add_thread(text.strip(), now + max(_number(t.get("due_in_hours"), 24.0), 0.0) * 3600, episode_id)
        mem.mark_consolidated([r["id"] for r in rows])
        return proposals
