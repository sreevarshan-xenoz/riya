"""Background memory consolidation: turn quiet-time transcripts into facts, an episode and follow-up threads."""
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
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
CHUNK_ROWS = 60
CHUNK_CHARS = 8000
CHUNK_GAP_S = 1800  # a longer silence starts a new conversation
MAX_FAILURES = 3
log = logging.getLogger("riya.consolidate")
REMEMBER_RE = re.compile(r"^\s*(?:riya[,!\s]+)?(?:(?:please|also)\s+)*remember(?:\s+that)?\s+(.+?)\s*[.!]*$", re.IGNORECASE)


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


_PERSON = {"my": "their", "mine": "theirs", "i": "they", "me": "them", "i'm": "they're", "i've": "they've",
           "i'll": "they'll", "i'd": "they'd", "myself": "themselves"}
# a standalone word: not touching a letter, digit, _ . / @ - or apostrophe ("I/O", "i.e.", "me@x.com", "my-app.me");
# a full stop right after is fine when it ends the sentence ("it's mine.")
_FIRST_PERSON = re.compile(r"(?<![\w'\u2019./@-])(?:i['\u2019](?:m|ve|ll|d)|myself|mine|my|me"
                           r"|i(?:\s+(?:am|was|have)\b)?)(?![\w'\u2019/@-]|\.\S)", re.I)
_VERB = {"am": "are", "was": "were", "have": "have"}  # "I am" -> "they are"
_QUOTED = re.compile("(\"[^\"]*\"|\u201c[^\u201d]*\u201d|(?<!\\w)'[^']*'(?!\\w)|(?<!\\w)\u2018[^\u2019]*\u2019(?!\\w)"
                     "|`[^`]*`)")  # their quoted words stay theirs: my friend said "I love you"
_NUMERAL_AFTER = re.compile(r"\b(?:war|chapter|part|type|phase|level|act|season|episode|henry|louis|elizabeth|george"
                            r"|charles|james|william)\s+$", re.I)  # World War I, Henry I
_PLACE_MINE = re.compile(r"\b(?:a|the|gold|coal|this|that)\s+$", re.I)  # a coal mine
_CODE = re.compile(r"`|\w\(|[{};=]")  # "for i in range(3)": a lowercase i there is a variable


def third_person(text: str) -> str:
    """The user's words about themselves, rewritten for her memory: "my favourite colour" -> "their favourite colour".
    "My" stays capitalised; "I" becomes a plain "they", as the text follows the "The user said:" prefix, and "I am",
    "I was", "I have" become "they are", "they were", "they have". Quoted words, roman numerals, a mine you dig, addresses, file names and code variables are left alone."""
    code = bool(_CODE.search(text))

    def rewrite(seg: str) -> str:
        def swap(m: re.Match) -> str:
            (raw, *verb), before = m.group().split(), seg[:m.start()]
            word = raw.replace("\u2019", "'").lower()
            if (raw == "I" and _NUMERAL_AFTER.search(before) or word == "mine" and _PLACE_MINE.search(before)
                    or raw == "i" and code):
                return m.group()
            out = _PERSON[word]
            out = out.capitalize() if raw[0].isupper() and not word.startswith("i") else out
            return " ".join([out, *(_VERB[v.lower()] for v in verb)])
        return _FIRST_PERSON.sub(swap, seg)
    return "".join(part if i % 2 else rewrite(part) for i, part in enumerate(_QUOTED.split(text)))


def remember_intent(text: str) -> str | None:
    if text.rstrip().endswith("?"):
        return None
    m = REMEMBER_RE.match(text)
    if not m or m.group(1).lower() == "that":
        return None
    return m.group(1)


def _number(value, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def first_chunk(rows: list[dict]) -> list[dict]:
    """The oldest conversation: at most CHUNK_ROWS rows or CHUNK_CHARS characters, ending at a gap over 30 min."""
    out, chars = [], 0
    for r in rows:
        if out and (r["at"] - out[-1]["at"] > CHUNK_GAP_S or len(out) >= CHUNK_ROWS
                    or chars + len(r["text"]) > CHUNK_CHARS):
            break
        out.append(r)
        chars += len(r["text"])
    return out


class Consolidator:
    def __init__(self, memory: Memory, model_getter: Callable[[], ChatModel], clock, quiet_s: float = 600):
        self.memory, self.model_getter, self.clock, self.quiet_s = memory, model_getter, clock, quiet_s
        self._failures: tuple[int, int] | None = None  # (first transcript id of the failing chunk, failures in a row)

    def due(self, last_turn_at: float) -> bool:
        return bool(self.memory.unconsolidated()) and self.clock.now() - last_turn_at >= self.quiet_s

    def run(self, model: ChatModel | None = None) -> Proposals:
        """Consolidates the oldest chunk of transcript. model overrides the configured one for this run."""
        rows = first_chunk(self.memory.unconsolidated())
        if not rows:
            return Proposals.empty()
        messages = [{"role": "system", "content": CONSOLIDATE_PROMPT},
                    {"role": "user", "content": "\n".join(f"{r['role']}: {r['text']}" for r in rows)}]
        try:
            proposals = _parse(complete(model or self.model_getter(), messages, max_tokens=600))
        except ModelUnavailable:  # an outage isn't the chunk's fault: leave it, retry on the normal cadence, uncounted
            return Proposals.empty()
        if proposals is None:  # an unusable reply: leave the transcripts so a later run retries; 3 in a row give up
            self._failed(rows)
            return Proposals.empty()
        self._failures = None
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
        said_at = rows[-1]["at"]  # follow-ups are dated from the conversation, not from this run
        for t in proposals.threads:
            text = t.get("text")
            if isinstance(text, str) and text.strip():
                mem.add_thread(text.strip(), said_at + max(_number(t.get("due_in_hours"), 24.0), 0.0) * 3600,
                               episode_id)
        mem.mark_consolidated([r["id"] for r in rows])
        return proposals

    def _failed(self, rows: list[dict]) -> None:
        first = rows[0]["id"]
        n = self._failures[1] + 1 if self._failures and self._failures[0] == first else 1
        self._failures = (first, n)
        if n < MAX_FAILURES:
            return
        day = datetime.fromtimestamp(rows[0]["at"]).strftime("%Y-%m-%d")
        log.warning("consolidation failed %d times on transcript %d..%d; keeping a minimal episode", n, first,
                    rows[-1]["id"])
        self.memory.add_episode(rows[0]["at"], rows[-1]["at"], f"Conversation on {day}: summary unavailable",
                                outcome="unconsolidated")
        self.memory.mark_consolidated([r["id"] for r in rows])
        self._failures = None

    def run_pending(self, last_turn_at: float, model: ChatModel | None = None, max_chunks: int = 10) -> None:
        """Consolidates every due chunk now, oldest first, stopping at the first run that makes no progress."""
        for _ in range(max_chunks):
            if not self.due(last_turn_at):
                return
            before = len(self.memory.unconsolidated())
            self.run(model)
            if len(self.memory.unconsolidated()) >= before:
                return
