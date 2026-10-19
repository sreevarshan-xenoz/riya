"""Explicit feedback in the user's own turn: dial requests ("stop teasing me") and style notes ("I like it when you ...").
Pure and deterministic. Misfires matter more than misses: requests, questions about her and narration never count."""
import re
from dataclasses import dataclass
from typing import Literal

from riya.core.consolidate import third_person
from riya.core.learn.dials import DELTA

MAX_WORDS = 10
NOTE_MAX = 120
MAX_TEXT = 500  # only this many characters of a turn are ever read (keeps a 10k-char turn fast)


@dataclass(frozen=True)
class Feedback:
    dial: str | None
    delta: float
    note: str | None
    polarity: Literal["like", "dislike", "rule"] | None = None


_SUBJ = r"(?:that's|that was|that is|this is|it's|you're|you are|your (?:answers|replies) are) (?:way |a bit |kinda )?too"
_YOU = r"(?:you're|you are) (?:way |a bit |kinda )?"
# (dial, sign, pattern); tried in this order, the first match decides
_DIAL_PATTERNS = [(d, sg, re.compile(rf"(?<![\w']){p}(?![\w'])")) for d, sg, p in [
    ("teasing", -1, r"(?:(?:stop|quit) teasing(?: me)?|less teasing|tease (?:me )?less|(?:no more|enough) teasing)"),
    ("teasing", 1, r"(?:tease me more|you can tease(?: me)?(?: more)?|more teasing)"),
    ("verbosity", -1, rf"(?:(?:shorter|briefer) (?:answers|replies|responses)|talk less|keep it short(?:er)?|{_SUBJ} long)"),
    ("verbosity", 1, rf"(?:longer (?:answers|replies|responses)|talk more|{_SUBJ} short)"),
    ("emoji", -1, r"(?:(?:no|fewer|less) (?:more )?emojis?|(?:stop using|don'?t use) (?:any |so many )?emojis?)"),
    ("emoji", 1, r"(?:(?:use )?more emojis?)"),
    ("sweetness", 1, r"(?:be (?:sweeter|cuter|more (?:sweet|cute)))"),
    ("sweetness", -1, r"(?:less cutesy|talk normally|(?:stop|quit) being (?:so )?cutesy)"),
    ("clinginess", -1, rf"(?:{_YOU}too clingy|(?:stop|quit) being (?:so )?clingy|less clingy)"),
    ("clinginess", 1, r"(?:i like (?:that|it when) you miss me)"),
]]
_LEAD = re.compile(r"(?:(?:hey|ok|okay|so|and|also|riya|please|pls|just|seriously|honestly)[,!\s]+)*")
_NOTE_LEAD = re.compile(r"(?:(?:hey|ok|okay|riya)[,!\s]+)*", re.I)
_QUESTION = re.compile(r"(?:do(?! not\b)|does|did|are|is|was|were|why|what|how|when|where|who|which)\b")
# after a dial phrase that starts the turn: end, punctuation, a courtesy word or a clause opener
_FOLLOW_START = re.compile(r"\s*(?:$|[,.!;:~…)]|(?:please|thanks|ok|okay|riya|and|but|because|so)\b)")
_COURTESY = re.compile(r"(?:[\s,.!;:~…)]|\b(?:please|pls|riya|ok|okay|from now on|thanks|thank you|now|then|a bit"
                       r"|a little)\b)*")
_NEG = {"not", "never", "no", "dont"}
_NARRATION = {"said", "says", "told", "tells", "asked", "asks", "saying", "wrote", "writes"}
_OTHER = {"i", "i'm", "i'll", "i'd", "i've", "we", "he", "she", "they", "my", "his", "her", "their", "our", "someone",
          "people"}
_QUOTES = "\"“”‘"
_NOTES = [(re.compile(p, re.I), pol, kind) for p, pol, kind in [
    (r"i (?:really )?like it when you (.+)", "like", None),
    (r"i (?:really )?(?:don't|do not) like it when you (.+)", "dislike", None),
    (r"please (?:don't|do not) (.+)", "rule", "not"),
    (r"please always (.+)", "rule", "always"),
]]
_YOU_TO_ME = re.compile(r"(?<![\w'])(?:yourself|you're|your|you)(?![\w'])", re.I)
_YOU_MAP = {"you": "me", "your": "my", "yourself": "myself", "you're": "I'm"}


def _you_to_me(text: str) -> str:
    return _YOU_TO_ME.sub(lambda m: _YOU_MAP[m.group().lower()], text)


def _ok_before(before: str) -> bool:
    """No negation near it, no narration, no other subject, no quote before the phrase."""
    toks = re.findall(r"[\w']+", before)
    if any(t in _NEG or t.endswith("n't") for t in toks[-3:]):
        return False
    return not (any(c in before for c in _QUOTES) or any(t in _NARRATION or t in _OTHER for t in toks))


def _dial(s: str) -> Feedback | None:
    lead = _LEAD.match(s).end()
    for dial, sign, pat in _DIAL_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        after = s[m.end():]
        if not _ok_before(s[:m.start()]):
            return None
        if m.start() == lead and _FOLLOW_START.match(after) or len(s.split()) <= MAX_WORDS and _COURTESY.fullmatch(after):
            return Feedback(dial, sign * DELTA, None)
        return None
    return None


def _note(norm: str) -> Feedback | None:
    body = norm[_NOTE_LEAD.match(norm).end():]
    for pat, polarity, kind in _NOTES:
        m = pat.match(body)
        if not m:
            continue
        clause = re.split(r"[.!?]", m.group(1), maxsplit=1)[0].strip()
        if len(clause.split()) < 2:
            return None
        c = _you_to_me(third_person(clause))
        if polarity == "like":
            note = f"They like it when I {c}"
        elif polarity == "dislike":
            note = f"They don't like it when I {c}"
        else:
            note = f"They asked me not to {c}" if kind == "not" else f"They asked me to always {c}"
        return Feedback(None, 0.0, note[:NOTE_MAX - 1].rstrip(" ,;:-") + ".", polarity)
    return None


def detect_feedback(text: str) -> Feedback | None:
    """Explicit feedback in the user's own turn, or None. It counts only when it is the turn's main intent."""
    norm = re.sub(r"\s+", " ", text[:MAX_TEXT].replace("’", "'")).strip()
    s = norm.lower()
    if not s or s.endswith("?") or _QUESTION.match(s) or _QUESTION.match(s[_LEAD.match(s).end():]):
        return None
    return _dial(s) or _note(norm)


ACKS: dict[tuple[str, int], str] = {
    ("teasing", -1): "Okay~ less teasing, got it.", ("teasing", 1): "Ooh, more teasing? You asked for it~",
    ("verbosity", -1): "Got it, I'll keep it short.", ("verbosity", 1): "Okay, I'll tell you a bit more~",
    ("emoji", -1): "Okay, fewer emojis.", ("emoji", 1): "Yay, more emojis then~",
    ("sweetness", -1): "Okay, I'll tone down the cute.", ("sweetness", 1): "Hehe, sweeter it is~",
    ("clinginess", -1): "Okay… I'll give you a little more space.",
    ("clinginess", 1): "Aww, then I'll keep missing you~",
}
NOTE_ACKS: dict[str, str] = {"like": "Hehe, noted~ I'll remember you like that.",
                             "dislike": "Okay, I won't do that anymore.", "rule": "Okay, got it~ I'll remember."}


def ack_line(fb: Feedback) -> str:
    """The templated acknowledgement; she never argues."""
    return ACKS[(fb.dial, 1 if fb.delta > 0 else -1)] if fb.dial else NOTE_ACKS[fb.polarity]
