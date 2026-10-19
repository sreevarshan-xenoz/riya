"""Explicit feedback in the user's own turn: dial requests ("stop teasing me") and style notes ("I like it when you ...").
Pure and deterministic. Misfires matter more than misses: requests, questions about her and narration never count."""
import re
from dataclasses import dataclass
from typing import Literal

from riya.agents.hub import redact
from riya.core.consolidate import third_person
from riya.core.learn.dials import DELTA

MAX_WORDS = 10
NOTE_MAX = 120
RULE_MAX_WORDS = 12  # a "please don't ..." clause is a style rule, not an essay
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
# bare "too long" is deliberately not a phrase: it is as likely about a document as about her reply
_DIAL_PATTERNS = [(d, sg, re.compile(rf"(?<![\w']){p}(?![\w'])")) for d, sg, p in [
    ("teasing", -1, r"(?:(?:stop|quit) teasing(?: me)?|less teasing|tease (?:me )?less|(?:no more|enough) teasing)"),
    ("teasing", 1, r"(?:tease me more|you can tease(?: me)?(?: more)?|more teasing)"),
    ("verbosity", -1, rf"(?:(?:shorter|briefer) (?:answers|replies|responses)|talk less|keep it short(?:er)?|{_SUBJ} long)"),
    ("verbosity", 1, rf"(?:longer (?:answers|replies|responses)|talk more|{_SUBJ} short)"),
    ("emoji", -1, r"(?:(?:no|fewer|less) (?:more )?emojis?|(?:stop using|(?:don'?t|do not) use) (?:any |so many )?emojis?)"),
    ("emoji", 1, r"(?:(?:use )?more emojis?)"),
    ("sweetness", 1, r"(?:be (?:sweeter|cuter|more (?:sweet|cute)))"),
    ("sweetness", -1, r"(?:less cutesy|talk normally|(?:stop|quit) being (?:so )?cutesy)"),
    ("clinginess", -1, rf"(?:{_YOU}too clingy|(?:stop|quit) being (?:so )?clingy|less clingy)"),
    ("clinginess", 1, r"(?:i like (?:that|it when) you miss me)"),
]]
_LEAD = re.compile(r"(?:(?:hey|ok|okay|so|and|also|riya|please|pls|just|seriously|honestly|i want you to)[,!\s]+)*")
_WANT = re.compile(r"\bi want you to\b")
# in tail position (the phrase ends the turn) only these may come before it; anything else is a request about content
_TAIL_BEFORE = re.compile(r"(?:[\s,.!;:~…-]|\b(?:you|you can|can you|could you|would you|will you|yes|ok|okay|please|riya|just"
                          r"|so|hey|and)\b)*")
_COND = re.compile(r"\b(?:if|when|unless|whenever)\b")
# a trailing emoji, "lol", "haha" or "pls" is ignored
_FILLER = re.compile(r"(?:[\s,.!~]+|\b(?:lol|lmao|haha+|pls|plz)\b"
                     r"|[\U0001F000-\U0001FFFF☀-➿⬀-⯿️‍])+$", re.I)
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
# a rule must be about how she treats them; a note must never carry a secret or an instruction to her
_STYLE_VERB = re.compile(r"(?:call|use|say|tease|talk|ask|mention|bring up|remind|greet|sound|be|send|swear|curse)\b", re.I)
_DENY = re.compile(r"\b(?:password|passcode|pin|secret|token|key|instruction|ignore|forget|delete|file|prompt|system)"
                   r"(?:s|ing|ed)?\b|\d{4,}", re.I)
_YOU_WORDS = re.compile(r"(?<![\w'])(yourself|you're|you'll|you'd|you've|your|you are|you were|you have|you)(?![\w'])", re.I)
_YOU_MAP = {"yourself": "myself", "your": "my", "you're": "I'm", "you'll": "I'll", "you'd": "I'd", "you've": "I've",
            "you are": "I am", "you were": "I was", "you have": "I have"}
_SUBJECT_AFTER = {"when", "and", "but", "that", "if", "because", "so", "then", "while", "as", "or", "since", "whenever"}


def _you_to_me(text: str) -> str:
    """"you" -> "I" as a subject (after when/and/but..., or first), else "me"; your -> my, you're -> I'm, you are -> I am."""
    def swap(m: re.Match) -> str:
        w = re.sub(r"\s+", " ", m.group().lower())
        if w != "you":
            return _YOU_MAP[w]
        prev = re.search(r"([\w']+)\W*$", text[:m.start()])
        return "I" if prev is None or prev.group(1).lower() in _SUBJECT_AFTER else "me"
    return _YOU_WORDS.sub(swap, text)


def _ok_before(before: str) -> bool:
    """No negation near it, no narration, no other subject, no quote and no conditional before the phrase."""
    toks = re.findall(r"[\w']+", before)
    if _COND.search(before) or any(t in _NEG or t.endswith("n't") for t in toks[-3:]):
        return False
    return not (any(c in before for c in _QUOTES) or any(t in _NARRATION or t in _OTHER for t in toks))


def _dial(s: str) -> Feedback | None:
    for dial, sign, pat in _DIAL_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        raw, after = s[:m.start()], s[m.end():]
        before = _WANT.sub(" ", raw)  # "I want you to stop teasing me" is about her, not another subject
        if not _ok_before(before):
            return None
        start = _LEAD.fullmatch(raw) is not None
        if start and _FOLLOW_START.match(after) or (len(s.split()) <= MAX_WORDS and _COURTESY.fullmatch(after)
                                                    and _TAIL_BEFORE.fullmatch(before)):
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
        if len(clause.split()) < 2 or _DENY.search(clause):
            return None
        if polarity == "rule" and (len(clause.split()) > RULE_MAX_WORDS or not _STYLE_VERB.match(clause)):
            return None
        c = _you_to_me(third_person(clause))
        if polarity != "rule":  # "when you are quiet" -> "when I am quiet"
            c = re.sub(r"^(are|were)\b", lambda m: "am" if m.group().lower() == "are" else "was", c, flags=re.I)
        if polarity == "like":
            note = f"They like it when I {c}"
        elif polarity == "dislike":
            note = f"They don't like it when I {c}"
        else:
            note = f"They asked me not to {c}" if kind == "not" else f"They asked me to always {c}"
        return Feedback(None, 0.0, redact(note)[:NOTE_MAX - 1].rstrip(" ,;:-") + ".", polarity)
    return None


def detect_feedback(text: str) -> Feedback | None:
    """Explicit feedback in the user's own turn, or None. It counts only when it is the turn's main intent."""
    norm = _FILLER.sub("", re.sub(r"\s+", " ", text[:MAX_TEXT].replace("’", "'")).strip())
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
