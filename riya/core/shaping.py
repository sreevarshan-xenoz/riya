"""Reply shaping, pure: what kind of turn this is, its token budget and sentence cap, and the emoji policy.
Small local models ramble and love emoji; the length and emoji rules are enforced here, in code, not just asked for.
See docs/superpowers/specs/2026-10-18-riya-v0.2-sounds-like-her-design.md \u00A74.3."""
import re
from collections.abc import Iterator
from typing import Literal

BUDGETS = {"small": 110, "about_me": 160, "explain": 380, "greeting": 90, "bid": 40}  # max_tokens per kind
CAPS = {"small": 3, "about_me": 6, "explain": 8, "greeting": 3}  # most sentences per kind

_VS = "[\uFE0F\U0001F3FB-\U0001F3FF]*"  # variation selector and skin tones ride along with their emoji
_PICTO = ("[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u27BF\u2B50\u2B55\u2B1B\u2B1C\u231A\u231B"
          "\u23E9-\u23FA\u2934\u2935\u2B05-\u2B07\u203C\u2049]")
_TEXT_GLYPHS = "\u2713\u2714\u266A\u263A"  # check marks, a note, a smiley: pictographs, emoji only with a VS16
EMOJI = re.compile(rf"[\U0001F1E6-\U0001F1FF]{{2}}|[0-9#*]\uFE0F?\u20E3"
                   rf"|(?![{_TEXT_GLYPHS}](?!\uFE0F)){_PICTO}{_VS}(?:\u200D{_PICTO}{_VS})*")
_EMOJI_WITH_GAP = re.compile(rf"([ \t]*)({EMOJI.pattern})")
CODE_BLOCK = re.compile(r"```.*?(?:```|\Z)", re.S)  # a fenced block, or an unclosed one to the end
_CODE_SPLIT = re.compile(r"(```.*?(?:```|\Z))", re.S)  # split(): the odd items are the fenced blocks
_BOLD = re.compile(r"(?<!\w)\*\*(?=\S)(.+?)\*\*")
_STARS = re.compile(r"(?<!\w)\*(?=\S)([^*]+)\*")  # *giggles*
_PARENS = re.compile(r"\(([^()]*)\)")  # (laughs)
STAGE_WORDS = frozenset("""laughs laughing giggles giggling sighs sighing blushes blushing smiles smiling grins grinning
pouts pouting winks whispers whispering nods huffs teasing hehe hmph yawns yawning stretches""".split())
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,!?…;:])")
_LABEL = re.compile(r"^\s*\**riya\**\s*(?::|[-\u2013\u2014](?=\s))\**\s*", re.I)  # "Riya:", "**Riya:**", "Riya -"
FENCE = "```"
CODE_SAID = "here's the code"
GENERIC_PHRASES: tuple[str, ...] = (  # assistant-speak she never says; the persona scorecard counts them too
    "you've got this", "no pressure", "let me know", "i'm here to", "i'm here for you", "feel free",
    "how can i help", "happy to help", "don't hesitate", "just say the word", "we'll figure it out together",
    "take a deep breath")
GENERIC_MAX_WORDS = 6
_IDENTITY = re.compile(r"\b(?:are you (?:human|real|alive|an? (?:ai|person|bot|robot|program|real (?:person|girl)))"
                       r"|what are you(?=\s*(?:[?!.,]|$))|is this an? ai)\b", re.I)


def turn_kind(text: str, *, about_me: bool, needs_thinking: bool) -> Literal["small", "explain", "about_me"]:
    """About-me wins, then anything that needs thinking is an explanation, the rest is small talk."""
    return "about_me" if about_me else "explain" if needs_thinking else "small"


def cap_sentences(sentences: Iterator[str], n: int) -> Iterator[str]:
    """Yield at most n sentences, closing the upstream iterator as the last one is handed over, so the model stops
    generating. A sentence that opens a fenced code block is joined with the following ones until the fence closes."""
    it = iter(sentences)
    close = getattr(it, "close", None)
    try:
        for done in range(n):
            sentence = next(it, None)
            while sentence is not None and sentence.count(FENCE) % 2 == 1:
                more = next(it, None)
                if more is None:
                    break
                sentence += "\n" + more
            if sentence is None:
                return
            if done == n - 1 and close:
                close()
            yield sentence
    finally:
        if close:
            close()


def is_identity_question(text: str) -> bool:
    """"are you human / real / an AI / a bot...", "what are you", "is this an AI": she must say she's an AI."""
    return bool(_IDENTITY.search(text))


def is_generic(sentence: str) -> bool:
    """A short (at most 6 words) sentence made of assistant-speak: it holds a GENERIC_PHRASES phrase. Code never is."""
    if FENCE in sentence or len(sentence.split()) > GENERIC_MAX_WORDS:
        return False
    t = sentence.lower().replace("’", "'")
    return any(p in t for p in GENERIC_PHRASES)


def drop_generic(sentences: Iterator[str], others: bool = False, dropped: list | None = None) -> Iterator[str]:
    """Drops each short generic sentence (is_generic) as soon as another sentence is known to be in the reply. One that
    is the reply's only sentence stays; others says another sentence was already said before this stream. Holds a
    generic sentence back only until the next one arrives. Each one dropped is appended to dropped, when given."""
    it = iter(sentences)
    pending = None
    try:
        for sentence in it:
            if pending is not None and dropped is not None:
                dropped.append(pending)
            pending = None  # another sentence exists, so the held generic one goes
            if is_generic(sentence):
                if others:
                    if dropped is not None:
                        dropped.append(sentence)
                    continue
                pending = sentence
                continue
            others = True
            yield sentence
        if pending is not None:
            yield pending
    finally:
        close = getattr(it, "close", None)
        if close:
            close()


def strip_label(text: str) -> str:
    """Drop a leading "Riya:" / "Riya -" speaker label: the small model sometimes copies the style examples' format."""
    return _LABEL.sub("", text, count=1)


def count_emoji(text: str) -> int:
    """Emoji clusters (a ZWJ sequence, flag or skin-toned emoji is one); Tamil, CJK and accented letters aren't."""
    return len(EMOJI.findall(text))


def limit_emoji(text: str, max_n: int = 1) -> str:
    """Keep the first max_n emoji and drop the rest, taking the space before a dropped one along. Fenced code is
    never touched."""
    seen = dropped = 0

    def keep(m: re.Match) -> str:
        nonlocal seen, dropped
        seen += 1
        if seen <= max_n:
            return m.group()
        dropped += 1
        return ""

    out = "".join(seg if i % 2 else _EMOJI_WITH_GAP.sub(keep, seg) for i, seg in enumerate(_CODE_SPLIT.split(text)))
    return out.strip() if dropped else text


def tts_text(text: str) -> str:
    """What the voice should say: a fenced block becomes "here's the code"; no emoji, no "~"; only short stage
    directions go.
    *giggles*: not right after a word character, at most 6 words. (laughs softly): 1-3 words, one of them a STAGE_WORDS
    word, so asides like "(the powerhouse)" and "f(x)" are read as written. Whitespace is collapsed."""
    removed = 0

    def stars(m: re.Match) -> str:
        nonlocal removed
        if len(m.group(1).split()) > 6:
            return m.group()
        removed += 1
        return " "

    def parens(m: re.Match) -> str:
        nonlocal removed
        words = m.group(1).split()
        if len(words) > 3 or not STAGE_WORDS.intersection(re.findall(r"[a-z]+", m.group(1).lower())):
            return m.group()
        removed += 1
        return " "

    out, n = CODE_BLOCK.subn(f" {CODE_SAID} ", text)
    out = _PARENS.sub(parens, _STARS.sub(stars, _BOLD.sub(r"\g<1>", out)))
    out, k = EMOJI.subn("", out)
    out, t = re.subn("~", " ", out)  # her "~" is for the eyes only
    out = " ".join(out.split())
    return _SPACE_BEFORE_PUNCT.sub(r"\g<1>", out) if n + k + t + removed else out
