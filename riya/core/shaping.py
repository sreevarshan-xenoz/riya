"""Reply shaping, pure: what kind of turn this is, its token budget and sentence cap, and the emoji policy.
Small local models ramble and love emoji; the length and emoji rules are enforced here, in code, not just asked for.
See docs/superpowers/specs/2026-10-18-riya-v0.2-sounds-like-her-design.md §4.3."""
import re
from collections.abc import Iterator
from typing import Literal

BUDGETS = {"small": 110, "about_me": 160, "explain": 380, "greeting": 90, "bid": 40}  # max_tokens per kind
CAPS = {"small": 3, "about_me": 6, "explain": 8, "greeting": 3}  # most sentences per kind

_VS = "[️\U0001F3FB-\U0001F3FF]*"  # variation selector and skin tones ride along with their emoji
_PICTO = ("[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF☀-➿⭐⭕⬛⬜⌚⌛"
          "⏩-⏺⤴⤵⬅-⬇‼⁉]")
_TEXT_GLYPHS = "✓✔♪☺"  # check marks, a note, a smiley: pictographs, emoji only with a VS16
EMOJI = re.compile(rf"[\U0001F1E6-\U0001F1FF]{{2}}|[0-9#*]️?⃣"
                   rf"|(?![{_TEXT_GLYPHS}](?!️)){_PICTO}{_VS}(?:‍{_PICTO}{_VS})*")
_EMOJI_WITH_GAP = re.compile(rf"([ \t]*)({EMOJI.pattern})")
_STAGE = re.compile(r"\*[^*]*\*|\([^)]*\)")  # *giggles* and (sighs)
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,!?…;:])")
FENCE = "```"


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


def count_emoji(text: str) -> int:
    """Emoji clusters (a ZWJ sequence, flag or skin-toned emoji is one); Tamil, CJK and accented letters aren't."""
    return len(EMOJI.findall(text))


def limit_emoji(text: str, max_n: int = 1) -> str:
    """Keep the first max_n emoji and drop the rest, taking the space before a dropped one along."""
    seen = dropped = 0

    def keep(m: re.Match) -> str:
        nonlocal seen, dropped
        seen += 1
        if seen <= max_n:
            return m.group()
        dropped += 1
        return ""

    out = _EMOJI_WITH_GAP.sub(keep, text)
    return out.strip() if dropped else text


def tts_text(text: str) -> str:
    """What the voice should say: no emoji, no *stage directions* or (asides), whitespace collapsed."""
    out, n1 = EMOJI.subn("", text)
    out, n2 = _STAGE.subn(" ", out)
    out = " ".join(out.split())
    return _SPACE_BEFORE_PUNCT.sub(r"\1", out) if n1 + n2 else out
