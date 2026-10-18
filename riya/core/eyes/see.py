"""Seeing: the vision request, the note it returns, and the pure detector for "she's being asked about the screen"."""
import base64
import json
import logging
import re
from dataclasses import dataclass

from riya.core.models import ChatModel, complete

ACTIVITY_WORDS, DETAILS_WORDS, NOTE_MAX_TOKENS = 12, 30, 200
log = logging.getLogger("riya.eyes")

NOTE_PROMPT = (
    "This is a screenshot of the window the user is working in. Reply with JSON only, no other text, in exactly this "
    'shape: {"app": string, "activity": string, "details": string, "notable": string or null}.\n'
    "- app: the application, e.g. VS Code, Chrome, Spotify.\n"
    "- activity: what the user is doing, at most 12 words.\n"
    "- details: a little more context, at most 30 words.\n"
    "- notable: an error, a failing test or anything a friend would want to mention, at most 30 words, otherwise null.\n"
    "Describe what the user is doing. Do not copy personal data verbatim, in any field including notable: no "
    "passwords, no full email or message bodies, no account numbers; summarize instead.\n"
    "If the image shows something sensitive (password fields, bank or card details, private messages or DMs, email "
    "bodies, medical or legal documents, a password-manager window, incognito or private browsing), set activity to "
    '"something private", details to an empty string and notable to null.')
PRIVATE = "something private"
NOTABLE_WORDS = 30


@dataclass
class GlanceNote:
    app: str
    activity: str
    details: str
    notable: str | None


def vision_messages(jpeg: bytes, text: str, system: str | None = None) -> list[dict]:
    """Chat messages whose user turn carries the text and the image (OpenAI-style content parts)."""
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    parts = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": parts}]


def _clip(value, words: int) -> str:
    return " ".join(value.split()[:words]) if isinstance(value, str) else ""


def _first_object(text: str) -> dict | None:
    """The first JSON object that parses, trying each "{" in turn (so chatter around or after it is fine)."""
    dec, i = json.JSONDecoder(), text.find("{")
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        i = text.find("{", i + 1)
    return None


def parse_note(text: str) -> GlanceNote | None:
    """The first JSON object in the reply as a note (fields clipped); None if there isn't a usable one. Whatever the
    model sent, a "something private" activity keeps no details and no notable."""
    data = _first_object(text) if isinstance(text, str) else None
    if data is None:
        return None
    app, activity = data.get("app"), _clip(data.get("activity"), ACTIVITY_WORDS)
    if not isinstance(app, str) or not app.strip() or not activity:
        return None
    if activity.casefold().rstrip(".") == PRIVATE:
        return GlanceNote(app.strip(), activity, "", None)
    notable = _clip(data.get("notable"), NOTABLE_WORDS)
    return GlanceNote(app.strip(), activity, _clip(data.get("details"), DETAILS_WORDS),
                      None if notable.casefold() in ("", "null", "none") else notable)


def describe(model: ChatModel, jpeg: bytes) -> GlanceNote | None:
    """Ask the (online) model what's on the screen. Never raises: any failure is just no note."""
    try:
        return parse_note(complete(model, vision_messages(jpeg, NOTE_PROMPT), max_tokens=NOTE_MAX_TOKENS))
    except Exception as e:  # ModelUnavailable, a bad reply, anything: she simply has no note this time
        log.info("no glance note: %s", e)
        return None


# A bare "this" is not enough (an accidental screenshot is a privacy cost): the verbs need a screen anchor.
_NOUN = r"(?:error|bug|code|page|message|exception|traceback|output|screen|window|tab)"
_DEM = rf"(?:(?:this|that|these) {_NOUN}|(?:my|the) (?:screen|window|tab|page))"
_END = r"\s*[?!.]*\s*$"
_SCREEN_QUESTION = re.compile("|".join([
    r"\bwhat am i (?:looking|staring) at\b",
    r"\bwhat(?:'s| is) (?:wrong|up|going on|happening|the (?:problem|issue|error|bug|matter)) here\b",
    rf"\bwhat(?:'s| is) (?:wrong|up|going on|happening|the (?:problem|issue|error|bug|matter)) (?:with|on|in|at) "
    rf"(?:{_DEM}|(?:this|that)(?={_END}|,))",
    rf"\bwhat(?:'s| is) (?:this|that) {_NOUN}\b",
    rf"\bwhat does (?:this|(?:this|that) {_NOUN}) (?:say|mean)\b",
    rf"\b(?:why|how) (?:is|does|did|do|can|come)\b.{{0,30}}\b(?:this|that) {_NOUN}\b",
    rf"\b(?:is|does|are|am i) (?:this|that|these) {_NOUN}\b",
    rf"\b(?:look(?:ing)? at|take a look at|have a look at|check(?: out)?|read|explain|translate|summari[sz]e|review|"
    rf"debug|fix|help me with|help with) {_DEM}\b",
    rf"^\s*(?:please )?(?:read|see) (?:this|that|here){_END}",
    rf"\b(?:look at|check) (?:this|that)(?: out)?(?:[\s,]+(?:lol|lmao|haha|please|pls|riya|now|rn|real quick))*{_END}",
    rf"\bsee {_DEM}\b",
    rf"\b(?:can|could|do|did|will) you (?:see|read|look at|make out)\b.{{0,25}}\b(?:{_DEM}|here|error)\b",
    rf"\b(?:can|could|do|did|will) you (?:see|read|look at|make out) (?:this|that){_END}",
    r"\b(?:what|show|tell|look|see|read|check)\b.{0,20}\b(?:on|at) (?:my |the )?screen\b",
    r"\bon (?:my )?screen\b",
]), re.I)


def is_screen_question(text: str) -> bool:
    """A question or imperative about the screen: "what's wrong with this", "read this", or a verb with a screen anchor
    ("explain this error", "look at my screen"). Bare "this/that" elsewhere ("look at that sunset") doesn't count.
    Some false positives are fine: the privacy checks still apply before she looks."""
    return bool(_SCREEN_QUESTION.search(text.replace("’", "'"))) if isinstance(text, str) else False
