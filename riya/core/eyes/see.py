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
    "- notable: an error, a failing test or anything a friend would want to mention, otherwise null.\n"
    "Describe what the user is doing. Do not copy personal data verbatim: no passwords, no full email or message "
    "bodies, no account numbers; summarize instead.\n"
    'If the image shows something sensitive (a password field, bank or card details), set activity to "something '
    'private", details to an empty string and notable to null.')


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


def parse_note(text: str) -> GlanceNote | None:
    """The outermost JSON object in the reply as a note (activity and details clipped); None if it isn't one."""
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    app, activity = data.get("app"), _clip(data.get("activity"), ACTIVITY_WORDS)
    if not isinstance(app, str) or not app.strip() or not activity:
        return None
    return GlanceNote(app.strip(), activity, _clip(data.get("details"), DETAILS_WORDS),
                      _clip(data.get("notable"), 10**6) or None)


def describe(model: ChatModel, jpeg: bytes) -> GlanceNote | None:
    """Ask the (online) model what's on the screen. Never raises: any failure is just no note."""
    try:
        return parse_note(complete(model, vision_messages(jpeg, NOTE_PROMPT), max_tokens=NOTE_MAX_TOKENS))
    except Exception as e:  # ModelUnavailable, a bad reply, anything: she simply has no note this time
        log.info("no glance note: %s", e)
        return None


_THIS = r"(?:this|that|these|those|here|(?:my|the) screen|(?:my|the) (?:window|tab|page))"
_ERR = r"(?:error|bug|code|page|message|exception|traceback|problem|issue|thing|test|output|screen|window|tab)"
_SCREEN_QUESTION = re.compile("|".join([
    r"\bwhat am i (?:looking|staring) at\b",
    r"\bwhat(?:'s| is) (?:wrong|up|going on|happening|the (?:problem|issue|error|bug)) (?:with|on|in|at) " + _THIS,
    r"\bwhat(?:'s| is) (?:going on|happening) here\b",
    r"\bwhat(?:'s| is) (?:this|that) " + _ERR + r"\b",
    r"\bwhat does (?:this|that|it) (?:say|mean)\b",
    r"\b(?:why|how) (?:is|does|did|do|can|come)\b.{0,30}\b(?:this|that) " + _ERR + r"\b",
    r"\b(?:look(?:ing)? at|take a look at|have a look at|check(?: out)?|read|explain|translate|summari[sz]e|review|"
    r"debug|fix|help me with|help with) " + _THIS + r"\b",
    r"\b(?:can|could|do|did|will) you (?:see|read|look at|make out)\b.{0,25}\b(?:this|that|here|screen|error)\b",
    r"\b(?:please|just|you) see (?:this|that|here)\b|^see (?:this|that|here)\b",
    r"\b(?:what|show|tell|look|see|read|check)\b.{0,20}\b(?:on|at) (?:my |the )?screen\b",
    r"\bon (?:my )?screen\b",
]), re.I)


def is_screen_question(text: str) -> bool:
    """Deictic screen words ("this", "here", "my screen", "this error") with a question or an imperative. False
    positives are fine: the privacy checks still apply before she looks."""
    return bool(_SCREEN_QUESTION.search(text.replace("’", "'"))) if isinstance(text, str) else False
