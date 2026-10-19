"""Online function calling: when `match` misses a clear command, the online model may propose exactly one tool call.

The request carries only the user's turn, the local time and the NAMES of their apps and commands. The reply is
untrusted: `propose` returns a call only if it is the single offered tool, its arguments parse, `write_file` text is
the user's own words, and `validate_call` accepts it. Nothing here runs a tool, and nothing raises into the engine.
"""
from __future__ import annotations

import json
import logging
import re

from riya.core.models import ToolReply  # noqa: F401  (re-exported: what call_tools returns)

from .intent import _local_now, is_command_form
from .tools import REGISTRY, validate_call
from .types import Call

MAX_ARGS_BYTES = 4096
MAX_TURN = 2000  # characters of the turn that are sent
MAX_NAMES = 50
_OWN_WORDS = frozenset({"write_file", "clipboard_copy"})
log = logging.getLogger(__name__)

_JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "duration": "string"}
_DESC = {
    "time_now": "Tell the current time and date.",
    "timer_set": "Start a countdown timer.",
    "reminder_set": "Set a reminder for a clock time or a delay.",
    "reminder_list": "List the active reminders and timers.",
    "reminder_cancel": "Cancel a reminder or timer, by id or by words from its text.",
    "note_add": "Save a short note.",
    "note_list": "List the saved notes.",
    "calc": "Work out an arithmetic expression.",
    "weather": "Say the weather; the place may be left out for the saved one.",
    "lookup": "Look a topic up in an encyclopedia.",
    "open_url": "Open a web address in the browser.",
    "open_app": "Open one of the user's named apps.",
    "media": "Press a media key: play/pause, next or previous track.",
    "volume": "Change the system volume or toggle mute.",
    "clipboard_copy": "Copy text to the clipboard.",
    "run_command": "Run one of the user's named commands.",
    "write_file": "Save text the user dictated into a new file. The text must be the user's own words.",
}
_PARAM_DESC = {
    "duration": "A duration such as '10 minutes' or '1 hour 30 minutes'.",
    "when": "When, such as 'at 5 pm', 'tomorrow at 9' or 'in 20 minutes'.",
    "name": "The exact name from the user's list.",
}
_FILE = re.compile(r"\bfiles?\b")
_SYSTEM = ("You route a desktop companion's commands to tools. If the user's message clearly asks for one of the "
           "offered tools, call exactly one tool. Otherwise answer in one short sentence without calling a tool.")


def _clean(s: str, n: int = 60) -> str:
    return " ".join(s.split())[:n]


def _names(items) -> list[str]:
    out = [_clean(i["name"]) for i in (items or []) if isinstance(i, dict) and isinstance(i.get("name"), str)]
    return [n for n in out if n][:MAX_NAMES]


def online_messages(text: str, now, settings, tz) -> list[dict]:
    """[system, user]: the turn, the local time, and the names (only) of the user's apps and commands."""
    t = _local_now(now, tz)
    lines = [_SYSTEM, f"Local time: {t:%A %Y-%m-%d %H:%M}."]
    if apps := _names(getattr(settings, "apps", None)):
        lines.append("The user's apps: " + ", ".join(apps) + ".")
    if cmds := _names(getattr(settings, "commands", None)):
        lines.append("The user's commands: " + ", ".join(cmds) + ".")
    return [{"role": "system", "content": "\n".join(lines)}, {"role": "user", "content": text[:MAX_TURN]}]


def _enabled(settings, name: str) -> bool:
    return bool((getattr(settings, "tools_enabled", None) or {}).get(name, False))


def _wants_t2(settings, text: str) -> bool:
    low = text.casefold()
    return bool(_FILE.search(low)) or any(re.search(rf"(?<!\w){re.escape(n.casefold())}(?!\w)", low)
                                          for n in _names(getattr(settings, "commands", None)))


def _property(spec, key: str) -> dict:
    p: dict = {"type": _JSON_TYPE[spec.type]}
    if desc := _PARAM_DESC.get(key) or (_PARAM_DESC.get(spec.type) if spec.type == "duration" else None):
        p["description"] = desc
    if spec.enum is not None:
        p["enum"] = list(spec.enum)
    return p


def tool_schemas(settings, text: str) -> list[dict]:
    """OpenAI-style `tools` for the enabled registry tools; the T2 ones only if the turn names a command or says "file"."""
    t2 = _wants_t2(settings, text)
    out = []
    for name, tool in REGISTRY.items():
        if not _enabled(settings, name) or (tool.tier >= 2 and not t2):
            continue
        params = {"type": "object", "properties": {k: _property(sp, k) for k, sp in tool.params.items()},
                  "required": [k for k, sp in tool.params.items() if sp.required]}
        out.append({"type": "function", "function": {"name": name, "description": _DESC[name], "parameters": params}})
    return out


def parse_arguments(raw) -> dict | None:
    """The arguments JSON string as a dict; None for anything else (not a string, over 4 KB, bad JSON, not an object)."""
    if not isinstance(raw, str) or len(raw) > MAX_ARGS_BYTES or len(raw.encode("utf-8", "replace")) > MAX_ARGS_BYTES:
        return None
    try:
        v = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return v if isinstance(v, dict) else None


def _collapse(s: str) -> str:
    return " ".join(s.split())


def propose(model, text: str, now, settings, tz=None) -> Call | None:
    """The model's single call for this turn (arguments as parsed), or None. Never raises, never executes anything."""
    try:
        if model is None or not getattr(settings, "hands_online", False) or not getattr(settings, "online_enabled", False):
            return None
        if not is_command_form(text):
            return None
        tools = tool_schemas(settings, text)
        reply = model.call_tools(online_messages(text, now, settings, tz), tools)
        if len(reply.calls) != 1:
            return None
        name, raw = reply.calls[0]
        args = parse_arguments(raw)
        if args is None or name not in {t["function"]["name"] for t in tools}:  # only what was offered
            return None
        if name in _OWN_WORDS:  # text that is written or copied must be the user's own words
            t = args.get("text")
            if not isinstance(t, str) or not _collapse(t) or _collapse(t) not in _collapse(text):
                return None
        call = Call(name, args)
        return call if validate_call(call, settings) is not None else None
    except Exception as e:  # incl. ModelUnavailable: the normal reply path just runs
        log.debug("online call dropped: %s", type(e).__name__)
        return None
