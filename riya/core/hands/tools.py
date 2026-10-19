"""The tool registry: tiers, ParamSpec validation (`validate_call`), summaries and the runs that need no effects.

Runs that need timers, reminders and notes (Task 5) raise NotImplementedError here until their task fills them in. `Settings` is named only under TYPE_CHECKING: settings.py imports TIERS from here.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .effects import VOLUME_PRESSES, output_tail
from .intent import parse_duration
from .net import lookup, weather
from .safe import calc, format_number, safe_filename, unique_path, valid_url
from .types import RESULT_MAX, Call, ParamSpec, ToolResult

if TYPE_CHECKING:
    from riya.core.settings import Settings

__all__ = ["Call", "ParamSpec", "ToolResult", "ToolCtx", "Tool", "REGISTRY", "TIERS", "DEFAULT_OFF", "RESULT_MAX",
           "validate_call"]

TIERS: dict[str, int] = {
    **dict.fromkeys(("time_now", "timer_set", "reminder_set", "reminder_list", "reminder_cancel", "note_add",
                     "note_list", "calc", "weather", "lookup"), 0),
    **dict.fromkeys(("open_url", "open_app", "media", "volume", "clipboard_copy"), 1),
    **dict.fromkeys(("run_command", "write_file"), 2),
}
DEFAULT_OFF = frozenset({"run_command", "write_file"})
MAX_TIMER_S = 24 * 3600
WRITE_MAX_CHARS = 65_536
WRITE_MAX_BYTES = 65_536
WRITE_TRIES = 5


@dataclass
class ToolCtx:
    """What a run may use: settings, the db, effects, http, the clock, the Riya folder and the time zone."""
    settings: Any = None
    db: Any = None
    effects: Any = None
    http: Any = None
    clock: Any = None
    riya_dir: Any = None
    tz: Any = None


@dataclass(frozen=True)
class Tool:
    name: str
    tier: int
    params: dict[str, ParamSpec]
    summary: Callable[[dict], str]
    run: Callable[[dict, ToolCtx], ToolResult]
    one_of: tuple[str, ...] = ()


def _todo(args: dict, ctx: ToolCtx) -> ToolResult:
    raise NotImplementedError("this tool's run is written by the task that owns its dependency")


def _time_now(args: dict, ctx: ToolCtx) -> ToolResult:
    now = ctx.clock.now()
    t = datetime.fromtimestamp(now, ctx.tz) if ctx.tz else datetime.fromtimestamp(now).astimezone()
    return ToolResult(True, f"It's {t.hour % 12 or 12}:{t:%M %p} on {t:%A}, {t.day} {t:%B}.")


def _calc(args: dict, ctx: ToolCtx) -> ToolResult:
    v = calc(args["expr"])
    if v is None:
        return ToolResult(False, "I couldn't work that out.")
    return ToolResult(True, f"= {format_number(v)} ({args['expr']})", {"value": v})  # the answer survives the cut


_MARK = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _ctl(s) -> str:
    """s with control, format (bidi) and line/paragraph separator characters shown as visible markers."""
    return "".join((_MARK.get(ch) or f"\\u{ord(ch):04x}") if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch
                   for ch in str(s))


def _q(s: str, n: int = 60) -> str:
    """A one-line quoted value: cut to n chars, backslashes and quotes escaped, control characters made visible."""
    s = s if len(s) <= n else s[:n] + "..."
    return '"' + _ctl(s.replace("\\", "\\\\").replace('"', '\\"')) + '"'


def _effect(ctx: ToolCtx, name: str, *a) -> bool:
    """Call one effect; False if it raised (a failed effect is a kind result, never an exception)."""
    try:
        getattr(ctx.effects, name)(*a)
        return True
    except Exception:
        return False


def _open_url(args: dict, ctx: ToolCtx) -> ToolResult:
    if not _effect(ctx, "open_url", args["url"]):
        return ToolResult(False, "I couldn't open that.")
    return ToolResult(True, f"Opened {_ctl(args['url'])[:150]}.")


def _open_app(args: dict, ctx: ToolCtx) -> ToolResult:
    app = _named(getattr(ctx.settings, "apps", None), args["name"])
    if app is None or not isinstance(app.get("target"), str):
        return ToolResult(False, f"{_q(args['name'], 40)} isn't in your apps list.")
    if not _effect(ctx, "open_app", app["target"]):
        return ToolResult(False, f"I couldn't open {_q(app['name'], 40)}.")
    return ToolResult(True, f"Opened {_q(app['name'], 40)}.")


_MEDIA = {"play_pause": "Toggled play/pause.", "next": "Skipped to the next track.",
          "previous": "Back to the previous track."}
_VOLUME = {"up": ("volume_up", VOLUME_PRESSES, "Volume up."), "down": ("volume_down", VOLUME_PRESSES, "Volume down."),
           "mute": ("mute", 1, "Toggled mute.")}


def _media(args: dict, ctx: ToolCtx) -> ToolResult:
    if not _effect(ctx, "media_key", args["action"], 1):
        return ToolResult(False, "I couldn't press the media key.")
    return ToolResult(True, _MEDIA[args["action"]])


def _volume(args: dict, ctx: ToolCtx) -> ToolResult:
    key, presses, said = _VOLUME[args["action"]]
    if not _effect(ctx, "media_key", key, presses):
        return ToolResult(False, "I couldn't change the volume.")
    return ToolResult(True, said)


def _clipboard_copy(args: dict, ctx: ToolCtx) -> ToolResult:
    if not _effect(ctx, "copy", args["text"]):
        return ToolResult(False, "I couldn't reach the clipboard.")
    return ToolResult(True, f"Copied {len(args['text'])} characters to the clipboard.")


def _run_command(args: dict, ctx: ToolCtx) -> ToolResult:
    """Run the snapshot's argv (never a shell); ok only for exit code 0. The output tail is cut to fit the 200 chars."""
    name, timeout = _q(args["name"], 40), min(args["timeout_s"], 60)
    try:
        code, out = ctx.effects.run(args["argv"], args["cwd"], timeout)
    except Exception:
        return ToolResult(False, f"{name} couldn't start.")
    if code is None:  # the effect says why, e.g. "timed out after 60 s"
        return ToolResult(False, f"{name} {output_tail(out, RESULT_MAX - len(name) - 1)}")
    head = f"{name} finished:" if code == 0 else f"{name} failed (exit {code}):"
    tail = output_tail(out, RESULT_MAX - len(head) - 1)
    return ToolResult(code == 0, f"{head} {tail}" if tail else head.rstrip(":") + " (no output).", {"exit": code})


def _write_file(args: dict, ctx: ToolCtx) -> ToolResult:
    """Write into the Riya folder: at most 64 KB of UTF-8, exclusive create, a numeric suffix instead of overwriting."""
    data = args["text"].encode("utf-8", "replace")
    if len(data) > WRITE_MAX_BYTES:
        return ToolResult(False, "That's too long to save (the limit is 64 KB).")
    if ctx.riya_dir is None:
        return ToolResult(False, "I don't have a folder to save into.")
    try:
        Path(ctx.riya_dir).mkdir(parents=True, exist_ok=True)
        for _ in range(WRITE_TRIES):  # unique_path then open is racy: "x" decides, and a loser takes the next suffix
            path = unique_path(ctx.riya_dir, args["filename"])
            if path is None:
                break
            try:
                with open(path, "xb") as f:
                    f.write(data)
            except FileExistsError:
                continue
            except OSError:
                path.unlink(missing_ok=True)
                raise
            return ToolResult(True, f"Saved {_q(path.name, 60)} in your Riya folder.", {"path": str(path)})
    except OSError:
        return ToolResult(False, "I couldn't save that file.")
    return ToolResult(False, "I couldn't find a free file name.")


def _weather(args: dict, ctx: ToolCtx) -> ToolResult:
    saved = getattr(ctx.settings, "weather_place", None)
    place = args.get("place") or saved
    if not place:
        return ToolResult(False, "Which city should I check the weather for?")
    r = weather(ctx.http, place)
    if r.ok and "place" in args and not saved:  # the first named city becomes the default (the engine saves it)
        r.data = {**(r.data or {}), "save_place": args["place"].strip()}
    return r


def _lookup(args: dict, ctx: ToolCtx) -> ToolResult:
    return lookup(ctx.http, args["query"])


def _P(type, **kw) -> ParamSpec:
    return ParamSpec(type, **kw)


def _tool(name, params, summary, run=_todo, one_of=()) -> tuple[str, Tool]:
    return name, Tool(name, TIERS[name], params, summary, run, one_of)


_NAME = {"name": _P("str")}


def _action(*values) -> dict:
    return {"action": _P("str", enum=values)}


REGISTRY: dict[str, Tool] = dict([
    _tool("time_now", {}, lambda a: "tell the time", _time_now),
    _tool("timer_set", {"duration": _P("duration", min=1, max=MAX_TIMER_S)},
          lambda a: f"set a timer for {format_number(a['duration'])} seconds"),
    _tool("reminder_set", {"when": _P("str", max_len=100), "text": _P("str")},
          lambda a: f"remind you {a['when']}: {a['text']}"),
    _tool("reminder_list", {}, lambda a: "list your reminders"),
    _tool("reminder_cancel", {"id": _P("int", required=False, min=1, max=2**31 - 1), "text": _P("str", required=False)},
          lambda a: f"cancel reminder {a['id']}" if "id" in a else f"cancel the reminder about {a['text']}",
          one_of=("id", "text")),
    _tool("note_add", {"text": _P("str", max_len=1000)}, lambda a: f"note down {_q(a['text'])}"),
    _tool("note_list", {}, lambda a: "list your notes"),
    _tool("calc", {"expr": _P("str")}, lambda a: f"work out {a['expr']}", _calc),
    _tool("weather", {"place": _P("str", required=False, max_len=80)},
          lambda a: f"check the weather in {a['place']}" if "place" in a else "check the weather", _weather),
    _tool("lookup", {"query": _P("str")}, lambda a: f"look up {_q(a['query'])}", _lookup),
    _tool("open_url", {"url": _P("str", max_len=2000)}, lambda a: f"open {_ctl(a['url'])}", _open_url),
    _tool("open_app", _NAME, lambda a: f"open {_q(a['name'])}", _open_app),
    _tool("media", _action("play_pause", "next", "previous"), lambda a: f"media: {a['action'].replace('_', '/')}", _media),
    _tool("volume", _action("up", "down", "mute"), lambda a: f"volume {a['action']}", _volume),
    _tool("clipboard_copy", {"text": _P("str", max_len=4000)}, lambda a: f"copy {_q(a['text'])} to the clipboard", _clipboard_copy),
    _tool("run_command", _NAME, lambda a: f"run {_q(a['name'])} ({_ctl(' '.join(map(str, a['argv'])))})", _run_command),
    _tool("write_file", {"filename": _P("str"), "text": _P("str", max_len=WRITE_MAX_CHARS)},
          lambda a: f"write {_q(a['filename'])} ({len(a['text'].encode('utf-8', 'replace'))} bytes): {_q(a['text'], 80)}",
          _write_file),
])
assert set(REGISTRY) == set(TIERS)


def _check(spec: ParamSpec, v):
    """The normalized value, or raise ValueError."""
    t = spec.type
    if t == "str":
        if not isinstance(v, str) or not v.strip() or len(v) > spec.max_len:
            raise ValueError
    elif t == "int":
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError
    else:
        if t == "duration" and isinstance(v, str):
            if len(v) > spec.max_len or (v := parse_duration(v)) is None:
                raise ValueError
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError
        if t == "duration" and v <= 0:
            raise ValueError
        v = float(v)
    if t != "str" and ((spec.min is not None and v < spec.min) or (spec.max is not None and v > spec.max)):
        raise ValueError
    if spec.enum is not None and v not in spec.enum:
        raise ValueError
    return v


def _named(items, name: str):
    """The stored {name, ...} entry whose name equals `name` (exactly, else ignoring case), or None."""
    items = [i for i in (items or []) if isinstance(i, dict) and isinstance(i.get("name"), str)]
    return next((i for i in items if i["name"] == name), None) or next(
        (i for i in items if i["name"].casefold() == name.casefold()), None)


def validate_call(call: Call, settings: Settings) -> Call | None:
    """A new, normalized Call, or None when the tool, a key or a value is unknown or invalid (see the spec, 4.1)."""
    try:
        tool, raw = REGISTRY.get(call.tool), call.args
        if tool is None or not isinstance(raw, dict) or set(raw) - set(tool.params):
            return None
        if any(k not in raw for k, p in tool.params.items() if p.required):
            return None
        if tool.one_of and not any(k in raw for k in tool.one_of):
            return None
        args = {k: _check(tool.params[k], v) for k, v in raw.items()}
        if tool.name == "open_url":
            args["url"] = valid_url(args["url"])
            if args["url"] is None:
                return None
        elif tool.name == "open_app":
            if (app := _named(getattr(settings, "apps", None), args["name"])) is None:
                return None
            args["name"] = app["name"]
        elif tool.name == "write_file":  # the confirmed name is the written name
            args["filename"] = safe_filename(args["filename"])
            if args["filename"] is None:
                return None
        elif tool.name == "run_command":
            if (cmd := _named(getattr(settings, "commands", None), args["name"])) is None:
                return None
            argv, timeout = cmd["argv"], cmd["timeout_s"]
            if not isinstance(argv, list) or not argv or isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                return None
            args = {"name": cmd["name"], "argv": list(argv), "cwd": cmd.get("cwd"),
                    "timeout_s": max(1, min(60, timeout))}
        return Call(call.tool, args)
    except Exception:  # the gate never raises: a bad call (even a 400-digit int) is just None
        return None
