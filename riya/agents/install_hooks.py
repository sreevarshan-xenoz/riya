"""Merge Riya's Claude Code `http` hook handlers into settings.json (and take them out again).

    python -m riya.agents.install_hooks [--print | --apply | --remove] [--settings PATH]

Handlers are recognised by their URL, so everything else in the file is left alone.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from riya.agents.hooks import default_token_path, load_or_create_token

HOOK_URL = "http://127.0.0.1:8766/hook"
ENV_VAR = "RIYA_HOOK_TOKEN"
RESTART_LINE = "Restart Claude Code (and open a new terminal) afterwards."
SETX_TIMEOUT_S = 10
HANDLER = {"type": "http", "url": HOOK_URL, "timeout": 3,
           "headers": {"X-Riya-Token": f"${ENV_VAR}"}, "allowedEnvVars": [ENV_VAR]}
# (event, matcher or None for "no matcher key"), one group per event
EVENTS = [
    ("UserPromptSubmit", None),
    ("PermissionRequest", "*"),
    ("Notification", "permission_prompt|idle_prompt"),
    ("Stop", None),
    ("StopFailure", None),
    ("PostToolUse", "*"),
    ("PostToolUseFailure", "*"),
    ("SubagentStop", "*"),
    ("SessionStart", None),
    ("SessionEnd", None),
    ("PreToolUse", "*"),
]


class SettingsError(Exception):
    pass


def default_settings_path() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / ".claude" / "settings.json"


def snippet() -> dict:
    hooks = {}
    for event, matcher in EVENTS:
        group = {"matcher": matcher} if matcher is not None else {}
        hooks[event] = [{**group, "hooks": [copy.deepcopy(HANDLER)]}]
    return {"hooks": hooks}


def _is_riya(h) -> bool:
    return isinstance(h, dict) and h.get("url") == HOOK_URL


def riya_events(data: dict | None) -> list[str]:
    """The EVENTS names (in EVENTS order) whose groups hold a Riya handler. Read-only; tolerates malformed settings."""
    hooks = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(hooks, dict):
        return []
    out = []
    for event, _ in EVENTS:
        groups = hooks.get(event)
        if isinstance(groups, list) and any(
                isinstance(g, dict) and isinstance(g.get("hooks"), list) and any(map(_is_riya, g["hooks"]))
                for g in groups):
            out.append(event)
    return out


def read_settings(path: Path) -> dict | None:
    """The parsed settings, or None when the file is missing. Raises SettingsError if unusable."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise SettingsError(f"cannot read {path}: {e}") from e
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as e:  # RecursionError: absurdly deep nesting
        why = e if isinstance(e, ValueError) else type(e).__name__
        raise SettingsError(f"{path} is not valid JSON ({why}); nothing was changed") from e
    if not isinstance(data, dict):
        raise SettingsError(f"{path}: the top level is not a JSON object; nothing was changed")
    if "hooks" in data and not isinstance(data["hooks"], dict):
        raise SettingsError(f"{path}: 'hooks' is not a JSON object; nothing was changed")
    return data


def warnings(data: dict | None) -> list[str]:
    """One line per setting that would stop Riya's hooks from firing (never modified)."""
    if not data:
        return []
    out = []
    urls, envs = data.get("allowedHttpHookUrls"), data.get("httpHookAllowedEnvVars")
    if isinstance(urls, list) and not any(isinstance(u, str) and fnmatch.fnmatchcase(HOOK_URL, u) for u in urls):
        out.append(f"Warning: allowedHttpHookUrls does not include {HOOK_URL}, so Claude Code will block Riya's hooks.")
    if isinstance(envs, list) and ENV_VAR not in envs:
        out.append(f"Warning: httpHookAllowedEnvVars does not include {ENV_VAR}, so the token header will not be sent and Riya's hooks will be rejected.")
    if data.get("disableAllHooks") is True:
        out.append("Warning: disableAllHooks is true, so no hooks (Riya's included) will fire.")
    return out


def blockers(data: dict | None) -> list[str]:
    """Why Claude Code's own settings would block Riya's hooks, one short reason per setting, most decisive first.
    Read-only: Riya never edits these keys. [] when nothing blocks them."""
    if not isinstance(data, dict):
        return []
    out = []
    if data.get("disableAllHooks") is True:
        out.append("disableAllHooks is true")
    urls, envs = data.get("allowedHttpHookUrls"), data.get("httpHookAllowedEnvVars")
    if isinstance(urls, list) and not any(isinstance(u, str) and fnmatch.fnmatchcase(HOOK_URL, u) for u in urls):
        out.append(f"allowedHttpHookUrls doesn't include {HOOK_URL}")
    if isinstance(envs, list) and ENV_VAR not in envs:
        out.append(f"httpHookAllowedEnvVars doesn't include {ENV_VAR}")
    return out


def merge(data: dict) -> int:
    """Add Riya's handler to each event that lacks one. Returns how many events were added."""
    hooks = data.setdefault("hooks", {})
    added = 0
    for event, matcher in EVENTS:
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            raise SettingsError(f"'hooks.{event}' is not a list; nothing was changed")
        if any(isinstance(g, dict) and isinstance(g.get("hooks"), list) and any(map(_is_riya, g["hooks"]))
               for g in groups):
            continue
        group = {"matcher": matcher} if matcher is not None else {}
        groups.append({**group, "hooks": [copy.deepcopy(HANDLER)]})
        added += 1
    return added


def strip(data: dict) -> int:
    """Delete only Riya's handlers, then groups/events/hooks that this emptied. Returns handlers removed."""
    hooks = data.get("hooks")
    removed = 0
    if not isinstance(hooks, dict):
        return 0
    for event in list(hooks):
        groups = hooks[event]
        if not isinstance(groups, list):
            continue
        kept = []
        for g in groups:
            inner = g.get("hooks") if isinstance(g, dict) else None
            if isinstance(inner, list) and any(map(_is_riya, inner)):
                left = [h for h in inner if not _is_riya(h)]
                removed += len(inner) - len(left)
                if not left:
                    continue
                g = {**g, "hooks": left}
            kept.append(g)
        if kept:
            hooks[event] = kept
        elif len(kept) != len(groups):
            del hooks[event]
    if removed and not hooks:
        del data["hooks"]
    return removed


def _strip_plan(data: dict | None) -> tuple[list[str], int]:
    """What strip would delete, on a copy: the events losing a Riya handler (EVENTS order, then any other event in file
    order) and how many handlers in all."""
    if not isinstance(data, dict) or not isinstance(data.get("hooks"), dict):
        return [], 0
    hooks = data["hooks"]
    order = [e for e, _ in EVENTS if e in hooks] + [e for e in hooks if e not in dict(EVENTS)]
    events, handlers = [], 0
    for event in order:
        n = strip({"hooks": {event: copy.deepcopy(hooks[event])}})
        if n:
            events.append(event)
            handlers += n
    return events, handlers


def preview(mode: Literal["apply", "remove"], settings_path: Path) -> str:
    """What --apply or --remove would change in settings_path, in one paragraph. Reads only; raises SettingsError when
    the file is unusable (for apply, also when --apply itself would refuse it)."""
    data = read_settings(Path(settings_path))
    have = riya_events(data)
    if mode == "apply":
        if data is not None:
            merge(copy.deepcopy(data))  # a dry run on a copy: the same SettingsError --apply would give
        blocked = blockers(data)
        note = f"Claude Code will block her hooks: {'; '.join(blocked)}. " if blocked else ""
        events = [event for event, _ in EVENTS if event not in have]
        if events:
            return (f"Riya will add her hook to {len(events)} Claude Code event(s) in {settings_path}: "
                    f"{', '.join(events)}. The file is backed up first, and the {ENV_VAR} user environment variable is "
                    f"set to her token. {note}{RESTART_LINE}")
        return (f"Riya's hook is already in all {len(EVENTS)} events in {settings_path}; only the {ENV_VAR} variable is "
                f"set again. {note}{RESTART_LINE}")
    if mode == "remove":
        events, handlers = _strip_plan(data)
        if events:
            count = f" ({handlers} handlers)" if handlers != len(events) else ""
            return (f"Riya will remove her hook from {len(events)} Claude Code event(s){count} in {settings_path}: "
                    f"{', '.join(events)}. The file is backed up first; the {ENV_VAR} variable is left alone. "
                    f"{RESTART_LINE}")
        return f"Riya's hook isn't in {settings_path}; nothing will change. {RESTART_LINE}"
    raise ValueError(f"unknown preview mode {mode!r}")


def backup(path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.riya-backup-{stamp}")
    n = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}.riya-backup-{stamp}-{n}")
        n += 1
    dest.write_bytes(path.read_bytes())
    return dest


def encode(data: dict) -> bytes:
    """Serialise first, so an unencodable value (a lone surrogate) aborts before anything is written."""
    try:
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    except (UnicodeError, ValueError, TypeError) as e:
        raise SettingsError(f"cannot encode the new settings as UTF-8 ({e}); nothing was changed") from e


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def do_print(token_path: Path, settings_path: Path, out=print) -> int:
    out(json.dumps(snippet(), indent=2))
    out(f"\nToken file: {token_path}")
    out(f"Settings file: {settings_path}")
    try:
        out("\n" + preview("apply", settings_path))
    except SettingsError:
        pass
    out("\nTo install: merge the 'hooks' entries above into your Claude Code settings.json, or run\n"
        "  python -m riya.agents.install_hooks --apply\n"
        f"and set the {ENV_VAR} user environment variable to the contents of the token file\n"
        "(--apply does that with setx). Then restart Claude Code sessions and open a new terminal.\n"
        "To undo: python -m riya.agents.install_hooks --remove")
    try:
        found = warnings(read_settings(settings_path))
    except SettingsError:
        found = []
    for line in found:
        out(line)
    return 0


FAIL_STEPS = ("settings", "write", "setx")  # what a failed --apply or --remove reports through `why`


def _failed(why: list | None, step: str, reason: str) -> None:
    """Record (step, reason) for an in-process caller: the step that really failed, with no full path in reason."""
    if why is not None:
        why.append((step, reason))


def _settings_reason(e: SettingsError, settings_path: Path) -> str:
    text = str(e).replace(str(settings_path), settings_path.name)
    return text.removesuffix("; nothing was changed")


def do_apply(token_path: Path, settings_path: Path, runner, out=print, now=None, why: list | None = None) -> int:
    try:
        data = read_settings(settings_path)
        existed = data is not None
        data = {} if data is None else data
        added = merge(data)
        payload = encode(data)
    except SettingsError as e:
        out(f"Error: {e}")
        _failed(why, "settings", _settings_reason(e, settings_path))
        return 1
    try:
        token = load_or_create_token(token_path)
        if existed:
            out(f"Backup: {backup(settings_path, now)}")
        write_atomic(settings_path, payload)
    except OSError as e:
        out(f"Error: could not write the settings ({e}); the original file was not changed")
        _failed(why, "write", f"couldn't write the settings ({e.strerror or type(e).__name__})")
        return 1
    out(f"Settings updated: {settings_path} ({added} event(s) added, {len(EVENTS) - added} already present)")
    for line in warnings(data):
        out(line)
    reason = "setx failed"
    try:
        res = runner(["setx", ENV_VAR, token], capture_output=True, text=True, timeout=SETX_TIMEOUT_S)
        ok = res.returncode == 0
    except subprocess.TimeoutExpired:  # say, antivirus holding it: never let a hung setx hold the caller
        ok, reason = False, f"setx timed out after {SETX_TIMEOUT_S:g} s"
    except OSError:
        ok = False
    if not ok:
        out(f"Warning: setx failed. Set the {ENV_VAR} user environment variable to the contents of {token_path} yourself.")
        _failed(why, "setx", reason)
        return 1
    out(f"Set the {ENV_VAR} user environment variable.")
    out("Restart your Claude Code sessions (and open a new terminal) for the variable to take effect.")
    out("If the hooks don't fire, run /hooks in Claude Code or restart it.")
    return 0


def do_remove(settings_path: Path, out=print, now=None, why: list | None = None) -> int:
    try:
        data = read_settings(settings_path)
        removed = strip(data) if data is not None else 0
        payload = encode(data) if removed else b""
    except SettingsError as e:
        out(f"Error: {e}")
        _failed(why, "settings", _settings_reason(e, settings_path))
        return 1
    if data is None:
        out(f"{settings_path} does not exist; nothing to remove.")
        return 0
    if not removed:
        out("No Riya handlers found; nothing changed.")
        return 0
    try:
        out(f"Backup: {backup(settings_path, now)}")
        write_atomic(settings_path, payload)
    except OSError as e:
        out(f"Error: could not write the settings ({e}); the original file was not changed")
        _failed(why, "write", f"couldn't write the settings ({e.strerror or type(e).__name__})")
        return 1
    out(f"Removed {removed} Riya handler(s) from {settings_path}.")
    out(f"The {ENV_VAR} environment variable was left alone. To remove it:\n"
        f"  reg delete HKCU\\Environment /v {ENV_VAR} /f")
    return 0


def main(argv=None, runner=subprocess.run, token_path=None, settings_path=None, out=print, now=None,
         why: list | None = None) -> int:
    """The CLI. An in-process caller may pass `why`, a list that gets one (step, reason) entry when --apply or
    --remove fails, step in FAIL_STEPS, so it can tell which step failed without parsing the printed lines."""
    p = argparse.ArgumentParser(prog="python -m riya.agents.install_hooks", description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--print", dest="mode", action="store_const", const="print", help="show the snippet (default)")
    mode.add_argument("--apply", dest="mode", action="store_const", const="apply", help="merge into settings and setx the token")
    mode.add_argument("--remove", dest="mode", action="store_const", const="remove", help="delete only Riya's handlers")
    p.add_argument("--settings", help="settings.json path (default: %%USERPROFILE%%/.claude/settings.json)")
    args = p.parse_args(argv)
    settings = Path(args.settings) if args.settings else Path(settings_path) if settings_path else default_settings_path()
    settings = settings.resolve()  # a symlinked settings.json stays a symlink: backup and replace hit the target
    token = Path(token_path) if token_path else default_token_path()
    if args.mode == "apply":
        return do_apply(token, settings, runner, out, now, why)
    if args.mode == "remove":
        return do_remove(settings, out, now, why)
    return do_print(token, settings, out)


if __name__ == "__main__":
    sys.exit(main())
