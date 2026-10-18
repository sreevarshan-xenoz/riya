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

from riya.agents.hooks import default_token_path, load_or_create_token

HOOK_URL = "http://127.0.0.1:8766/hook"
ENV_VAR = "RIYA_HOOK_TOKEN"
HANDLER = {"type": "http", "url": HOOK_URL, "timeout": 3,
           "headers": {"X-Riya-Token": f"${ENV_VAR}"}, "allowedEnvVars": [ENV_VAR]}
# (event, matcher or None for "no matcher key"), one group per event
EVENTS = [
    ("UserPromptSubmit", None),
    ("PermissionRequest", "*"),
    ("Notification", "permission_prompt|idle_prompt"),
    ("Stop", None),
    ("StopFailure", None),
    ("PostToolUseFailure", "*"),
    ("SubagentStop", "*"),
    ("SessionStart", None),
    ("SessionEnd", None),
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
    except ValueError as e:
        raise SettingsError(f"{path} is not valid JSON ({e}); nothing was changed") from e
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


def backup(path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.riya-backup-{stamp}")
    n = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}.riya-backup-{stamp}-{n}")
        n += 1
    dest.write_bytes(path.read_bytes())
    return dest


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
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


def do_apply(token_path: Path, settings_path: Path, runner, out=print, now=None) -> int:
    try:
        data = read_settings(settings_path)
        existed = data is not None
        data = {} if data is None else data
        added = merge(data)
    except SettingsError as e:
        out(f"Error: {e}")
        return 1
    token = load_or_create_token(token_path)
    if existed:
        out(f"Backup: {backup(settings_path, now)}")
    write_atomic(settings_path, data)
    out(f"Settings updated: {settings_path} ({added} event(s) added, {len(EVENTS) - added} already present)")
    for line in warnings(data):
        out(line)
    try:
        res = runner(["setx", ENV_VAR, token], capture_output=True, text=True)
        ok = res.returncode == 0
    except OSError:
        ok = False
    if not ok:
        out(f"Warning: setx failed. Set the {ENV_VAR} user environment variable to the contents of {token_path} yourself.")
        return 1
    out(f"Set the {ENV_VAR} user environment variable.")
    out("Restart your Claude Code sessions (and open a new terminal) for the variable to take effect.")
    out("If the hooks don't fire, run /hooks in Claude Code or restart it.")
    return 0


def do_remove(settings_path: Path, out=print, now=None) -> int:
    try:
        data = read_settings(settings_path)
    except SettingsError as e:
        out(f"Error: {e}")
        return 1
    if data is None:
        out(f"{settings_path} does not exist; nothing to remove.")
        return 0
    removed = strip(data)
    if not removed:
        out("No Riya handlers found; nothing changed.")
        return 0
    out(f"Backup: {backup(settings_path, now)}")
    write_atomic(settings_path, data)
    out(f"Removed {removed} Riya handler(s) from {settings_path}.")
    out(f"The {ENV_VAR} environment variable was left alone. To remove it:\n"
        f"  reg delete HKCU\\Environment /v {ENV_VAR} /f")
    return 0


def main(argv=None, runner=subprocess.run, token_path=None, settings_path=None, out=print, now=None) -> int:
    p = argparse.ArgumentParser(prog="python -m riya.agents.install_hooks", description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--print", dest="mode", action="store_const", const="print", help="show the snippet (default)")
    mode.add_argument("--apply", dest="mode", action="store_const", const="apply", help="merge into settings and setx the token")
    mode.add_argument("--remove", dest="mode", action="store_const", const="remove", help="delete only Riya's handlers")
    p.add_argument("--settings", help="settings.json path (default: %%USERPROFILE%%/.claude/settings.json)")
    args = p.parse_args(argv)
    settings = Path(args.settings) if args.settings else Path(settings_path) if settings_path else default_settings_path()
    token = Path(token_path) if token_path else default_token_path()
    if args.mode == "apply":
        return do_apply(token, settings, runner, out, now)
    if args.mode == "remove":
        return do_remove(settings, out, now)
    return do_print(token, settings, out)


if __name__ == "__main__":
    sys.exit(main())
