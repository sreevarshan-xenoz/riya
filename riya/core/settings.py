"""User settings: typed defaults, validated patches, quiet hours and the Windows autostart shortcut."""
import copy
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

from riya.agents.logwatch import DEFAULT_ENTRIES
from riya.core.eyes.policy import validate_patterns
from riya.core.hands.tools import DEFAULT_OFF, TIERS
from riya.core.store import KV

MODES = ("buddy", "panel", "window")
_QUIET_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")
MAX_GLANCE_CAP = 2000
GLANCE_GAP_RANGE = (10, 86400)  # seconds
MAX_DENY_PATTERNS, MAX_PATTERN_LEN = 100, 300
MAX_AGENT_LOGS, MAX_LOG_NAME = 20, 40
AGENT_LOG_KEYS = ("name", "glob", "done_re", "fail_re", "lost_re")
MAX_APPS, MAX_COMMANDS, MAX_HANDS_NAME, MAX_TARGET, MAX_ARGV, MAX_PLACE = 50, 20, 40, 500, 20, 80
COMMAND_KEYS = ("name", "argv", "cwd", "timeout_s")
TIMEOUT_RANGE = (1, 60)  # seconds


def _default_tools() -> dict[str, bool]:
    return {n: n not in DEFAULT_OFF for n in TIERS}


@dataclass
class Settings:
    mode: str = "buddy"
    voice: str = "alba"
    volume: float = 0.8
    mute_voice: bool = False
    mute_mic: bool = False
    autostart: bool = False
    online_enabled: bool = True
    online_consolidation: bool = False
    may_start_conversations: bool = True
    quiet_hours: str | None = "23:30-08:00"
    buddy_position: list[int] | None = None
    sprite_pack: str | None = None
    eyes_enabled: bool = True
    glance_daily_cap: int = 150
    glance_min_gap_s: int = 45
    eyes_deny_titles: list[str] = field(default_factory=list)  # extra regexes, on top of the built-in denylist
    eyes_deny_processes: list[str] = field(default_factory=list)
    dont_look: bool = False
    agent_logs: list[dict] = field(default_factory=lambda: copy.deepcopy(DEFAULT_ENTRIES))  # the Colab log watcher
    tools_enabled: dict[str, bool] = field(default_factory=_default_tools)  # her hands; the two T2 tools start off
    tools_ask: list[str] = field(default_factory=list)  # T1 tools raised to "ask first"
    apps: list[dict] = field(default_factory=list)  # {name, target}: what "open <name>" may launch
    commands: list[dict] = field(default_factory=list)  # {name, argv, cwd, timeout_s}: what "run <name>" may run
    weather_place: str | None = None
    hands_online: bool = True
    learning_enabled: bool = True  # off freezes what she learned: no detection, notes, twin records, events or drift


def load_settings(kv: KV) -> Settings:
    stored = kv.get("settings", {})
    known = {f.name for f in fields(Settings)}
    out = Settings(**{k: v for k, v in stored.items() if k in known})
    if isinstance(stored.get("tools_enabled"), dict):  # tools added later keep their defaults
        out.tools_enabled = _default_tools() | stored["tools_enabled"]
    return out


def save_settings(kv: KV, s: Settings) -> None:
    kv.set("settings", asdict(s))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_agent_logs(v) -> bool:
    """A list of {name, glob, done_re, fail_re, lost_re}: all strings, a unique non-empty name and a glob; a pattern may
    be empty (that kind is never matched) but must compile."""
    if not isinstance(v, list) or len(v) > MAX_AGENT_LOGS:
        return False
    names = set()
    for e in v:
        if not isinstance(e, dict) or set(e) != set(AGENT_LOG_KEYS) or not all(
                isinstance(e[k], str) and len(e[k]) <= MAX_PATTERN_LEN for k in AGENT_LOG_KEYS):
            return False
        if not e["name"].strip() or len(e["name"]) > MAX_LOG_NAME or e["name"] in names or not e["glob"].strip():
            return False
        names.add(e["name"])
        for k in ("done_re", "fail_re", "lost_re"):
            try:
                re.compile(e[k])
            except re.error as err:
                raise ValueError(f"invalid pattern in agent_logs {e['name']!r} {k}: {err}") from None
    return True


def _text(v, n: int) -> bool:
    return isinstance(v, str) and bool(v.strip()) and len(v) <= n


def _check_tools_enabled(v) -> bool:
    return isinstance(v, dict) and all(k in TIERS and isinstance(b, bool) for k, b in v.items())


def _check_tools_ask(v) -> bool:
    return isinstance(v, list) and all(isinstance(n, str) and TIERS.get(n) == 1 for n in v)  # only T1 may ask


def _check_apps(v) -> bool:
    if not isinstance(v, list) or len(v) > MAX_APPS:
        return False
    if not all(isinstance(a, dict) and set(a) == {"name", "target"} and _text(a["name"], MAX_HANDS_NAME)
               and _text(a["target"], MAX_TARGET) for a in v):
        return False
    return len({a["name"].casefold() for a in v}) == len(v)


def _check_commands(v) -> bool:
    if not isinstance(v, list) or len(v) > MAX_COMMANDS:
        return False
    for c in v:
        if not isinstance(c, dict) or set(c) != set(COMMAND_KEYS) or not _text(c["name"], MAX_HANDS_NAME):
            return False
        argv, t = c["argv"], c["timeout_s"]
        if not isinstance(argv, list) or not 1 <= len(argv) <= MAX_ARGV or not all(_text(a, MAX_TARGET) for a in argv):
            return False
        if not (c["cwd"] is None or _text(c["cwd"], MAX_TARGET)):
            return False
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not TIMEOUT_RANGE[0] <= t <= TIMEOUT_RANGE[1]:
            return False
    return len({c["name"].casefold() for c in v}) == len(v)


def _check(key: str, v) -> None:
    ok: bool
    if key == "mode":
        ok = v in MODES
    elif key == "voice":
        ok = isinstance(v, str) and bool(v)
    elif key == "volume":
        ok = (_is_int(v) or isinstance(v, float)) and 0 <= v <= 1
    elif key == "quiet_hours":
        ok = v is None or (isinstance(v, str) and _QUIET_RE.match(v) is not None)
    elif key == "buddy_position":
        ok = v is None or (isinstance(v, list) and len(v) == 2 and all(_is_int(x) for x in v))
    elif key == "sprite_pack":
        ok = v is None or isinstance(v, str)
    elif key == "glance_daily_cap":
        ok = _is_int(v) and 0 <= v <= MAX_GLANCE_CAP
    elif key == "glance_min_gap_s":
        ok = (_is_int(v) or isinstance(v, float)) and GLANCE_GAP_RANGE[0] <= v <= GLANCE_GAP_RANGE[1]
    elif key in ("eyes_deny_titles", "eyes_deny_processes"):
        ok = (isinstance(v, list) and len(v) <= MAX_DENY_PATTERNS
              and all(isinstance(p, str) and len(p) <= MAX_PATTERN_LEN for p in v))
        if ok and (bad := validate_patterns(v)):
            raise ValueError(f"invalid pattern in {key!r}: {', '.join(map(repr, bad))}")
    elif key == "agent_logs":
        ok = _check_agent_logs(v)
    elif key == "tools_enabled":
        ok = _check_tools_enabled(v)
    elif key == "tools_ask":
        ok = _check_tools_ask(v)
    elif key == "apps":
        ok = _check_apps(v)
    elif key == "commands":
        ok = _check_commands(v)
    elif key == "weather_place":
        ok = v is None or _text(v, MAX_PLACE)
    else:
        ok = isinstance(v, bool)
    if not ok:
        raise ValueError(f"invalid value for setting {key!r}: {v!r}")


def apply_patch(s: Settings, patch: dict) -> Settings:
    known = {f.name for f in fields(Settings)}
    for key, value in patch.items():
        if key not in known:
            raise ValueError(f"unknown setting {key!r}")
        _check(key, value)
    if "tools_enabled" in patch:  # a partial map toggles just those tools
        patch = {**patch, "tools_enabled": {**s.tools_enabled, **patch["tools_enabled"]}}
    return replace(s, **patch)


def in_quiet_hours(spec: str | None, t: datetime) -> bool:
    m = _QUIET_RE.match(spec) if spec else None
    if not m:
        return False
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    start, end, now = h1 * 60 + m1, h2 * 60 + m2, t.hour * 60 + t.minute
    if start == end:
        return False
    return start <= now < end if start < end else now >= start or now < end


def default_startup_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"


def set_autostart(enabled: bool, launcher: Path, startup_dir: Path | None = None) -> None:
    if startup_dir is None:
        startup_dir = default_startup_dir()
    shortcut = startup_dir / "riya.cmd"
    if enabled:
        startup_dir.mkdir(parents=True, exist_ok=True)
        shortcut.write_text(
            f'@start "" powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File "{launcher}"', encoding="utf-8"
        )
    else:
        shortcut.unlink(missing_ok=True)
