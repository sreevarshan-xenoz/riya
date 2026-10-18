"""User settings: typed defaults, validated patches, quiet hours and the Windows autostart shortcut."""
import copy
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

from riya.agents.logwatch import DEFAULT_ENTRIES
from riya.core.eyes.policy import validate_patterns
from riya.core.store import KV

MODES = ("buddy", "panel", "window")
_QUIET_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")
MAX_GLANCE_CAP = 2000
GLANCE_GAP_RANGE = (10, 86400)  # seconds
MAX_DENY_PATTERNS, MAX_PATTERN_LEN = 100, 300
MAX_AGENT_LOGS, MAX_LOG_NAME = 20, 40
AGENT_LOG_KEYS = ("name", "glob", "done_re", "fail_re", "lost_re")


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


def load_settings(kv: KV) -> Settings:
    stored = kv.get("settings", {})
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in stored.items() if k in known})


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


def set_autostart(enabled: bool, launcher: Path, startup_dir: Path | None = None) -> None:
    if startup_dir is None:
        startup_dir = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
    shortcut = startup_dir / "riya.cmd"
    if enabled:
        startup_dir.mkdir(parents=True, exist_ok=True)
        shortcut.write_text(
            f'@start "" powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File "{launcher}"', encoding="utf-8"
        )
    else:
        shortcut.unlink(missing_ok=True)
