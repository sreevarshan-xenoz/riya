"""The Health page's checks: ten rows, each passive and read-only, each in its own thread with a timeout.

A check never opens the mic, plays audio, grabs the screen, writes a file or reaches past a local `http_get` probe;
every path it reads comes in through `HealthCtx`. A check that hangs or raises becomes a warn row; it never blocks the
others and `run_checks` never raises. No detail ever holds the key, only its mask. Imports nothing from the engine or
the protocol."""
from __future__ import annotations

import glob
import json
import re
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath

from riya.agents import install_hooks
from riya.agents.logwatch import _root
from riya.core import secrets as keystore
from riya.core.secrets import KeyInfo
from riya.core.settings import Settings
from riya.core.store import SCHEMA_VERSION

CHECK_TIMEOUT_S = 5.0
DETAIL_MAX = 120
STATUSES = ("ok", "warn", "off")
CONNECTED_S = 24 * 3600
ROWS = ("local", "online", "voice", "mic", "look", "eyes", "claude_code", "colab", "memory", "autostart")
FIXES = ("online_test", "voice_sample", "mic_check", "eyes_test", "hooks_connect", "hooks_disconnect", "autostart")
HTTP_MAX_BYTES = 64 * 1024


def clip(text: str, n: int = DETAIL_MAX) -> str:
    """One line: whitespace collapsed; over n characters it is cut to n - 1 and ends with an ellipsis."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


@dataclass(frozen=True)
class HealthItem:
    key: str
    status: str
    detail: str
    fix: str | None = None

    def __post_init__(self):
        if self.key not in ROWS:
            raise ValueError(f"unknown health row {self.key!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown health status {self.status!r}")
        if self.fix is not None and self.fix not in FIXES:
            raise ValueError(f"unknown health fix {self.fix!r}")
        # scrub first: a remembered key inside a detail (say, an error text) must not survive the clip
        object.__setattr__(self, "detail", clip(keystore.scrub(str(self.detail))))


def clip_left(text: str, n: int) -> str:
    """text cut from the left, so its tail stays visible: over n characters it is "…" + the last n - 1."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else "…" + text[-(n - 1):]


def _with_path(prefix: str, path) -> str:
    """prefix + path, the path clipped from the left so the whole fits in DETAIL_MAX."""
    return prefix + clip_left(str(path), max(DETAIL_MAX - len(prefix), 2))


_USERINFO = re.compile(r"(//)[^/@\s]*@")


def _no_credentials(url: str) -> str:
    return _USERINFO.sub("//", url)


def default_http_get(url: str, timeout_s: float) -> bytes:
    """GET url (http or https only) and return at most 64 KiB. For the local llama-server probe, so no proxy."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are fetched")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout_s) as r:
        return r.read(HTTP_MAX_BYTES)


@dataclass
class HealthCtx:
    now: float
    settings: Settings
    local_url: str  # "" means none
    key: KeyInfo
    last_tests: dict[str, dict]  # online_test, eyes_test, mic_check, voice_sample: {at, ok, detail, mask}
    voice_running: bool
    vision_ready: bool
    receiver_ok: bool | None  # None: not started
    hooks_seen_at: float | None
    claude_settings: Path
    repo_root: Path
    db_path: str  # "" means in memory
    schema_version: int
    startup_dir: Path
    session_start: float | None = None  # when this core started; None: no test counts as this session's
    http_get: Callable[[str, float], bytes] = default_http_get
    tz: object = field(default=None)


def pack_in_use(sprite_pack: str | None, repo_root: Path) -> tuple[Path | None, bool]:
    """The first of body/src/pack-path.js's three candidates with a manifest.json, and whether it is the placeholder."""
    repo_root = Path(repo_root)
    candidates = []
    if isinstance(sprite_pack, str) and sprite_pack.strip():
        candidates.append(repo_root / sprite_pack.strip())  # an absolute path replaces the repo root, as path.resolve
    candidates += [repo_root / "sprites" / "riya-v1", repo_root / "body" / "placeholder-pack"]
    for i, d in enumerate(candidates):
        if (d / "manifest.json").is_file():
            return d, i == len(candidates) - 1
    return None, False


def _t(ctx: HealthCtx, at: float) -> str:
    return datetime.fromtimestamp(at, ctx.tz).strftime("%H:%M")


def _last_test(ctx: HealthCtx, name: str, for_key: bool = False) -> dict | None:
    """The last test of that kind; one made with a key counts only while it is still the current key."""
    t = ctx.last_tests.get(name)
    if not isinstance(t, dict) or not isinstance(t.get("at"), (int, float)):
        return None
    if for_key and t.get("mask") != ctx.key.mask:
        return None
    return t


def check_local(ctx: HealthCtx) -> HealthItem:
    if not ctx.local_url:
        return HealthItem("local", "warn", "No local server is configured.")
    try:
        raw = ctx.http_get(ctx.local_url.rstrip("/") + "/v1/models", 3.0)
    except Exception:
        return HealthItem("local", "warn", f"llama-server isn't answering at {_no_credentials(ctx.local_url)}")
    name = ""
    try:
        data = json.loads(raw)
        model_id = data["data"][0]["id"]
        name = PureWindowsPath(model_id).name if isinstance(model_id, str) else ""
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    return HealthItem("local", "ok", f"llama-server is up · {name}" if name else "llama-server is up")


_SOURCES = {"app": " saved in Riya", ".env": " set via .env (edit that file to remove it)",
            "env": " set via the RIYA_ONLINE_API_KEY environment variable"}


def check_online(ctx: HealthCtx) -> HealthItem:
    if not ctx.settings.online_enabled:
        return HealthItem("online", "off", "Online brain is switched off in Settings.")
    if not ctx.key.mask:
        return HealthItem("online", "off", "No key yet: paste a free Gemini key below.")
    text, status = f"Key {ctx.key.mask}{_SOURCES.get(ctx.key.source or '', '')}", "ok"
    t = _last_test(ctx, "online_test", for_key=True)
    if t is None:
        text += " · not tested yet"
    elif t.get("ok"):
        text += f" · test worked {_t(ctx, t['at'])}"
    else:
        text += f" · last test failed {_t(ctx, t['at'])}: {t.get('detail', '')}"
        status = "warn"
    return HealthItem("online", status, text, "online_test")


def check_voice(ctx: HealthCtx) -> HealthItem:
    voice = ctx.settings.voice
    if not ctx.voice_running:
        return HealthItem("voice", "warn", "Her voice isn't running (text only). Restart Riya without --no-voice.")
    if ctx.settings.mute_voice:
        return HealthItem("voice", "off", f"Muted · voice {voice}")
    t = _last_test(ctx, "voice_sample")
    if t is not None and not t.get("ok"):
        return HealthItem("voice", "warn", f"Voice {voice} · the last sample failed: {t.get('detail', '')}", "voice_sample")
    if t is not None and ctx.session_start is not None and t["at"] >= ctx.session_start:  # heard this session
        return HealthItem("voice", "ok", f"Speaking works · voice {voice}", "voice_sample")
    return HealthItem("voice", "ok", f"Voice ready (not tested yet) · voice {voice}", "voice_sample")


def check_mic(ctx: HealthCtx) -> HealthItem:
    """Passive: it reports the last Mic check the user clicked; it never opens the device itself."""
    if ctx.settings.mute_mic:
        return HealthItem("mic", "off", "Mic muted in Settings.")
    if not ctx.voice_running:
        return HealthItem("mic", "warn", "Her ears aren't running (text only).", "mic_check")
    t = _last_test(ctx, "mic_check")
    if t is None:
        return HealthItem("mic", "ok", "Listening · not checked yet", "mic_check")
    if not t.get("ok"):
        return HealthItem("mic", "warn", f"The last mic check failed: {t.get('detail', '')}", "mic_check")
    return HealthItem("mic", "ok", f"Listening · {t.get('detail', '')}", "mic_check")


def check_look(ctx: HealthCtx) -> HealthItem:
    pack, placeholder = pack_in_use(ctx.settings.sprite_pack, ctx.repo_root)
    if pack is None:
        return HealthItem("look", "warn", "No art pack found.")
    if placeholder:
        return HealthItem("look", "warn", _with_path("Placeholder art · her own pack goes in ",
                                                     Path(ctx.repo_root) / "sprites" / "riya-v1"))
    return HealthItem("look", "ok", _with_path("Her own art · ", pack))


def check_eyes(ctx: HealthCtx) -> HealthItem:
    s = ctx.settings
    if not s.eyes_enabled:
        return HealthItem("eyes", "off", "Eyes off.")
    if not ctx.vision_ready:
        return HealthItem("eyes", "warn", "Her eyes need the online key.")
    if not s.online_enabled:
        return HealthItem("eyes", "warn", "Her eyes need the online brain switched on.")
    text, status = "On", "ok"
    t = _last_test(ctx, "eyes_test", for_key=True)
    if t is None:
        text += " · not tested yet"
    elif t.get("ok"):
        text += f" · eye test worked {_t(ctx, t['at'])}"
    else:
        text += f" · the last eye test failed {_t(ctx, t['at'])}: {t.get('detail', '')}"
        status = "warn"
    if s.dont_look:
        text += " · don't-look is on"
    return HealthItem("eyes", status, text, "eyes_test")


def check_claude_code(ctx: HealthCtx) -> HealthItem:
    """Read-only: the receiver's state, the hooks in Claude Code's settings.json (never written), what in that file
    blocks them, and the last event. The file decides the state; the last event only refines the wording."""
    if ctx.receiver_ok is not True:
        return HealthItem("claude_code", "warn", "Receiver down: port 8766 is busy or the receiver didn't start.")
    try:
        data = install_hooks.read_settings(Path(ctx.claude_settings))
    except install_hooks.SettingsError:
        return HealthItem("claude_code", "warn", "Couldn't read Claude Code's settings.json.")
    if not install_hooks.riya_events(data):  # absent (say, just disconnected), whatever the last event time
        return HealthItem("claude_code", "off", "Not connected.", "hooks_connect")
    if blocked := install_hooks.blockers(data):  # only the user changes those keys: no fix button
        return HealthItem("claude_code", "warn", f"Hooks are blocked by your Claude Code settings: {'; '.join(blocked)}")
    seen = ctx.hooks_seen_at
    if seen is not None and ctx.now - seen <= CONNECTED_S:
        return HealthItem("claude_code", "ok", f"Connected · last event {_t(ctx, seen)}", "hooks_disconnect")
    if seen is None:
        return HealthItem("claude_code", "warn", "Applied · restart Claude Code (and open a new terminal).",
                          "hooks_disconnect")
    return HealthItem("claude_code", "warn", "Applied · no event in 24 h; restart Claude Code if it's open.",
                      "hooks_disconnect")


def check_colab(ctx: HealthCtx) -> HealthItem:
    entries = ctx.settings.agent_logs
    if not entries:
        return HealthItem("colab", "off", "No Colab log folders set.")
    roots = list(dict.fromkeys(_root(e["glob"]) for e in entries))
    for r in roots:
        if not Path(r).is_dir():
            return HealthItem("colab", "warn", f"Folder not found: {r}")
    return HealthItem("colab", "ok", f"Watching {len(roots)} folder(s)")


def _newest_backup(db: Path) -> Path | None:
    best, best_key = None, None
    for p in db.parent.glob(glob.escape(db.name) + ".pre-v*.bak"):
        try:
            key = (p.stat().st_mtime, p.name)
        except OSError:
            continue
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def check_memory(ctx: HealthCtx) -> HealthItem:
    v = ctx.schema_version
    status = "ok" if v == SCHEMA_VERSION else "warn"
    if not ctx.db_path:
        return HealthItem("memory", status, f"In memory · schema {v}")
    db = Path(ctx.db_path)
    size = 0
    for p in (db, db.with_name(db.name + "-wal")):
        try:
            size += p.stat().st_size
        except OSError:
            pass
    bak = _newest_backup(db)
    backup = f"backup {bak.name}" if bak else "no backup yet"
    return HealthItem("memory", status, f"{size / (1024 * 1024):.1f} MB · schema {v} · {backup}")


def check_autostart(ctx: HealthCtx) -> HealthItem:
    if not ctx.settings.autostart:
        return HealthItem("autostart", "off", "Doesn't start with Windows.", "autostart")
    if (Path(ctx.startup_dir) / "riya.cmd").exists():
        return HealthItem("autostart", "ok", "Starts with Windows.", "autostart")
    return HealthItem("autostart", "warn", "Autostart is on but its shortcut is missing.", "autostart")


CHECKS: dict[str, Callable[[HealthCtx], HealthItem]] = {
    "local": check_local, "online": check_online, "voice": check_voice, "mic": check_mic, "look": check_look,
    "eyes": check_eyes, "claude_code": check_claude_code, "colab": check_colab, "memory": check_memory,
    "autostart": check_autostart,
}


_inflight: dict[tuple, threading.Thread] = {}  # (row, check function) -> its thread while a run is still going
_inflight_lock = threading.Lock()


def run_checks(ctx: HealthCtx, timeout_s: float = CHECK_TIMEOUT_S, checks: dict | None = None) -> list[HealthItem]:
    """Every check in its own daemon thread, all joined against one shared deadline. A check still running gives a warn
    row, a raise gives a warn row naming only the exception type. A check whose previous run is still stuck gets no new
    thread (the row says "No answer yet"), so a blocked call can't pile up threads across refreshes. Never raises; the
    result follows `checks` order."""
    checks = CHECKS if checks is None else checks
    done: dict[str, HealthItem] = {}
    lock = threading.Lock()

    def work(key, fn):
        try:
            item = fn(ctx)
            if not isinstance(item, HealthItem):
                raise TypeError("a check must return a HealthItem")
            if item.key != key:
                raise ValueError("a check returned another row's item")
        except BaseException as e:  # noqa: BLE001 - nothing may escape the thread, and str(e) could hold a secret
            item = HealthItem(key, "warn", f"This check failed ({type(e).__name__}).")
        with lock:
            done[key] = item

    threads, stuck = [], set()
    for key, fn in checks.items():
        with _inflight_lock:
            for gk in [gk for gk, th in _inflight.items() if not th.is_alive()]:
                del _inflight[gk]
            if (key, fn) in _inflight:
                stuck.add(key)
                continue
            t = threading.Thread(target=work, args=(key, fn), name=f"riya-health-{key}", daemon=True)
            _inflight[(key, fn)] = t
            t.start()  # inside the lock, so a concurrent run never sees it as "not alive" and doubles it
        threads.append(t)
    deadline = time.monotonic() + timeout_s
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    with lock:
        finished = dict(done)
    return [finished.get(key) or HealthItem(
        key, "warn", "No answer yet: the last check is still running." if key in stuck
        else f"No answer within {timeout_s:g} s.") for key in checks]
