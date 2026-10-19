"""The online-model API key: where it lives (%LOCALAPPDATA%\\Riya\\secrets.env), which source wins, how it is masked, and
how it is scrubbed from every log record. Always imported as `from riya.core import secrets as keystore`.

The key never leaves the core except as `mask(key)`; nothing here logs a value."""
from __future__ import annotations

import logging
import os
import sys
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from riya.agents.install_hooks import write_atomic

KEY_VAR = "RIYA_ONLINE_API_KEY"
MAX_KEY_LEN = 200
MASK = "••••"
REDACTED = "[redacted key]"
REDACT_MIN = 8  # a shorter value (a test key like "k") would shred every log line, so it is never scrubbed
Source = Literal["app", ".env", "env"]
log = logging.getLogger("riya.secrets")


class InvalidKey(ValueError):
    """The key is unusable. The messages never contain the input."""
    EMPTY = "The key is empty."
    TOO_LONG = f"The key is longer than {MAX_KEY_LEN} characters."
    BAD_CHARS = "The key can't contain spaces or control characters."


def default_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya" / "secrets.env"


def validate(raw: str) -> str:
    """The key with its outer whitespace trimmed; InvalidKey when it is empty, too long, or holds any whitespace or
    control character (Unicode category C*)."""
    key = raw.strip()
    if not key:
        raise InvalidKey(InvalidKey.EMPTY)
    if len(key) > MAX_KEY_LEN:
        raise InvalidKey(InvalidKey.TOO_LONG)
    if any(c.isspace() or unicodedata.category(c).startswith("C") for c in key):
        raise InvalidKey(InvalidKey.BAD_CHARS)
    return key


def mask(key: str | None) -> str | None:
    if key is None:
        return None
    return MASK + key[-4:] if len(key) >= REDACT_MIN else MASK


def load(path: Path) -> str | None:
    """The key in the file (its last KEY_VAR= line), or None when the file is missing, unreadable or holds an invalid
    key. Never raises and never logs a value."""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        log.warning("can't read %s (%s)", path, type(e).__name__)
        return None
    raw = None
    for line in lines:
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == KEY_VAR:
            raw = value
    if raw is None:
        return None
    try:
        return validate(raw)
    except InvalidKey as e:
        log.warning("ignoring the key in %s: %s", path, e)
        return None


def save(key: str, path: Path) -> str:
    """Validate and write the key atomically; returns it. InvalidKey or OSError leaves the old file untouched."""
    key = validate(key)
    remember(key)
    write_atomic(Path(path), f"{KEY_VAR}={key}\n".encode("utf-8"))
    return key


def clear(path: Path) -> bool:
    """True when a file was removed."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return False
    return True


# ---- redaction -----------------------------------------------------------------------------------------------------

_lock = threading.Lock()
_remembered: set[str] = set()


def remember(key: str | None) -> None:
    if key:
        with _lock:
            _remembered.add(key)


def forget_all() -> None:
    with _lock:
        _remembered.clear()


def scrub(text: str) -> str:
    """text with every remembered key (of REDACT_MIN or more characters) replaced by REDACTED."""
    with _lock:
        keys = sorted((k for k in _remembered if len(k) >= REDACT_MIN), key=len, reverse=True)
    for k in keys:
        if k in text:
            text = text.replace(k, REDACTED)
    return text


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # a bad format string must not lose the record or leak through its args
            msg = f"{record.msg} {record.args}"
        record.msg, record.args = scrub(msg), None
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True


_state: dict = {}  # filter, factory (ours), previous factory, root; empty when not installed
_state_lock = threading.Lock()


def install_redaction(root: logging.Logger | None = None) -> RedactFilter:
    """Scrub every record: the filter goes on the root logger and each of its handlers, and the log-record factory is
    wrapped so a record from any logger (uvicorn's own handlers, propagate=False) is scrubbed when it is created.
    Idempotent."""
    with _state_lock:
        root = root or logging.getLogger()
        if not _state:
            flt = RedactFilter()
            previous = logging.getLogRecordFactory()

            def factory(*args, **kwargs):
                record = previous(*args, **kwargs)
                if _state:
                    flt.filter(record)
                return record

            logging.setLogRecordFactory(factory)
            _state.update(filter=flt, factory=factory, previous=previous, roots=[])
        flt = _state["filter"]
        if root not in _state["roots"]:
            _state["roots"].append(root)
        for target in (root, *root.handlers):
            target.addFilter(flt)
        return flt


def uninstall_redaction() -> None:
    with _state_lock:
        if not _state:
            return
        flt = _state["filter"]
        if logging.getLogRecordFactory() is _state["factory"]:
            logging.setLogRecordFactory(_state["previous"])
        for root in _state["roots"]:
            root.removeFilter(flt)
            for h in root.handlers:
                h.removeFilter(flt)
        _state.clear()


# ---- which key wins ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyInfo:
    mask: str | None
    source: Source | None

    @property
    def clearable(self) -> bool:
        return self.source == "app"


class OnlineKey:
    """The online key's sources: a key saved in secrets.env ("app") beats the one the repo .env or the process
    environment gave at start (`base`). Clear removes only the saved one."""

    def __init__(self, path: Path, base: str | None, base_source: Literal[".env", "env"] | None):
        self.path, self.base, self.base_source = Path(path), base, base_source

    @classmethod
    def at_start(cls, environ: Mapping[str, str], had_env_key: bool, path: Path) -> OnlineKey:
        """`had_env_key`: the process environment itself had the key before .env was loaded (so it is "env", else ".env")."""
        base = (environ.get(KEY_VAR) or "").strip() or None
        source = None if base is None else "env" if had_env_key else ".env"
        remember(base)
        remember(load(path))
        return cls(path, base, source)

    def current(self) -> tuple[str | None, Source | None]:
        saved = load(self.path)
        if saved is not None:
            remember(saved)
            return saved, "app"
        return self.base, self.base_source

    def info(self) -> KeyInfo:
        key, source = self.current()
        return KeyInfo(mask(key), source)

    def set(self, raw: str) -> str:
        return save(raw, self.path)

    def clear(self) -> bool:
        return clear(self.path)

    def env(self, environ: Mapping[str, str]) -> dict[str, str]:
        out = dict(environ)
        out.pop(KEY_VAR, None)
        key, _ = self.current()
        if key:
            out[KEY_VAR] = key
        return out
