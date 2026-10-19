"""Safe evaluators for tool input: arithmetic (never `eval`), URLs and file names. Pure; nothing here touches the OS."""
from __future__ import annotations

import ast
import math
import operator
import re
from pathlib import Path
from urllib.parse import urlsplit

MAX_EXPR, MAX_EXP, MAX_VALUE = 200, 100, 1e100
MAX_URL, MAX_NAME = 2000, 80
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _num(v):
    """v if it's a finite real int or float within bounds, else raise."""
    if type(v) not in (int, float) or not math.isfinite(v) or abs(v) > MAX_VALUE:
        raise ValueError
    return v


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _num(_UN[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_EXP:
            raise ValueError
        return _num(_BIN[type(node.op)](a, b))
    raise ValueError


def calc(expr) -> int | float | None:
    """Evaluate plain arithmetic (+ - * / // % ** ^ x × ÷, parentheses); None for anything else, never `eval`."""
    if not isinstance(expr, str) or len(expr) > MAX_EXPR:
        return None
    s = expr.strip().replace("^", "**").replace("÷", "/")
    s = re.sub(r"(?<=[\d)])\s*[x×]\s*(?=[\d(.])", " * ", s)
    s = s.replace("×", "*")
    try:
        return _eval(ast.parse(s, mode="eval"))
    except (ValueError, SyntaxError, ArithmeticError, RecursionError, MemoryError, TypeError):
        return None


def format_number(x) -> str:
    """84.0 -> '84', 3.5 -> '3.5'; floats get at most 10 significant digits."""
    if isinstance(x, int):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.10g}"


# ---- urls ----
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_HOST = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?$")
_URL_BAD = re.compile(r"[^\x21-\x7e]|[\\\"<>`]")  # anything but printable non-space ASCII, plus quote-ish characters


def valid_url(text) -> str | None:
    """An http(s) URL to open: a bare domain gets https://; None for other schemes, UNC paths, userinfo or odd input."""
    if not isinstance(text, str) or not text or len(text) > MAX_URL or _URL_BAD.search(text) or text.startswith("//"):
        return None
    explicit = re.match(r"^https?://", text, re.I) is not None
    if not explicit and (_SCHEME.match(text) and not re.match(r"^[^/:]+:\d+(?:/|$)", text) or "://" in text):
        return None  # javascript:, file:, mailto: ...
    url = text if explicit else "https://" + text
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname, parts.port
    except ValueError:
        return None
    if not host or "@" in parts.netloc or not _HOST.match(host) or (not explicit and "." not in host):
        return None
    return url


# ---- file names ----
_NAME_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.I)
_EXTS = (".txt", ".md", ".log", ".csv", ".json")


def safe_filename(name) -> str | None:
    """A plain Windows-safe file name of at most 80 chars: no paths, reserved names or odd extensions; None if empty."""
    if not isinstance(name, str):
        return None
    s = _NAME_BAD.sub("", name).strip(" .")
    if not s:
        return None
    e = next((e for e in _EXTS if s.lower().endswith(e)), None)
    stem, ext = (s[:-len(e)], s[-len(e):]) if e else (s, ".txt")
    stem = stem.rstrip(" .")
    if not stem:
        return None
    if _RESERVED.match(stem.split(".")[0].rstrip()):
        stem = "_" + stem
    stem = stem[:MAX_NAME - len(ext)].rstrip(" .")
    return stem + ext if stem else None


def unique_path(folder, name) -> Path | None:
    """folder/name, or 'stem (2).ext', 'stem (3).ext' ... for the first that doesn't exist; None when none is free."""
    folder, p = Path(folder), Path(name)
    if p.name != name:
        return None
    for i in range(1, 1000):
        cand = folder / (name if i == 1 else f"{p.stem} ({i}){p.suffix}")
        if not cand.exists():
            return cand
    return None
