"""OS effects behind one interface: open a URL or app, media keys, the clipboard and a command (argv, shell=False).

`WindowsEffects` is the real thing; `FakeEffects` records calls for tests. Nothing here runs at import time, and the
Windows DLLs are only touched inside the methods. Tool output is untrusted: `output_tail` redacts it, then keeps the end.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path
from ctypes import wintypes
from typing import Protocol

from riya.agents.hub import redact

from .safe import format_number, valid_url

__all__ = ["Effects", "WindowsEffects", "FakeEffects", "VOLUME_PRESSES", "TAIL_MAX", "key_inputs", "output_tail"]

VOLUME_PRESSES = 5
TAIL_MAX = 200
_TAIL_SCAN = 200_000  # the most output chars looked at before the cut
KEYS = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD}
MAX_PRESSES = 20
INPUT_KEYBOARD, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 1, 0x0001, 0x0002
CREATE_NO_WINDOW, CREATE_NEW_PROCESS_GROUP = 0x08000000, 0x00000200
CONTINUATION = bytes(range(0x80, 0xC0))  # UTF-8 continuation bytes
OUTPUT_READ_MAX = 64 * 1024  # the most command output ever read back (the end of it)
KILL_WAIT_S = 5
CLIPBOARD_TRIES = 5
CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002
ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):  # unused, but it sets the union's size (INPUT is 40 bytes on 64-bit)
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]


def key_inputs(vk: int, presses: int = 1) -> list[INPUT]:
    """`presses` key-down/key-up pairs for virtual key `vk`, every event flagged KEYEVENTF_EXTENDEDKEY."""
    out = []
    for _ in range(presses):
        for flags in (KEYEVENTF_EXTENDEDKEY, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP):
            out.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)))
    return out


def SendInput(inputs: list[INPUT]) -> None:
    """Send the events with user32.SendInput; OSError if Windows didn't take them all."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    arr = (INPUT * len(inputs))(*inputs)
    if user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT)) != len(arr):
        raise OSError("SendInput was blocked")


def output_tail(text, n: int = TAIL_MAX) -> str:
    """The last n chars of a command's output as one clean line, after the v0.4 redaction of the whole of it."""
    s = redact((text if isinstance(text, str) else str(text))[-_TAIL_SCAN:])
    s = "".join(" " if ch.isspace() or unicodedata.category(ch) in ("Zl", "Zp") else ch
                for ch in s if not unicodedata.category(ch).startswith("C") or ch.isspace())
    return re.sub(r" {2,}", " ", s).strip()[-n:].lstrip() if n > 0 else ""


class Effects(Protocol):
    def open_url(self, url: str) -> None: ...
    def open_app(self, target: str) -> None: ...
    def media_key(self, key: str, presses: int = 1) -> None: ...
    def copy(self, text: str) -> None: ...
    def run(self, argv: list[str], cwd: str | None, timeout_s: float) -> tuple[int | None, str]: ...


def _oem_codec() -> str:
    """The console's OEM code page (what redirected console tools mostly write), e.g. "cp437"."""
    try:
        return f"cp{ctypes.WinDLL('kernel32').GetOEMCP()}"
    except (OSError, AttributeError):
        return "cp437"


def _decode(b) -> str:
    """Command output as text: strict UTF-8 if it is valid UTF-8, else the console OEM code page (with replacement)."""
    if isinstance(b, str):
        return b
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return b.decode(_oem_codec(), "replace")
    except LookupError:
        return b.decode("utf-8", "replace")


def _kill_tree(p) -> None:
    """Kill the child Riya spawned and everything it started (taskkill /T /F on that pid, fixed argv, no shell)."""
    exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
    try:
        subprocess.run([str(exe), "/T", "/F", "/PID", str(p.pid)], shell=False, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=KILL_WAIT_S * 2,
                       creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        p.kill()  # at least the child itself, if taskkill failed
    except OSError:
        pass
    try:
        p.wait(timeout=KILL_WAIT_S)
    except subprocess.SubprocessError:
        pass


class WindowsEffects:
    """The real effects (Windows only). Tests construct it or inspect `key_inputs`; they never call these methods."""

    def open_url(self, url: str) -> None:
        if valid_url(url) != url:  # the gate already checked: never hand startfile anything but a plain http(s) URL
            raise ValueError("not an http(s) URL")
        os.startfile(url)

    def open_app(self, target: str) -> None:
        os.startfile(target)

    def media_key(self, key: str, presses: int = 1) -> None:
        if key not in KEYS or not 1 <= presses <= MAX_PRESSES:
            raise ValueError("unknown key or press count")
        SendInput(key_inputs(KEYS[key], presses))

    def copy(self, text: str) -> None:
        """Replace the clipboard with `text` (CF_UNICODETEXT); OSError on failure."""
        text = re.sub(r"\r?\n", "\r\n", text.replace("\x00", ""))
        data = text.encode("utf-16-le", "replace") + b"\x00\x00"
        u, k = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
        u.GetDesktopWindow.restype = wintypes.HWND  # an owner window: with a NULL owner SetClipboardData can fail
        u.OpenClipboard.argtypes, u.OpenClipboard.restype = (wintypes.HWND,), wintypes.BOOL
        u.SetClipboardData.argtypes, u.SetClipboardData.restype = (wintypes.UINT, wintypes.HANDLE), wintypes.HANDLE
        k.GlobalAlloc.argtypes, k.GlobalAlloc.restype = (wintypes.UINT, ctypes.c_size_t), wintypes.HGLOBAL
        k.GlobalLock.argtypes, k.GlobalLock.restype = (wintypes.HGLOBAL,), wintypes.LPVOID
        k.GlobalUnlock.argtypes, k.GlobalFree.argtypes = (wintypes.HGLOBAL,), (wintypes.HGLOBAL,)
        for _ in range(CLIPBOARD_TRIES):  # another app may hold it for a moment
            if u.OpenClipboard(u.GetDesktopWindow()):
                break
            time.sleep(0.05)
        else:
            raise OSError("the clipboard is busy")
        try:
            if not u.EmptyClipboard():
                raise OSError("couldn't clear the clipboard")
            h = k.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not h:
                raise OSError("out of memory")
            p = k.GlobalLock(h)
            if not p:
                k.GlobalFree(h)
                raise OSError("couldn't lock memory")
            ctypes.memmove(p, data, len(data))
            k.GlobalUnlock(h)
            if not u.SetClipboardData(CF_UNICODETEXT, h):
                k.GlobalFree(h)  # on success the clipboard owns the memory
                raise OSError("couldn't set the clipboard")
        finally:
            u.CloseClipboard()

    def run(self, argv: list[str], cwd: str | None, timeout_s: float) -> tuple[int | None, str]:
        """(exit code, the last 64 KB of merged output); (None, reason) on a timeout or when it couldn't start.

        Never a shell. Output goes to a temp file, not a pipe, so a grandchild holding the handle can't block us; on a
        timeout the child's whole tree is killed and the wait is bounded.
        """
        try:
            with tempfile.TemporaryFile() as out:
                p = subprocess.Popen(argv, shell=False, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=subprocess.STDOUT, creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
                try:
                    code = p.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    _kill_tree(p)
                    return None, f"timed out after {format_number(timeout_s)} s"
                out.seek(0, os.SEEK_END)
                start = max(0, out.tell() - OUTPUT_READ_MAX)
                out.seek(start)
                data = out.read()
                return code, _decode(data.lstrip(CONTINUATION) if start else data)  # a cut may split a UTF-8 char
        except (OSError, ValueError):
            return None, "couldn't start"


class FakeEffects:
    """Records every call in `.calls`; `run` returns `run_result`. `fail` (an exception or any truthy value) makes
    every call raise after it's recorded."""

    def __init__(self, run_result: tuple[int | None, str] = (0, "ok"), fail=None):
        self.run_result, self.fail, self.calls = run_result, fail, []

    def _do(self, *call) -> None:
        self.calls.append(call)
        f = self.fail
        if isinstance(f, BaseException) or (isinstance(f, type) and issubclass(f, BaseException)):
            raise f
        if f:
            raise OSError(str(f))

    def open_url(self, url: str) -> None:
        self._do("open_url", url)

    def open_app(self, target: str) -> None:
        self._do("open_app", target)

    def media_key(self, key: str, presses: int = 1) -> None:
        self._do("media_key", key, presses)

    def copy(self, text: str) -> None:
        self._do("copy", text)

    def run(self, argv: list[str], cwd: str | None, timeout_s: float) -> tuple[int | None, str]:
        self._do("run", list(argv), cwd, timeout_s)
        return self.run_result
