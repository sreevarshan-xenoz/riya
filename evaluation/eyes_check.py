"""Live check of her eyes: open Notepad with known text, grab that window in memory, ask the online vision model.

  python -m evaluation.eyes_check

Needs RIYA_ONLINE_API_KEY (env or .env), otherwise it prints "skipped (no online key)". The screenshot stays in memory
(nothing is written but the small .txt, which is deleted). PASS when the answer to "what does this say?" has 4721.
Notepad is closed and the temp file deleted afterwards. Always exits 0; not part of pytest.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from riya.core.eyes.capture import WindowSource, grab_jpeg, make_window_source, set_dpi_awareness
from riya.core.eyes.see import describe, vision_messages
from riya.core.models import ChatModel, complete, load_dotenv, load_online_from_env

TEXT = "Riya eyes check 4721"
CODE = "4721"
QUESTION = "what does this say?"
VISION_TIMEOUT_S = 20.0  # same as the engine's vision model (riya/core/__main__.py)
ANSWER_MAX_TOKENS = 400  # flash models may spend tokens thinking
FIND_TIMEOUT_S = 10.0
PAINT_S = 1.0  # let the window paint before the grab
SKIP = "skipped (no online key)"
WINDOWS = sys.platform == "win32"


def passed(answer: str) -> bool:
    """The answer to "what does this say?" has the code in it."""
    return CODE in answer


def ask(model: ChatModel, jpeg: bytes) -> tuple[str, bool]:
    """Print the glance note and the answer to the question; return (the answer, whether it passes). Never raises."""
    note, why = describe(model, jpeg)
    print(f"note: {note}" if note is not None else f"note: none ({why})")
    try:
        answer = complete(model, vision_messages(jpeg, QUESTION), max_tokens=ANSWER_MAX_TOKENS)
    except Exception as e:
        print(f"answer: request failed ({type(e).__name__}: {e})")
        return "", False
    print(f"answer: {answer.strip()}")
    return answer, passed(answer)


def wait_for_window(source: WindowSource, user32, title_part: str, timeout_s: float = FIND_TIMEOUT_S,
                    sleep=time.sleep, clock=time.monotonic):
    """The WindowInfo of the window titled with `title_part`, brought to the front; None if it never shows up."""
    deadline, part, hwnd = clock() + timeout_s, title_part.lower(), 0
    while clock() < deadline:
        fg = source.foreground()
        if fg is not None and part in fg.title.lower():
            return fg
        hwnd = next((h for h, t in top_windows(user32) if part in t.lower()), hwnd)
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.keybd_event(0x12, 0, 0, 0)  # a tapped Alt lets a background process take the foreground
            user32.keybd_event(0x12, 0, 2, 0)
            user32.SetForegroundWindow(hwnd)
        sleep(0.3)
    return None


def top_windows(user32) -> list[tuple[int, str]]:
    """(hwnd, title) of the visible top-level windows that have a title."""
    import ctypes
    from ctypes import wintypes

    out = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            out.append((int(hwnd), buf.value))
        return True

    user32.EnumWindows(each, 0)
    return out


def _user32():
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    u.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.IsWindow.argtypes = [wintypes.HWND]
    u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t]
    u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    return u


def close_window(user32, hwnd: int, wait_s: float = 3.0) -> bool:
    """Ask the window to close (WM_CLOSE: Notepad asks to save if there's anything unsaved). Never kills a process.
    True when it closed; otherwise prints a note and leaves it open."""
    user32.PostMessageW(hwnd, 0x0010, 0, 0)
    end = time.monotonic() + wait_s
    while user32.IsWindow(hwnd) and time.monotonic() < end:
        time.sleep(0.1)
    if user32.IsWindow(hwnd):
        print("note: the Notepad window is still open (a save prompt?): left as it is, close it yourself")
        return False
    return True


def cleanup(user32, hwnd: int | None, tmp: Path, close=close_window) -> None:
    """Close our Notepad window (only one that wasn't open before: hwnd is None otherwise) and delete the temp dir.
    Each step is guarded, so nothing here raises."""
    if hwnd:
        try:
            close(user32, hwnd)
        except Exception as e:
            print(f"note: couldn't close Notepad ({type(e).__name__}: {e})")
    for _ in range(10):  # Notepad can hold the file for a moment
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                return
            time.sleep(0.3)
        except Exception:
            return


def live(model: ChatModel, popen=subprocess.Popen) -> bool:
    """Open Notepad on a temp .txt, look at it, ask. True on PASS. Closes the Notepad window it opened (never a window
    that was open before, never by killing a process) and deletes the temp file whatever happens."""
    tmp = Path(tempfile.mkdtemp(prefix="riya_eyes_"))
    user32 = hwnd = None
    ok = False
    try:
        set_dpi_awareness()
        user32, source = _user32(), make_window_source()
        path = tmp / f"riya_eyes_check_{os.getpid()}.txt"
        path.write_text(TEXT, encoding="utf-8")
        before = {h for h, _ in top_windows(user32)}
        popen(["notepad.exe", str(path)])
        info = wait_for_window(source, user32, path.stem)
        if info is None:
            print(f"FAIL: no window titled {path.name} came to the front within {FIND_TIMEOUT_S:.0f} s")
            hwnd = next((h for h, t in top_windows(user32) if path.stem.lower() in t.lower() and h not in before), None)
            return False
        if info.hwnd in before:  # a tab in an already open Notepad: that window is the user's, leave it
            print("note: the file opened in an already open Notepad window, which will be left open")
        else:
            hwnd = info.hwnd
        time.sleep(PAINT_S)
        info = source.window(info.hwnd) or info
        jpeg = grab_jpeg(info)
        if not jpeg:
            print("FAIL: nothing of the window is visible on its monitor")
            return False
        print(f"window: {info.title!r} ({info.process}), {len(jpeg)} bytes in memory")
        _, ok = ask(model, jpeg)
        print("PASS" if ok else f"FAIL: {CODE} is not in the answer")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
    finally:
        cleanup(user32, hwnd, tmp)
    return ok


def main(argv: list[str] | None = None, env=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.eyes_check", description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):  # titles and notes can hold characters the console can't encode
            sys.stdout.reconfigure(errors="replace")
        if env is None:
            load_dotenv()
            env = os.environ
        model = load_online_from_env(env, name="vision", first_token_timeout_s=VISION_TIMEOUT_S)
        if model is None:
            print(SKIP)
        elif not WINDOWS:
            print("skipped (Windows only)")
        else:
            live(model)
    except Exception as e:  # a check script: report, never fail the caller
        print(f"FAIL: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
