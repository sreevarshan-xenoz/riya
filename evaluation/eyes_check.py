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
SKIP = "skipped (no online key)"


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


def close_window(user32, hwnd: int, wait_s: float = 3.0) -> None:
    """Ask the window to close (WM_CLOSE); kill its process if it's still there after wait_s."""
    import ctypes
    from ctypes import wintypes

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    user32.PostMessageW(hwnd, 0x0010, 0, 0)
    end = time.monotonic() + wait_s
    while user32.IsWindow(hwnd) and time.monotonic() < end:
        time.sleep(0.1)
    if user32.IsWindow(hwnd) and pid.value:
        subprocess.run(["taskkill", "/PID", str(pid.value), "/F"], capture_output=True, check=False)


def live(model: ChatModel) -> bool:
    """Open Notepad on a temp .txt, look at it, ask. True on PASS. Cleans up Notepad and the file whatever happens."""
    set_dpi_awareness()
    user32, source = _user32(), make_window_source()
    tmp = Path(tempfile.mkdtemp(prefix="riya_eyes_"))
    proc = hwnd = None
    ok = False
    try:
        path = tmp / f"riya_eyes_check_{os.getpid()}.txt"
        path.write_text(TEXT, encoding="utf-8")
        before = {h for h, _ in top_windows(user32)}
        proc = subprocess.Popen(["notepad.exe", str(path)])
        info = wait_for_window(source, user32, path.stem)
        if info is None:
            print(f"FAIL: no window titled {path.name} came to the front within {FIND_TIMEOUT_S:.0f} s")
            hwnd = next((h for h, t in top_windows(user32) if path.stem.lower() in t.lower()), None)
            return False
        hwnd = info.hwnd if info.hwnd not in before else None  # a tab in an already open Notepad: leave that window
        time.sleep(1.0)  # let it paint
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
        if hwnd:
            close_window(user32, hwnd)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        for _ in range(10):  # Notepad can hold the file for a moment
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.3)
    return ok


def main(argv: list[str] | None = None, env=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.eyes_check", description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # window titles and notes can hold characters the console can't encode
        sys.stdout.reconfigure(errors="replace")
    if env is None:
        load_dotenv()
        env = os.environ
    model = load_online_from_env(env, name="vision", first_token_timeout_s=VISION_TIMEOUT_S)
    if model is None:
        print(SKIP)
    elif sys.platform != "win32":
        print("skipped (Windows only)")
    else:
        live(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
