"""Capture: the Windows foreground window as JPEG bytes. Pixels stay in memory, never on disk."""
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

Rect = tuple[int, int, int, int]
REPO_ROOT = Path(__file__).resolve().parents[3]
_ELECTRON_DIR = (REPO_ROOT / "body" / "node_modules" / "electron").as_posix().lower()
_dpi_done = False


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    process: str
    rect: Rect
    monitor: Rect
    minimized: bool
    is_riya: bool


class WindowSource(Protocol):
    def foreground(self) -> WindowInfo | None: ...


class FakeWindowSource:
    def __init__(self, info: WindowInfo | None = None):
        self.info = info

    def foreground(self) -> WindowInfo | None:
        return self.info


def is_riya_window(path: str, title: str) -> bool:
    """The Electron body: from the repo's node_modules, or any electron.exe titled "Riya..."."""
    p = path.replace("\\", "/").lower()
    return p.startswith(_ELECTRON_DIR + "/") or (p.rsplit("/", 1)[-1] == "electron.exe" and title.startswith("Riya"))


def clamp_rect(rect: Rect, bounds: Rect) -> Rect:
    """`rect` cut to `bounds`; empty (right <= left or bottom <= top) when they don't overlap."""
    return (max(rect[0], bounds[0]), max(rect[1], bounds[1]), min(rect[2], bounds[2]), min(rect[3], bounds[3]))


def _set_dpi() -> None:
    import ctypes

    try:
        f = ctypes.windll.user32.SetProcessDpiAwarenessContext
        f.argtypes = [ctypes.c_void_p]
        if f(-4):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return
    except (AttributeError, OSError):
        pass
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE


def set_dpi_awareness() -> None:
    """Per-monitor DPI awareness, once, so window rects and pixels agree. Never raises."""
    global _dpi_done
    if _dpi_done or sys.platform != "win32":
        return
    _dpi_done = True
    try:
        _set_dpi()
    except Exception:  # already set, or an older Windows: capture still works
        pass


class WindowsWindowSource:
    """The foreground window via user32, dwmapi and kernel32 (ctypes)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._c, self._w = ctypes, wintypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        self._user32, self._kernel32, self._dwm = u, k, ctypes.windll.dwmapi

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        self._MONITORINFO = MONITORINFO
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.IsIconic.argtypes = [wintypes.HWND]
        u.MonitorFromWindow.restype = wintypes.HANDLE
        u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    def _image_path(self, hwnd) -> str:
        c, k = self._c, self._kernel32
        pid = self._w.DWORD()
        self._user32.GetWindowThreadProcessId(hwnd, c.byref(pid))
        handle = k.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf, size = c.create_unicode_buffer(1024), self._w.DWORD(1024)
            return buf.value if k.QueryFullProcessImageNameW(handle, 0, buf, c.byref(size)) else ""
        finally:
            k.CloseHandle(handle)

    def _frame(self, hwnd) -> Rect:
        r = self._w.RECT()
        if self._dwm.DwmGetWindowAttribute(hwnd, 9, self._c.byref(r), self._c.sizeof(r)) != 0:  # EXTENDED_FRAME_BOUNDS
            if not self._user32.GetWindowRect(hwnd, self._c.byref(r)):
                return (0, 0, 0, 0)
        return (r.left, r.top, r.right, r.bottom)

    def foreground(self) -> WindowInfo | None:
        try:
            u, c = self._user32, self._c
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return None
            title = c.create_unicode_buffer(512)
            u.GetWindowTextW(hwnd, title, 512)
            path = self._image_path(hwnd)
            mon = self._MONITORINFO()
            mon.cbSize = c.sizeof(self._MONITORINFO)
            if not u.GetMonitorInfoW(u.MonitorFromWindow(hwnd, 2), c.byref(mon)):  # MONITOR_DEFAULTTONEAREST
                return None
            m = mon.rcMonitor
            return WindowInfo(int(hwnd), title.value, path.replace("\\", "/").rsplit("/", 1)[-1], self._frame(hwnd),
                              (m.left, m.top, m.right, m.bottom), bool(u.IsIconic(hwnd)),
                              is_riya_window(path, title.value))
        except Exception:
            return None


def make_window_source() -> WindowSource:
    return WindowsWindowSource() if sys.platform == "win32" else FakeWindowSource()


def grab_jpeg(info: WindowInfo, max_side: int = 1280, quality: int = 70) -> bytes:
    """The window's visible region as JPEG bytes (long side <= max_side); b"" when nothing of it is on its monitor."""
    left, top, right, bottom = clamp_rect(info.rect, info.monitor)
    if right <= left or bottom <= top:
        return b""
    import mss  # lazy: core tests import this module without mss installed
    from PIL import Image

    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality)
    return out.getvalue()


def thumb(jpeg: bytes) -> bytes:
    """A 64x36 grayscale raw thumbnail, for telling whether the screen changed."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as img:
        return img.convert("L").resize((64, 36)).tobytes()


def mad(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two equal-size thumbnails, 0..1 (1.0 when they can't be compared)."""
    if not a or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / (len(a) * 255)
