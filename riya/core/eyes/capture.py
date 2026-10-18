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
    cls: str = ""  # window class (GetClassNameW): tells the desktop and the taskbar from real windows
    ex_style: int = 0  # extended style (GWL_EXSTYLE): click-through overlays
    cloaked: bool = False  # DWM-cloaked (another virtual desktop, a suspended app): not on screen


class WindowSource(Protocol):
    def foreground(self) -> WindowInfo | None: ...

    def window(self, hwnd: int) -> WindowInfo | None:
        """That window as it is now (current title, rect, minimized), or None when it's gone."""


class FakeWindowSource:
    """info is the foreground window; windows are the other open ones (a window seen in front stays open)."""

    def __init__(self, info: WindowInfo | None = None, windows: dict[int, WindowInfo] | None = None):
        self.info, self.windows = info, dict(windows or {})

    def foreground(self) -> WindowInfo | None:
        if self.info is not None:
            self.windows[self.info.hwnd] = self.info
        return self.info

    def window(self, hwnd: int) -> WindowInfo | None:
        if self.info is not None and self.info.hwnd == hwnd:
            return self.info
        return self.windows.get(hwnd)


def is_riya_window(path: str, title: str) -> bool:
    """The Electron body: from the repo's node_modules, or any electron.exe titled "Riya..."."""
    p = path.replace("\\", "/").lower()
    return p.startswith(_ELECTRON_DIR + "/") or (p.rsplit("/", 1)[-1] == "electron.exe" and title.startswith("Riya"))


def clean_title(title: str) -> str:
    """A window title safe to encode: lone surrogates become U+FFFD, emoji stay."""
    return title.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def clamp_rect(rect: Rect, bounds: Rect) -> Rect:
    """`rect` cut to `bounds`; empty (right <= left or bottom <= top) when they don't overlap."""
    return (max(rect[0], bounds[0]), max(rect[1], bounds[1]), min(rect[2], bounds[2]), min(rect[3], bounds[3]))


def _set_dpi() -> None:
    import ctypes

    try:
        f = ctypes.WinDLL("user32", use_last_error=True).SetProcessDpiAwarenessContext
        f.argtypes = [ctypes.c_void_p]
        if f(-4):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return
    except (AttributeError, OSError):
        pass
    ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE


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
        # private handles: argtypes on the shared ctypes.windll objects would be overwritten by other modules
        u, k = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32, self._kernel32, self._dwm = u, k, ctypes.WinDLL("dwmapi", use_last_error=True)

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        self._MONITORINFO = MONITORINFO
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.IsIconic.argtypes = [wintypes.HWND]
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        # 32-bit Python has only GetWindowLongW (the same thing there)
        self._ex_style = getattr(u, "GetWindowLongPtrW", None) or u.GetWindowLongW
        self._ex_style.restype = ctypes.c_ssize_t
        self._ex_style.argtypes = [wintypes.HWND, ctypes.c_int]
        u.IsWindow.argtypes = [wintypes.HWND]
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

    def _cloaked(self, hwnd) -> bool:
        v = self._w.DWORD()
        return self._dwm.DwmGetWindowAttribute(hwnd, 14, self._c.byref(v), 4) == 0 and v.value != 0  # DWMWA_CLOAKED

    def foreground(self) -> WindowInfo | None:
        try:
            return self._info(self._user32.GetForegroundWindow())
        except Exception:
            return None

    def window(self, hwnd: int) -> WindowInfo | None:
        try:
            return self._info(hwnd) if hwnd and self._user32.IsWindow(hwnd) else None
        except Exception:
            return None

    def _info(self, hwnd) -> WindowInfo | None:
        try:
            u, c = self._user32, self._c
            if not hwnd:
                return None
            n = min(max(u.GetWindowTextLengthW(hwnd), 0), 4096) + 1
            buf = c.create_unicode_buffer(n)
            u.GetWindowTextW(hwnd, buf, n)
            title = clean_title(buf.value)
            path = self._image_path(hwnd)
            mon = self._MONITORINFO()
            mon.cbSize = c.sizeof(self._MONITORINFO)
            if not u.GetMonitorInfoW(u.MonitorFromWindow(hwnd, 2), c.byref(mon)):  # MONITOR_DEFAULTTONEAREST
                return None
            m = mon.rcMonitor
            cls = c.create_unicode_buffer(257)
            u.GetClassNameW(hwnd, cls, 257)
            return WindowInfo(int(hwnd), title, path.replace("\\", "/").rsplit("/", 1)[-1], self._frame(hwnd),
                              (m.left, m.top, m.right, m.bottom), bool(u.IsIconic(hwnd)),
                              is_riya_window(path, title), cls.value, self._ex_style(hwnd, -20) & 0xFFFFFFFF,
                              self._cloaked(hwnd))
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

    if not jpeg:
        return b""
    with Image.open(io.BytesIO(jpeg)) as img:
        return img.convert("L").resize((64, 36)).tobytes()


def mad(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two equal-size thumbnails, 0..1 (1.0 when they can't be compared)."""
    if not a or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / (len(a) * 255)
