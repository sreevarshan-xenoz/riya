import builtins
import io
import os
import sys
import types

import pytest
from PIL import Image

from riya.core.eyes import capture
from riya.core.eyes.capture import (FakeWindowSource, WindowInfo, WindowsWindowSource, clamp_rect, grab_jpeg, mad,
                                    set_dpi_awareness, thumb)

MON = (0, 0, 1920, 1080)


def info(rect=(100, 100, 400, 300), monitor=MON, **kw):
    base = dict(hwnd=1, title="t", process="p.exe", rect=rect, monitor=monitor, minimized=False, is_riya=False)
    return WindowInfo(**{**base, **kw})


def mss_stub(monkeypatch, grabs):
    """An mss module whose screen is a flat 300x200 RGB buffer; `grabs` records each requested region."""

    class Shot:
        def __init__(self, w, h):
            self.size, self.width, self.height = (w, h), w, h
            self.rgb = bytes([200, 100, 50]) * (w * h)

    class Sct:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def grab(self, region):
            grabs.append(region)
            return Shot(region["width"], region["height"])

    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=lambda: Sct()))


def test_clamp_inside_is_unchanged():
    assert clamp_rect((100, 100, 400, 300), MON) == (100, 100, 400, 300)


def test_clamp_rect_partly_off_monitor():
    assert clamp_rect((-50, -20, 300, 200), MON) == (0, 0, 300, 200)
    assert clamp_rect((1800, 1000, 2100, 1300), MON) == (1800, 1000, 1920, 1080)


def test_clamp_second_monitor_and_fully_off():
    second = (1920, 0, 3840, 1080)
    assert clamp_rect((1800, 100, 2200, 500), second) == (1920, 100, 2200, 500)
    left, top, right, bottom = clamp_rect((5000, 0, 5200, 100), MON)
    assert right <= left or bottom <= top


def test_grab_returns_jpeg_of_clamped_region(monkeypatch):
    grabs = []
    mss_stub(monkeypatch, grabs)
    data = grab_jpeg(info(rect=(-50, -20, 300, 200)))
    assert data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9"
    assert grabs == [{"left": 0, "top": 0, "width": 300, "height": 200}]
    assert Image.open(io.BytesIO(data)).size == (300, 200)


def test_grab_downscales_long_side(monkeypatch):
    mss_stub(monkeypatch, [])
    data = grab_jpeg(info(rect=(0, 0, 1900, 1000)), max_side=1280)
    assert max(Image.open(io.BytesIO(data)).size) == 1280


def test_grab_empty_region_is_empty_bytes(monkeypatch):
    grabs = []
    mss_stub(monkeypatch, grabs)
    assert grab_jpeg(info(rect=(5000, 0, 5200, 100))) == b"" and grabs == []
    assert grab_jpeg(info(rect=(10, 10, 10, 50))) == b""


def test_grab_never_touches_disk(monkeypatch):
    mss_stub(monkeypatch, [])
    real_save = Image.Image.save

    def boom(*a, **k):
        raise AssertionError("disk write")

    def save(self, fp, *a, **k):
        if isinstance(fp, (str, bytes, os.PathLike)):
            raise AssertionError("save to a path")
        return real_save(self, fp, *a, **k)

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(Image.Image, "save", save)
    monkeypatch.setattr(os, "replace", boom)
    assert grab_jpeg(info())[:2] == b"\xff\xd8"


def jpeg(color, size=(200, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_thumb_is_64x36_grayscale_raw():
    t = thumb(jpeg((255, 255, 255)))
    assert len(t) == 64 * 36 and min(t) >= 250


def test_mad_range():
    white, black = thumb(jpeg((255, 255, 255))), thumb(jpeg((0, 0, 0)))
    assert mad(white, white) == 0.0
    assert mad(white, black) > 0.97
    assert mad(black, white) == mad(white, black)
    half = thumb(jpeg((128, 128, 128)))
    assert 0.45 < mad(black, half) < 0.55


def test_mad_mismatched_or_empty_is_maximal():
    assert mad(b"", b"") == 1.0 and mad(b"\x00" * 4, b"\x00" * 5) == 1.0


def test_fake_window_source():
    i = info()
    assert FakeWindowSource(i).foreground() is i and FakeWindowSource(None).foreground() is None


def test_fake_window_source_reads_a_window_by_handle():
    a, b = info(hwnd=1), info(hwnd=2, title="other")
    src = FakeWindowSource(a, windows={2: b})
    assert src.window(1) is a and src.window(2) is b and src.window(3) is None
    src.foreground()
    src.info = b
    assert src.window(1) is a                     # a window seen in front stays open until it's removed
    del src.windows[1]
    assert src.window(1) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_window_by_handle_matches_the_foreground():
    src = WindowsWindowSource()
    w = src.foreground()
    if w is not None:
        again = src.window(w.hwnd)
        assert again is not None and (again.hwnd, again.title, again.process) == (w.hwnd, w.title, w.process)
    assert src.window(0) is None and src.window(0x7FFFFFF0) is None


def test_dpi_awareness_never_raises_and_is_idempotent():
    set_dpi_awareness()
    set_dpi_awareness()


def test_dpi_awareness_swallows_failures(monkeypatch):
    monkeypatch.setattr(capture, "_dpi_done", False)
    monkeypatch.setattr(capture.sys, "platform", "win32")
    monkeypatch.setattr(capture, "_set_dpi", lambda: (_ for _ in ()).throw(OSError("nope")))
    set_dpi_awareness()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_foreground_does_not_raise():
    w = WindowsWindowSource().foreground()
    assert w is None or (isinstance(w, WindowInfo) and len(w.rect) == 4 and len(w.monitor) == 4)


def test_is_riya_rules():
    root = capture.REPO_ROOT
    electron = str(root / "body" / "node_modules" / "electron" / "dist" / "electron.exe")
    assert capture.is_riya_window(electron, "anything")
    assert capture.is_riya_window(r"C:\x\electron.exe", "Riya")
    assert capture.is_riya_window(r"C:\x\Electron.EXE", "Riya - overlay")
    assert not capture.is_riya_window(r"C:\x\electron.exe", "VS Code")
    assert not capture.is_riya_window(r"C:\x\chrome.exe", "Riya")


def test_clean_title_lone_surrogate_and_emoji():
    assert capture.clean_title("a\ud83dz") == "a\ufffdz"
    assert capture.clean_title("Riya \U0001F600 \u65e5\u672c") == "Riya \U0001F600 \u65e5\u672c"
    capture.clean_title("a\ud83dz").encode("utf-8")


def test_thumb_of_empty_bytes_is_empty():
    assert thumb(b"") == b""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
@pytest.mark.parametrize("window_first", [True, False])
def test_window_and_idle_sources_do_not_clobber_each_other(window_first):
    from riya.core.presence import WindowsIdleSource

    if window_first:
        w, idle = WindowsWindowSource(), WindowsIdleSource()
    else:
        idle, w = WindowsIdleSource(), WindowsWindowSource()
    assert w.foreground() is not None
    assert isinstance(idle.fullscreen(), bool)


def test_window_info_defaults_keep_old_constructions():
    w = WindowInfo(1, "t", "p.exe", (0, 0, 1, 1), MON, False, False)
    assert (w.cls, w.ex_style, w.cloaked) == ("", 0, False)


def _find(cls):
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    u.FindWindowW.restype = wintypes.HWND
    u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    return u.FindWindowW(cls, None)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
@pytest.mark.parametrize("cls", ["Progman", "Shell_TrayWnd"])
def test_windows_desktop_and_taskbar_are_refused_as_no_window(cls):
    """Read-only: the real desktop and taskbar windows, by class (nothing is created, moved or focused)."""
    from riya.core.clock import FakeClock
    from riya.core.eyes.policy import GlancePolicy

    hwnd = _find(cls)
    if not hwnd:
        pytest.skip(f"no {cls} window (no explorer shell)")
    w = WindowsWindowSource().window(hwnd)
    assert w is not None and w.cls == cls and w.process.lower() == "explorer.exe"
    assert GlancePolicy(FakeClock()).allowed(info=w, presence="focused", eyes_enabled=True, dont_look=False,
                                             online_ok=True, force=True) == "no window"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_info_reads_class_and_ex_style():
    w = WindowsWindowSource().foreground()
    if w is not None:
        assert isinstance(w.cls, str) and w.cls and isinstance(w.ex_style, int) and w.ex_style >= 0


def test_fake_window_source_stacks():
    a, b = info(hwnd=1), info(hwnd=2)
    src = FakeWindowSource(a, stacks={1: [b]})
    assert src.above(1) == [b] and src.above(2) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_above_walks_up_the_real_z_order():
    """Read-only: the windows above the real foreground window and above the desktop."""
    src = WindowsWindowSource()
    fg = src.foreground()
    if fg is not None:
        above = src.above(fg.hwnd)
        assert all(isinstance(w, WindowInfo) and w.hwnd != fg.hwnd for w in above)
    desk = _find("Progman")
    if desk:
        assert len(src.above(desk)) >= 1                   # at least the taskbar sits above the desktop
    with pytest.raises(RuntimeError):
        src.above(0x7FFFFFF0)                              # not a window: it can't tell
