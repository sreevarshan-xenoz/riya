"""The agent's window: which process sent a hook (the owner of its TCP connection), which top-level window belongs to
that process chain, and a tiny helper that brings a window forward (`python -m riya.agents.peer focus <hwnd>`).
stdlib only; ctypes is touched only inside WindowsPeerOS and _user32. FakePeerOS lives here because it is the core's
default off Windows and in tests, so no test can reach the real Windows calls."""
import os
import re
import socket
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import PureWindowsPath
from typing import Protocol

SERVER_PORT = 8766  # the hook receiver's port (a test keeps it equal to hooks.HOOK_PORT)
MAX_PEERS = 200  # sessions remembered at most; the oldest is dropped
MAX_CHAIN = 16  # parents walked at most
SW_RESTORE = 9
MAX_FOLDER = 260
MAX_CWD = 512  # only the tail of a cwd is parsed
MAX_TITLE = 512
MAX_WINDOWS = 4000
STOP_PROCESSES = frozenset({"explorer.exe", "svchost.exe", "services.exe", "wininit.exe", "winlogon.exe",
                            "csrss.exe", "smss.exe", "system"})
CONSOLE_HOSTS = frozenset({"conhost.exe", "openconsole.exe"})
ESTABLISHED = 5  # MIB_TCP_STATE_ESTAB
TCP_HINT = 16 * 1024  # the first guess at the TCP table's size
UNKNOWN_TIME = float("nan")  # PeerOS.created for a process that runs but can't be read (access denied): no comparison
STILL_ACTIVE = 259  # GetExitCodeProcess of a running process
ERROR_ACCESS_DENIED = 5


def known(t: float | None) -> bool:
    """A real creation time (not None, not UNKNOWN_TIME)."""
    return t is not None and t == t


@dataclass(frozen=True)
class TcpRow:
    local_port: int
    remote_port: int
    pid: int
    state: int = ESTABLISHED


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    name: str  # lowercased


@dataclass(frozen=True)
class TopWindow:
    hwnd: int
    pid: int
    title: str


@dataclass(frozen=True)
class Peer:
    pid: int
    folder: str
    created: float | None = None  # the process's creation time when it was noted: a reused pid has another one


class PeerOS(Protocol):
    def tcp_rows(self) -> list[TcpRow]:
        """Established IPv4 loopback connections only (the receiver binds only 127.0.0.1)."""

    def processes(self) -> dict[int, Proc]: ...

    def windows(self) -> list[TopWindow]:
        """Visible, unowned, uncloaked top-level windows in z-order, topmost first."""

    def created(self, pid: int) -> float | None:
        """The process creation time (seconds, any fixed epoch); None when that pid isn't running; UNKNOWN_TIME when it
        runs but its time can't be read (access denied), so a walk passes it without comparing."""


class FakePeerOS:
    """A PeerOS from plain data: the core's default off Windows and in tests. calls records each method used; names in
    raises make that method raise OSError. created maps pid -> creation time; a pid in procs without one is alive at
    0.0, and a pid in neither is not running. A time of UNKNOWN_TIME stands for a running process whose time is
    unreadable."""

    def __init__(self, rows: Iterable[TcpRow] = (), procs: Iterable[Proc] = (), wins: Iterable[TopWindow] = (),
                 created: Mapping[int, float] | None = None):
        self.rows, self.procs, self.wins = list(rows), list(procs), list(wins)
        self.times: dict[int, float] = dict(created or {})
        self.calls: list[str] = []
        self.raises: set[str] = set()

    def _use(self, name: str) -> None:
        self.calls.append(name)
        if name in self.raises:
            raise OSError(f"fake {name} failed")

    def tcp_rows(self) -> list[TcpRow]:
        self._use("tcp_rows")
        return list(self.rows)

    def processes(self) -> dict[int, Proc]:
        self._use("processes")
        return {p.pid: p for p in self.procs}

    def windows(self) -> list[TopWindow]:
        self._use("windows")
        return list(self.wins)

    def created(self, pid: int) -> float | None:
        self._use("created")
        if pid in self.times:
            return self.times[pid]
        return 0.0 if any(p.pid == pid for p in self.procs) else None


class WindowsPeerOS:
    """The real tables (ctypes, private WinDLL handles as in presence.WindowsIdleSource). Read-only."""

    AF_INET, TCP_TABLE_OWNER_PID_ALL = 2, 5
    LOOPBACK = 0x0100007F  # 127.0.0.1 as a little-endian DWORD
    TH32CS_SNAPPROCESS, GW_OWNER, DWMWA_CLOAKED = 2, 4, 14
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._c, self._w = ctypes, wintypes
        self._iphlp = ctypes.WinDLL("iphlpapi", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._dwm = ctypes.WinDLL("dwmapi", use_last_error=True)

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", wintypes.WCHAR * 260)]

        self._ENTRY = PROCESSENTRY32W
        self._hint = TCP_HINT  # the last table size: one kernel call per lookup, not two
        self._ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        i, k, u, d = self._iphlp, self._kernel32, self._user32, self._dwm
        i.GetExtendedTcpTable.restype = wintypes.ULONG
        i.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
                                          wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
        k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.Process32FirstW.restype = k.Process32NextW.restype = wintypes.BOOL
        k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.OpenProcess.restype = ctypes.c_void_p
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.GetProcessTimes.restype = wintypes.BOOL
        k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        u.EnumWindows.argtypes = [self._ENUMPROC, wintypes.LPARAM]
        u.IsWindowVisible.restype = wintypes.BOOL
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.GetWindow.restype = wintypes.HWND
        u.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        d.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    def tcp_rows(self) -> list[TcpRow]:
        import struct
        c, w = self._c, self._w
        size = w.DWORD(self._hint)
        for _ in range(4):  # the table can grow between the sizing call and the read
            buf = c.create_string_buffer(size.value)
            err = self._iphlp.GetExtendedTcpTable(buf, c.byref(size), False, self.AF_INET,
                                                  self.TCP_TABLE_OWNER_PID_ALL, 0)
            if err == 0:
                break
            if err != 122:  # ERROR_INSUFFICIENT_BUFFER: retry with the size it asked for
                raise OSError(f"GetExtendedTcpTable failed ({err})")
        else:
            raise OSError("GetExtendedTcpTable never settled")
        self._hint = max(TCP_HINT, len(buf))
        (n,) = struct.unpack_from("<I", buf, 0)
        n = min(n, (len(buf) - 4) // 24)
        rows = []
        for j in range(n):  # MIB_TCPROW_OWNER_PID: state, laddr, lport, raddr, rport, pid (DWORDs)
            state, laddr, lport, raddr, rport, pid = struct.unpack_from("<6I", buf, 4 + 24 * j)
            if state == ESTABLISHED and laddr == self.LOOPBACK and raddr == self.LOOPBACK:
                rows.append(TcpRow(socket.ntohs(lport & 0xFFFF), socket.ntohs(rport & 0xFFFF), pid, state))
        return rows

    def processes(self) -> dict[int, Proc]:
        c, k = self._c, self._kernel32
        snap = k.CreateToolhelp32Snapshot(self.TH32CS_SNAPPROCESS, 0)
        if snap in (None, c.c_void_p(-1).value):
            raise OSError("CreateToolhelp32Snapshot failed")
        out: dict[int, Proc] = {}
        try:
            e = self._ENTRY()
            e.dwSize = c.sizeof(e)
            ok = k.Process32FirstW(snap, c.byref(e))
            while ok:
                out[e.th32ProcessID] = Proc(e.th32ProcessID, e.th32ParentProcessID, e.szExeFile.lower())
                ok = k.Process32NextW(snap, c.byref(e))
        finally:
            k.CloseHandle(snap)
        return out

    def windows(self) -> list[TopWindow]:
        c, w, u = self._c, self._w, self._user32
        found: list[TopWindow] = []

        def visit(hwnd, _lparam):
            try:
                if not hwnd or not u.IsWindowVisible(hwnd) or u.GetWindow(hwnd, self.GW_OWNER):
                    return True
                cloaked = c.c_int(0)
                if self._dwm.DwmGetWindowAttribute(hwnd, self.DWMWA_CLOAKED, c.byref(cloaked), c.sizeof(cloaked)) == 0 \
                        and cloaked.value:
                    return True
                n = u.GetWindowTextLengthW(hwnd)
                if n <= 0:
                    return True
                text = c.create_unicode_buffer(min(n, MAX_TITLE) + 1)
                u.GetWindowTextW(hwnd, text, len(text))
                if not text.value:
                    return True
                pid = w.DWORD(0)
                u.GetWindowThreadProcessId(hwnd, c.byref(pid))
                found.append(TopWindow(int(hwnd), pid.value, text.value))
            except Exception:  # one odd window must not end the walk
                pass
            return len(found) < MAX_WINDOWS

        cb = self._ENUMPROC(visit)  # kept alive for the call
        u.EnumWindows(cb, 0)
        return found


    def created(self, pid: int) -> float | None:
        """The creation time of a running process (FILETIME seconds); None when the pid isn't running; UNKNOWN_TIME when
        it exists but can't be opened (access denied: a protected or other-user process). A process that has exited
        isn't running even while a handle keeps it listed (GetExitCodeProcess != STILL_ACTIVE)."""
        c, w, k = self._c, self._w, self._kernel32
        if pid <= 0:
            return None
        handle = k.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return UNKNOWN_TIME if c.get_last_error() == ERROR_ACCESS_DENIED else None
        try:
            code = w.DWORD(0)
            if k.GetExitCodeProcess(handle, c.byref(code)) and code.value != STILL_ACTIVE:
                return None
            made, gone, kern, user = (w.FILETIME() for _ in range(4))
            if not k.GetProcessTimes(handle, c.byref(made), c.byref(gone), c.byref(kern), c.byref(user)):
                return UNKNOWN_TIME
            return ((made.dwHighDateTime << 32) | made.dwLowDateTime) / 1e7
        finally:
            k.CloseHandle(handle)


def default_peer_os() -> PeerOS:
    return WindowsPeerOS() if sys.platform == "win32" else FakePeerOS()


def owner_pid(local_port: int, remote_port: int, os_: PeerOS, own_pid: int | None = None) -> int | None:
    """The pid owning the connection row with that local and remote port. At a hook that is the client's side
    (local = request.client.port, remote = SERVER_PORT); the server-side row is the core's own pid, which is refused."""
    me = own_pid or os.getpid()
    for r in os_.tcp_rows():
        if r.local_port == local_port and r.remote_port == remote_port and r.state == ESTABLISHED \
                and r.pid > 0 and r.pid != me:  # a closing or ownerless row never hides the live one
            return r.pid
    return None


def parent_chain(pid: int, procs: Mapping[int, Proc], created: Callable[[int], float | None] | None = None) -> list[int]:
    """[pid, parent, ...], stopping at a missing pid, a repeat, a pid <= 0, MAX_CHAIN entries, or the first system or
    shell process (explorer and friends), which is excluded. With created (pid -> creation time), it also stops at a
    process that is not running or was created after its child: a parent born later than its child is a reused pid. A
    process with UNKNOWN_TIME is walked past, and the next one is compared with the last known time."""
    chain: list[int] = []
    child_at: float | None = None
    while len(chain) < MAX_CHAIN and pid > 0 and pid not in chain:
        p = procs.get(pid)
        if p is None or p.name in STOP_PROCESSES:
            break
        if created is not None:
            at = created(pid)
            if at is None or (known(at) and child_at is not None and at > child_at):
                break
            if known(at):
                child_at = at
        chain.append(pid)
        pid = p.ppid
    return chain


def pick_window(chain: list[int], procs: Mapping[int, Proc], windows: list[TopWindow], folder: str) -> int | None:
    """The first process in the chain that has a window decides. Its windows are those it owns plus those owned by its
    direct console-host children, in z-order. Among them: the first whose title holds folder, else the first."""
    want = folder.casefold() if folder and folder.strip() else ""
    for pid in chain:
        owners = {pid} | {p.pid for p in procs.values() if p.ppid == pid and p.name in CONSOLE_HOSTS}
        mine = [w for w in windows if w.pid in owners]
        if mine:
            if want:
                for w in mine:
                    if want in w.title.casefold():
                        return w.hwnd
            return mine[0].hwnd
    return None


def window_for_pid(pid: int, folder: str, os_: PeerOS) -> int | None:
    if pid <= 0:
        return None
    procs = os_.processes()
    return pick_window(parent_chain(pid, procs, os_.created), procs, os_.windows(), folder)


def peer_alive(p: Peer, os_: PeerOS) -> bool:
    """The agent a Peer names is still that process: running, and (when its creation time was noted) created at that
    same time, so a pid that was reused since the hook fired never counts. An unreadable time on either side can't be
    compared: running is all that can be checked."""
    at = os_.created(p.pid)
    return at is not None and (not known(at) or not known(p.created) or at == p.created)


def _folder_of(cwd: str) -> str:
    return PureWindowsPath(cwd[-MAX_CWD:]).name[:MAX_FOLDER]


class PeerMap:
    """session -> Peer (the agent's pid and its cwd's folder name), thread-safe. The receiver notes every hook; a
    cached session only refreshes its folder, so the table lookup runs once per session (or per SessionStart)."""

    def __init__(self, lookup: Callable[[int], int | None], created: Callable[[int], float | None] | None = None):
        self.lookup = lookup
        self.created = created  # pid -> creation time, stored with the pid (for peer_alive at click time)
        self._lock = threading.Lock()
        self._peers: dict[str, Peer] = {}

    @classmethod
    def for_os(cls, os_: PeerOS) -> "PeerMap":
        return cls(lambda port: owner_pid(port, SERVER_PORT, os_), os_.created)

    def note(self, session: str, client_port: int, cwd: str | None = None, fresh: bool = False) -> None:
        with self._lock:
            cur = self._peers.get(session)
            if cur is not None and not fresh:
                if isinstance(cwd, str):
                    self._peers[session] = replace(cur, folder=_folder_of(cwd))
                return
            folder = _folder_of(cwd) if isinstance(cwd, str) else (cur.folder if cur else "")
        try:
            pid = self.lookup(client_port)
        except Exception:
            pid = None
        if not pid:
            return
        try:
            born = self.created(pid) if self.created is not None else None
        except Exception:
            born = None
        if not known(born):
            born = None  # an unreadable time is stored as "not noted"
        with self._lock:
            self._peers.pop(session, None)  # re-stored last: the oldest is the first key
            self._peers[session] = Peer(pid, folder, born)
            while len(self._peers) > MAX_PEERS:
                del self._peers[next(iter(self._peers))]

    def get(self, session: str) -> Peer | None:
        with self._lock:
            return self._peers.get(session)


def _user32():
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    for name in ("IsWindow", "IsWindowVisible", "IsIconic", "SetForegroundWindow"):
        fn = getattr(u, name)
        fn.restype, fn.argtypes = wintypes.BOOL, [wintypes.HWND]
    u.ShowWindow.restype, u.ShowWindow.argtypes = wintypes.BOOL, [wintypes.HWND, ctypes.c_int]
    return u


def focus(hwnd: int, user32=None) -> bool:
    """Brings hwnd forward: a window that is gone or hidden is refused, a minimised one is restored first."""
    u = user32 or _user32()
    if not u.IsWindow(hwnd) or not u.IsWindowVisible(hwnd):
        return False
    if u.IsIconic(hwnd):
        u.ShowWindow(hwnd, SW_RESTORE)
    return bool(u.SetForegroundWindow(hwnd))


def main(argv: list[str] | None = None, user32=None) -> int:
    """`focus <hwnd>` (a decimal int in 1..2**63-1): 0 when the window came forward, 1 when not, 2 on bad usage."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "focus" or not re.fullmatch(r"[0-9]{1,19}", argv[1]) \
            or not 1 <= int(argv[1]) <= 2**63 - 1:
        print("usage: python -m riya.agents.peer focus <hwnd>", file=sys.stderr)
        return 2
    return 0 if focus(int(argv[1]), user32) else 1


if __name__ == "__main__":
    sys.exit(main())
