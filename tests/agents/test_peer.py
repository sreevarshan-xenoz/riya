"""v0.9 Task 2: the peer PID of a hook (by its TCP row), its window (by process chain and folder), and the focus
helper. Everything goes through FakePeerOS / a fake user32: no test touches a real Windows API."""
import sys

import pytest

from riya.agents import hooks, peer
from riya.agents.peer import (MAX_CHAIN, MAX_PEERS, FakePeerOS, Peer, PeerMap, Proc, TcpRow, TopWindow, owner_pid,
                              parent_chain, pick_window, window_for_pid)
from tests.fakes import make_core


def procs(*rows):
    return {p.pid: p for p in (Proc(*r) for r in rows)}


def test_owner_pid_is_the_client_side_row():
    rows = [TcpRow(8766, 50000, 111), TcpRow(50000, 8766, 4242)]
    os_ = FakePeerOS(rows=rows)
    assert owner_pid(50000, 8766, os_, own_pid=111) == 4242
    assert owner_pid(50001, 8766, os_, own_pid=111) is None
    assert owner_pid(50002, 8766, FakePeerOS(rows=[TcpRow(50002, 8766, 0)]), own_pid=111) is None
    assert owner_pid(50003, 8766, FakePeerOS(rows=[TcpRow(50003, 8766, 111)]), own_pid=111) is None
    assert os_.calls == ["tcp_rows"] * 2


def test_owner_pid_skips_closing_ownerless_and_own_rows_before_the_live_one():
    rows = [TcpRow(50000, 8766, 4242, state=11), TcpRow(50000, 8766, 0), TcpRow(50000, 8766, 111),
            TcpRow(50000, 8766, 4242)]
    assert owner_pid(50000, 8766, FakePeerOS(rows=rows), own_pid=111) == 4242
    assert owner_pid(50000, 8766, FakePeerOS(rows=rows[:3]), own_pid=111) is None


def test_parent_chain_stops_safely():
    table = procs((4242, 4100, "node.exe"), (4100, 4050, "pwsh.exe"), (4050, 4000, "openconsole.exe"),
                  (4000, 10, "windowsterminal.exe"), (10, 1, "explorer.exe"))
    assert parent_chain(4242, table) == [4242, 4100, 4050, 4000]
    assert parent_chain(10, table) == []
    assert parent_chain(4242, procs((4242, 4100, "a"), (4100, 4242, "b"))) == [4242, 4100]  # a cycle stops
    assert parent_chain(4242, procs((4242, 99, "a"))) == [4242]  # a missing parent stops
    assert parent_chain(77, {}) == []
    assert parent_chain(0, table) == [] and parent_chain(-3, table) == []
    long = procs(*[(i, i + 1, "p.exe") for i in range(1, 60)])
    assert parent_chain(1, long) == list(range(1, MAX_CHAIN + 1))


def test_window_choice_stays_in_the_agents_chain():
    table = procs((4242, 4100, "node.exe"), (4100, 4050, "pwsh.exe"), (4050, 4000, "openconsole.exe"),
                  (4000, 10, "windowsterminal.exe"), (10, 1, "explorer.exe"))
    wins = [TopWindow(5, 10, "riya - File Explorer"), TopWindow(77, 4000, "riya \u2014 Windows Terminal"),
            TopWindow(78, 4000, "other")]
    assert window_for_pid(4242, "RIYA", FakePeerOS(procs=table.values(), wins=wins)) == 77
    assert window_for_pid(4242, "nomatch", FakePeerOS(procs=table.values(), wins=wins)) == 77
    assert window_for_pid(4242, "", FakePeerOS(procs=table.values(), wins=wins[::-1])) == 78  # first in z-order
    # the second window of the terminal wins on a title match
    wins2 = [TopWindow(78, 4000, "other"), TopWindow(77, 4000, "riya - pwsh")]
    assert window_for_pid(4242, "riya", FakePeerOS(procs=table.values(), wins=wins2)) == 77
    # a classic console: the conhost child owns the window
    cmd = procs((5100, 5000, "node.exe"), (5000, 10, "cmd.exe"), (5001, 5000, "conhost.exe"), (10, 1, "explorer.exe"))
    cw = [TopWindow(90, 5001, "C:\\riya")]
    assert window_for_pid(5100, "riya", FakePeerOS(procs=cmd.values(), wins=cw)) == 90
    # a console host that is not a child of a chain process is not counted
    stray = procs((5100, 5000, "node.exe"), (5000, 10, "cmd.exe"), (6001, 7, "conhost.exe"))
    assert window_for_pid(5100, "riya", FakePeerOS(procs=stray.values(), wins=[TopWindow(91, 6001, "riya")])) is None
    # VS Code
    code = procs((6100, 6050, "node.exe"), (6050, 6000, "node.exe"), (6000, 10, "code.exe"), (10, 1, "explorer.exe"))
    vw = [TopWindow(1, 6000, "notes - Visual Studio Code"), TopWindow(2, 6000, "engine.py - riya - Visual Studio Code")]
    assert window_for_pid(6100, "riya", FakePeerOS(procs=code.values(), wins=vw)) == 2
    # no windows anywhere, and pid 0
    assert window_for_pid(6100, "riya", FakePeerOS(procs=code.values())) is None
    assert window_for_pid(0, "riya", FakePeerOS(procs=code.values(), wins=vw)) is None
    assert pick_window([], {}, vw, "riya") is None


def test_the_first_process_with_a_window_decides():
    table = procs((1, 2, "node.exe"), (2, 3, "pwsh.exe"), (3, 10, "wt.exe"), (10, 0, "explorer.exe"))
    wins = [TopWindow(30, 3, "terminal"), TopWindow(20, 2, "pwsh")]
    assert pick_window(parent_chain(1, table), table, wins, "terminal") == 20


def test_peer_map_caches_per_session():
    asked = []

    def lookup(port):
        asked.append(port)
        return 4242

    m = PeerMap(lookup)
    m.note("s1", 50000, "E:/riya")
    for i in range(9):
        m.note("s1", 50001 + i, "E:/riya")
    assert asked == [50000] and m.get("s1") == Peer(4242, "riya")
    m.note("s1", 50009, "C:\\work\\other")  # a cached session only updates its folder
    assert asked == [50000] and m.get("s1") == Peer(4242, "other")
    m.note("s1", 50010, None)
    assert m.get("s1") == Peer(4242, "other")
    m.note("s1", 50011, "E:/riya", fresh=True)
    assert asked == [50000, 50011] and m.get("s1") == Peer(4242, "riya")
    assert m.get("nope") is None


def test_peer_map_misses_and_errors_store_nothing_and_retry():
    answers = iter([None, RuntimeError("boom"), 77])

    def lookup(port):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    m = PeerMap(lookup)
    m.note("s", 1, "E:/riya")
    m.note("s", 2, "E:/riya")
    assert m.get("s") is None
    m.note("s", 3, "E:/riya")
    assert m.get("s") == Peer(77, "riya")
    z = PeerMap(lambda p: 0)
    z.note("s", 1)
    assert z.get("s") is None


def test_peer_map_folder_is_bounded_and_the_cap_is_max_peers():
    m = PeerMap(lambda p: 5)
    m.note("a", 1, "C:\\x\\" + "d" * 1000)
    assert len(m.get("a").folder) == 260
    m.note("b", 1, 12345)  # not a str: no folder
    assert m.get("b") == Peer(5, "")
    m2 = PeerMap(lambda p: 5)
    for i in range(MAX_PEERS + 20):
        m2.note(f"s{i}", 1, "E:/riya")
    assert len(m2._peers) == MAX_PEERS and m2.get("s0") is None and m2.get(f"s{MAX_PEERS + 19}") is not None


def test_peer_map_for_os_uses_the_client_side_row():
    os_ = FakePeerOS(rows=[TcpRow(8766, 50000, 111), TcpRow(50000, 8766, 4242)])
    m = PeerMap.for_os(os_)
    m.note("s", 50000, "E:/riya")
    assert m.get("s") == Peer(4242, "riya")


def test_the_port_matches_the_receivers():
    assert peer.SERVER_PORT == hooks.HOOK_PORT


def test_a_test_core_never_gets_the_windows_peer_os():
    core = make_core()[0]
    assert isinstance(core.peer_os, FakePeerOS) and isinstance(core.peers, PeerMap)


def test_fake_peer_os_records_calls_copies_and_raises():
    rows = [TcpRow(1, 2, 3)]
    os_ = FakePeerOS(rows=rows, procs=[Proc(1, 0, "a")], wins=[TopWindow(9, 1, "t")])
    assert os_.tcp_rows() is not os_.tcp_rows() and os_.processes() == {1: Proc(1, 0, "a")}
    assert os_.windows() == [TopWindow(9, 1, "t")]
    assert os_.calls == ["tcp_rows", "tcp_rows", "processes", "windows"]
    os_.raises = {"windows"}
    with pytest.raises(OSError):
        os_.windows()


def test_default_peer_os_is_real_only_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert isinstance(peer.default_peer_os(), FakePeerOS)


class FakeUser32:
    def __init__(self, window=True, visible=True, iconic=False, fg=True):
        self.window, self.visible, self.iconic, self.fg, self.calls = window, visible, iconic, fg, []

    def IsWindow(self, h):
        self.calls.append(("IsWindow", h))
        return self.window

    def IsWindowVisible(self, h):
        self.calls.append(("IsWindowVisible", h))
        return self.visible

    def IsIconic(self, h):
        self.calls.append(("IsIconic", h))
        return self.iconic

    def ShowWindow(self, h, cmd):
        self.calls.append(("ShowWindow", h, cmd))
        return 1

    def SetForegroundWindow(self, h):
        self.calls.append(("SetForegroundWindow", h))
        return self.fg


def test_focus_restores_then_foregrounds_only_a_visible_window():
    u = FakeUser32(iconic=True)
    assert peer.focus(77, u) is True
    shown = [c for c in u.calls if c[0] in ("ShowWindow", "SetForegroundWindow")]
    assert shown == [("ShowWindow", 77, 9), ("SetForegroundWindow", 77)]
    u = FakeUser32()
    assert peer.focus(77, u) is True and not any(c[0] == "ShowWindow" for c in u.calls)
    assert peer.focus(77, FakeUser32(fg=False)) is False
    for u in (FakeUser32(window=False), FakeUser32(visible=False)):
        assert peer.focus(77, u) is False
        assert not any(c[0] in ("SetForegroundWindow", "ShowWindow") for c in u.calls)


def test_cli(capsys):
    u = FakeUser32()
    assert peer.main(["focus", "77"], user32=u) == 0
    assert ("SetForegroundWindow", 77) in u.calls
    assert peer.main(["focus", "5"], user32=FakeUser32(fg=False)) == 1
    for argv in (["focus", "x"], ["focus", "-1"], ["focus"], ["kill", "77"], [], ["focus", "0"],
                 ["focus", str(2**63)], ["focus", "7", "8"], ["focus", "+7"], ["focus", "1_0"]):
        u = FakeUser32()
        assert peer.main(argv, user32=u) == 2 and u.calls == []
    assert "usage" in capsys.readouterr().err
    assert peer.main(["focus", str(2**63 - 1)], user32=FakeUser32()) == 0


def test_fake_peer_os_created():
    os_ = FakePeerOS(procs=[Proc(1, 0, "a"), Proc(2, 1, "b")], created={2: 7.5, 9: 3.0})
    assert os_.created(1) == 0.0  # listed without a time: alive at 0.0
    assert os_.created(2) == 7.5 and os_.created(9) == 3.0
    assert os_.created(77) is None
    assert os_.calls == ["created"] * 4
    os_.raises = {"created"}
    with pytest.raises(OSError):
        os_.created(1)


def test_parent_chain_stops_at_a_missing_or_younger_parent():
    table = procs((4242, 4100, "node.exe"), (4100, 4000, "pwsh.exe"), (4000, 10, "wt.exe"), (10, 1, "explorer.exe"))
    alive = FakePeerOS(procs=table.values(), created={4242: 300.0, 4100: 200.0, 4000: 100.0})
    assert parent_chain(4242, table, alive.created) == [4242, 4100, 4000]
    # a parent born after its child is a reused pid (its real parent is gone): the chain stops before it
    reused = FakePeerOS(procs=table.values(), created={4242: 300.0, 4100: 200.0, 4000: 900.0})
    assert parent_chain(4242, table, reused.created) == [4242, 4100]
    younger = FakePeerOS(procs=table.values(), created={4242: 300.0, 4100: 301.0})
    assert parent_chain(4242, table, younger.created) == [4242]
    # equal times (no times known) never stop it
    assert parent_chain(4242, table, FakePeerOS(procs=table.values()).created) == [4242, 4100, 4000]
    # the pid itself or a parent that is not running (no time) stops it
    gone = FakePeerOS(procs=[Proc(4242, 4100, "node.exe"), Proc(4100, 4000, "pwsh.exe")])
    assert parent_chain(4242, table, gone.created) == [4242, 4100]
    assert parent_chain(4242, table, FakePeerOS().created) == []
    assert parent_chain(4242, table) == [4242, 4100, 4000]  # no created: as before


def test_window_for_pid_ignores_a_reused_parents_window():
    table = [Proc(4242, 4000, "node.exe"), Proc(4000, 10, "windowsterminal.exe"), Proc(10, 0, "explorer.exe")]
    wins = [TopWindow(77, 4000, "riya - Windows Terminal")]
    fine = FakePeerOS(procs=table, wins=wins, created={4242: 100.0, 4000: 50.0})
    assert window_for_pid(4242, "riya", fine) == 77
    reused = FakePeerOS(procs=table, wins=wins, created={4242: 100.0, 4000: 500.0})
    assert window_for_pid(4242, "riya", reused) is None


def test_peer_alive_needs_the_same_process():
    from riya.agents.peer import peer_alive
    os_ = FakePeerOS(procs=[Proc(5, 0, "a")], created={5: 10.0})
    assert peer_alive(Peer(5, "riya", 10.0), os_) is True
    assert peer_alive(Peer(5, "riya", 11.0), os_) is False  # reused: another creation time
    assert peer_alive(Peer(5, "riya"), os_) is True  # no time noted: running is all that can be checked
    assert peer_alive(Peer(6, "riya"), os_) is False and peer_alive(Peer(6, "riya", 1.0), os_) is False


def test_peer_map_stores_the_creation_time_and_keeps_it_on_a_folder_refresh():
    os_ = FakePeerOS(rows=[TcpRow(50000, 8766, 4242)], procs=[Proc(4242, 0, "node.exe")], created={4242: 12.5})
    m = PeerMap.for_os(os_)
    m.note("s", 50000, "E:/riya")
    assert m.get("s") == Peer(4242, "riya", 12.5)
    m.note("s", 50000, "E:/other")  # a cached session only refreshes its folder
    assert m.get("s") == Peer(4242, "other", 12.5)
    broken = PeerMap(lambda p: 7, lambda pid: (_ for _ in ()).throw(OSError("no")))
    broken.note("t", 1, "E:/riya")
    assert broken.get("t") == Peer(7, "riya", None)  # a failed time read never loses the peer


def test_an_unreadable_time_is_walked_past_without_a_comparison():
    """Access denied: the process runs but its creation time can't be read (UNKNOWN_TIME)."""
    from riya.agents.peer import UNKNOWN_TIME, known, peer_alive
    assert not known(UNKNOWN_TIME) and not known(None) and known(0.0)
    table = procs((4242, 4100, "node.exe"), (4100, 4000, "pwsh.exe"), (4000, 10, "wt.exe"), (10, 1, "explorer.exe"))
    denied = FakePeerOS(procs=table.values(), created={4242: 300.0, 4100: UNKNOWN_TIME, 4000: 100.0})
    assert parent_chain(4242, table, denied.created) == [4242, 4100, 4000]
    # the next process is still compared with the last known time (the child's)
    young = FakePeerOS(procs=table.values(), created={4242: 300.0, 4100: UNKNOWN_TIME, 4000: 900.0})
    assert parent_chain(4242, table, young.created) == [4242, 4100]
    # the agent itself unreadable: still walked
    me = FakePeerOS(procs=table.values(), created={4242: UNKNOWN_TIME, 4100: 200.0, 4000: 100.0})
    assert parent_chain(4242, table, me.created) == [4242, 4100, 4000]
    wins = [TopWindow(77, 4000, "riya - Windows Terminal")]
    walked = FakePeerOS(procs=table.values(), wins=wins, created={4100: UNKNOWN_TIME, 4242: 300.0, 4000: 100.0})
    assert window_for_pid(4242, "riya", walked) == 77
    # peer_alive: running is all that can be checked
    os_ = FakePeerOS(procs=[Proc(5, 0, "a")], created={5: UNKNOWN_TIME})
    assert peer_alive(Peer(5, "riya", 10.0), os_) is True and peer_alive(Peer(6, "riya", 10.0), os_) is False
    assert peer_alive(Peer(5, "riya", UNKNOWN_TIME), FakePeerOS(procs=[Proc(5, 0, "a")], created={5: 3.0})) is True
    # an unreadable time is stored as "not noted"
    m = PeerMap.for_os(FakePeerOS(rows=[TcpRow(50000, 8766, 5)], procs=[Proc(5, 0, "a")], created={5: UNKNOWN_TIME}))
    m.note("s", 50000, "E:/riya")
    assert m.get("s") == Peer(5, "riya", None)
