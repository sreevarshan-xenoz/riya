import numpy as np

from riya.core.clock import FakeClock
from riya.core.listen import ListenGate, barge_in_ok, mouth_levels


def test_closed_gate_needs_her_name():
    g = ListenGate(FakeClock())
    assert g.accept("what's the weather", gaming=False) is None
    assert g.accept("Riya, what's the weather", gaming=False) == "Riya, what's the weather" and g.is_open()


def test_gate_open_for_two_minutes_after_turn():
    clock = FakeClock()
    g = ListenGate(clock)
    g.open("hotkey")
    clock.advance(119)
    assert g.accept("and tomorrow?", gaming=False) == "and tomorrow?"
    g.note_turn()
    clock.advance(121)
    assert g.accept("random tv noise", gaming=False) is None


def test_gaming_only_hotkey_wakes():
    g = ListenGate(FakeClock())
    assert g.accept("Riya help", gaming=True) is None
    g.open("hotkey")
    assert g.accept("help", gaming=True) == "help"


def test_barge_in_needs_clear_margin():
    assert not barge_in_ok(0.05, 0.04)
    assert barge_in_ok(0.2, 0.04)
    assert not barge_in_ok(0.01, 0.0)


def test_mouth_levels_30fps():
    sr = 24000
    t = np.arange(sr) / sr
    loud = mouth_levels((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    assert len(loud) == 30 and min(loud) > 0.5
    assert max(mouth_levels(np.zeros(sr, dtype=np.float32), sr)) == 0


def test_wake_word_must_be_a_whole_word():
    g = ListenGate(FakeClock())
    assert g.accept("Victoria called Mariya", gaming=False) is None and not g.is_open()
    assert g.accept("hey RIYAA are you there", gaming=False) == "hey RIYAA are you there"
