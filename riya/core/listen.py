"""When she listens (her name, the hotkey or a recent turn), echo-safe barge-in, and mouth levels for the body."""
import re

import numpy as np

from riya.core.presence import Clock

WAKE_WORDS = ("riya", "ria", "reya", "riyaa")
LISTEN_WINDOW_S = 120
MOUTH_FPS = 30
MOUTH_FULL_RMS = 0.15
_WAKE_RE = re.compile(r"\b(?:" + "|".join(WAKE_WORDS) + r")\b", re.IGNORECASE)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


class ListenGate:
    """Closed: only her name (never while gaming) or an explicit open() lets speech through. Open: everything does,
    until open_window_s pass without a turn."""

    def __init__(self, clock: Clock, open_window_s: float = LISTEN_WINDOW_S):
        self.clock, self.window = clock, open_window_s
        self.until = float("-inf")
        self.reason: str | None = None

    def open(self, reason: str) -> None:
        self.reason, self.until = reason, self.clock.now() + self.window

    def is_open(self) -> bool:
        return self.clock.now() < self.until

    def note_turn(self) -> None:
        self.until = self.clock.now() + self.window

    def accept(self, transcript: str, gaming: bool) -> str | None:
        if self.is_open():
            return transcript
        if not gaming and _WAKE_RE.search(transcript):
            self.open("name")
            return transcript
        return None


def barge_in_ok(mic_rms: float, playback_rms: float, ratio: float = 2.5, floor: float = 0.02) -> bool:
    """The user must be clearly louder than her own playback picked up by the mic."""
    return mic_rms > max(ratio * playback_rms, floor)


def mouth_level(window: np.ndarray) -> float:
    return min(1.0, rms(window) / MOUTH_FULL_RMS)


def mouth_levels(chunk: np.ndarray, sr: int, fps: int = MOUTH_FPS) -> list[float]:
    """One mouth-openness level in [0, 1] per full 1/fps window of audio."""
    win = sr // fps
    return [mouth_level(chunk[i:i + win]) for i in range(0, len(chunk) - win + 1, win)]
