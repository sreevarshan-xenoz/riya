"""The decision twin: a decayed Beta(alpha, beta) per kind and bucket, guessing whether she's welcome right now. Pure."""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from riya.core.initiative import LONG_STRETCH_S

KINDS = ("bid", "game_offer", "report", "greeting")
PRIOR = 2.0  # alpha = beta = 2: no opinion yet
HALF_LIFE_S = 30 * 86400  # evidence above the prior halves every 30 days
FLOOR = 0.15  # an ignored bucket still fires this often
SHORT_GREET_P = 0.4


@dataclass(frozen=True)
class Cell:
    alpha: float  # engaged + prior
    beta: float  # ignored + prior
    updated_at: float


def bucket(now: float, tz, presence: str, work_stretch_s: float) -> str:
    """'<band>/<presence>/<short|long>' with the band from the hour in tz."""
    h = datetime.fromtimestamp(now, tz).hour
    band = "night" if h >= 22 or h < 5 else "morning" if h < 12 else "afternoon" if h < 17 else "evening"
    return f"{band}/{presence}/{'long' if work_stretch_s > LONG_STRETCH_S else 'short'}"


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def decayed(cell: Cell | None, now: float) -> Cell:
    """The cell as of now: evidence above the prior fades with the time since updated_at."""
    if (cell is None or not (_finite(cell.alpha) and _finite(cell.beta) and _finite(cell.updated_at))
            or cell.alpha + cell.beta <= 0):
        return Cell(PRIOR, PRIOR, now)  # nothing yet, or a corrupt cell (alpha, beta or time): read as the prior
    f = 0.5 ** (max(0.0, now - cell.updated_at) / HALF_LIFE_S)
    return Cell(PRIOR + (cell.alpha - PRIOR) * f, PRIOR + (cell.beta - PRIOR) * f, now)


def updated(cell: Cell | None, engaged: bool, now: float) -> Cell:
    c = decayed(cell, now)
    return Cell(c.alpha + engaged, c.beta + (not engaged), now)


def mean(cell: Cell | None, now: float) -> float:
    c = decayed(cell, now)
    return c.alpha / (c.alpha + c.beta)


def evidence(cell: Cell | None, now: float) -> float:
    c = decayed(cell, now)
    return c.alpha + c.beta - 2 * PRIOR


def chance(cell: Cell | None, now: float) -> float:
    """Fresh or enjoyed (p >= 0.5) keeps the normal rate; an ignored bucket falls toward the floor."""
    return max(FLOOR, min(1.0, mean(cell, now) / 0.5))


class CellStore(Protocol):
    def get_cell(self, kind: str, bucket: str) -> Cell | None: ...
    def put_cell(self, kind: str, bucket: str, cell: Cell) -> None: ...


class MemoryCells:
    def __init__(self):
        self.cells: dict[tuple[str, str], Cell] = {}

    def get_cell(self, kind, bucket):
        return self.cells.get((kind, bucket))

    def put_cell(self, kind, bucket, cell):
        self.cells[(kind, bucket)] = cell


class Twin:
    def __init__(self, store: CellStore, clock):
        self.store, self.clock = store, clock

    def record(self, kind: str, bucket: str, engaged: bool) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        now = self.clock.now()
        self.store.put_cell(kind, bucket, updated(self.store.get_cell(kind, bucket), engaged, now))

    def p_engage(self, kind: str, bucket: str) -> float:
        return mean(self.store.get_cell(kind, bucket), self.clock.now())

    def chance(self, kind: str, bucket: str) -> float:
        return chance(self.store.get_cell(kind, bucket), self.clock.now())

    def should(self, kind: str, bucket: str, rng) -> bool:
        return rng.random() < self.chance(kind, bucket)

    def greet_style(self, bucket: str) -> Literal["short", "normal"]:
        return "short" if self.p_engage("greeting", bucket) < SHORT_GREET_P else "normal"
