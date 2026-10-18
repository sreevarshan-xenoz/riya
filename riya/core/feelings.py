"""Riya's feelings: momentary affect, slow mood, missing-you, an estimate of the user's mood, and the face she shows.

Layers: affect (valence/arousal/concern/surprise) decays toward mood within minutes; mood follows affect over hours;
missing is set when the user returns after an absence and fades while they are together; the body (energy) drains
with continuous talk and recovers during breaks; expression is derived from all of it plus short-lived overrides.
"""
import math
import re
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from riya.core.appraise import NEG, POS, appraise
from riya.core.protocol import EXPRESSIONS  # noqa: F401  (re-exported)

AFFECT_HALF_LIFE_S = 300
MOOD_TAU_S = 6 * 3600
MISSING_HALF_LIFE_S = 240
MAX_TURN_DELTA = 0.35
MOOD_BASELINE = 0.1
MOOD_RETURN_TAU_S = 3 * 86400  # mood drifts back to baseline over days
USER_MOOD_HALF_LIFE_S = 1800
BASE_AROUSAL, BASE_CURIOSITY, BASE_WARMTH, BASE_PLAYFULNESS = 0.3, 0.5, 0.6, 0.4
_MISSING_POINTS = ((0, 0.0), (2, 0.10), (24, 0.50), (48, 0.85))


def missing_curve(hours: float) -> float:
    if hours >= 48:
        return 0.85 + 0.15 * (1 - math.exp(-(hours - 48) / 48))
    for (h0, m0), (h1, m1) in zip(_MISSING_POINTS, _MISSING_POINTS[1:]):
        if hours <= h1:
            return m0 + (m1 - m0) * (max(hours, h0) - h0) / (h1 - h0)
    return 0.0


# Valence and mood are coupled linear ODEs: v' = -k(v - m), m' = (v - m)/MOOD_TAU - (m - base)/MOOD_RETURN_TAU.
# Their deviation from MOOD_BASELINE evolves as exp(A*dt), solved once by eigendecomposition so that one big tick
# gives exactly the same result as many small ones.
_K = math.log(2) / AFFECT_HALF_LIFE_S
_A = np.array([[-_K, _K], [1 / MOOD_TAU_S, -1 / MOOD_TAU_S - 1 / MOOD_RETURN_TAU_S]])
_EIGVALS, _EIGVECS = np.linalg.eig(_A)
_EIGINV = np.linalg.inv(_EIGVECS)


def _valence_mood_step(valence: float, mood: float, dt: float) -> tuple[float, float]:
    y = _EIGVECS @ (np.exp(_EIGVALS * dt) * (_EIGINV @ np.array([valence - MOOD_BASELINE, mood - MOOD_BASELINE])))
    return float(y[0].real) + MOOD_BASELINE, float(y[1].real) + MOOD_BASELINE


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _toward(x: float, target: float, factor: float) -> float:
    return target + (x - target) * factor


@dataclass
class UserMood:
    valence: float = 0.0
    confidence: float = 0.0
    label: str = "unknown"


@dataclass
class FeelingState:
    valence: float = 0.1
    arousal: float = 0.3
    warmth: float = 0.6
    concern: float = 0.0
    playfulness: float = 0.4
    surprise: float = 0.0
    mood: float = 0.1
    missing: float = 0.0
    energy: float = 0.8
    curiosity: float = 0.5
    user_mood: UserMood = field(default_factory=UserMood)


def _user_label(valence: float) -> str:
    if valence <= -0.5:
        return "down"
    if valence < -0.15:
        return "a bit off"
    if valence >= 0.5:
        return "upbeat"
    if valence > 0.15:
        return "good"
    return "neutral"


class Feelings:
    def __init__(self, clock, traits: dict[str, float]):
        self.clock, self.traits = clock, traits
        self.state = FeelingState()
        self.prev_words: set[str] = set()
        self._last = clock.now()  # time of the last tick (or of the save, after a restore)
        self._override: tuple[str, float | None] | None = None  # (expression, expires_at or None)

    # ---- events -------------------------------------------------------------------------------------
    def _bump(self, name: str, delta: float, lo: float = -1.0, hi: float = 1.0, cap: float = MAX_TURN_DELTA) -> None:
        setattr(self.state, name, _clip(getattr(self.state, name) + _clip(delta, -cap, cap), lo, hi))

    def _set_override(self, name: str, seconds: float | None) -> None:
        self._override = (name, None if seconds is None else self.clock.now() + seconds)

    def on_return(self, away_s: float) -> None:
        s = self.state
        s.missing = min(1.0, missing_curve(max(0.0, away_s) / 3600) * self.traits.get("clinginess", 0.8) / 0.8)
        cap = MAX_TURN_DELTA * 2
        self._bump("valence", 0.4 * s.missing, cap=cap)
        self._bump("arousal", 0.3 * s.missing, 0.0, 1.0, cap=cap)
        if away_s > 300:  # rested while they were away
            s.energy = max(s.energy, min(0.9, s.energy + 0.02 * away_s / 60))
        self._set_override("missed_you" if s.missing >= 0.4 else "happy", 20)

    def estimate_user(self, text: str) -> None:
        words = re.findall(r"[a-z']+", text.lower())
        n_affect = sum(w in POS or w in NEG for w in words)
        if not n_affect:
            return
        valence, _, _ = appraise(text, set())
        old = self.state.user_mood
        if old.confidence > 0:
            valence = 0.7 * valence + 0.3 * old.valence
        self.state.user_mood = UserMood(valence, min(0.8, 0.3 + 0.1 * n_affect), _user_label(valence))

    def on_user_text(self, text: str) -> None:
        self.estimate_user(text)
        s = self.state
        valence, arousal, novelty = appraise(text, self.prev_words)
        self.prev_words = set(re.findall(r"[a-z']+", text.lower()))
        self._bump("valence", 0.4 * (valence - s.valence))
        self._bump("arousal", 0.3 * (arousal - s.arousal), 0.0, 1.0)
        um = s.user_mood
        if um.valence < 0:  # empathy
            self._bump("concern", 0.5 * -um.valence * um.confidence, 0.0, 1.0)
        warm = 0.1 * self.traits.get("sweetness", 0.5)
        self._bump("warmth", warm if valence >= 0 else -warm, 0.0, 1.0)
        self._bump("surprise", 0.4 * arousal * novelty, 0.0, 1.0)
        self._bump("curiosity", 0.1 * novelty, 0.0, 1.0)
        s.energy = _clip(s.energy - 0.01, 0.0, 1.0)  # continuous talk is tiring

    def on_event(self, name: str) -> None:
        if name == "bid_ignored":
            self._set_override("pouty", 30)
        elif name == "teased":
            self._set_override("teasing", 10)
        elif name == "thinking":
            self._set_override("thinking", None)
        elif name == "thinking_done" and self._override and self._override[0] == "thinking":
            self._override = None

    # ---- time ---------------------------------------------------------------------------------------
    def tick(self, present: bool) -> None:
        now = self.clock.now()
        dt, self._last = max(0.0, now - self._last), now
        if not dt:
            return
        s = self.state
        f = 0.5 ** (dt / AFFECT_HALF_LIFE_S)
        s.valence, s.mood = _valence_mood_step(s.valence, s.mood, dt)
        s.arousal = _toward(s.arousal, BASE_AROUSAL, f)
        s.concern = _toward(s.concern, 0.0, f)
        s.surprise = _toward(s.surprise, 0.0, f)
        s.curiosity = _toward(s.curiosity, BASE_CURIOSITY, f)
        s.warmth = _toward(s.warmth, BASE_WARMTH, f)
        s.playfulness = _toward(s.playfulness, BASE_PLAYFULNESS, f)
        if present:
            s.missing *= 0.5 ** (dt / MISSING_HALF_LIFE_S)
        elif s.energy < 0.9:
            s.energy = min(0.9, s.energy + 0.02 * dt / 60)
        um = s.user_mood
        if um.confidence:
            c = um.confidence * 0.5 ** (dt / USER_MOOD_HALF_LIFE_S)
            s.user_mood = UserMood(um.valence, c, um.label) if c >= 0.1 else UserMood()

    # ---- readout ------------------------------------------------------------------------------------
    def expression(self) -> str:
        if self._override and (self._override[1] is None or self.clock.now() < self._override[1]):
            return self._override[0]
        s = self.state
        for name, hit in (("sleepy", s.energy < 0.25), ("surprised", s.surprise > 0.6), ("worried", s.concern > 0.5),
                          ("sad", s.valence < -0.4), ("excited", s.valence > 0.4 and s.arousal > 0.6),
                          ("flustered", s.warmth > 0.8 and s.arousal > 0.5), ("happy", s.valence > 0.3),
                          ("curious", s.curiosity > 0.6)):
            if hit:
                return name
        return "neutral"

    def prompt_tag(self) -> str:
        s = self.state
        tag = (f"[Riya's state: feeling {self.expression()} | valence {s.valence:+.2f} arousal {s.arousal:.2f} "
               f"warmth {s.warmth:.2f} mood {s.mood:+.2f} energy {s.energy:.2f} curiosity {s.curiosity:.2f}")
        if s.missing >= 0.4:
            tag += f" | she missed them (missing {s.missing:.2f}) and is glad they are back"
        if s.concern > 0.3:
            tag += f" | worried about them ({s.concern:.2f})"
        if s.user_mood.confidence >= 0.4:
            tag += f" | they seem {s.user_mood.label}"
        return tag + "]"

    # ---- persistence --------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"state": asdict(self.state), "saved_at": self.clock.now()}

    @classmethod
    def from_dict(cls, d: dict, clock, traits: dict[str, float]) -> "Feelings":
        f = cls(clock, traits)
        raw = dict(d.get("state", {}))
        um = UserMood(**raw.pop("user_mood", {}))
        known = {x.name for x in fields(FeelingState)} - {"user_mood"}
        f.state = FeelingState(**{k: v for k, v in raw.items() if k in known}, user_mood=um)
        f._last = d.get("saved_at", clock.now())
        return f
