"""Affect core: emotions as functional control signals, not decoration.

Grounded in:
- Homeostatic RL (Keramati & Gutkin 2014; Yoshida et al. 2025): reward = reduction of drive, where drive is
  the distance of internal variables from their setpoints.
- Curiosity / learning progress (Oudeyer): positive affect from *reducing* prediction error, not from error itself.
- Core affect (Russell): valence x arousal, plus a confidence/control axis from the self-model.

The resulting vector modulates behaviour (exploration, planning depth, System 1 vs 2) and is what the
language backbone receives (`HybridLM(affect=...)`) so Jarvis can speak from its actual internal state.
"""
from dataclasses import dataclass, field

import numpy as np

AFFECT_DIM = 8


@dataclass
class AffectState:
    valence: float = 0.0  # good/bad, from drive reduction + learning progress
    arousal: float = 0.2  # activation, from urgency + surprise
    confidence: float = 0.5  # from the self-model: how well do I predict myself/the world
    mood: float = 0.0  # slow average of valence
    energy_drive: float = 0.0
    health_drive: float = 0.0
    curiosity: float = 0.5  # information need; rises when learning stalls, falls while learning
    surprise: float = 0.0

    def vector(self):
        return np.array([self.valence, self.arousal, self.confidence, self.mood, self.energy_drive,
                         self.health_drive, self.curiosity, self.surprise], dtype=np.float32)


@dataclass
class AffectCore:
    setpoints: np.ndarray = field(default_factory=lambda: np.array([0.8, 1.0], dtype=np.float32))
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.5], dtype=np.float32))
    exponent: float = 2.0
    mood_rate: float = 0.02
    ablate: bool = False  # H1 control: affect is still computed/logged but no longer influences anything
    state: AffectState = field(default_factory=AffectState)
    _err_fast: float = 0.0
    _err_slow: float = 0.0

    def drive(self, body):
        """Homeostatic drive D(h) = sum_i w_i |h*_i - h_i|^p (Keramati & Gutkin)."""
        return float((self.weights * np.abs(self.setpoints - body) ** self.exponent).sum())

    def homeostatic_reward(self, body, next_body):
        return self.drive(body) - self.drive(next_body)

    def update(self, body, next_body, prediction_error: float, confidence: float | None = None, change_prob: float = 0.0):
        """Call once per environment step. Returns the homeostatic reward.
        change_prob: 'the world just changed' probability from the context belief; it raises surprise and
        arousal (unexpected-uncertainty signal, cf. Yu & Dayan 2005: norepinephrine)."""
        prediction_error = max(prediction_error, change_prob)
        s = self.state
        r = self.homeostatic_reward(body, next_body)
        # Learning progress = fast error average dropping below slow average.
        self._err_fast += 0.2 * (prediction_error - self._err_fast)
        self._err_slow += 0.02 * (prediction_error - self._err_slow)
        progress = self._err_slow - self._err_fast

        dev = np.abs(self.setpoints - next_body)
        s.energy_drive, s.health_drive = float(dev[0]), float(dev[1])
        s.surprise = float(prediction_error)
        s.curiosity = float(np.clip(s.curiosity + 0.01 * (0.5 - s.curiosity) - 0.5 * progress, 0, 1))
        s.valence = float(np.tanh(8.0 * r + 4.0 * progress))
        urgency = float((self.weights * dev).max())
        s.arousal = float(1 / (1 + np.exp(-(3.0 * urgency + 2.0 * s.surprise - 1.5))))
        if confidence is not None:
            s.confidence = float(confidence)
        s.mood += self.mood_rate * (s.valence - s.mood)
        return r

    # --- behaviour modulation ---------------------------------------------------------------
    def signal(self):
        """The affect vector other modules see (constant when ablated)."""
        return AffectState().vector() if self.ablate else self.state.vector()

    def curiosity(self):
        return 0.5 if self.ablate else self.state.curiosity

    def exploration_temperature(self):
        """Curious and safe -> explore; distressed -> exploit what is known to work."""
        if self.ablate:
            return 0.5
        s = self.state
        return float(np.clip(0.2 + 0.8 * s.curiosity * (1 - s.arousal), 0.05, 1.0))

    def use_deliberation(self, s1_confidence: float):
        """System 2 (planning) when the fast policy is unsure, surprise is high, or stakes are high."""
        if self.ablate:
            return s1_confidence < 0.7
        s = self.state
        return s1_confidence < 0.7 or s.surprise > 0.5 or s.arousal > 0.7

    # --- expression ---------------------------------------------------------------------------
    def label(self):
        s = self.state
        if s.health_drive > 0.4:
            return "hurt"
        if s.energy_drive > 0.5:
            return "hungry"
        if s.surprise > 0.6 and s.valence < 0:
            return "alarmed"
        if s.valence > 0.3:
            return "satisfied" if s.arousal < 0.5 else "excited"
        if s.valence < -0.3:
            return "frustrated" if s.confidence < 0.5 else "displeased"
        if s.curiosity > 0.6:
            return "curious"
        return "calm" if s.mood >= 0 else "uneasy"

    def describe(self):
        s = self.state
        return (f"[feeling {self.label()} | valence {s.valence:+.2f} arousal {s.arousal:.2f} "
                f"confidence {s.confidence:.2f} mood {s.mood:+.2f} curiosity {s.curiosity:.2f}]")
