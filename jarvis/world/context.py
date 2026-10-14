"""Context belief: what does each kind of object do to my body, and has that just changed?

The agent keeps a Beta belief P(harmful | object kind) per object channel, learned only from its own experience
(it stepped onto X and its health dropped, or it did not). Learning is change-point aware via the Bayes Factor
Surprise (SMiLe rule; Liakoni et al. 2021, cf. Nassar et al. 2010): an outcome that is far more likely under the
prior than under the current belief signals a possible change of the world, and the belief is pulled back toward
the prior in proportion to the change-point probability gamma before the new evidence is added.

gamma is also the agent's 'something changed' signal, used by the affect core (arousal) and to gate plasticity.

The belief vector is fed into the world-model encoder and stored with every transition, so old experience and new
experience are learned *conditionally* instead of fighting: the model can switch rather than overwrite.
"""
import numpy as np

from .env import FRUIT_A, FRUIT_B, LAVA, MOVES

CHANNELS = (FRUIT_A, FRUIT_B, LAVA)
CTX_DIM = len(CHANNELS) + 1  # P(harm) per channel in [-1, 1] + change-point probability


def touched_channel(view, action: int):
    """Which object channel the agent steps onto with `action`, read from its own egocentric view (or None)."""
    c = view.shape[-1] // 2
    dr, dc = MOVES[action]
    r, k = c + dr, c + dc
    if view[0, r, k]:  # wall: the agent does not move
        return None
    for i, ch in enumerate(CHANNELS):
        if view[ch, r, k]:
            return i
    return None


class SmileBelief:
    """Single-hypothesis belief with Bayes-Factor-Surprise forgetting (SMiLe-like). Kept as a baseline: each update
    lowers the surprise of the next contradicting outcome, so re-learning slows down (see BOCPD below)."""

    def __init__(self, hazard: float = 0.1, prior: float = 1.0, max_count: float = 20.0):
        self.hazard, self.prior, self.max_count = hazard, prior, max_count
        self.a = np.full(len(CHANNELS), prior)  # pseudo-counts of 'harmful'
        self.b = np.full(len(CHANNELS), prior)  # pseudo-counts of 'harmless'
        self.cp = 0.0

    def p_harm(self):
        return self.a / (self.a + self.b)

    def update_channel(self, i, harmful):
        p_now = self.a[i] / (self.a[i] + self.b[i])
        surprise = 0.5 / max(p_now if harmful else 1 - p_now, 1e-6)  # prior predictive / current predictive
        m = self.hazard / (1 - self.hazard)
        gamma = m * surprise / (1 + m * surprise)
        self.a[i] = (1 - gamma) * self.a[i] + gamma * self.prior + harmful
        self.b[i] = (1 - gamma) * self.b[i] + gamma * self.prior + (1 - harmful)
        total = self.a[i] + self.b[i]
        if total > self.max_count:
            self.a[i] *= self.max_count / total
            self.b[i] *= self.max_count / total
        return gamma

    def observe(self, view, action: int, body, next_body):
        """Update from one step; returns the change-point probability (0 when nothing was touched)."""
        i = touched_channel(view, action)
        self.cp *= 0.5  # the 'just changed' signal fades between events
        if i is None:
            return 0.0
        gamma = self.update_channel(i, float(next_body[1] < body[1] - 0.1))
        self.cp = max(self.cp, gamma)
        return gamma

    def vector(self):
        return np.concatenate([self.p_harm() * 2 - 1, [self.cp]]).astype(np.float32)


class ContextBelief(SmileBelief):
    """Bayesian online change-point detection (Adams & MacKay 2007) with a Beta-Bernoulli model per object channel.

    Keeps a posterior over the run length r (events since the last change) and Beta statistics for each r. A streak
    of contradicting outcomes quickly makes 'a change happened r events ago' the best explanation, so the belief
    switches in a few events, while one isolated odd outcome is absorbed (tuned: 3 contradicting events to flip
    P(harm) past 0.5 vs 6 for the SMiLe baseline). The change signal is the posterior mass on a change within the
    last 5 events.
    """

    def __init__(self, hazard: float = 0.02, prior: float = 1.0, max_run: int = 80):
        self.hazard, self.prior, self.L = hazard, prior, max_run
        n = len(CHANNELS)
        self.R = np.zeros((n, max_run)); self.R[:, 0] = 1.0  # noqa: E702  run-length posterior per channel
        self.A = np.full((n, max_run), prior)
        self.B = np.full((n, max_run), prior)
        self.events = np.zeros(n, dtype=int)
        self.cp = 0.0

    def p_harm(self):
        return (self.R * self.A / (self.A + self.B)).sum(1)

    def update_channel(self, i, harmful):
        R, A, B, H = self.R[i], self.A[i], self.B[i], self.hazard
        pred = np.where(harmful, A / (A + B), B / (A + B))
        growth = R * pred * (1 - H)
        new_R = np.empty_like(R)
        new_R[0] = (R * pred * H).sum()
        new_R[1:] = growth[:-1]
        new_R[-1] += growth[-1]  # truncate: the longest run absorbs the overflow
        new_R /= new_R.sum()
        self.A[i, 1:], self.B[i, 1:] = A[:-1] + harmful, B[:-1] + (1 - harmful)
        self.A[i, 0], self.B[i, 0] = self.prior + harmful, self.prior + (1 - harmful)
        self.R[i] = new_R
        self.events[i] += 1
        # P(a change within the last 5 events). Only meaningful once the channel has history: a fresh channel is
        # 'new', not 'changed'.
        return float(new_R[:5].sum()) if self.events[i] > 10 else 0.0


class OracleContext:
    """Upper-bound control: the true hidden rule, in the same format as ContextBelief.vector()."""

    def __init__(self, env):
        self.env, self.cp = env, 0.0

    def observe(self, view, action, body, next_body):
        return 0.0

    def vector(self):
        a_bad = self.env.poison == FRUIT_A
        return np.array([1.0 if a_bad else -1.0, -1.0 if a_bad else 1.0, 1.0, 0.0], dtype=np.float32)
