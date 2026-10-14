"""Stage 1 (Notion page 27): calibrated confidence gate between System 1 and System 2.

FailureHead predicts P(the proposed action leads to a harmful outcome next step: health drop > 0.1 or death) from the
frozen world-model latent. It is fitted after training on replay (train seeds), then temperature-scaled on validation
seeds only. The gate sends a decision to System 2 (MPC) when
  p_failure >= p_fail_max  OR  predicted death risk >= p_dead_max  OR  expected surprise > validation 90th pct.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import mlp


class FailureHead(nn.Module):
    def __init__(self, dz, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.net = mlp(dz + n_actions, 128, 1)
        self.log_temp = nn.Parameter(torch.zeros(()), requires_grad=False)

    def logits(self, z, a):
        return self.net(torch.cat([z, F.one_hot(a, self.n_actions).float()], -1)).squeeze(-1)

    def prob(self, z, a):
        return torch.sigmoid(self.logits(z, a) / self.log_temp.exp())


def harmful_labels(body, next_body, dead):
    return ((body[:, 1] - next_body[:, 1]) > 0.1).float().clamp(max=1) + dead.float() > 0


def ece(p, y, bins=10):
    """Expected calibration error with equal-count bins."""
    p, y = np.asarray(p), np.asarray(y, dtype=float)
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err)


def fit_temperature(head, z, a, y):
    """Temperature scaling on held-out validation data (only the scalar temperature is fitted)."""
    with torch.no_grad():
        logit = head.logits(z, a)
    t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logit / t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    head.log_temp.data.copy_(t.detach())
    return float(t.exp())


class ConfidenceGate:
    def __init__(self, head, p_fail_max=0.20, p_dead_max=0.10, surprise_max=float("inf")):
        self.head, self.p_fail_max, self.p_dead_max, self.surprise_max = head, p_fail_max, p_dead_max, surprise_max

    @torch.no_grad()
    def fire(self, wm, z, a):
        """Boolean mask (B,): True -> use System 2. Also returns the reason codes for logging."""
        p_fail = self.head.prob(z, a)
        p_dead = torch.sigmoid(wm.dead_head(wm.predict(z, a)).squeeze(-1))
        surprise = wm.expected_surprise(z, a)
        reasons = torch.stack([p_fail >= self.p_fail_max, p_dead >= self.p_dead_max, surprise > self.surprise_max], 1)
        return reasons.any(1), reasons, p_fail
