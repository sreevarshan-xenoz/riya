"""Action-conditioned JEPA world model: predict the *latent* of the next observation (not pixels),
plus interoceptive heads (predicted body state, death) and a metacognitive head that predicts the
model's own prediction error. Collapse is prevented by an EMA target encoder + VICReg variance/covariance terms.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Encoder(nn.Module):
    def __init__(self, obs_shape, body_dim, dz):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.fc = mlp(32 * h * w + body_dim, 256, dz)

    def forward(self, view, body):
        return self.fc(torch.cat([self.conv(view).flatten(1), body], -1))


def vicreg_regulariser(z):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var_loss = F.relu(1 - std).mean()
    cov = (z.T @ z) / max(1, z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, off.pow(2).sum() / z.size(1)


class WorldModel(nn.Module):
    def __init__(self, obs_shape, body_dim: int = 2, n_actions: int = 5, dz: int = 64, hidden: int = 256, ctx_dim: int = 0,
                 salience_gain: float = 0.0):
        """ctx_dim > 0: the encoder also receives a context-belief vector (jarvis.world.context), so predictions
        are conditioned on what the agent currently believes about the world's hidden rules.
        salience_gain > 0: body-prediction loss weighted up to (1 + gain)x on transitions with large body change."""
        super().__init__()
        self.n_actions, self.dz, self.ctx_dim, self.salience_gain = n_actions, dz, ctx_dim, salience_gain
        self.encoder = Encoder(obs_shape, body_dim + ctx_dim, dz)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = mlp(dz + n_actions, hidden, dz, n=3)
        self.body_head = mlp(dz, 64, body_dim)
        self.dead_head = mlp(dz, 64, 1)
        self.surprise_head = mlp(dz + n_actions, 64, 1)

    def encode(self, view, body, ctx=None, target: bool = False):
        if self.ctx_dim:
            body = torch.cat([body, ctx], -1)
        return (self.target if target else self.encoder)(view, body)

    def predict(self, z, a):
        return z + self.predictor(torch.cat([z, F.one_hot(a, self.n_actions).float()], -1))

    def body(self, z):
        return torch.sigmoid(self.body_head(z))

    def expected_surprise(self, z, a):
        return F.softplus(self.surprise_head(torch.cat([z, F.one_hot(a, self.n_actions).float()], -1))).squeeze(-1)

    @torch.no_grad()
    def one_step_error(self, view, body, a, next_view, next_body, ctx=None, next_ctx=None):
        zp = self.predict(self.encode(view, body, ctx), a)
        return (zp - self.encode(next_view, next_body, next_ctx, target=True)).pow(2).mean(-1)

    def loss(self, seq):
        """seq: dict of tensors shaped (B, K, ...) for K consecutive transitions within one episode.
        Unrolls the predictor K steps in latent space (open loop) so planning rollouts stay accurate."""
        B, K = seq["action"].shape
        ctx = seq.get("ctx")
        z = self.encode(seq["view"][:, 0], seq["body"][:, 0], None if ctx is None else ctx[:, 0])
        var_l, cov_l = vicreg_regulariser(z)
        body_l = F.mse_loss(self.body(z), seq["body"][:, 0])
        pred_l = dead_l = surp_l = 0.0
        alive = torch.ones(B, device=z.device)
        first_err = None
        ev_se, ev_n = 0.0, 0.0
        for k in range(K):
            a = seq["action"][:, k]
            zp = self.predict(z, a)
            with torch.no_grad():
                nctx = seq.get("next_ctx")
                zt = self.encode(seq["next_view"][:, k], seq["next_body"][:, k], None if nctx is None else nctx[:, k], target=True)
            err = (zp - zt).pow(2).mean(-1)
            if k == 0:
                first_err = err.detach()
                surp_l = F.mse_loss(self.expected_surprise(z.detach(), a), first_err)
            w = alive / alive.sum().clamp(min=1)
            pred_l = pred_l + (w * err).sum()
            # Salience weighting: transitions that change the body a lot (eating, poison, lava) are rare but are
            # what matters; weight them up to (1 + salience_gain)x, like emotionally intense moments in memory.
            delta = (seq["next_body"][:, k] - seq["body"][:, k]).abs().sum(-1)
            sal = 1 + self.salience_gain * (delta / 0.3).clamp(max=1)
            body_se = (self.body(zp) - seq["next_body"][:, k]).pow(2).mean(-1)
            body_l = body_l + (w * sal * body_se).sum() / (w * sal).sum().clamp(min=1e-6) * w.sum()
            event = (delta > 0.1).float() * alive
            ev_se, ev_n = ev_se + (body_se.detach() * event).sum(), ev_n + event.sum()
            dead = seq["dead"][:, k]
            dead_l = dead_l + (w * F.binary_cross_entropy_with_logits(self.dead_head(zp).squeeze(-1), dead, reduction="none")).sum()
            alive = alive * (1 - dead)
            z = zp
        total = pred_l / K + 5.0 * body_l / K + dead_l / K + surp_l + 1.0 * var_l + 0.04 * cov_l
        metrics = {"pred": pred_l.item() / K, "body": body_l.item() / K, "dead": dead_l.item() / K,
                   "surprise_head": surp_l.item(), "var": var_l.item(),
                   # body prediction error on salient transitions only: does the model know what fruit/lava do?
                   "body_event": float(ev_se / ev_n) if ev_n > 0 else None}
        return total, metrics, first_err

    @torch.no_grad()
    def update_target(self, tau: float = 0.99):
        for pt, p in zip(self.target.parameters(), self.encoder.parameters()):
            pt.lerp_(p, 1 - tau)
