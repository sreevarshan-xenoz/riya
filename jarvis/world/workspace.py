"""Global Workspace + self-model.

Global Workspace Theory (Baars; Dehaene): many specialist processes run in parallel, but only a few
contents win a limited-capacity workspace and are *broadcast* to every other process. Here:

  specialists  : perception latent, predicted outcome, affect vector, episodic memory recall
  competition  : `slots` learned queries attend over the candidates (capacity << number of candidates)
  broadcast    : the winning contents condition the fast policy (System 1) and the self-model
  self-model   : predicts the agent's OWN next valence, own surprise, and whether its fast policy will
                 agree with deliberate planning (-> confidence / metacognition)

The attention weights are logged every step: they are the inspectable 'contents of the workspace'.
This implements functional indicator properties from the consciousness-science literature
(Butlin, Long et al.); it makes no claim that the system is phenomenally conscious.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import mlp

SOURCES = ["perception", "prediction", "affect", "memory"]


class GlobalWorkspace(nn.Module):
    def __init__(self, dz: int, affect_dim: int, n_actions: int, d: int = 64, slots: int = 2):
        super().__init__()
        dims = {"perception": dz, "prediction": dz, "affect": affect_dim, "memory": dz + 1}
        self.proj = nn.ModuleDict({k: nn.Linear(dims[k], d) for k in SOURCES})
        self.source_emb = nn.Parameter(torch.randn(len(SOURCES), d) * 0.02)
        self.queries = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.key = nn.Linear(d, d)
        self.policy = mlp(slots * d, 128, n_actions)
        self.self_model = mlp(slots * d, 64, 3)  # [next valence, own surprise, P(System1 agrees with System2)]
        self.d = d

    def forward(self, sources: dict):
        tokens = torch.stack([self.proj[k](sources[k]) for k in SOURCES], 1) + self.source_emb
        attn = torch.softmax(self.queries @ self.key(tokens).transpose(1, 2) / self.d ** 0.5, -1)  # (B, slots, S)
        broadcast = (attn @ tokens).flatten(1)
        sm = self.self_model(broadcast)
        return self.policy(broadcast), sm, attn

    def loss(self, sources, plan_action, has_plan, next_valence, surprise):
        logits, sm, attn = self(sources)
        mask = has_plan.float()
        n = mask.sum().clamp(min=1)
        policy_l = (F.cross_entropy(logits, plan_action.clamp(min=0), reduction="none") * mask).sum() / n
        agree = (logits.argmax(-1) == plan_action).float()
        agree_l = (F.binary_cross_entropy_with_logits(sm[:, 2], agree, reduction="none") * mask).sum() / n
        self_l = F.mse_loss(torch.tanh(sm[:, 0]), next_valence) + F.mse_loss(F.softplus(sm[:, 1]), surprise)
        # mild entropy bonus keeps the competition from collapsing onto one source too early
        ent = -(attn * attn.clamp(min=1e-8).log()).sum(-1).mean()
        total = policy_l + agree_l + self_l - 0.01 * ent
        return total, {"policy": policy_l.item(), "agree": agree_l.item(), "self": self_l.item(),
                       "s1_acc": ((agree * mask).sum() / n).item()}
