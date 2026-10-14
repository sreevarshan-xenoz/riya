"""Vectorized SurvivalGrid + batched agent for fast, multi-seed experiments (Notion page 27, Stage 0).

The environment rules are untouched: VecSurvival is a list of ordinary SurvivalGrid instances, one per seed.
One shared brain (world model, workspace, replay, episodic memory) acts for all of them in a single batched forward
pass; each environment keeps its own affect core and context belief, because each is a separate 'life'.

Policies (page 27 baselines):
  random  B0  legal random actions
  s1      B1  System 1 only: workspace policy, no memory retrieval, no planner
  memory  B3  System 1 + episodic memory retrieval, no planner
  full    B2  v0: affect + memory + world model + System 1 / System 2 (MPC) gate
  gated   S1  Stage 1: System 1 unless the calibrated confidence gate (jarvis/world/gate.py) fires
  mpc         always System 2 (always-on MPC reference for the Stage-1 compute/success trade-off)
"""
import numpy as np
import torch
import torch.nn.functional as F

from .affect import AffectCore
from .context import ContextBelief, OracleContext
from .env import SurvivalGrid


class VecSurvival:
    def __init__(self, seeds, **env_kwargs):
        self.envs = [SurvivalGrid(seed=int(s), **env_kwargs) for s in seeds]
        self.seeds = list(seeds)

    def reset_all(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        out = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]


class BatchAgent:
    def __init__(self, wm, ws, memory, n_envs, device, context="none", envs=None, policy="full",
                 ablate_affect=False, horizon=6, n_samples=256, gate=None):
        self.wm, self.ws, self.memory, self.device, self.policy, self.gate = wm, ws, memory, device, policy, gate
        self.n_actions, self.horizon, self.n_samples = wm.n_actions, horizon, n_samples
        self.affects = [AffectCore(ablate=ablate_affect) for _ in range(n_envs)]
        if context == "belief":
            self.contexts = [ContextBelief() for _ in range(n_envs)]
        elif context == "oracle":
            self.contexts = [OracleContext(e) for e in envs]
        else:
            self.contexts = None
        a0 = self.affects[0]
        self.setpoints = torch.as_tensor(a0.setpoints, device=device)
        self.weights = torch.as_tensor(a0.weights, device=device)
        self.exponent = a0.exponent

    def ctx_batch(self):
        if self.contexts is None:
            return None
        return torch.as_tensor(np.stack([c.vector() for c in self.contexts]), device=self.device)

    def encode(self, obs):
        view = torch.as_tensor(np.stack([o["view"] for o in obs]), device=self.device)
        body = torch.as_tensor(np.stack([o["body"] for o in obs]), device=self.device)
        return self.wm.encode(view, body, self.ctx_batch())

    def _drive(self, body):
        return (self.weights * (self.setpoints - body).abs() ** self.exponent).sum(-1)

    @torch.no_grad()
    def plan_batch(self, z, curiosity):
        """Random-shooting MPC for M environments at once: z (M, dz), curiosity (M,). Returns best first actions."""
        M, N, H, A = z.size(0), self.n_samples, self.horizon, self.n_actions
        acts = torch.randint(0, A, (M, N, H), device=self.device)
        acts[:, :, 0] = torch.arange(N, device=self.device) % A
        zc = z[:, None, :].expand(M, N, -1).reshape(M * N, -1)
        cur = curiosity[:, None].expand(M, N).reshape(-1)
        flat = acts.reshape(M * N, H)
        body = self.wm.body(zc)
        alive = torch.ones(M * N, device=self.device)
        score = torch.zeros(M * N, device=self.device)
        for t in range(H):
            a = flat[:, t]
            score += alive * (0.95 ** t) * 0.2 * cur * self.wm.expected_surprise(zc, a)
            zc = self.wm.predict(zc, a)
            nb = self.wm.body(zc)
            p_dead = torch.sigmoid(self.wm.dead_head(zc).squeeze(-1))
            score += alive * (0.95 ** t) * ((self._drive(body) - self._drive(nb)) * 10 - 5 * p_dead)
            alive, body = alive * (1 - p_dead), nb
        best = torch.full((M, A), -1e9, device=self.device).scatter_reduce(1, acts[:, :, 0], score.view(M, N), "amax")
        return best.argmax(1)

    @torch.no_grad()
    def act(self, obs):
        B = len(obs)
        if self.policy == "random":
            return np.random.randint(self.n_actions, size=B), [{"mode": "random", "plan": -1, "s1_conf": 0.0,
                                                                  "z": None, "mem": None} for _ in range(B)]
        z = self.encode(obs)
        z_np = z.cpu().numpy()
        use_mem = self.policy in ("memory", "full")
        mem = np.stack([self.memory.recall(z_np[i]) if use_mem else np.zeros(self.wm.dz + 1, np.float32)
                        for i in range(B)])
        aff = torch.as_tensor(np.stack([a.signal() for a in self.affects]), device=self.device)
        with torch.no_grad():
            preds = torch.stack([self.wm.predict(z, torch.full((B,), a, device=self.device, dtype=torch.long))
                                 for a in range(self.n_actions)], 1).mean(1)
        logits, sm, attn = self.ws({"perception": z, "prediction": preds, "affect": aff,
                                    "memory": torch.as_tensor(mem, device=self.device)})
        s1_conf = torch.sigmoid(sm[:, 2]).cpu().numpy()
        temps = np.array([a.exploration_temperature() for a in self.affects], dtype=np.float32)
        probs = F.softmax(logits / torch.as_tensor(temps, device=self.device)[:, None], -1)
        actions = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
        modes, plans = ["S1"] * B, np.full(B, -1)
        reasons = None
        if self.policy in ("full", "gated", "mpc"):
            if self.policy == "full":
                delib = np.array([self.affects[i].use_deliberation(float(s1_conf[i])) for i in range(B)])
            elif self.policy == "mpc":
                delib = np.ones(B, dtype=bool)
            else:
                fire, reasons, _ = self.gate.fire(self.wm, z, torch.as_tensor(actions, device=self.device).long())
                delib = fire.cpu().numpy()
                reasons = reasons.cpu().numpy()
            idx = np.nonzero(delib)[0]
            if len(idx):
                cur = torch.as_tensor([self.affects[i].curiosity() for i in idx], device=self.device, dtype=torch.float32)
                best = self.plan_batch(z[idx], cur).cpu().numpy()
                for j, i in enumerate(idx):
                    plans[i], modes[i] = best[j], "S2"
                    explore = np.random.rand() < 0.3 * temps[i]
                    actions[i] = np.random.randint(self.n_actions) if explore else best[j]
        infos = [{"mode": modes[i], "plan": int(plans[i]), "s1_conf": float(s1_conf[i]), "z": z_np[i], "mem": mem[i],
                  "attn": attn[i].mean(0).cpu().numpy(),
                  "gate_reasons": None if reasons is None else reasons[i].tolist()} for i in range(B)]
        return actions, infos
