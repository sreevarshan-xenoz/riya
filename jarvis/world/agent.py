"""The agent loop: perceive -> workspace -> (System 1 policy | System 2 planning in the world model) -> act,
with an affect core regulating it and an episodic memory of emotionally salient moments."""
import numpy as np
import torch
import torch.nn.functional as F


class EpisodicMemory:
    """Stores latents of salient moments (|valence| high) with their valence; recalls the nearest one."""

    def __init__(self, dz: int, capacity: int = 2048, threshold: float = 0.3):
        self.z = np.zeros((capacity, dz), dtype=np.float32)
        self.v = np.zeros(capacity, dtype=np.float32)
        self.n, self.i, self.capacity, self.threshold = 0, 0, capacity, threshold

    def write(self, z, valence: float):
        if abs(valence) < self.threshold:
            return
        self.z[self.i], self.v[self.i] = z, valence
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def recall(self, z):
        if self.n == 0:
            return np.zeros(self.z.shape[1] + 1, dtype=np.float32)
        zs = self.z[: self.n]
        sim = zs @ z / (np.linalg.norm(zs, axis=1) * np.linalg.norm(z) + 1e-6)
        j = int(sim.argmax())
        return np.concatenate([zs[j], [self.v[j]]]).astype(np.float32)


def _tensor(x, device):
    t = torch.as_tensor(x, device=device)
    return t.float() if t.dtype == torch.uint8 else t  # views are stored as uint8 to keep checkpoints small


class Replay:
    def __init__(self, capacity: int = 100_000, surprise_frac: float = 0.0, alpha: float = 0.6):
        """surprise_frac: share of each world-model batch drawn in proportion to priority^alpha (prioritized
        experience replay; Schaul et al. 2016). Priority = latest one-step prediction error of that window;
        new transitions enter at the current max priority, so fresh evidence (e.g. a rule change) is replayed
        promptly. The remaining share stays uniform to protect old knowledge (retention)."""
        self.capacity, self.data, self.n, self.i = capacity, None, 0, 0
        self.surprise_frac, self.alpha = surprise_frac, alpha
        self.priority = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, **tr):
        if self.data is None:
            self.data = {}
        for k, v in tr.items():
            if k not in self.data:  # first use, or a field added after this buffer was checkpointed
                self.data[k] = np.zeros((self.capacity, *np.shape(v)), dtype=np.asarray(v).dtype)
            self.data[k][self.i] = v
        self.priority[self.i] = self.max_priority
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def _valid(self, s, K):
        ep = self.data["episode"]
        return ep[s] == ep[s + K - 1]  # episode ids are global and increasing, so this also rejects the ring seam

    def sample_seq(self, batch: int, K: int, device, recent_frac: float = 0.0, recent_window: int = 3000):
        """Windows of K consecutive transitions from a single episode. Returns (tensors, start indices).
        recent_frac: share of the batch drawn from the last `recent_window` transitions (arousal-gated recency)."""
        n_pri = int(batch * getattr(self, "surprise_frac", 0.0))
        n_rec = int(batch * recent_frac) if self.n > recent_window + K else 0
        starts = []
        if n_rec:
            newest = (self.i - 1) % self.capacity
            while len(starts) < n_rec:
                s = (newest - K - np.random.randint(0, recent_window, n_rec * 2)) % self.capacity
                s = s[s < self.n - K]
                starts.extend(s[self._valid(s, K)].tolist())
            starts = starts[:n_rec]
            n_pri = min(n_pri, batch - n_rec)
            pri_starts, starts = starts, []
        else:
            pri_starts = []
        if n_pri:
            p = self.priority[: self.n - K].astype(np.float64) ** self.alpha + 1e-6
            p /= p.sum()
            while len(starts) < n_pri:
                s = np.random.choice(self.n - K, n_pri * 2, p=p)
                starts.extend(s[self._valid(s, K)].tolist())
            starts = starts[:n_pri]
        starts = pri_starts + starts
        while len(starts) < batch:
            s = np.random.randint(0, self.n - K, batch * 2)
            starts.extend(s[self._valid(s, K)].tolist())
        starts = np.array(starts[:batch])
        idx = starts[:, None] + np.arange(K)
        keys = ["view", "body", "action", "next_view", "next_body", "dead"] + [k for k in ("ctx", "next_ctx") if k in self.data]
        out = {k: _tensor(self.data[k][idx], device) for k in keys}
        out["action"] = out["action"].long()
        return out, starts

    def update_priority(self, starts, errors):
        if not hasattr(self, "priority"):  # replay pickled by an older version
            return
        errors = np.asarray(errors, dtype=np.float32) + 1e-4
        self.priority[starts] = errors
        self.max_priority = max(self.max_priority * 0.999, float(errors.max()))

    def sample(self, batch: int, device, keys):
        idx = np.random.randint(0, self.n, batch)
        return {k: _tensor(self.data[k][idx], device) for k in keys}


class Agent:
    def __init__(self, wm, ws, affect, n_actions: int, device, horizon: int = 6, n_samples: int = 256, context=None):
        self.wm, self.ws, self.affect, self.context = wm, ws, affect, context
        self.n_actions, self.device, self.horizon, self.n_samples = n_actions, device, horizon, n_samples
        self.memory = EpisodicMemory(wm.dz)
        self.setpoints = torch.as_tensor(affect.setpoints, device=device)
        self.weights = torch.as_tensor(affect.weights, device=device)

    def _t(self, x):
        return torch.as_tensor(np.asarray(x)[None], device=self.device)

    def ctx(self):
        """Current context-belief vector as a batch of 1 (None when the agent has no context model)."""
        return None if self.context is None else self._t(self.context.vector())

    def encode_obs(self, obs, ctx=None):
        return self.wm.encode(self._t(obs["view"]), self._t(obs["body"]), self.ctx() if ctx is None else ctx)

    def sources(self, z, affect_vec, mem_vec):
        with torch.no_grad():
            preds = torch.stack([self.wm.predict(z, torch.full((z.size(0),), a, device=z.device, dtype=torch.long))
                                 for a in range(self.n_actions)], 1).mean(1)
        return {"perception": z, "prediction": preds, "affect": affect_vec, "memory": mem_vec}

    def _drive(self, body):
        return (self.weights * (self.setpoints - body).abs() ** self.affect.exponent).sum(-1)

    @torch.no_grad()
    def plan(self, z):
        """System 2: random-shooting MPC in latent space. Scores imagined futures by predicted homeostatic
        reward (drive reduction), death risk, and expected surprise weighted by current curiosity."""
        N, H = self.n_samples, self.horizon
        acts = torch.randint(0, self.n_actions, (N, H), device=self.device)
        acts[:, 0] = torch.arange(N, device=self.device) % self.n_actions
        zc = z.expand(N, -1)
        body = self.wm.body(zc)
        alive = torch.ones(N, device=self.device)
        score = torch.zeros(N, device=self.device)
        curiosity = self.affect.curiosity()
        for t in range(H):
            a = acts[:, t]
            score += alive * (0.95 ** t) * 0.2 * curiosity * self.wm.expected_surprise(zc, a)
            zc = self.wm.predict(zc, a)
            nb = self.wm.body(zc)
            p_dead = torch.sigmoid(self.wm.dead_head(zc).squeeze(-1))
            score += alive * (0.95 ** t) * ((self._drive(body) - self._drive(nb)) * 10 - 5 * p_dead)
            alive, body = alive * (1 - p_dead), nb
        best = torch.full((self.n_actions,), -1e9, device=self.device).scatter_reduce(0, acts[:, 0], score, "amax")
        return int(best.argmax())

    @torch.no_grad()
    def act(self, obs, random_policy: bool = False):
        z = self.encode_obs(obs)
        mem = self.memory.recall(z[0].cpu().numpy())
        src = self.sources(z, self._t(self.affect.signal()), self._t(mem))
        logits, sm, attn = self.ws(src)
        s1_conf = float(torch.sigmoid(sm[0, 2]))
        info = {"z": z[0].cpu().numpy(), "mem": mem, "s1_conf": s1_conf, "attn": attn[0].mean(0).cpu().numpy(), "plan": -1}
        if random_policy:
            info["mode"] = "random"
            return int(np.random.randint(self.n_actions)), info
        temp = self.affect.exploration_temperature()
        if self.affect.use_deliberation(s1_conf):
            info["mode"], info["plan"] = "S2", self.plan(z)
            a = info["plan"] if np.random.rand() > 0.3 * temp else int(np.random.randint(self.n_actions))
        else:
            info["mode"] = "S1"
            a = int(torch.multinomial(F.softmax(logits[0] / temp, -1), 1))
        return a, info
