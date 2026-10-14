"""Survival gridworld with a body. The agent has interoceptive state (energy, health) separate from its
egocentric view, so drives and affect have something real to regulate.

- energy drains every step; fruit restores it
- two fruit types (A, B): one is nourishing, the other is poisonous
- every `shift_every` episodes the poisonous type swaps (hidden rule change -> tests belief revision)
- lava hurts; health slowly regenerates while energy is high
"""
import numpy as np

WALL, FRUIT_A, FRUIT_B, LAVA = range(4)
N_CHANNELS = 4
ACTIONS = ["stay", "up", "down", "left", "right"]
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


class SurvivalGrid:
    def __init__(self, size: int = 11, view: int = 2, n_fruit: int = 6, n_lava: int = 6,
                 shift_every: int = 50, max_steps: int = 300, seed: int | None = None, shift_jitter: float = 0.0):
        """shift_jitter: rule-shift intervals are drawn from shift_every * U(1 - j, 1 + j), so the agent cannot
        anticipate a flip by counting episodes."""
        self.size, self.view, self.n_fruit, self.n_lava = size, view, n_fruit, n_lava
        self.shift_every, self.max_steps, self.shift_jitter = shift_every, max_steps, shift_jitter
        self.rng = np.random.default_rng(seed)
        # Separate generator for the flip schedule, so every agent faces the same flip episodes for a given seed
        # (the layout generator is consumed differently depending on how the agent behaves).
        self.shift_rng = np.random.default_rng(None if seed is None else seed + 10_007)
        self.episode = 0
        self.next_shift = self._interval()
        self.poison = FRUIT_B  # hidden rule

    def _interval(self):
        j = self.shift_jitter
        return max(1, int(round(self.shift_every * self.shift_rng.uniform(1 - j, 1 + j)))) if j else self.shift_every

    @property
    def obs_shape(self):
        w = 2 * self.view + 1
        return (N_CHANNELS, w, w)

    def _empty_cell(self):
        while True:
            r, c = self.rng.integers(1, self.size - 1, 2)
            if not self.grid[:, r, c].any() and (r, c) != tuple(self.pos):
                return r, c

    def reset(self):
        if self.episode and self.episode >= getattr(self, "next_shift", self.shift_every):
            self.poison = FRUIT_A if self.poison == FRUIT_B else FRUIT_B
            self.next_shift = self.episode + self._interval()
        self.episode += 1
        self.grid = np.zeros((N_CHANNELS, self.size, self.size), dtype=np.float32)
        self.grid[WALL, 0, :] = self.grid[WALL, -1, :] = self.grid[WALL, :, 0] = self.grid[WALL, :, -1] = 1
        self.pos = np.array([self.size // 2, self.size // 2])
        for _ in range(self.n_lava):
            self.grid[(LAVA, *self._empty_cell())] = 1
        for i in range(self.n_fruit):
            self.grid[(FRUIT_A if i % 2 else FRUIT_B, *self._empty_cell())] = 1
        self.energy, self.health, self.t = 0.7, 1.0, 0
        return self._obs()

    def body(self):
        return np.array([self.energy, self.health], dtype=np.float32)

    def _obs(self):
        v = self.view
        padded = np.pad(self.grid, ((0, 0), (v, v), (v, v)))
        padded[WALL] = np.pad(self.grid[WALL], v, constant_values=1)
        r, c = self.pos + v
        return {"view": padded[:, r - v : r + v + 1, c - v : c + v + 1].copy(), "body": self.body()}

    def step(self, action: int):
        dr, dc = MOVES[action]
        nxt = self.pos + (dr, dc)
        if not self.grid[WALL, nxt[0], nxt[1]]:
            self.pos = nxt
        r, c = self.pos
        event = None
        self.energy -= 0.01
        if self.grid[LAVA, r, c]:
            self.health -= 0.25
            event = "lava"
        for fruit in (FRUIT_A, FRUIT_B):
            if self.grid[fruit, r, c]:
                self.grid[fruit, r, c] = 0
                if fruit == self.poison:
                    self.health -= 0.3
                    self.energy += 0.05
                    event = "poison"
                else:
                    self.energy += 0.35
                    event = "food"
                self.grid[(fruit, *self._empty_cell())] = 1
        if self.energy > 0.5:
            self.health += 0.005
        self.energy, self.health = float(np.clip(self.energy, 0, 1)), float(np.clip(self.health, 0, 1))
        self.t += 1
        dead = self.energy <= 0 or self.health <= 0
        done = dead or self.t >= self.max_steps
        return self._obs(), done, {"dead": dead, "event": event, "poison": "A" if self.poison == FRUIT_A else "B"}
