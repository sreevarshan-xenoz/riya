"""Stage-0 reproducibility tests for the page-27 benchmark."""
import json
import os
import subprocess
import sys

import numpy as np

from jarvis.world.env import SurvivalGrid
from jarvis.world.vec import VecSurvival

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _trajectory(seed, n=150):
    env = SurvivalGrid(seed=seed, shift_every=5, shift_jitter=0.4, max_steps=50)
    rng = np.random.default_rng(123)
    obs, out = env.reset(), []
    for _ in range(n):
        obs, done, info = env.step(int(rng.integers(5)))
        out.append((obs["view"].sum(), float(obs["body"][0]), info["poison"]))
        if done:
            obs = env.reset()
    return out


def test_same_seed_same_trajectory():
    assert _trajectory(7) == _trajectory(7)
    assert _trajectory(7) != _trajectory(8)


def test_flip_schedule_independent_of_behaviour():
    a, b = SurvivalGrid(seed=3, shift_every=4, shift_jitter=0.4), SurvivalGrid(seed=3, shift_every=4, shift_jitter=0.4)
    rules_a, rules_b = [], []
    for k in range(30):
        a.reset(), b.reset()
        for _ in range(k % 7):  # different behaviour per episode
            a.step(1)
        rules_a.append(a.poison), rules_b.append(b.poison)
    assert rules_a == rules_b


def test_no_hidden_rule_in_observation():
    env = SurvivalGrid(seed=0)
    obs = env.reset()
    assert set(obs) == {"view", "body"}  # the poison identity is never part of what the agent sees


def test_vec_env_matches_single_envs():
    vec = VecSurvival([1, 2], shift_every=5, shift_jitter=0.4, max_steps=40)
    singles = [SurvivalGrid(seed=1, shift_every=5, shift_jitter=0.4, max_steps=40),
               SurvivalGrid(seed=2, shift_every=5, shift_jitter=0.4, max_steps=40)]
    vo, so = vec.reset_all(), [e.reset() for e in singles]
    for t in range(30):
        acts = [t % 5, (t * 3) % 5]
        vo, vd, _ = vec.step(acts)
        so = [e.step(a)[0] for e, a in zip(singles, acts)]
        assert all(np.array_equal(v["view"], s["view"]) for v, s in zip(vo, so))
        if any(vd):
            break


def test_runner_train_eval_and_resume(tmp_path):
    out = tmp_path / "train"

    def cmd(episodes):
        return [sys.executable, "-m", "evaluation.run_world", "train", "--variant", "v0.1", "--n_seeds", "2",
                "--episodes", str(episodes), "--batch", "16", "--updates", "2", "--transitions_per_update", "20",
                "--replay", "5000", "--out", str(out)]

    subprocess.run(cmd(2), cwd=ROOT, check=True, capture_output=True, timeout=600)
    rows = [json.loads(l) for l in open(out / "episodes.jsonl")]
    assert len(rows) == 4 and {"seed", "success", "poison", "planner_calls", "code_hash"} <= set(rows[0])
    # resume with a larger budget continues instead of restarting
    subprocess.run(cmd(3), cwd=ROOT, check=True, capture_output=True, timeout=600)
    assert len(open(out / "episodes.jsonl").readlines()) == 6
    ev = tmp_path / "eval"
    subprocess.run([sys.executable, "-m", "evaluation.run_world", "eval", "--ckpt", str(out / "brain.pt"), "--policy", "full",
                    "--split", "test", "--n_seeds", "2", "--episodes", "1", "--out", str(ev)],
                   cwd=ROOT, check=True, capture_output=True, timeout=600)
    assert len(open(ev / "episodes.jsonl").readlines()) == 2
    # Stage 1: fit + calibrate the confidence gate on validation seeds, then evaluate the gated policy
    gate = tmp_path / "gate.pt"
    subprocess.run([sys.executable, "-m", "evaluation.fit_gate", "--train_dir", str(out), "--steps", "20",
                    "--val_episodes", "1", "--out", str(gate)], cwd=ROOT, check=True, capture_output=True, timeout=900)
    gv = tmp_path / "eval-gated"
    subprocess.run([sys.executable, "-m", "evaluation.run_world", "eval", "--ckpt", str(out / "brain.pt"), "--policy",
                    "gated", "--gate", str(gate), "--split", "test", "--n_seeds", "2", "--episodes", "1", "--out", str(gv)],
                   cwd=ROOT, check=True, capture_output=True, timeout=600)
    assert len(open(gv / "episodes.jsonl").readlines()) == 2
