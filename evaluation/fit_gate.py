"""Fit the Stage-1 failure head on a frozen brain, calibrate it on validation seeds, save gate.pt.

  python -m evaluation.fit_gate --train_dir runs/bench/train-v0.1 --val_episodes 10 --out runs/bench/gate-v0.1.pt

1. Train FailureHead on the training replay (train seeds) with the world model frozen.
2. Collect validation transitions by running the frozen 'full' agent on validation seeds (no learning).
3. Temperature-scale on half of validation, report ECE/Brier on the other half; set the surprise threshold to the
   validation 90th percentile. Thresholds are never tuned on test seeds.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.run_world import MANIFEST, VARIANTS, build, seeds
from jarvis.world.gate import FailureHead, ece, fit_temperature, harmful_labels
from jarvis.world.vec import VecSurvival


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--val_episodes", type=int, default=10)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain = torch.load(f"{a.train_dir}/brain.pt", map_location=device, weights_only=False)
    state = torch.load(f"{a.train_dir}/state.pt", map_location=device, weights_only=False)
    variant = brain["variant"]
    val_seeds = seeds("validation")
    envs = VecSurvival(val_seeds, max_steps=MANIFEST["episode_steps"], **MANIFEST["regimes"]["swap"])
    wm, ws, _, agent = build(variant, device, len(val_seeds), envs, "full")
    wm.load_state_dict(brain["wm"]), ws.load_state_dict(brain["ws"])
    agent.memory = brain["memory"]
    wm.eval()
    replay = state["replay"]
    head = FailureHead(wm.dz, wm.n_actions).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    keys = ["view", "body", "action", "next_body", "dead", "ctx"]
    for step in range(a.steps):
        b = replay.sample(512, device, keys)
        with torch.no_grad():
            z = wm.encode(b["view"], b["body"], b["ctx"] if wm.ctx_dim else None)
        y = harmful_labels(b["body"], b["next_body"], b["dead"]).float()
        pos = y.mean().clamp(min=1e-3)
        loss = F.binary_cross_entropy_with_logits(head.logits(z, b["action"].long()), y,
                                                  pos_weight=torch.clamp((1 - pos) / pos, max=20.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"failure head trained ({a.steps} steps), harmful rate in replay {pos.item():.3f}", flush=True)

    # validation data from the frozen agent (no parameter updates)
    Z, A, Y, S = [], [], [], []
    obs = envs.reset_all()
    done_eps = np.zeros(len(val_seeds), int)
    while (done_eps < a.val_episodes).any():
        ctx = agent.ctx_batch()
        acts, infos = agent.act(obs)
        nxt, done, info = envs.step(acts)
        with torch.no_grad():
            view = torch.as_tensor(np.stack([o["view"] for o in obs]), device=device)
            body = torch.as_tensor(np.stack([o["body"] for o in obs]), device=device)
            z = wm.encode(view, body, ctx)
            at = torch.as_tensor(acts, device=device).long()
            S.append(wm.expected_surprise(z, at).cpu())
        Y.append(torch.as_tensor([(obs[i]["body"][1] - nxt[i]["body"][1] > 0.1) or info[i]["dead"]
                                  for i in range(len(val_seeds))], dtype=torch.float32))  # before any reset
        for i in range(len(val_seeds)):
            if agent.contexts is not None:
                agent.contexts[i].observe(obs[i]["view"], int(acts[i]), obs[i]["body"], nxt[i]["body"])
            agent.affects[i].update(obs[i]["body"], nxt[i]["body"], 0.0, infos[i]["s1_conf"])
            if done[i]:
                done_eps[i] += 1
                nxt[i] = envs.envs[i].reset()
        Z.append(z.cpu()), A.append(at.cpu())
        obs = nxt
    Z, A, Y, S = torch.cat(Z).to(device), torch.cat(A).to(device), torch.cat(Y).to(device), torch.cat(S)
    n = len(Y) // 2
    raw = head.prob(Z[n:], A[n:]).detach().cpu().numpy()
    temp = fit_temperature(head, Z[:n], A[:n], Y[:n])
    cal = head.prob(Z[n:], A[n:]).detach().cpu().numpy()
    y_eval = Y[n:].cpu().numpy()
    report = {"variant": variant, "temperature": temp, "val_transitions": len(Y), "harmful_rate_val": float(Y.mean()),
              "ece_raw": ece(raw, y_eval), "ece_calibrated": ece(cal, y_eval),
              "brier_calibrated": float(np.mean((cal - y_eval) ** 2)),
              "surprise_p90": float(np.percentile(S.numpy(), 90))}
    print(json.dumps(report, indent=1), flush=True)
    torch.save({"head": head.state_dict(), "dz": wm.dz, "n_actions": wm.n_actions, "report": report,
                "thresholds": {"p_fail_max": 0.20, "p_dead_max": 0.10, "surprise_max": report["surprise_p90"]}}, a.out)
    print("GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
