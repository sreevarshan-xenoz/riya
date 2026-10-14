"""Train the world-model core: JEPA world model + affect core + global workspace in the survival gridworld.

  python train_world.py --out_dir runs/world --episodes 2000
  python train_world.py --out_dir runs/world-ctx --context belief --arousal_plasticity --shift_jitter 0.4

Context modes (how the agent knows which hidden rule is in force):
  none    no context input (original v0)
  belief  Bayesian belief about what each object does, learned from own experience, change-point aware
  oracle  the true rule (upper-bound control)

What to watch: survival, poison per episode, and per rule flip (see summarize_world.py): P10 = poison eaten in the
first 10 episodes after a flip, episodes to recover, flip-back P10, and the replay share from the current rule.
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from jarvis.world import AFFECT_DIM, SOURCES, AffectCore, Agent, GlobalWorkspace, Replay, SurvivalGrid, WorldModel
from jarvis.world.context import CTX_DIM, ContextBelief, OracleContext


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--random_episodes", type=int, default=30, help="pure exploration before planning starts")
    p.add_argument("--updates_per_episode", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seq", type=int, default=5, help="open-loop unroll length for world-model training")
    p.add_argument("--shift_every", type=int, default=50, help="mean episodes between hidden poison-rule swaps")
    p.add_argument("--shift_jitter", type=float, default=0.0, help="randomise swap intervals by +-this fraction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--max_minutes", type=float, default=0)
    p.add_argument("--ablate_affect", action="store_true", help="H1 control run: affect computed but has no influence")
    p.add_argument("--surprise_frac", type=float, default=0.0,
                   help="share of world-model batches sampled by prediction-error priority (0 = uniform replay)")
    p.add_argument("--context", choices=["none", "belief", "oracle"], default="none")
    p.add_argument("--arousal_plasticity", action="store_true",
                   help="change-point probability raises learning rate (up to 3x) and the share of recent replay")
    p.add_argument("--flush_on_flip", action="store_true", help="oracle control: empty the replay buffer at each flip")
    p.add_argument("--salience", type=float, default=0.0,
                   help="weight body-prediction loss up to (1+salience)x on transitions with large body change")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = SurvivalGrid(shift_every=args.shift_every, shift_jitter=args.shift_jitter, seed=args.seed)
    ctx_dim = CTX_DIM if args.context != "none" else 0
    wm = WorldModel(env.obs_shape, ctx_dim=ctx_dim, salience_gain=args.salience).to(device)
    ws = GlobalWorkspace(wm.dz, AFFECT_DIM, wm.n_actions).to(device)
    affect = AffectCore(ablate=args.ablate_affect)
    context = {"none": None, "belief": ContextBelief(), "oracle": OracleContext(env)}[args.context]
    agent = Agent(wm, ws, affect, wm.n_actions, device, context=context)
    opt_wm = torch.optim.AdamW(wm.parameters(), lr=args.lr)
    opt_ws = torch.optim.AdamW(ws.parameters(), lr=args.lr)
    replay = Replay(surprise_frac=args.surprise_frac)
    ckpt_path, log_path = os.path.join(args.out_dir, "world.pt"), os.path.join(args.out_dir, "log.jsonl")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    start_ep = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # own checkpoint: includes numpy replay
        wm.load_state_dict(ck["wm"]), ws.load_state_dict(ck["ws"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_ws.load_state_dict(ck["opt_ws"])
        replay, agent.memory, affect.state = ck["replay"], ck["memory"], ck["affect"]
        env.episode, env.poison, start_ep = ck["env_episode"], ck["env_poison"], ck["episode"]
        env.next_shift = ck.get("env_next_shift", (env.episode // args.shift_every + 1) * args.shift_every)
        if ck.get("context") is not None and args.context == "belief":
            agent.context = context = ck["context"]
        print(f"resumed at episode {start_ep}")

    def save(ep):
        torch.save({"wm": wm.state_dict(), "ws": ws.state_dict(), "opt_wm": opt_wm.state_dict(), "opt_ws": opt_ws.state_dict(),
                    "replay": replay, "memory": agent.memory, "affect": affect.state, "env_episode": env.episode,
                    "env_poison": env.poison, "env_next_shift": env.next_shift, "episode": ep,
                    "context": context if args.context == "belief" else None}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    zeros_ctx = np.zeros(ctx_dim, dtype=np.float32)
    rule_id = {"A": 0, "B": 1}
    prev_rule = None
    t_start = time.time()
    for ep in range(start_ep, args.episodes):
        obs, done = env.reset(), False
        rule = "A" if env.poison == 1 else "B"
        if args.flush_on_flip and prev_rule is not None and rule != prev_rule:
            replay.n = replay.i = 0
        prev_rule = rule
        stats = collections.Counter()
        feelings, attn_sum, steps, cp_max = collections.Counter(), np.zeros(len(SOURCES)), 0, 0.0
        random_phase = ep < args.random_episodes
        while not done:
            ctx = context.vector() if context is not None else zeros_ctx
            a, info = agent.act(obs, random_policy=random_phase)
            nxt, done, step_info = env.step(a)
            cp = context.observe(obs["view"], a, obs["body"], nxt["body"]) if context is not None else 0.0
            next_ctx = context.vector() if context is not None else zeros_ctx
            cp_max = max(cp_max, cp)
            c0, c1 = (agent._t(ctx), agent._t(next_ctx)) if ctx_dim else (None, None)
            err = float(wm.one_step_error(agent._t(obs["view"]), agent._t(obs["body"]), torch.tensor([a], device=device),
                                          agent._t(nxt["view"]), agent._t(nxt["body"]), c0, c1))
            affect_before = affect.signal()
            affect.update(obs["body"], nxt["body"], err, confidence=info["s1_conf"], change_prob=cp)
            agent.memory.write(info["z"], affect.state.valence)
            replay.add(view=obs["view"].astype(np.uint8), body=obs["body"], action=np.int64(a),
                       next_view=nxt["view"].astype(np.uint8), next_body=nxt["body"],
                       dead=np.float32(step_info["dead"]), episode=np.int64(ep), plan=np.int64(info["plan"]),
                       affect=affect_before, mem=info["mem"], next_valence=np.float32(affect.state.valence),
                       surprise=np.float32(err), ctx=ctx, next_ctx=next_ctx, rule=np.int64(rule_id[rule]))
            stats[info["mode"]] += 1
            if step_info["event"]:
                stats[step_info["event"]] += 1
            feelings[affect.label()] += 1
            attn_sum += info["attn"]
            obs, steps = nxt, steps + 1

        # Arousal-gated plasticity: a detected change makes learning faster and more recency-focused.
        lr_mult = float(np.clip(1 + 4 * cp_max, 1, 3)) if args.arousal_plasticity else 1.0
        recent_frac = float(np.clip(0.5 * cp_max, 0, 0.5)) if args.arousal_plasticity else 0.0
        for opt in (opt_wm, opt_ws):
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_mult

        wm_m, ws_m, current_share = {}, {}, None
        if replay.n > args.batch * 4:
            for _ in range(args.updates_per_episode):
                seq, starts = replay.sample_seq(args.batch, args.seq, device, recent_frac=recent_frac)
                loss, wm_m, first_err = wm.loss(seq)
                replay.update_priority(starts, first_err.cpu().numpy())
                opt_wm.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                opt_wm.step()
                wm.update_target()

                b = replay.sample(args.batch, device, ["view", "body", "plan", "affect", "mem", "next_valence", "surprise", "ctx"])
                with torch.no_grad():
                    z = wm.encode(b["view"], b["body"], b["ctx"] if ctx_dim else None)
                src = agent.sources(z, b["affect"], b["mem"])
                loss, ws_m = ws.loss(src, b["plan"].long(), b["plan"] >= 0, b["next_valence"], b["surprise"])
                opt_ws.zero_grad(set_to_none=True)
                loss.backward()
                opt_ws.step()
            current_share = float((replay.data["rule"][starts] == rule_id[rule]).mean())

        decisions = stats["S1"] + stats["S2"]
        row = {"episode": ep, "survival": steps, "food": stats["food"], "poison": stats["poison"], "lava": stats["lava"],
               "rule": rule, "S1%": round(100 * stats["S1"] / max(1, decisions), 1),
               "mood": round(affect.state.mood, 3), "feelings": dict(feelings.most_common(3)),
               "workspace": {k: round(float(v), 3) for k, v in zip(SOURCES, attn_sum / max(1, steps))},
               "cp_max": round(cp_max, 3), "lr_mult": round(lr_mult, 2),
               "replay_current_share": None if current_share is None else round(current_share, 3),
               "belief": None if context is None else [round(float(x), 2) for x in context.vector()],
               **{f"wm_{k}": round(v, 4) for k, v in wm_m.items() if v is not None},
               **{f"ws_{k}": round(v, 4) for k, v in ws_m.items()}}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"ep {ep} | alive {steps} | food {stats['food']} poison {stats['poison']} lava {stats['lava']} | "
              f"poison={rule} | S1 {row['S1%']}% | cp {cp_max:.2f} | {affect.describe()}", flush=True)

        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if (ep + 1) % args.save_every == 0 or ep + 1 == args.episodes or out_of_time:
            save(ep + 1)
        if out_of_time:
            print("time budget reached; rerun the same command to resume")
            break


if __name__ == "__main__":
    main()
