"""Page-27 benchmark runner (vectorized: all seeds of a split step together through one batched brain).

  # train a frozen checkpoint on the train seeds (v0 = plain, v0.1 = belief + arousal plasticity + salience)
  python -m evaluation.run_world train --variant v0.1 --episodes 60 --out runs/bench/v0.1
  # evaluate a policy with frozen weights on the locked test seeds
  python -m evaluation.run_world eval --ckpt runs/bench/v0.1/brain.pt --policy full --split test --episodes 60 \
      --out runs/bench/eval-v0.1-full

Every finished episode appends one JSON line (page-27 artifact schema, fields we can measure) to <out>/episodes.jsonl.
Checkpoints are atomic and resumable (Colab sessions get reclaimed).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from jarvis.world import AFFECT_DIM, GlobalWorkspace, Replay, WorldModel
from jarvis.world.agent import EpisodicMemory
from jarvis.world.context import CTX_DIM
from jarvis.world.vec import BatchAgent, VecSurvival

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = json.load(open(os.path.join(HERE, "manifest.json")))
VARIANTS = {  # name: (context, arousal_plasticity, salience)
    "v0": ("none", False, 0.0),
    "v0.1": ("belief", True, 19.0),
    "oracle": ("oracle", False, 19.0),
}


def seeds(split):
    lo, hi = MANIFEST["seeds"][split]
    return list(range(lo, hi + 1))


def code_hash():
    h = hashlib.sha1()
    root = os.path.dirname(HERE)
    for sub in ("jarvis/world", "evaluation"):
        for name in sorted(os.listdir(os.path.join(root, sub))):
            if name.endswith(".py"):
                h.update(open(os.path.join(root, sub, name), "rb").read())
    return h.hexdigest()[:12]


def build(variant, device, n_envs, envs, policy="full", ablate_affect=False):
    context, _, salience = VARIANTS[variant]
    ctx_dim = CTX_DIM if context != "none" else 0
    wm = WorldModel(envs.envs[0].obs_shape, ctx_dim=ctx_dim, salience_gain=salience).to(device)
    ws = GlobalWorkspace(wm.dz, AFFECT_DIM, wm.n_actions).to(device)
    memory = EpisodicMemory(wm.dz)
    agent = BatchAgent(wm, ws, memory, n_envs, device, context=context, envs=envs.envs, policy=policy,
                       ablate_affect=ablate_affect)
    return wm, ws, memory, agent


def run(args):
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)  # keep the user's laptop responsive
    torch.manual_seed(args.seed_offset)
    np.random.seed(args.seed_offset)
    split = "train" if args.cmd == "train" else args.split
    sd = seeds(split)[: args.n_seeds] if args.n_seeds else seeds(split)
    regime = MANIFEST["regimes"][args.regime]
    envs = VecSurvival(sd, max_steps=MANIFEST["episode_steps"], **regime)
    ckpt_in = args.ckpt if args.cmd == "eval" else None
    variant = args.variant
    if ckpt_in:
        meta = torch.load(ckpt_in, map_location=device, weights_only=False)
        variant = meta["variant"]
    wm, ws, memory, agent = build(variant, device, len(sd), envs, args.policy, args.ablate_affect)
    if args.policy == "gated":
        from jarvis.world.gate import ConfidenceGate, FailureHead

        g = torch.load(args.gate, map_location=device, weights_only=False)
        head = FailureHead(g["dz"], g["n_actions"]).to(device)
        head.load_state_dict(g["head"])
        agent.gate = ConfidenceGate(head, **g["thresholds"])
    _, arousal_plasticity, _ = VARIANTS[variant]
    opt_wm = torch.optim.AdamW(wm.parameters(), lr=args.lr)
    opt_ws = torch.optim.AdamW(ws.parameters(), lr=args.lr)
    replay = Replay(capacity=args.replay)
    state_path = os.path.join(args.out, "state.pt")
    ep_counts = np.zeros(len(sd), dtype=int)

    if ckpt_in:
        wm.load_state_dict(meta["wm"]), ws.load_state_dict(meta["ws"])
        agent.memory = memory = meta["memory"]
    if os.path.exists(state_path):  # resume this run
        st = torch.load(state_path, map_location=device, weights_only=False)
        wm.load_state_dict(st["wm"]), ws.load_state_dict(st["ws"])
        opt_wm.load_state_dict(st["opt_wm"]), opt_ws.load_state_dict(st["opt_ws"])
        replay, agent.memory, ep_counts = st["replay"], st["memory"], st["ep_counts"]
        memory = agent.memory
        for e, (epi, poison, nxt) in zip(envs.envs, st["env_state"]):
            e.episode, e.poison, e.next_shift = epi, poison, nxt
        if st.get("contexts") is not None:
            agent.contexts = st["contexts"]
        print(f"resumed: {ep_counts.sum()} episodes done", flush=True)
    frozen = args.cmd == "eval"
    for m in (wm, ws):
        m.train(not frozen)

    def save():
        torch.save({"wm": wm.state_dict(), "ws": ws.state_dict(), "opt_wm": opt_wm.state_dict(),
                    "opt_ws": opt_ws.state_dict(), "replay": replay, "memory": agent.memory, "ep_counts": ep_counts,
                    "env_state": [(e.episode, e.poison, e.next_shift) for e in envs.envs],
                    "contexts": agent.contexts if variant == "v0.1" else None}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    run_id = f"{args.cmd}-{variant}-{args.policy}-{split}-{int(time.time())}"
    cfg_hash = hashlib.sha1(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:10]
    meta_common = {"run_id": run_id, "code_hash": code_hash(), "config_hash": cfg_hash, "variant": variant,
                   "policy": args.policy, "split": split, "regime_id": args.regime, "bench": MANIFEST["version"],
                   "checkpoint_id": os.path.basename(ckpt_in) if ckpt_in else None}
    obs = envs.reset_all()
    acc = [dict(steps=0, food=0, poison=0, lava=0, s2=0, err=0.0, cp=0.0, valence=0.0, mem_writes=0) for _ in sd]
    t_ep = [time.time()] * len(sd)
    zeros_ctx = np.zeros(CTX_DIM if agent.contexts is not None else 0, dtype=np.float32)
    log = open(os.path.join(args.out, "episodes.jsonl"), "a")
    transitions, last_save = 0, time.time()
    pending = [[] for _ in sd]
    target = args.episodes
    while (ep_counts < target).any():
        ctx = [c.vector() for c in agent.contexts] if agent.contexts is not None else [zeros_ctx] * len(sd)
        actions, infos = agent.act(obs)
        nxt, done, info = envs.step(actions)
        for i in range(len(sd)):
            if ep_counts[i] >= target:
                continue
            cp = agent.contexts[i].observe(obs[i]["view"], int(actions[i]), obs[i]["body"], nxt[i]["body"]) \
                if agent.contexts is not None else 0.0
            nctx = agent.contexts[i].vector() if agent.contexts is not None else zeros_ctx
            err = 0.0
            if args.policy != "random":
                with torch.no_grad():
                    t = lambda x: torch.as_tensor(np.asarray(x)[None], device=device)  # noqa: E731
                    err = float(wm.one_step_error(t(obs[i]["view"]), t(obs[i]["body"]), torch.tensor([int(actions[i])], device=device),
                                                  t(nxt[i]["view"]), t(nxt[i]["body"]),
                                                  t(ctx[i]) if len(zeros_ctx) else None, t(nctx) if len(zeros_ctx) else None))
            aff = agent.affects[i]
            aff_before = aff.signal()
            aff.update(obs[i]["body"], nxt[i]["body"], err, confidence=infos[i]["s1_conf"], change_prob=cp)
            if infos[i]["z"] is not None and abs(aff.state.valence) >= agent.memory.threshold:
                agent.memory.write(infos[i]["z"], aff.state.valence)
                acc[i]["mem_writes"] += 1
            if not frozen:  # buffered per env; flushed contiguously at episode end (sequence sampling needs it)
                pending[i].append(dict(view=obs[i]["view"].astype(np.uint8), body=obs[i]["body"], action=np.int64(actions[i]),
                           next_view=nxt[i]["view"].astype(np.uint8), next_body=nxt[i]["body"],
                           dead=np.float32(info[i]["dead"]), episode=np.int64(ep_counts[i] * 1000 + i),
                           plan=np.int64(infos[i]["plan"]), affect=aff_before,
                           mem=infos[i]["mem"] if infos[i]["mem"] is not None else np.zeros(wm.dz + 1, np.float32),
                           next_valence=np.float32(aff.state.valence), surprise=np.float32(err), ctx=ctx[i], next_ctx=nctx,
                           rule=np.int64(0 if info[i]["poison"] == "A" else 1)))
                transitions += 1
            a = acc[i]
            a["steps"] += 1
            a["s2"] += infos[i]["mode"] == "S2"
            a["err"] += err
            a["cp"] = max(a["cp"], cp)
            a["valence"] += aff.state.valence
            if info[i]["event"]:
                a[info[i]["event"]] += 1
            if done[i]:
                health = float(nxt[i]["body"][1])
                rec = {**meta_common, "seed": sd[i], "episode": int(ep_counts[i]), "rule": info[i]["poison"],
                       "steps": a["steps"], "success": bool(a["steps"] >= MANIFEST["episode_steps"] and health >= 0.5),
                       "terminated": bool(info[i]["dead"]), "energy": round(float(nxt[i]["body"][0]), 3),
                       "health": round(health, 3), "food": a["food"], "poison": a["poison"], "lava": a["lava"],
                       "planner_calls": int(a["s2"]), "latent_prediction_error": round(a["err"] / a["steps"], 5),
                       "change_prob_max": round(a["cp"], 3), "mean_valence": round(a["valence"] / a["steps"], 3),
                       "memory_writes": a["mem_writes"], "wall_time_ms": int((time.time() - t_ep[i]) * 1000),
                       "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0}
                log.write(json.dumps(rec) + "\n")
                for tr in pending[i]:
                    replay.add(**tr)
                pending[i] = []
                ep_counts[i] += 1
                acc[i] = dict(steps=0, food=0, poison=0, lava=0, s2=0, err=0.0, cp=0.0, valence=0.0, mem_writes=0)
                t_ep[i] = time.time()
                nxt[i] = envs.envs[i].reset()
        obs = nxt

        if not frozen and replay.n > args.batch * 4 and transitions >= args.transitions_per_update:
            transitions = 0
            cp_max = max(a["cp"] for a in acc)
            lr_mult = float(np.clip(1 + 4 * cp_max, 1, 3)) if arousal_plasticity else 1.0
            for opt in (opt_wm, opt_ws):
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_mult
            for _ in range(args.updates):
                seq, starts = replay.sample_seq(args.batch, 5, device, recent_frac=0.5 * cp_max if arousal_plasticity else 0.0)
                loss, _, _ = wm.loss(seq)
                opt_wm.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                opt_wm.step()
                wm.update_target()
                b = replay.sample(args.batch, device, ["view", "body", "plan", "affect", "mem", "next_valence", "surprise", "ctx"])
                with torch.no_grad():
                    z = wm.encode(b["view"], b["body"], b["ctx"] if wm.ctx_dim else None)
                    preds = torch.stack([wm.predict(z, torch.full((z.size(0),), k, device=device, dtype=torch.long))
                                         for k in range(wm.n_actions)], 1).mean(1)
                src = {"perception": z, "prediction": preds, "affect": b["affect"], "memory": b["mem"]}
                loss, _ = ws.loss(src, b["plan"].long(), b["plan"] >= 0, b["next_valence"], b["surprise"])
                opt_ws.zero_grad(set_to_none=True)
                loss.backward()
                opt_ws.step()
        if time.time() - last_save > args.save_minutes * 60:
            log.flush()
            if not frozen:
                save()
            last_save = time.time()
            print(f"{ep_counts.sum()}/{target * len(sd)} episodes", flush=True)
    log.close()
    if not frozen:
        save()
        torch.save({"wm": wm.state_dict(), "ws": ws.state_dict(), "memory": agent.memory, "variant": variant,
                    "code_hash": code_hash()}, os.path.join(args.out, "brain.pt"))
    print("RUN_DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="v0.1", choices=list(VARIANTS))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--policy", default="full", choices=["random", "s1", "memory", "full", "gated", "mpc"])
    p.add_argument("--gate", default=None, help="gate.pt from evaluation/fit_gate.py (for --policy gated)")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--regime", default="swap", choices=["stable", "swap"])
    p.add_argument("--episodes", type=int, default=60, help="episodes per seed")
    p.add_argument("--n_seeds", type=int, default=0, help="use only the first n seeds of the split (smoke/pilot)")
    p.add_argument("--ablate_affect", action="store_true")
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--transitions_per_update", type=int, default=50)
    p.add_argument("--replay", type=int, default=200_000)
    p.add_argument("--save_minutes", type=float, default=10)
    p.add_argument("--threads", type=int, default=0, help="cap CPU threads (0 = torch default)")
    p.add_argument("--cpu", action="store_true", help="never use the GPU")
    a = p.parse_args()
    if a.cmd == "eval" and not a.ckpt and a.policy != "random":
        p.error("eval needs --ckpt (frozen brain) unless --policy random")
    if a.cmd == "eval" and not a.ckpt:
        a.variant = "v0"
    run(a)


if __name__ == "__main__":
    main()
