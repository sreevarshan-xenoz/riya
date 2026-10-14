"""Affect-grounded SFT for the butler's voice (Notion page 23, session 2).

Starts from a pretrained HybridLM checkpoint and trains on affect-labelled dialogue (jarvis/voice): loss only on the
response tokens, the 8-dim affect vector fed through `affect_proj`, affect dropout (vector zeroed for a share of
rows, enabling classifier-free guidance on affect at inference), and a share of plain pretraining text to limit
forgetting.

Human-dialogue rows (EmpatheticDialogues, DailyDialog) teach affect -> tone grounding under a neutral "human speaker"
system prompt; the butler persona prompt (jarvis/voice/persona.py) is reserved for synthetic butler rows so the
butler never learns to claim a human life story it does not have.

  python train_voice.py --init runs/lm-small/ckpt.pt --data data/voice/empathetic_dialogues.jsonl \
      data/voice/daily_dialog.jsonl --out_dir runs/voice --steps 3000
"""
import argparse
import json
import os
import random
import time

import numpy as np
import tiktoken
import torch

from jarvis.lm import HybridLM, LMConfig
from jarvis.lm.data import TokenLoader
from jarvis.lm.optim import build_optimizers, wsd_lr_factor
from jarvis.voice.chat import format_row
from jarvis.voice.persona import system_prompt

HUMAN_SYSTEM = "A person talking with someone. Their feelings show in how they speak."
BUTLER_SYSTEM = system_prompt()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="pretrained LM checkpoint (train_lm.py output)")
    p.add_argument("--data", nargs="+", required=True, help="voice JSONL files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pretrain_dir", default=None, help="token .bin dir mixed in to limit forgetting")
    p.add_argument("--pretrain_frac", type=float, default=0.15)
    p.add_argument("--affect_dropout", type=float, default=0.15)
    p.add_argument("--neutral_keep", type=float, default=0.25, help="keep this share of 'no emotion' rows")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--muon_lr", type=float, default=0.005)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_minutes", type=float, default=0)
    return p.parse_args()


def load_rows(paths, neutral_keep, rng):
    rows = {"train": [], "val": []}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("label") in ("no emotion", "neutral") and rng.random() > neutral_keep:
                continue
            split = "val" if r.get("split") in ("validation", "val", "test") else "train"
            rows[split].append(r)
    return rows


def encode(row, enc, block_size):
    system = BUTLER_SYSTEM if row.get("source", "").startswith("synth") else HUMAN_SYSTEM
    f = format_row(row, system)
    p, c = enc.encode_ordinary(f["prompt"]), enc.encode_ordinary(f["completion"])
    c = c[: block_size]
    p = p[-(block_size + 1 - len(c)) :] if len(p) + len(c) > block_size + 1 else p
    ids = p + c
    labels = [-1] * (len(p) - 1) + c  # predict token t+1 from t; loss only on completion tokens
    return ids[:-1], labels


def make_batch(rows, enc, args, device, drop: bool):
    xs, ys, affs = [], [], []
    for r in rows:
        x, y = encode(r, enc, args.block_size)
        xs.append(x)
        ys.append(y)
        a = np.asarray(r["affect"], dtype=np.float32)
        affs.append(np.zeros_like(a) if drop and random.random() < args.affect_dropout else a)
    T = max(len(x) for x in xs)
    X = torch.full((len(xs), T), 50256, dtype=torch.long)
    Y = torch.full((len(xs), T), -1, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, : len(x)], Y[i, : len(y)] = torch.tensor(x), torch.tensor(y)
    return X.to(device), Y.to(device), torch.from_numpy(np.stack(affs)).to(device)


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(0)
    enc = tiktoken.get_encoding("gpt2")
    rows = load_rows(args.data, args.neutral_keep, rng)
    print(f"rows: {len(rows['train'])} train / {len(rows['val'])} val")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    resume = os.path.exists(ckpt_path)
    src = torch.load(ckpt_path if resume else args.init, map_location=device, weights_only=True)
    cfg = LMConfig(**src["config"])
    model = HybridLM(cfg).to(device)
    model.load_state_dict(src["model"])
    opts = build_optimizers(model, args.lr, args.muon_lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    if resume:
        for opt, state in zip(opts, src["opts"]):
            opt.load_state_dict(state)
        scaler.load_state_dict(src["scaler"])
        step = src["step"]
        print(f"resuming voice SFT at step {step}")
    pretrain = TokenLoader(args.pretrain_dir, "train", args.block_size, args.batch_size, device) if args.pretrain_dir else None
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    def save():
        torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts], "scaler": scaler.state_dict(),
                    "step": step, "config": cfg.to_dict()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        weights_path = os.path.join(args.out_dir, "weights.pt")
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "step": step,
                    "config": cfg.to_dict()}, weights_path + ".tmp")
        os.replace(weights_path + ".tmp", weights_path)

    @torch.no_grad()
    def evaluate(n=20):
        model.eval()
        out = {}
        for name, drop_all in (("val", False), ("val_noaffect", True)):
            losses = []
            for i in range(n):
                batch = rows["val"][i * args.batch_size : (i + 1) * args.batch_size]
                X, Y, A = make_batch(batch, enc, args, device, drop=False)
                with amp:
                    _, loss = model(X, Y, affect=torch.zeros_like(A) if drop_all else A)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out  # val_noaffect - val > 0 means the model is actually using the affect vector

    t_start = t0 = time.time()
    while step < args.steps:
        factor = wsd_lr_factor(step, args.steps, warmup=100)
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * factor
        for _ in range(args.grad_accum):
            if pretrain and random.random() < args.pretrain_frac:
                X, Y = pretrain.batch()
                A = torch.zeros(X.size(0), cfg.affect_dim, device=device)
            else:
                X, Y, A = make_batch(random.sample(rows["train"], args.batch_size), enc, args, device, drop=True)
            with amp:
                _, loss = model(X, Y, affect=A)
            scaler.scale(loss / args.grad_accum).backward()
        for opt in opts:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.3f} | lr x{factor:.2f} | {(time.time() - t0) / 50:.2f}s/step", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate()
            print(f"== step {step} val {ev['val']:.3f} | val without affect {ev['val_noaffect']:.3f} "
                  f"(affect gain {ev['val_noaffect'] - ev['val']:+.4f})", flush=True)
            with open(os.path.join(args.out_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, **ev}) + "\n")
        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if step % args.save_every == 0 or step == args.steps or out_of_time:
            save()
        if out_of_time:
            break


if __name__ == "__main__":
    main()
