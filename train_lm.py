"""Pretrain the hybrid LM backbone. Designed for Colab T4: fp16 autocast + GradScaler, checkpoints to
Google Drive, auto-resume after disconnects.

  python train_lm.py --data_dir /content/drive/MyDrive/jarvis/data/tinystories \
                     --out_dir /content/drive/MyDrive/jarvis/runs/lm-small --preset small
"""
import argparse
import json
import os
import time
from dataclasses import replace

import torch

from jarvis.lm import PRESETS, HybridLM, LMConfig
from jarvis.lm.data import TokenLoader
from jarvis.lm.optim import build_optimizers, wsd_lr_factor


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--pattern", default=None, help="override block pattern, e.g. DDDA or RRRA or AAAA (ablations)")
    p.add_argument("--bitnet", action="store_true", help="BitNet b1.58-style ternary linears")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW LR (embeddings, norms)")
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--no_muon", action="store_true")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--max_minutes", type=float, default=0, help="stop + save cleanly before Colab kills the session")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, iters=20):
    model.eval()
    losses = [model(*loader.batch())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    weights_path = os.path.join(args.out_dir, "weights.pt")
    resume = torch.load(ckpt_path, map_location=device, weights_only=True) if os.path.exists(ckpt_path) else None
    light = None
    if resume is None and os.path.exists(weights_path):  # recovery from the small fp16 copy (no optimizer state)
        light = torch.load(weights_path, map_location=device, weights_only=True)
    if resume or light:
        cfg = LMConfig(**(resume or light)["config"])
        print(f"resuming from step {(resume or light)['step']}" + (" (weights only, fresh optimizer)" if light else ""))
    else:
        cfg = replace(PRESETS[args.preset], block_size=args.block_size, bitnet=args.bitnet)
        if args.pattern:
            cfg = replace(cfg, pattern=args.pattern)

    model = HybridLM(cfg).to(device)
    opts = build_optimizers(model, args.lr, args.muon_lr, use_muon=not args.no_muon)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    if resume:
        model.load_state_dict(resume["model"])
        for opt, state in zip(opts, resume["opts"]):
            opt.load_state_dict(state)
        scaler.load_state_dict(resume["scaler"])
        step = resume["step"]
    elif light:
        model.load_state_dict({k: v.float() for k, v in light["model"].items()})
        step = light["step"]
    print(f"params: {model.num_params() / 1e6:.1f}M non-embedding, {model.num_params(False) / 1e6:.1f}M total | {cfg}")

    fwd = torch.compile(model) if args.compile else model
    train = TokenLoader(args.data_dir, "train", cfg.block_size, args.batch_size, device)
    val = TokenLoader(args.data_dir, "val", cfg.block_size, args.batch_size, device)
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    log_path = os.path.join(args.out_dir, "log.jsonl")

    def save():
        torch.save(
            {"model": model.state_dict(), "opts": [o.state_dict() for o in opts], "scaler": scaler.state_dict(),
             "step": step, "config": cfg.to_dict()},
            ckpt_path + ".tmp",
        )
        os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic: a Colab disconnect mid-save can't corrupt it
        # Small fp16 weights-only copy (~1/5 the size): what gets backed up and re-uploaded after a VM loss.
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "step": step,
                    "config": cfg.to_dict()}, weights_path + ".tmp")
        os.replace(weights_path + ".tmp", weights_path)

    t_start = t0 = time.time()
    while step < args.max_steps:
        factor = wsd_lr_factor(step, args.max_steps, args.warmup)
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * factor

        for _ in range(args.grad_accum):
            x, y = train.batch()
            with amp:
                _, loss = fwd(x, y)
            scaler.scale(loss / args.grad_accum).backward()
        for opt in opts:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            dt = time.time() - t0
            tok_s = 50 * args.batch_size * args.grad_accum * cfg.block_size / dt
            mem = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak" if device == "cuda" else ""
            print(f"step {step} | loss {loss.item():.3f} | gnorm {grad_norm:.2f} | lr x{factor:.2f} | {tok_s / 1e3:.1f}k tok/s{mem}")
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.max_steps:
            with amp:
                vl = evaluate(fwd, val)
            print(f"== step {step} val loss {vl:.3f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "train": loss.item(), "val": vl, "time": time.time()}) + "\n")
        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if step % args.save_every == 0 or step == args.max_steps or out_of_time:
            save()
            print(f"saved {ckpt_path}")
        if out_of_time:
            print("time budget reached; rerun the same command to resume")
            break


if __name__ == "__main__":
    main()
