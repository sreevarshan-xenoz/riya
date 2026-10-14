"""Fine-tune Riya v1: LoRA on an (uncensored) small base model, fp16 for Colab T4.

  python train_riya_lora.py --base huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated \
      --data /content/voice/riya_synth.jsonl --out_dir /content/runs/riya-lora

Riya's internal state enters as a short tag at the end of the system prompt (v1, Notion page 25):
  [Riya's state: feeling <description> | valence +0.31 arousal 0.62 confidence 0.70 energy 0.20 curiosity 0.80]
At runtime the affect core (jarvis/world/affect.py) writes the same tag. Loss is on Riya's replies only.
No refusal data is used, so the base model's uncensored behaviour is kept.
"""
import argparse
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from jarvis.voice.persona import NAME, system_prompt

AFFECT_NAMES = ["valence", "arousal", "confidence", "mood", "energy_drive", "health_drive", "curiosity", "surprise"]


def state_tag(affect, description=None):
    a = dict(zip(AFFECT_NAMES, affect))
    words = f"feeling {description} | " if description else ""
    return (f"[{NAME}'s state: {words}valence {a['valence']:+.2f} arousal {a['arousal']:.2f} "
            f"confidence {a['confidence']:.2f} mood {a['mood']:+.2f} energy {1 - a['energy_drive']:.2f} "
            f"health {1 - a['health_drive']:.2f} curiosity {a['curiosity']:.2f} surprise {a['surprise']:.2f}]")


def to_messages(row):
    system = system_prompt() + "\n" + state_tag(row["affect"], row.get("label"))
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": t["role"], "content": t["text"]} for t in row["context"]]
    msgs.append({"role": "assistant", "content": row["response"]})
    return msgs


def encode(tok, row, max_len):
    msgs = to_messages(row)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels[: len(ids)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--val_frac", type=float, default=0.05)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(0)

    rows = [json.loads(line) for path in a.data for line in open(path, encoding="utf-8")]
    rows = [r for r in rows if r.get("response") and len(r["response"]) > 2]
    random.shuffle(rows)
    n_val = max(20, int(len(rows) * a.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    with open(os.path.join(a.out_dir, "val_rows.jsonl"), "w") as f:  # held out for evaluation
        f.writelines(json.dumps(r) + "\n" for r in val)

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules="all-linear")
    model = get_peft_model(model, lora)
    for n, prm in model.named_parameters():  # LoRA weights in fp32 for stable fp16 mixed precision
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()

    def batches(data, shuffle=True):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), a.batch):
            enc = [encode(tok, data[j], a.max_len) for j in idx[i : i + a.batch]]
            T = max(len(x) for x, _ in enc)
            X = torch.full((len(enc), T), tok.pad_token_id)
            Y = torch.full((len(enc), T), -100)
            M = torch.zeros((len(enc), T), dtype=torch.long)
            for k, (x, y) in enumerate(enc):
                X[k, : len(x)], Y[k, : len(y)], M[k, : len(x)] = torch.tensor(x), torch.tensor(y), 1
            yield X.cuda(), Y.cuda(), M.cuda()

    steps = int(len(train) / a.batch * a.epochs)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(10, steps // 20), steps)
    scaler = torch.amp.GradScaler("cuda")

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for X, Y, M in batches(val, shuffle=False):
            with torch.autocast("cuda", dtype=torch.float16):
                losses.append(model(input_ids=X, attention_mask=M, labels=Y).loss.item())
        model.train()
        return sum(losses) / len(losses)

    print(f"train {len(train)} / val {len(val)} rows, {steps} steps; val loss before: {evaluate():.3f}", flush=True)
    step, epoch = 0, 0
    while step < steps:
        for X, Y, M in batches(train):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=X, attention_mask=M, labels=Y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f}", flush=True)
            if step % 200 == 0 or step == steps:
                print(f"== step {step} val loss {evaluate():.3f}", flush=True)
                model.save_pretrained(os.path.join(a.out_dir, "adapter"))
            if step >= steps:
                break
        epoch += 1
    merged = model.merge_and_unload()
    merged = merged.half()
    merged.save_pretrained(os.path.join(a.out_dir, "merged"), safe_serialization=True)
    tok.save_pretrained(os.path.join(a.out_dir, "merged"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
