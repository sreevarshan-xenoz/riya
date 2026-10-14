"""Teacher writes Riya's replies on the T4 with Hugging Face transformers (4-bit, no llama.cpp compile needed).

Reads /content/voice/synth_requests.jsonl, writes /content/voice/riya_synth.jsonl (resumable, flushed every batch).
Env: N (rows, default 5000), BATCH (default 16), TEACHER (HF repo id; default: first available abliterated model).
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

N = int(os.environ.get("N", "5000"))
BATCH = int(os.environ.get("BATCH", "16"))
OUT = "/content/voice/riya_synth.jsonl"
CANDIDATES = [os.environ["TEACHER"]] if os.environ.get("TEACHER") else [
    "huihui-ai/Huihui-Qwen3-8B-abliterated-v2", "huihui-ai/Qwen3-8B-abliterated", "mlabonne/Qwen3-8B-abliterated",
    "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated", "huihui-ai/Qwen3-4B-abliterated",
]


def pick_teacher():
    from huggingface_hub import HfApi

    api = HfApi()
    for repo in CANDIDATES:
        try:
            api.model_info(repo)
            return repo
        except Exception:
            continue
    raise SystemExit("no teacher candidate found on the Hub")


def main():
    teacher = pick_teacher()
    print("teacher:", teacher, flush=True)
    tok = AutoTokenizer.from_pretrained(teacher)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        teacher, device_map="cuda", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4"))
    model.eval()
    done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    reqs = [json.loads(l) for l in open("/content/voice/synth_requests.jsonl")][:N]
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)
    t0, made = time.time(), 0
    with open(OUT, "a") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i: i + BATCH]
            texts = []
            for r in batch:
                msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}]
                try:
                    texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=False))
                except TypeError:
                    texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=110, do_sample=True, temperature=0.8, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            for r, seq in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                reply = tok.decode(seq, skip_special_tokens=True).split("</think>")[-1].strip().strip('"')
                if not reply:
                    continue
                f.write(json.dumps({"id": r["id"], "source": "synth", "split": "train",
                                    "context": [{"role": "user", "text": r["user"]}], "response": reply,
                                    "affect": r["affect"], "label": r["affect_description"]}) + "\n")
                made += 1
            f.flush()
            print(f"{len(done) + made}/{len(reqs)} ({made / (time.time() - t0):.2f}/s)", flush=True)
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
