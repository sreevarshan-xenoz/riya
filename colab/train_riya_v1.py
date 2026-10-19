"""On Colab (a free T4 is enough): fine-tune Riya v1, a LoRA on the LFM2.5-1.2B weights she serves.

    python -m colab.train_riya_v1 --smoke   # 8 rows, one optimizer step, export, load in llama-server: prints SMOKE_OK
    python -m colab.train_riya_v1           # the full run: prints TRAIN_DONE

Reads sft.jsonl (riya.train.filter_sft), trains on the assistant tokens only (riya.train.fmt's mask), writes the
adapter, converts it with llama.cpp's convert_lora_to_gguf.py to riya-v1-lora.gguf, then proves llama-server loads it
(--lora, /lora-adapters) and that the HF chat template renders the same prompt as llama-server's. train_report.json
records the base, the config and the loss curve. LoRA weights in fp32 under autocast over fp16 base weights; nothing is quantised for training.

torch, transformers, peft and huggingface_hub are imported inside the functions that need them, so the pure helpers
(config, target regex, base resolution, command lines, report, batching) import and test on a CPU.
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.4 and §4.1."""
import argparse
import gc
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from riya.train.fmt import (IGNORE_INDEX, MAX_LEN, encode_row, read_jsonl, template_matches, validate_sft_row)

DEFAULT_BASE = "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated"  # the safetensors twin of the GGUF she serves
OFFICIAL_BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
BASE_SEARCH = "Huihui-LFM2.5-1.2B-Instruct-abliterated"
SERVED_GGUF_REPO = "mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF"  # its Q4_K_M is what she serves

# LFM2's short-conv blocks also have in_proj/out_proj, so the targets are pinned by path: attention and MLP only.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3")
TARGET_REGEX = r".*\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|feed_forward\.(?:w1|w2|w3))$"
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        "target_modules": TARGET_REGEX}
TRAIN = {"epochs": 2, "lr": 2e-4, "scheduler": "cosine", "warmup_ratio": 0.03, "max_len": MAX_LEN, "batch": 4,
         "grad_accum": 4, "seed": 0}
SMOKE_ROWS = 8
CLIP_NORM = 1.0

OUT_NAME = "riya-v1-lora.gguf"
REPORT_NAME = "train_report.json"
SMOKE_PORT = 8096
RIYA_V1_DIR = os.environ.get("RIYA_V1_DIR", "/content/drive/MyDrive/riya_v1")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/content/llama.cpp")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/content/bin/llama-server")
WORK_DIR = os.environ.get("RIYA_V1_WORK", "/content/riya_v1_work")  # local disk: the base, the served GGUF, logs

MISMATCH_BANNER = ("BASE MISMATCH: training on LiquidAI/LFM2.5-1.2B-Instruct, not the abliterated weights she "
                   "serves — the champion gate decides")
_QUANTISED = ("gguf", "mlx", "awq", "gptq", "4bit", "8bit", "fp8", "int4", "int8", "exl2", "bnb")


# ---- pure helpers -------------------------------------------------------------------------------------------

def peft_config_kwargs() -> dict:
    """The arguments of peft.LoraConfig (a copy of LORA)."""
    return dict(LORA)


def selected_modules(names: Iterable[str]) -> list[str]:
    """The module names LoRA adapts: those TARGET_REGEX matches in full, in order."""
    return [n for n in names if re.fullmatch(TARGET_REGEX, n)]


def bf16_capable(capability: tuple[int, int]) -> bool:
    """bf16 only on Ampere or newer (compute capability 8+). torch.cuda.is_bf16_supported() also says yes on a T4
    (sm75) by emulation, which is slow, so the decision is made from the capability itself."""
    return capability[0] >= 8


def train_settings(bf16_supported: bool) -> dict:
    """TRAIN plus the compute dtype: bf16 where the GPU supports it, fp16 otherwise (a T4)."""
    return TRAIN | {"dtype": "bfloat16" if bf16_supported else "float16"}


def plain_weights(repo_id: str) -> bool:
    """False for repos of quantised weights (GGUF, MLX, AWQ, ...): a LoRA must sit on the full-precision ones."""
    low = repo_id.lower()
    return not any(q in low for q in _QUANTISED)


def resolve_base(environ: Mapping[str, str], search: Callable[[str], list[str]]) -> tuple[str, bool]:
    """(HF repo id, base_mismatch): BASE_ID when set; else the abliterated safetensors repo found by search(BASE_SEARCH)
    (DEFAULT_BASE when it is among the hits, else the first hit); else the official base, which is a mismatch with
    the weights she serves. A search that raises counts as no hits."""
    pinned = (environ.get("BASE_ID") or "").strip()
    if pinned:
        return pinned, False
    try:
        hits = search(BASE_SEARCH)
    except Exception as e:
        print(f"base search failed ({e})", flush=True)
        hits = []
    if DEFAULT_BASE in hits:
        return DEFAULT_BASE, False
    if hits:
        return hits[0], False
    return OFFICIAL_BASE, True


def convert_cmd(llama_dir: str, adapter_dir: str, base_dir: str, out_path: str) -> list[str]:
    """llama.cpp's LoRA converter, against the same base directory the adapter was trained on."""
    return [sys.executable, f"{llama_dir}/convert_lora_to_gguf.py", adapter_dir, "--base", base_dir,
            "--outfile", out_path, "--outtype", "f16"]


def smoke_server_cmd(server: str, gguf: str, lora: str, port: int = SMOKE_PORT) -> list[str]:
    """llama-server on the served GGUF with the adapter at its default scale of 1.0."""
    return [server, "-m", gguf, "--lora", lora, "--port", str(port), "-c", "4096", "-ngl", "99", "--no-webui"]


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path)[-1]


def smoke_problems(served: list[str], lora: str, server_prompt: str, hf_prompt: str) -> list[str]:
    """What the smoke step found wrong: the adapter isn't in llama-server's /lora-adapters, or the HF chat template
    renders a different prompt from llama-server's."""
    problems = []
    if not any(_basename(p) == _basename(lora) for p in served):
        problems.append("llama-server didn't load the adapter")
    if not template_matches(server_prompt, hf_prompt):
        problems.append("the HF chat template differs from llama-server's")
    return problems


def make_report(*, base_id: str, base_mismatch: bool, rows: int, skipped: int, losses: list[tuple[int, float]],
                smoke: bool, gpu: str, settings: dict) -> dict:
    """train_report.json: the base (and whether it is the wrong one), the data, the loss curve and the config."""
    return {"base_id": base_id, "base_mismatch": base_mismatch, "rows": rows, "skipped_too_long": skipped,
            "loss_curve": [[s, v] for s, v in losses], "smoke": smoke, "gpu": gpu, "lora": peft_config_kwargs(),
            "train": settings, "adapter": OUT_NAME}


def epoch_batches(n: int, batch: int, seed: int, epoch: int) -> list[list[int]]:
    """Row indices 0..n-1 shuffled for this epoch (seeded) and cut into batches of up to `batch`."""
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return [order[i:i + batch] for i in range(0, n, batch)]


def total_steps(n: int, epochs: int, batch: int, accum: int) -> int:
    """Optimizer steps in the run: per epoch, the micro-batches grouped by `accum` (the last group may be short)."""
    return epochs * math.ceil(math.ceil(n / batch) / accum)


def pad_batch(items: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[list[list[int]], list[list[int]],
                                                                               list[list[int]]]:
    """(input ids, labels, attention mask) right-padded to the longest row; padded labels are IGNORE_INDEX."""
    width = max(len(ids) for ids, _ in items)
    ids = [i + [pad_id] * (width - len(i)) for i, _ in items]
    labels = [lab + [IGNORE_INDEX] * (width - len(lab)) for _, lab in items]
    mask = [[1] * len(i) + [0] * (width - len(i)) for i, _ in items]
    return ids, labels, mask


# ---- GPU-side steps (heavy imports stay inside) --------------------------------------------------------------

def hf_safetensors_search(query: str) -> list[str]:
    """HF repo ids matching the query that hold full-precision *.safetensors weights, most downloaded first."""
    from huggingface_hub import HfApi

    api = HfApi()
    hits = []
    for m in api.list_models(search=query, sort="downloads", limit=10):
        if not plain_weights(m.id):
            continue
        try:  # one flaky repo listing must not sink the whole search (that would train on the wrong base)
            if any(f.endswith(".safetensors") for f in api.list_repo_files(m.id)):
                hits.append(m.id)
        except Exception as e:
            print(f"couldn't list {m.id} ({e})", flush=True)
    return hits


def download_served_gguf() -> str:
    """The Q4_K_M GGUF she serves (the abliterated one), from SERVED_GGUF_REPO."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(SERVED_GGUF_REPO) if f.endswith(".gguf") and "q4_k_m" in f.lower()]
    if not files:
        raise SystemExit(f"no Q4_K_M GGUF in {SERVED_GGUF_REPO}; pass --served-gguf")
    return hf_hub_download(SERVED_GGUF_REPO, files[0], local_dir=f"{WORK_DIR}/served")


def _get_json(url: str, timeout: float = 10):
    import json
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post_json(url: str, body: dict, timeout: float = 30):
    import json
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def smoke_load(served_gguf: str, lora: str, messages: list[dict], hf_prompt: str) -> list[str]:
    """Starts llama-server on the served GGUF with --lora, waits for /health, reads /lora-adapters and the rendered
    chat prompt, and always stops the server it started. Returns smoke_problems(...)."""
    base = f"http://127.0.0.1:{SMOKE_PORT}"
    os.makedirs(WORK_DIR, exist_ok=True)
    proc = subprocess.Popen(smoke_server_cmd(LLAMA_SERVER, served_gguf, lora), stdout=subprocess.DEVNULL,
                            stderr=open(f"{WORK_DIR}/smoke_server.log", "w"))
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({proc.returncode}) loading the adapter; see "
                                 f"{WORK_DIR}/smoke_server.log")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("llama-server never became healthy")
        adapters = _get_json(f"{base}/lora-adapters")
        served = [a.get("path", "") for a in adapters]
        rendered = _post_json(f"{base}/apply-template", {"messages": messages, "add_generation_prompt": True})
        return smoke_problems(served, lora, rendered["prompt"], hf_prompt)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()  # our own child


def _run(args: argparse.Namespace) -> int:
    import json

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise SystemExit("no GPU: Runtime > Change runtime type > T4 GPU")
    out_dir = Path(args.out_dir) / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. the base: the weights she serves, else the official ones (flagged)
    base_id, mismatch = resolve_base(os.environ, hf_safetensors_search)
    if mismatch:
        bar = "!" * 78
        print(f"{bar}\n{MISMATCH_BANNER}\n{bar}", flush=True)
    print("base:", base_id, flush=True)
    base_dir = snapshot_download(base_id, local_dir=f"{WORK_DIR}/base/{base_id.replace('/', '__')}",
                                ignore_patterns=["*.gguf"])  # one folder per repo: no overlay on a rerun

    # 2. the model, with LoRA on attention and MLP only
    settings = train_settings(bf16_capable(torch.cuda.get_device_capability()))
    dtype = getattr(torch, settings["dtype"])
    print(f"GPU {torch.cuda.get_device_name(0)}: training in {settings['dtype']}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=dtype).cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**peft_config_kwargs()))
    chosen = selected_modules(n for n, _ in model.named_modules())
    assert chosen, "the LoRA target regex matched no module"
    assert not any(".conv." in n for n in chosen), "the LoRA targets include a short-conv layer"
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and ".conv." not in n for n, _ in trainable), "unexpected trainable weights"
    for _, p in trainable:
        p.data = p.data.float()  # fp32 LoRA weights; the forward runs under autocast
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    print(f"LoRA on {len(chosen)} modules, {sum(p.numel() for _, p in trainable):,} trainable parameters", flush=True)

    # 3. the data: each row's prompt masked, its answer trained on
    sft = read_jsonl(Path(args.sft))
    for row in sft:
        validate_sft_row(row)
    if args.smoke:
        sft = sft[:SMOKE_ROWS]
    data, skipped = [], 0
    for row in sft:
        enc = encode_row(tok, row["messages"], settings["max_len"])
        if enc is None:
            skipped += 1
        else:
            data.append(enc)
    if not data:
        raise SystemExit("no trainable rows")
    print(f"{len(data)} rows, {skipped} skipped for length", flush=True)

    # 4. the loop
    accum = min(settings["grad_accum"], math.ceil(len(data) / settings["batch"]))
    steps = 1 if args.smoke else total_steps(len(data), settings["epochs"], settings["batch"], accum)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=settings["lr"], weight_decay=0.0)
    warmup = 0 if args.smoke else math.ceil(settings["warmup_ratio"] * steps)  # the smoke step must have a real lr
    sched = get_cosine_schedule_with_warmup(opt, warmup, steps)
    fp16 = settings["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    losses: list[tuple[int, float]] = []
    step, acc_loss, micro = 0, 0.0, 0
    t0 = time.time()
    for epoch in range(1 if args.smoke else settings["epochs"]):
        batches = epoch_batches(len(data), settings["batch"], settings["seed"], epoch)
        for b_i, idx in enumerate(batches):
            ids, labels, mask = pad_batch([data[i] for i in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=dtype):
                out = model(input_ids=torch.tensor(ids).cuda(), attention_mask=torch.tensor(mask).cuda(),
                            labels=torch.tensor(labels).cuda())
            scaler.scale(out.loss / accum).backward()
            acc_loss += out.loss.item()
            micro += 1
            if micro == accum or b_i == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                losses.append((step, round(acc_loss / micro, 4)))
                acc_loss, micro = 0.0, 0
                if step % 10 == 0 or step == steps:
                    print(f"step {step}/{steps} loss {losses[-1][1]} ({time.time() - t0:.0f}s)", flush=True)
                if step >= steps:
                    break
        if step >= steps:
            break

    # 5. export: the PEFT adapter, then llama.cpp's LoRA GGUF against the same base
    adapter_dir = str(out_dir / "adapter")
    model.save_pretrained(adapter_dir)
    gpu = torch.cuda.get_device_name(0)
    del model, opt, sched, scaler, trainable, out
    gc.collect()
    torch.cuda.empty_cache()  # llama-server needs the memory next
    out_gguf = str(out_dir / OUT_NAME)
    subprocess.run(convert_cmd(LLAMA_DIR, adapter_dir, base_dir, out_gguf), check=True)

    # 6. report (before the smoke, so a failed load keeps the loss curve)
    report = make_report(base_id=base_id, base_mismatch=mismatch, rows=len(data), skipped=skipped, losses=losses,
                         smoke=args.smoke, gpu=gpu, settings=settings)
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 7. smoke: llama-server loads the adapter, and the HF template matches llama-server's
    served_gguf = args.served_gguf or download_served_gguf()
    prompt_msgs = sft[0]["messages"][:-1]
    hf_prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    problems = smoke_load(served_gguf, out_gguf, prompt_msgs, hf_prompt)
    if problems:
        raise SystemExit("smoke failed: " + "; ".join(problems))
    if mismatch:
        print(f"\n{MISMATCH_BANNER}\n", flush=True)
    print("SMOKE_OK" if args.smoke else "TRAIN_DONE", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Riya v1 trainer: sft.jsonl -> riya-v1-lora.gguf")
    ap.add_argument("--sft", default=f"{RIYA_V1_DIR}/sft.jsonl")
    ap.add_argument("--out-dir", default=RIYA_V1_DIR)
    ap.add_argument("--smoke", action="store_true", help="8 rows, one optimizer step, then export and load (under "
                                                         "<out-dir>/smoke)")
    ap.add_argument("--served-gguf", default=None, help="her served Q4_K_M GGUF (default: downloaded from "
                                                        f"{SERVED_GGUF_REPO})")
    return _run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
