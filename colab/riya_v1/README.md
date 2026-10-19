# Riya v1: the Colab runbook

Fine-tune a LoRA on the LFM2.5-1.2B weights Riya serves, then let the champion gate decide whether to adopt it.
Run the cells in order. "Local" cells run on your PC (PowerShell, in `E:\riya`); "Colab" cells run in a Colab notebook
on a free T4 (Runtime > Change runtime type > T4 GPU). Everything the two sides share lives in one Drive folder,
`G:/My Drive/riya_v1/` locally and `/content/drive/MyDrive/riya_v1` on Colab.

The data is synthetic (seeded fake names, facts and places). Nothing of yours goes to Colab except code.

## 1. Local: build the requests and sync the code

```powershell
.venv\Scripts\python -m riya.train.build_sft
New-Item -ItemType Directory -Force "G:/My Drive/riya_v1" | Out-Null
Copy-Item scratch_runs/riya_v1/requests.jsonl "G:/My Drive/riya_v1/"
python sync_to_drive.py
```

`build_sft` writes `scratch_runs/riya_v1/requests.jsonl` (3000 requests, seed 0). `sync_to_drive.py` mirrors the code to
`G:/My Drive/jarvis/code`.

## 2. Colab: mount Drive, get llama.cpp and the libraries

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/jarvis/code
```

Build llama.cpp. `setup_llamacpp` runs the build in the background (several minutes on a fresh session; it also
downloads candidate GGUFs), so this is three cells. Colab wipes `/content` between sessions, so run them in every new
session.

```python
import os
os.makedirs("/content/runs", exist_ok=True); os.makedirs("/content/bin", exist_ok=True)
!python -m colab.setup_llamacpp
```

Wait for the build to finish. This cell blocks until `BUILD_DONE` and stops with the log if the build dies first:

```python
import subprocess, time
log = "/content/runs/llamacpp.out"
while "BUILD_DONE" not in open(log).read():
    running = subprocess.run(["pgrep", "-f", "cmake --build"], capture_output=True).returncode == 0
    if not running and "BUILD_DONE" not in open(log).read():
        raise SystemExit("llama.cpp build stopped without BUILD_DONE. End of the log:\n" + open(log).read()[-2000:])
    time.sleep(10)
print("llama.cpp built")
```

Only then install the server binary:

```python
!cp /content/llama.cpp/build/bin/llama-server /content/bin/llama-server
```

Then the training libraries (llama.cpp's LoRA converter needs `gguf`; nothing is quantised, so there is no bitsandbytes):

```python
!pip install -q peft gguf sentencepiece huggingface_hub "transformers>=4.54"
```

## 3. Colab: run the teacher

```python
!python -m colab.gen_riya_v1
```

It serves the teacher (`TEACHER` env, default Huihui-Qwen3.5-9B-abliterated) with 8 slots, reads
`/content/drive/MyDrive/riya_v1/requests.jsonl` and appends to `teacher.jsonl` in the same folder. It is resumable:
run the cell again after a disconnect. It prints `GEN_DONE` at the end. The online fallback is off; to allow it for
rows the teacher failed on (10 requests a minute), run `%env TEACHER_ONLINE=1` and `%env RIYA_ONLINE_API_KEY=...` first.

## 4. Local: filter the teacher's answers

Drive for desktop syncs `teacher.jsonl` down on its own. Then:

```powershell
.venv\Scripts\python -m riya.train.filter_sft --teacher "G:/My Drive/riya_v1/teacher.jsonl"
```

Read the keep-rate report. Any **FLAGGED** kind (fewer than 40 kept rows) means going back to step 1 with a larger
`--n` before training. When it is clean, upload the kept rows:

```powershell
Copy-Item scratch_runs/riya_v1/sft.jsonl "G:/My Drive/riya_v1/"
```

## 5. Colab: the conversion smoke test

```python
!python -m colab.train_riya_v1 --smoke
```

It trains one optimizer step on 8 rows, exports the adapter, converts it with `convert_lora_to_gguf.py`, starts
`llama-server --lora`, checks `/lora-adapters` lists the adapter, and checks the HF chat template renders the same prompt
as llama-server's. It must print **`SMOKE_OK`** (its files go to `riya_v1/smoke/`). If it doesn't, stop: do not
start the full run.

## 6. Colab: train

```python
!python -m colab.train_riya_v1
!cat /content/drive/MyDrive/riya_v1/train_report.json
```

About 2 epochs at lr 2e-4, fp16 on a T4. It prints `TRAIN_DONE` and writes `riya-v1-lora.gguf` and
`train_report.json` to `riya_v1/`. Look at the loss curve in the report: it should fall and flatten.

> **Warning: check `"base_mismatch"` in `train_report.json`.** If it is `true`, the trainer could not find the
> abliterated safetensors weights that match the GGUF she serves, and trained on the official
> `LiquidAI/LFM2.5-1.2B-Instruct` instead. The adapter was then fitted to different weights from the ones it will be
> applied to. It may still work, but nothing guarantees it; **the champion gate is the only judge**, so don't adopt by
> hand. To train on the right weights, set the repo yourself and rerun: `%env BASE_ID=huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated`.

## 7. Local: install the adapter and run the gate

Copy `riya_v1/riya-v1-lora.gguf` from `G:/My Drive/` into the models folder, next to the served GGUF:

```powershell
Copy-Item "G:/My Drive/riya_v1/riya-v1-lora.gguf" "~/models/riya/"
.venv\Scripts\python -m evaluation.champion
```

The gate runs the persona eval three times for the base and for base + adapter on its own llama-server, then adopts the
adapter only if it clearly wins and regresses nothing. Close games and fullscreen windows first (it refuses below
2500 MB of free VRAM). Add `--dry-run` to see the decision without writing it.
