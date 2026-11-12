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
!pip uninstall -y torchao
```

Colab ships `torchao`, and `peft` imports it and fails on its version. Nothing here uses it, so remove it (the uninstall
is harmless when it is already gone).

## 3. Colab: run the teacher

```python
!python -m colab.gen_riya_v1
```

It serves the teacher (`TEACHER` env, default Huihui-Qwen3.5-9B-abliterated) with 8 slots, reads
`/content/drive/MyDrive/riya_v1/requests.jsonl` and appends to `teacher.jsonl` in the same folder. Each answer carries
its request's content hash `h`. It is resumable: run the cell again after a disconnect. After a rebuild in step 1,
only answers whose `h` still matches count as done, so changed requests are answered again (the filter ignores the old
answers as `stale_answer`). It prints `GEN_DONE` at the end. The online fallback is off; to allow it for
rows the teacher failed on (10 requests a minute), run `%env TEACHER_ONLINE=1` first; OpenCode's free model needs no key,
or set `%env RIYA_ONLINE_API_KEY=...` for higher limits.

## 4. Local: filter the teacher's answers

Drive for desktop syncs `teacher.jsonl` down on its own. Then:

```powershell
.venv\Scripts\python -m riya.train.filter_sft --teacher "G:/My Drive/riya_v1/teacher.jsonl"
```

If it stops with "built under an older prompt (stale fingerprint)" on requests built before v0.11, adding
`--allow-stale` is safe: v0.11 moved the fingerprint (its canary now carries the personal situation line), but no SFT
prompt changed, since no SFT scenario is a personal request (`test_no_sft_scenario_is_a_personal_request`). That
saves regenerating the teacher's answers. After any other prompt change, rebuild in step 1 instead.

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
`train_report.json` to `riya_v1/`. Look at the loss curve in the report: it should fall and flatten. If it prints
`TRAIN_ABORTED` instead, a loss went NaN or infinite (an fp16 overflow): no adapter was exported, and
`train_report.json` says `"aborted": "non-finite loss"` with the curve up to that step.

> **Warning: check `"base_mismatch"` in `train_report.json`.** If it is `true`, the trainer didn't train on the
> abliterated safetensors weights that match the GGUF she serves (`base_id` names the repo it used: the official
> `LiquidAI/LFM2.5-1.2B-Instruct`, a search hit of another name, or a `BASE_ID` you pinned that isn't
> `…/Huihui-LFM2.5-1.2B-Instruct-abliterated`). The adapter was then fitted to different weights from the ones it will
> be applied to. It may still work, but nothing guarantees it; **the champion gate is the only judge**, so don't adopt by
> hand. To train on the right weights, set the repo yourself and rerun: `%env BASE_ID=huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated`.



## 7. Colab: the gate's verdict

The gate runs on a Colab T4, not on your GPU: a fair run is 848 turns. Colab writes a checksummed **verdict** (both
transcripts, the adapter's and the base's sha256, the timings) and adopts nothing. Section 9 re-decides the verdict on
your laptop and adopts from it. A candidate run refuses, before any server starts, without a passing
`evaluation/gate_calibration.json` taken by this exact code on this base model: if it says there is none or that it is
stale, do section 8 first.

**Ship the code.** The gate's identity includes a hash of the probes and the scorers, so Colab must run exactly the code
you committed (not `jarvis/code`, which is the trainer's). Commit first, then, locally:

```powershell
git archive --format=zip -o scratch_runs/riya_code.zip HEAD
```

Upload `scratch_runs/riya_code.zip` as `/content/riya_code.zip` (Colab's Files panel), then unzip it fresh:

```bash
%%bash
rm -rf /content/code && python -c "import zipfile; zipfile.ZipFile('/content/riya_code.zip').extractall('/content/code')"
```

The base GGUF is the one section 2 downloaded. Check the inputs. The base's sha256 must start `11731cb4` (it ends
`530a`; it is byte-identical to the one she serves on your laptop):

```bash
%%bash
ls -l /content/bin/llama-server
sha256sum /content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf
mkdir -p /content/riya_v1 /content/drive/MyDrive/riya_v1/gate
cp /content/drive/MyDrive/riya_v1/riya-v1-lora.gguf /content/riya_v1/riya-v1-candidate-lora.gguf
```

**Install and verify the memory model.** The gate measures Riya with her real memory model, the pinned
`bge-small-en-v1.5` (v0.11), and refuses (exit 2, before any server starts) when a run would use anything else, such as
the simple hash fallback she uses without it. Install `fastembed` at the pinned version, point `RIYA_MODEL_DIR` at a
folder that survives the session, and check the model before any bank or verdict run. The first run downloads about
66 MB into that folder; every champion command below carries the same
`RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models` prefix (Kaggle: `/tmp/riya/models`). Offline, copy the
`models--Qdrant--bge-small-en-v1.5-onnx-Q` folder into `RIYA_MODEL_DIR` first and set `HF_HUB_OFFLINE=1` in every
command, or fastembed resolves the repo's current revision again. The check prints the model's id, which
must equal the `embedder` field of `gate.identity()` (section 8 compares them):

```bash
%%bash
pip install -q fastembed==0.9.0
mkdir -p /content/drive/MyDrive/riya_v1/models
cd /content/code && RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models PYTHONPATH=/content/code \
python -c "from riya.core.embed import FastEmbedder; print(FastEmbedder(verify=True).embedder_id)"
```

Kaggle follows the same flow with the same commands: the same `pip install -q fastembed==0.9.0`,
`RIYA_MODEL_DIR=/tmp/riya/models` in every command (after `mkdir -p /tmp/riya/models`), and the same pinned-model
check before the first run. `FastEmbedder(verify=True)` raises when the files don't match the pin. fastembed 0.9.0
downloads the repo's current revision, so any upstream commit (even a README-only one) renames the snapshot folder.
When the only problem is `snapshot … expected aa8f8b06…`, use the offline copy: empty `RIYA_MODEL_DIR`, copy the
`models--Qdrant--bge-small-en-v1.5-onnx-Q` folder in from Drive or the `riya-gate-inputs` dataset (keep a copy there
ahead of time), add `HF_HUB_OFFLINE=1` to every command (the check and each champion run), and run the check again.
Any other problem (a file's size or sha256) means a different model file, so don't run the gate on it.

Then the gate, with the candidate. `--verdict-out` implies `--dry-run`: nothing is adopted or written but the verdict.
It takes roughly 4-6 minutes (measured at 162-250 s per run at repeat 3 on a Kaggle T4; at `--repeat 4`, gate 0.11.0,
the gate's budget is a ceiling of 1200 s, which section 8 checks on `bank-1`) and prints the table and
`Verdict written: ... (total N s)`. Kaggle works the same as Colab, with the same commands:

```bash
%%bash
cd /content/code && SHA8=$(sha256sum /content/riya_v1/riya-v1-candidate-lora.gguf | cut -c1-8) && \
RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models LLAMA_SERVER=/content/bin/llama-server PYTHONPATH=/content/code PYTHONIOENCODING=utf-8 LLAMA_ARG_CACHE_RAM=0 \
python -m evaluation.champion --model /content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf \
  --lora /content/riya_v1/riya-v1-candidate-lora.gguf --db /content/riya_v1/champ.db \
  --url ws://127.0.0.1:9 --port 8097 --verdict-out /content/drive/MyDrive/riya_v1/verdict-$SHA8.json
```

The verdict lands in `G:/My Drive/riya_v1/verdict-<sha8>.json`. A different adapter file gets a different name.

Colab has about 12.7 GB of RAM. llama-server's host-side prompt cache defaults to 8 GiB, which has OOM-killed it here
before (`colab.gen_riya_v1` starts its own with `--cache-ram 0`). On Colab the gate passes `--cache-ram 0` to its own
llama-server itself: it knows it is on Colab from `COLAB_RELEASE_TAG`, which every Colab runtime sets (and `%%bash`
cells inherit). The commands also set `LLAMA_ARG_CACHE_RAM=0`, llama-server's environment variable for the same
setting, as a second guard; drop neither. On the laptop nothing changes: the CPU check in section 9 never gets the
flag. The gate's server also keeps its default context of 16384 with a q8_0 KV cache and one slot, which is small next
to the model. If a run dies with the server killed and no error, suspect the RAM: watch Colab's RAM meter during
`bank-1` (section 8).

## 8. Colab: calibrating the gate (after any probe, prompt, scorer, repeat or base-model change)

The calibration measures the gate on Colab's own model, with nothing to adopt: five live base-vs-base runs at `--repeat 4`
(the bank). Each one scores the base against itself, so every "improvement" or veto the gate finds is a false decision.
The criteria were fixed before any run (a bank of at least 3 runs with at most one false decision among them; 200
simulated draws from the bank at a false rate of at most 6%; the planted regressions caught; each run inside 1200 s). A
failing calibration is reported, never tuned. A bank taken before v0.11 (without the pinned embedder, or with other
retrieval code) is refused: take five new runs.

On a first calibration (no record yet), run section 7's **Ship the code** and **Check the inputs** cells first: this
section starts in `/content/code` and writes to `riya_v1/gate/`, which those cells make. The adapter copy in the check
cell fails if there's no adapter yet; that's fine here, since the bank needs none.

**First, check that Colab runs your code.** Print the gate's identity on Colab and on the laptop and compare the
five code fields (`gate_version`, `probe_sha256`, `prompt_fp`, `scorer_sha256`, `embedder`). If any differs, the zip is
stale or its line endings changed: stop and ship the code again (section 7). A mismatch would only show after about 30
minutes of Colab. Then print the loaded model's id on Colab; it must equal the `embedder` field:

```bash
%%bash
cd /content/code && PYTHONPATH=/content/code python -c "from evaluation import gate; import json; print(json.dumps(gate.identity(), indent=1))"
```

```powershell
.venv\Scripts\python -c "from evaluation import gate; import json; print(json.dumps(gate.identity(), indent=1))"
```

```bash
%%bash
cd /content/code && RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models PYTHONPATH=/content/code python -c "from riya.core.embed import FastEmbedder; print(FastEmbedder(verify=True).embedder_id)"
```

The laptop prints the same id once her model has downloaded there:

```powershell
.venv\Scripts\python -c "from riya.core.embed import FastEmbedder; print(FastEmbedder(verify=True).embedder_id)"
```

**Then the bank.** `bank-1` goes alone first, so you can read its total against the 1200 s budget (and its RAM) before
spending the other four:

```bash
%%bash
cd /content/code && for n in 1; do RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models LLAMA_SERVER=/content/bin/llama-server PYTHONPATH=/content/code PYTHONIOENCODING=utf-8 LLAMA_ARG_CACHE_RAM=0 \
  python -m evaluation.champion --model /content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf \
  --db /content/riya_v1/champ.db --url ws://127.0.0.1:9 --port 8097 --base-vs-base \
  --verdict-out /content/drive/MyDrive/riya_v1/gate/bank-$n.json; done
```

If `Verdict written: ... (total N s)` says more than 1200 s, stop: the gate is too slow as built, and the other four runs
aren't spent. Otherwise run the same command for `bank-2` and `bank-3`:

```bash
%%bash
cd /content/code && for n in 2 3; do RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models LLAMA_SERVER=/content/bin/llama-server PYTHONPATH=/content/code PYTHONIOENCODING=utf-8 LLAMA_ARG_CACHE_RAM=0 \
  python -m evaluation.champion --model /content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf \
  --db /content/riya_v1/champ.db --url ws://127.0.0.1:9 --port 8097 --base-vs-base \
  --verdict-out /content/drive/MyDrive/riya_v1/gate/bank-$n.json; done
```

Then `bank-4` and `bank-5`, the same command in its own cell (each run is separate, so a disconnect costs one):

```bash
%%bash
cd /content/code && for n in 4 5; do RIYA_MODEL_DIR=/content/drive/MyDrive/riya_v1/models LLAMA_SERVER=/content/bin/llama-server PYTHONPATH=/content/code PYTHONIOENCODING=utf-8 LLAMA_ARG_CACHE_RAM=0 \
  python -m evaluation.champion --model /content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf \
  --db /content/riya_v1/champ.db --url ws://127.0.0.1:9 --port 8097 --base-vs-base \
  --verdict-out /content/drive/MyDrive/riya_v1/gate/bank-$n.json; done
```

`--base-vs-base` needs no calibration record (it is what makes one), but it needs the pinned embedder: a run without
it exits 2 before any server starts. Back on your PC, calibrate from the five files on
the CPU (a few minutes, no model):

```powershell
New-Item -ItemType Directory -Force scratch_runs/riya_v1/gate/bank | Out-Null
Copy-Item "G:/My Drive/riya_v1/gate/bank-*.json" scratch_runs/riya_v1/gate/bank/
.venv\Scripts\python -m evaluation.gate_calibrate --bank scratch_runs/riya_v1/gate/bank
```

It prints each criterion and ends with `CALIBRATION: PASS` or `CALIBRATION: FAIL (reasons)`. Its seed and its 200 draws
are pre-registered in the code; any other number never passes. It writes `evaluation/gate_calibration.json` only on
PASS. A run that fails (or a diagnostic one, such as `--draws 50`) never touches that file: it writes
`scratch_runs/riya_v1/gate/calibration-failed.json` instead (or the `--out` path you name) and says where it went. Each
verdict and the record also name the Python and numpy they ran on, for information only: a note is printed when the
bank's differ from the laptop's. On PASS, **commit that file** and ship the code again (section 7): the
record is part of the code the gate checks. After a scorer-only change the same bank is re-scored on the CPU with this
same command; a change to the probes, the prompt, the retrieval code, the embedder, the repeat count or the base model needs five new live runs.

## 9. Local: adopt from the verdict

Copy the adapter to the **candidate** path next to the served GGUF, and the verdict into the gate folder. Never copy the
adapter over `riya-v1-lora.gguf` there: that is the adopted adapter, and only the gate replaces it.

```powershell
Copy-Item "G:/My Drive/riya_v1/riya-v1-lora.gguf" "~/models/riya/riya-v1-candidate-lora.gguf"
New-Item -ItemType Directory -Force scratch_runs/riya_v1/gate | Out-Null
Copy-Item "G:/My Drive/riya_v1/verdict-<sha8>.json" scratch_runs/riya_v1/gate/
```

First read the verdict. This re-scores the transcripts here and decides again, prints the table, and writes nothing and
starts nothing:

```powershell
.venv\Scripts\python -m evaluation.champion --check-verdict scratch_runs/riya_v1/gate/verdict-<sha8>.json
```

It refuses a verdict from other code than yours (the same identity as section 8), without a passing calibration, for
another adapter or base model than the files here (sha256), or one whose own decision disagrees with what your code
decides. Then adopt, if the table says adopt:

```powershell
.venv\Scripts\python -m evaluation.champion --adopt-verdict scratch_runs/riya_v1/gate/verdict-<sha8>.json
```

The adopt first runs a check on the CPU only: **her own llama-server starts with `-ngl 0`** and an empty
`CUDA_VISIBLE_DEVICES` (Colab's build isn't hers), renders her template, lists the adapter and makes one 4-token
generation, and is always stopped. Nothing touches the GPU, and it refuses while a fullscreen window is up. Then it
installs the candidate as `riya-v1-lora.gguf` in one atomic step, writes the champion record (`decided_on: "colab"`)
and switches Riya to it. It prints how long it took (about 2 minutes at most). Riya's own server still serves an adapter
only while it has a matching record.

Then read a few of the candidate's replies in the verdict, for feel: the gate measures honesty and style, not whether
she still sounds like Riya.

The old gate on your own GPU (a plain `evaluation.champion` with no verdict flag) still works with the same rules, but
it is not the path: it is 848 turns on your graphics card.
