"""Rebuild the T4 job chain after Colab reclaimed the VM: tokenize -> resume LM -> voice SFT, one detached chain.
Expects /content/jarvis.zip, /content/lm_ckpt.pt and the voice JSONL files uploaded flat to /content."""
import os
import subprocess
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
for d in ("/content/runs/lm-small", "/content/voice", "/content/data"):
    os.makedirs(d, exist_ok=True)
moves = {"/content/lm_ckpt.pt": "/content/runs/lm-small/ckpt.pt",
         "/content/lm_log.jsonl": "/content/runs/lm-small/log.jsonl",
         "/content/empathetic_dialogues.jsonl": "/content/voice/empathetic_dialogues.jsonl",
         "/content/daily_dialog.jsonl": "/content/voice/daily_dialog.jsonl"}
for src, dst in moves.items():
    if os.path.exists(src):
        os.replace(src, dst)
subprocess.run(["pip", "-q", "install", "tiktoken", "datasets"], capture_output=True)
# One sequential chain: no process-name polling, so nothing can match itself.
chain = """
set -e
python prepare_data.py tinystories /content/data/tinystories --max_tokens 470000000 > /content/runs/data.out 2>&1
python train_lm.py --data_dir /content/data/tinystories --out_dir /content/runs/lm-small --preset small \
  --batch_size 8 --grad_accum 4 --block_size 512 --max_steps 20000 --save_every 250 --max_minutes 600 \
  >> /content/runs/lm-small.out 2>&1
python train_voice.py --init /content/runs/lm-small/ckpt.pt \
  --data /content/voice/empathetic_dialogues.jsonl /content/voice/daily_dialog.jsonl \
  --pretrain_dir /content/data/tinystories --out_dir /content/runs/voice --steps 3000 --max_minutes 240 \
  >> /content/runs/voice.out 2>&1
"""
env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
p = subprocess.Popen(["bash", "-c", chain], cwd="/content/code", start_new_session=True, env=env,
                     stdout=open("/content/runs/chain.out", "a"), stderr=subprocess.STDOUT)
print("chain started pid", p.pid, "| files:", sorted(os.listdir("/content/runs/lm-small")), sorted(os.listdir("/content/voice")))
