"""Restart the LM -> voice chain on the T4 with the current code (data already tokenized)."""
import os
import subprocess

subprocess.run(["pkill", "-f", "set -e"])  # old chain shell first, so it cannot continue to the voice step
subprocess.run(["pkill", "-f", "train_lm.py"])
chain = """
set -e
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
print("chain restarted pid", p.pid, "| data:", os.listdir("/content/data/tinystories"))
