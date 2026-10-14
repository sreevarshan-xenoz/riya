"""Queue voice SFT on the T4: wait until LM pretraining exits, then fine-tune from its checkpoint."""
import os
import subprocess
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
os.makedirs("/content/voice", exist_ok=True)
for f in ("empathetic_dialogues.jsonl", "daily_dialog.jsonl"):
    if os.path.exists(f"/content/{f}"):
        os.replace(f"/content/{f}", f"/content/voice/{f}")
print(sorted(os.listdir("/content/voice")))
# '^python train_lm' is anchored to the start of a command line, so it cannot match this bash script itself.
script = """
while pgrep -f '^python train_lm.py' > /dev/null; do sleep 60; done
exec python train_voice.py --init /content/runs/lm-small/ckpt.pt \
  --data /content/voice/empathetic_dialogues.jsonl /content/voice/daily_dialog.jsonl \
  --pretrain_dir /content/data/tinystories --out_dir /content/runs/voice --steps 3000 --max_minutes 240
"""
log = open("/content/runs/voice.out", "a")
env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
p = subprocess.Popen(["bash", "-c", script], cwd="/content/code", stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env=env)
r = subprocess.run(["pgrep", "-f", "^python train_lm.py"], capture_output=True, text=True)
print(f"queued voice pid={p.pid}; LM running pids: {r.stdout.split()}")
