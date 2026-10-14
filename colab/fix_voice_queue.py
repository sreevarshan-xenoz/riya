"""Kill the mis-queued voice job (its wait pattern missed 'python3') and re-queue it with a correct wait."""
import os
import subprocess

print(subprocess.run("ps -eo pid,etime,cmd | grep -E 'train_(lm|voice)|while pgrep' | grep -v grep", shell=True,
                     capture_output=True, text=True).stdout)
subprocess.run(["pkill", "-f", "train_voice.py"])
subprocess.run(["pkill", "-f", "while pgrep -f '\\^python train_lm"])
lm = subprocess.run(["pgrep", "-f", "^python3? train_lm.py"], capture_output=True, text=True).stdout.split()
print("LM pids:", lm)
script = """
while pgrep -f '^python3? train_lm.py' > /dev/null; do sleep 60; done
exec python train_voice.py --init /content/runs/lm-small/ckpt.pt \
  --data /content/voice/empathetic_dialogues.jsonl /content/voice/daily_dialog.jsonl \
  --pretrain_dir /content/data/tinystories --out_dir /content/runs/voice --steps 3000 --max_minutes 240
"""
if lm:
    log = open("/content/runs/voice.out", "a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    p = subprocess.Popen(["bash", "-c", script], cwd="/content/code", stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, env=env)
    print("re-queued voice, waiter pid", p.pid)
print(subprocess.run("ps -eo pid,etime,cmd | grep -E 'train_(lm|voice)|while pgrep' | grep -v grep", shell=True,
                     capture_output=True, text=True).stdout)
