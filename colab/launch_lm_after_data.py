"""Queue LM pretraining on the VM: wait for TinyStories tokenization to finish, then train `small`."""
import subprocess

script = """
while pgrep -f prepare_data.py > /dev/null; do sleep 30; done
exec python train_lm.py --data_dir /content/data/tinystories --out_dir /content/runs/lm-small \
  --preset small --batch_size 32 --block_size 512 --max_steps 20000 --save_every 500 --max_minutes 600
"""
log = open("/content/runs/lm-small.out", "a")
p = subprocess.Popen(["bash", "-c", script], cwd="/content/code", stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
print(f"queued lm-small pid={p.pid}")
