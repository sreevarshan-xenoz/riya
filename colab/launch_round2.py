"""Round 2 on the VM: stop the round-1 world runs and the broken LM waiter, update code, start LM pretraining and the
context experiments (same seed, randomised flip timing, ~13 flips each)."""
import os
import subprocess
import zipfile

for pattern in ["train_world.py", "while pgrep"]:
    subprocess.run(["pkill", "-f", pattern])
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
r = subprocess.run(["python", "-m", "pytest", "-q"], cwd="/content/code", capture_output=True, text=True)
print(r.stdout[-200:])

common = ["--episodes", "400", "--shift_every", "30", "--shift_jitter", "0.4", "--save_every", "20", "--seed", "1"]
JOBS = {
    "lm-small": ["python", "train_lm.py", "--data_dir", "/content/data/tinystories", "--out_dir", "/content/runs/lm-small",
                 "--preset", "small", "--batch_size", "32", "--block_size", "512", "--max_steps", "20000",
                 "--save_every", "500", "--max_minutes", "600"],
    "w2-none": ["python", "train_world.py", "--out_dir", "/content/runs/w2-none", *common],
    "w2-belief": ["python", "train_world.py", "--out_dir", "/content/runs/w2-belief", "--context", "belief",
                  "--arousal_plasticity", *common],
    "w2-oracle": ["python", "train_world.py", "--out_dir", "/content/runs/w2-oracle", "--context", "oracle", *common],
}
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
for name, cmd in JOBS.items():
    log = open(f"/content/runs/{name}.out", "a")
    p = subprocess.Popen(cmd, cwd="/content/code", stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    print(f"started {name} pid={p.pid}")
