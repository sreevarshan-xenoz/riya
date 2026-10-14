"""On a CPU session: unpack code and resume one world run from the uploaded checkpoint/log in /content/runs/RUN.
Edit RUN/ARGS via the generator in move_world.ps1 (colab exec takes no arguments)."""
import os
import subprocess
import zipfile

RUN = "w2-oracle"
ARGS = "--context oracle --episodes 400 --shift_every 30 --shift_jitter 0.4 --save_every 20 --seed 1".split()
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
subprocess.run(["pip", "-q", "install", "tiktoken"], capture_output=True)
os.makedirs(f"/content/runs/{RUN}", exist_ok=True)
for f in ("world.pt", "log.jsonl"):  # uploaded flat to /content (the CLI cannot create folders)
    if os.path.exists(f"/content/{RUN}_{f}"):
        os.replace(f"/content/{RUN}_{f}", f"/content/runs/{RUN}/{f}")
print(sorted(os.listdir(f"/content/runs/{RUN}")))
log = open(f"/content/runs/{RUN}.out", "a")
env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"}
p = subprocess.Popen(["python", "train_world.py", "--out_dir", f"/content/runs/{RUN}", *ARGS], cwd="/content/code",
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(f"resumed {RUN} pid={p.pid}")

