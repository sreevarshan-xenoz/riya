"""On a CPU session: unpack code and start one world run (fresh or resumed). RUN/ARGS are filled in by launch_round3.ps1."""
import os
import subprocess
import zipfile

RUN = "w3-belief-sal"
ARGS = "--episodes 400 --shift_every 30 --shift_jitter 0.4 --save_every 20 --seed 1 --context belief --arousal_plasticity --salience 19".split()
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
os.makedirs(f"/content/runs/{RUN}", exist_ok=True)
log = open(f"/content/runs/{RUN}.out", "a")
env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}  # two runs share each 2-core session
p = subprocess.Popen(["python", "train_world.py", "--out_dir", f"/content/runs/{RUN}", *ARGS], cwd="/content/code",
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(f"started {RUN} pid={p.pid}")
