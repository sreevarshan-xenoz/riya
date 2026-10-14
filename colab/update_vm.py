"""Overwrite /content/code with the new /content/jarvis.zip WITHOUT deleting the folder (running/queued jobs
have it as their working directory), then start the surprise-replay world run next to the baseline."""
import subprocess
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
r = subprocess.run(["python", "-m", "pytest", "-q"], cwd="/content/code", capture_output=True, text=True)
print(r.stdout[-300:])
cmd = ["python", "train_world.py", "--out_dir", "/content/runs/world-surprise", "--episodes", "2000",
       "--save_every", "25", "--surprise_frac", "0.5"]
log = open("/content/runs/world-surprise.out", "a")
p = subprocess.Popen(cmd, cwd="/content/code", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(f"started world-surprise pid={p.pid}")
