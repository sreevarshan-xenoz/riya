"""(Re)start LM pretraining on the VM with the latest code (resumes from its checkpoint if one exists)."""
import os
import subprocess
import zipfile

subprocess.run(["pkill", "-f", "train_lm.py"])
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
cmd = ["python", "train_lm.py", "--data_dir", "/content/data/tinystories", "--out_dir", "/content/runs/lm-small",
       "--preset", "small", "--batch_size", "8", "--grad_accum", "4", "--block_size", "512", "--max_steps", "20000",
       "--save_every", "500", "--max_minutes", "600"]
log = open("/content/runs/lm-small.out", "a")
env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
p = subprocess.Popen(cmd, cwd="/content/code", stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(f"started lm-small pid={p.pid}")
