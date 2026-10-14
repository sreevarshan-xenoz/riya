"""Unpack /content/jarvis.zip into /content/code, install deps, run tests on the Colab VM."""
import os
import shutil
import subprocess
import zipfile

shutil.rmtree("/content/code", ignore_errors=True)
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
os.makedirs("/content/runs", exist_ok=True)
os.makedirs("/content/data", exist_ok=True)
r = subprocess.run("pip -q install tiktoken datasets pytest", shell=True, capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
r = subprocess.run("python -m pytest -q", shell=True, cwd="/content/code", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
