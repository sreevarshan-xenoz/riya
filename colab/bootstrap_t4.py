"""Fresh T4 bootstrap (compile-free): unpack code, then start the model download and the pass-1b pipeline."""
import os
import subprocess
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
os.makedirs("/content/runs", exist_ok=True)
for script in ("download_models.py", "pass1b.py"):
    r = subprocess.run(["python", f"/content/code/colab/{script}"], capture_output=True, text=True)
    print(script, "->", (r.stdout or r.stderr).strip()[-200:])
