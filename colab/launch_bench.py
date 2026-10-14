"""Run bench_lm.py detached (long GPU work must not run inside `colab exec`, whose reply times out)."""
import shutil
import subprocess
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
SCRIPT = "bench_gdn.py"  # edit to choose the benchmark
shutil.copy(f"/content/code/colab/{SCRIPT}", f"/content/{SCRIPT}")
log = open("/content/runs/bench.out", "w")
p = subprocess.Popen(["python", f"/content/{SCRIPT}"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(f"started bench pid={p.pid}")
