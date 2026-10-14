import os
import shutil
import subprocess

os.makedirs("/content/runs", exist_ok=True)
shutil.copy("/content/code/colab/bench_voice.py", "/content/bench_voice.py") if os.path.exists("/content/code") else None
log = open("/content/runs/voice_bench.out", "w")
p = subprocess.Popen(["python", "/content/bench_voice.py"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "OMP_NUM_THREADS": "2"})
print("voice bench started pid", p.pid)
