import os
import subprocess

os.makedirs("/content/runs/voice", exist_ok=True)
if os.path.exists("/content/pocket.wav"):
    os.replace("/content/pocket.wav", "/content/runs/voice/pocket.wav")
script = "pip -q install moonshine-voice soundfile > /dev/null 2>&1; python /content/bench_moonshine.py"
p = subprocess.Popen(["bash", "-c", script], stdout=open("/content/runs/moonshine.out", "w"), stderr=subprocess.STDOUT,
                     start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
print("moonshine bench started pid", p.pid)
