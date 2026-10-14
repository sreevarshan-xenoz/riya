"""Refresh code and restart pass 1b (model downloads and the extracted llama.cpp release are reused).
Never pkill by script text: pipeline shells contain other scripts' names. Only start if no pipeline is running."""
import subprocess
import zipfile

ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
if "downloading prebuilt llama.cpp" in ps:
    print("a pass-1b pipeline is already running; not starting another")
else:
    zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
    r = subprocess.run(["python", "/content/code/colab/pass1b.py"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-300:])
