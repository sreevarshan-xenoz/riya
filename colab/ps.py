import os
import subprocess

out = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
print("\n".join(l[:160] for l in out.splitlines() if any(k in l for k in ("gen_riya", "riya.bench", "llama-server", "train_riya", "curl", "tar ", "hf_hub", "download"))))
print("bin:", os.listdir("/content/bin") if os.path.exists("/content/bin") else "missing")
print("tmp/llama:", subprocess.run("du -sh /tmp/llama 2>/dev/null; ls /tmp/llama 2>/dev/null | head", shell=True, capture_output=True, text=True).stdout)
print("models:", os.listdir("/content/models") if os.path.exists("/content/models") else "missing")
print(open("/content/runs/models.out").read()[-400:] if os.path.exists("/content/runs/models.out") else "")
