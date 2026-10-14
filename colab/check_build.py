import subprocess

ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
print("\n".join(l[:120] for l in ps.splitlines() if any(k in l for k in ("cmake", "nvcc", "make", "bash -c", "train_"))))
print(open("/content/runs/llamacpp.out", errors="replace").read().splitlines()[-1][:150])
