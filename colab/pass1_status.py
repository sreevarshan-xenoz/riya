import os
import subprocess

ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
print("pipeline running:", any("STEP" in l or "llama-quantize" in l or "gen_riya" in l or "train_riya" in l or
                                "riya.bench" in l for l in ps.splitlines()) or "while [ ! -f" in ps)
p = "/content/runs/pass1.out"
print(open(p).read()[-1500:] if os.path.exists(p) else "no pass1.out yet")
