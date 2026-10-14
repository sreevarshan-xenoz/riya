import subprocess

out = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
print("\n".join(line[:140] for line in out.splitlines() if "train_" in line or "pgrep" in line))
