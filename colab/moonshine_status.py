import os

p = "/content/runs/moonshine.out"
lines = open(p, errors="replace").read().splitlines() if os.path.exists(p) else []
keep = [l for l in lines if "|" in l or "DONE" in l or "clip" in l or "FAILED" in l or "Error" in l]
print("\n".join(keep[-20:]) or "\n".join(lines[-5:]) or "no output yet")
