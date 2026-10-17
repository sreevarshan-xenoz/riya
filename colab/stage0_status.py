import glob
import os

for path in sorted(glob.glob("/content/runs/bench/*.out")):
    lines = open(path, errors="replace").read().splitlines()
    print(os.path.basename(path), "|", " / ".join(lines[-2:])[:200])
for path in sorted(glob.glob("/content/runs/bench/*/episodes.jsonl")):
    print(path.split("/")[-2], "episodes logged:", sum(1 for _ in open(path)))
print(open(glob.glob("/content/runs/stage0-*.out")[0]).read()[-200:] if glob.glob("/content/runs/stage0-*.out") else "")
