"""Stop world runs on the GPU VM (they resume from their last checkpoint elsewhere)."""
import glob
import json
import subprocess

subprocess.run(["pkill", "-f", "train_world.py"])
for path in sorted(glob.glob("/content/runs/w2-*/log.jsonl")):
    rows = open(path).read().splitlines()
    print(path, "episodes logged:", len(rows), "last:", json.loads(rows[-1])["episode"] if rows else None)
