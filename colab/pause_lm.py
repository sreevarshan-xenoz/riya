"""Pause LM pretraining on the T4 (Riya work has priority); it resumes later from runs/lm-small/ckpt.pt."""
import json
import os
import subprocess

subprocess.run(["pkill", "-f", "set -e"])  # the LM -> voice chain shell, so it does not move on to voice SFT
subprocess.run(["pkill", "-f", "train_lm.py"])
log = "/content/runs/lm-small/log.jsonl"
if os.path.exists(log):
    print("last LM eval:", open(log).read().splitlines()[-1])
ck = "/content/runs/lm-small/ckpt.pt"
print("checkpoint present:", os.path.exists(ck))
