"""Stop the round-2 world run on this CPU session (enough flips collected); round-3 runs keep going."""
import subprocess

subprocess.run(["pkill", "-f", "out_dir /content/runs/w2-"])
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
print([l[:90] for l in ps.splitlines() if "train_world" in l])
