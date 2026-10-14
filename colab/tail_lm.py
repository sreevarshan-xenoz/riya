"""Print the result lines of one job log on the VM. Edit NAME / PATTERN before running (colab exec takes no args)."""
NAME, PATTERN = "lm-small", "step"
with open(f"/content/runs/{NAME}.out", errors="replace") as f:
    lines = f.read().replace("\r", "\n").splitlines()
hits = [line for line in lines if PATTERN in line]
print("\n".join(hits[-12:] if PATTERN else lines[-12:]))

