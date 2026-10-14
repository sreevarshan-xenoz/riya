"""Print a compact status of jobs on the VM: processes, GPU, tail of each job log."""
import glob
import subprocess

print(subprocess.run("ps -eo pid,etime,pcpu,cmd | grep -E 'train_|prepare_' | grep -v grep", shell=True,
                     capture_output=True, text=True).stdout or "no jobs running")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
for path in sorted(glob.glob("/content/runs/*.out")):
    with open(path, errors="replace") as f:
        lines = f.read().replace("\r", "\n").splitlines()
    print(f"--- {path} ({len(lines)} lines)")
    print("\n".join(lines[-4:]))
