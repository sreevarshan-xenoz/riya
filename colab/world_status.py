"""On a CPU session: show the world run process and the last log line."""
import glob
import subprocess

print(subprocess.run("ps -eo pid,etime,pcpu,cmd | grep train_world | grep -v grep", shell=True,
                     capture_output=True, text=True).stdout.strip() or "no world run")
for path in glob.glob("/content/runs/*.out"):
    lines = open(path, errors="replace").read().splitlines()
    print(path, "|", lines[-1][:160] if lines else "(empty)")
