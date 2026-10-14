"""Start long jobs as detached processes on the VM (they keep running after `colab exec` returns)."""
import subprocess

JOBS = {
    "world": ["python", "train_world.py", "--out_dir", "/content/runs/world", "--episodes", "2000", "--save_every", "25"],
    "data": ["python", "prepare_data.py", "tinystories", "/content/data/tinystories", "--max_tokens", "470000000"],
}
for name, cmd in JOBS.items():
    log = open(f"/content/runs/{name}.out", "a")
    p = subprocess.Popen(cmd, cwd="/content/code", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"started {name} pid={p.pid}")
