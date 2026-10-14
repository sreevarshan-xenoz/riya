"""On the T4 VM: run the Riya bench for every downloaded candidate, on GPU (normal mode) and CPU-only (game mode).
Detached; results in /content/runs/bench/*.json and a summary in /content/runs/bench.out.
Note: the T4 is shared with LM training and the VM has only 2 CPU cores, so absolute latencies are pessimistic;
use them to rank candidates, then confirm the winner on the laptop."""
import glob
import os
import subprocess

os.makedirs("/content/runs/bench", exist_ok=True)
jobs = []
for model in sorted(glob.glob("/content/models/**/*.gguf", recursive=True)):
    name = os.path.basename(model).replace(".gguf", "")
    extra = "--reasoning-budget 0" if "qwen" in name.lower() else ""
    for mode, ngl in (("gpu", 99), ("cpu", 0)):
        turns = 30 if mode == "gpu" else 10
        jobs.append(f"python -m riya.bench --model {model} --ngl {ngl} --ctx 16384 --threads 2 --long_turns {turns} "
                    f"--extra '{extra}' --out /content/runs/bench/{name}-{mode}.json || echo FAILED {name}-{mode}")
script = "export LLAMA_SERVER=/content/llama.cpp/build/bin/llama-server\n" + "\n".join(jobs) + "\necho ALL_DONE\n"
log = open("/content/runs/bench.out", "w")
p = subprocess.Popen(["bash", "-c", script], cwd="/content/code", stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
print(f"bench started pid {p.pid} for {len(jobs)} runs")
