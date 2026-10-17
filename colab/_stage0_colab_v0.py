"""Page-27 Stage 0/1 on one Colab CPU session (role filled in by launch_stage0_colab.ps1). Detached; resumable.

role v0     : resume train-v0     -> eval B0 random, B1 s1, B2 full, B3 memory
role v0.1   : resume train-v0.1   -> eval v0.1 full -> fit gate -> eval gated + always-MPC
role oracle : resume train-oracle -> eval B4 oracle
Uploaded checkpoints arrive flat as /content/<run>_state.pt and /content/<run>_episodes.jsonl.
"""
import os
import subprocess
import zipfile

ROLE = "v0"
B = "/content/runs/bench"
run = f"train-{ROLE}"
if os.path.exists(f"/content/{run}_state.pt"):  # a fresh upload replaces whatever ran here before
    subprocess.run(["pkill", "-f", "STAGE0_TRAIN_DONE"])  # the previous chain shell (unique marker text)
    subprocess.run(["pkill", "-f", "evaluation.run_world"])
    subprocess.run(["rm", "-rf", f"{B}/{run}"])
zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
os.makedirs(f"{B}/{run}", exist_ok=True)
for f in ("state.pt", "episodes.jsonl"):
    if os.path.exists(f"/content/{run}_{f}"):
        os.replace(f"/content/{run}_{f}", f"{B}/{run}/{f}")

E = "python -m evaluation.run_world"
ev = lambda name, extra: f"[ -f {B}/eval-{name}/DONE ] || {{ {E} eval --split test --episodes 60 --out {B}/eval-{name} {extra} > {B}/eval-{name}.out 2>&1 && touch {B}/eval-{name}/DONE; }}"  # noqa: E731
steps = [f"[ -f {B}/{run}/brain.pt ] || {E} train --variant {ROLE} --episodes 60 --out {B}/{run} --save_minutes 5 > {B}/{run}.out 2>&1",
         "echo STAGE0_TRAIN_DONE"]
brain = f"--ckpt {B}/{run}/brain.pt"
if ROLE == "v0":
    steps += [ev("B0-random", "--policy random"), ev("B1-s1", f"{brain} --policy s1"),
              ev("B2-v0", f"{brain} --policy full"), ev("B3-memory", f"{brain} --policy memory")]
elif ROLE == "v0.1":
    steps += [ev("v0.1", f"{brain} --policy full"),
              f"[ -f {B}/gate-v0.1.pt ] || python -m evaluation.fit_gate --train_dir {B}/{run} --out {B}/gate-v0.1.pt > {B}/gate-v0.1.out 2>&1",
              ev("S1gate-v0.1", f"{brain} --policy gated --gate {B}/gate-v0.1.pt"), ev("mpc-v0.1", f"{brain} --policy mpc")]
else:
    steps += [ev("B4-oracle", f"{brain} --policy full")]
steps.append("echo ROLE_DONE")
p = subprocess.Popen(["bash", "-c", "\n".join(steps)], cwd="/content/code", start_new_session=True,
                     stdout=open(f"/content/runs/stage0-{ROLE}.out", "a"), stderr=subprocess.STDOUT,
                     env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"})
print(f"stage0 role {ROLE} started pid {p.pid}")
