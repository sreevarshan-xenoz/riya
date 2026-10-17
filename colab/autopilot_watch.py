"""Autopilot watcher (runs on the laptop, orchestration only — no compute).

Every INTERVAL seconds: back up logs/checkpoints from every Colab session, then exit with a one-line EVENT when
something needs attention, so the agent wakes up, acts, and re-arms the watcher:
  LOST <session>         a session was reclaimed by Colab
  DONE <session> <tag>   a job finished a stage (STAGE0_TRAIN_DONE, ROLE_DONE, BENCH_DONE, GEN_DONE, TRAIN_DONE, PASS1_DONE)
  FAIL <session> <line>  a FAILED / Traceback line appeared
State of already-reported events lives in scratch_runs/autopilot_seen.json.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLAB = "/root/.local/bin/colab"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEEN = ROOT / "scratch_runs" / "autopilot_seen.json"
LOCAL = ROOT / "scratch_runs" / "colab"
DRIVE = pathlib.Path("G:/My Drive/jarvis/runs")
SESSIONS = {
    "jarvis": {"log": "/content/runs/pass1.out",
               "files": ["runs/pass1.out", "runs/synth/riya_synth.jsonl", "runs/riya-v1/riya-v1-Q4_K_M.gguf",
                         "runs/train_riya.out", "runs/gen.out"]},
    "jarvis-cpu": {"log": "/content/runs/stage0-v0.out", "role": "v0"},
    "jarvis-cpu2": {"log": "/content/runs/stage0-v0.1.out", "role": "v0.1"},
    "jarvis-cpu3": {"log": "/content/runs/stage0-oracle.out", "role": "oracle"},
}
TAGS = ("STAGE0_TRAIN_DONE", "ROLE_DONE", "BENCH_DONE", "GEN_DONE", "TRAIN_DONE", "PASS1_DONE")


def wsl_path(p):
    p = pathlib.Path(p)
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def colab(*args, timeout=600):
    r = subprocess.run(["wsl", "-d", "Arch", "-e", COLAB, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


def download(session, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    code, _ = colab("download", "-s", session, remote, wsl_path(tmp))
    if code == 0 and tmp.exists():
        tmp.replace(local)
        return True
    return False


def files_for(name, cfg):
    if "role" in cfg:
        r = cfg["role"]
        files = [f"runs/bench/train-{r}/episodes.jsonl", f"runs/bench/train-{r}/state.pt", f"runs/bench/train-{r}/brain.pt",
                 f"runs/stage0-{r}.out"]
        evals = {"v0": ["B0-random", "B1-s1", "B2-v0", "B3-memory"], "v0.1": ["v0.1", "S1gate-v0.1", "mpc-v0.1"],
                 "oracle": ["B4-oracle"]}[r]
        files += [f"runs/bench/eval-{e}/episodes.jsonl" for e in evals]
        if r == "v0.1":
            files += ["runs/bench/gate-v0.1.pt", "runs/bench/gate-v0.1.out"]
        return files
    return cfg["files"]


def main():
    seen = json.loads(SEEN.read_text()) if SEEN.exists() else []
    while True:
        _, listing = colab("sessions")
        events = []
        for name, cfg in SESSIONS.items():
            if f"[{name}]" not in listing:
                ev = f"LOST {name}"
                if ev not in seen:
                    events.append(ev)
                continue
            for rel in files_for(name, cfg):
                download(name, f"/content/{rel}", LOCAL / rel)
            logfile = LOCAL / cfg["log"].replace("/content/", "")
            if download(name, cfg["log"], logfile):
                text = logfile.read_text(errors="replace")
                for tag in TAGS:
                    ev = f"DONE {name} {tag}"
                    if tag in text and ev not in seen:
                        events.append(ev)
                for line in text.splitlines():
                    if "FAILED" in line or "Traceback" in line:
                        ev = f"FAIL {name} {line.strip()[:120]}"
                        if ev not in seen:
                            events.append(ev)
        if DRIVE.parent.exists():
            shutil.copytree(LOCAL, DRIVE, dirs_exist_ok=True)
        if events:
            seen += events
            SEEN.write_text(json.dumps(seen, indent=1))
            print("\n".join(events), flush=True)
            return
        print(time.strftime("%H:%M"), "all quiet", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
