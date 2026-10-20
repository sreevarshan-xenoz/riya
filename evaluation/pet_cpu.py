"""Manual CPU budget check for Little Riya's body: how much CPU her Electron processes use over a stretch, against 1% of
the machine while she idles and 3% while she walks.

  python -m evaluation.pet_cpu --mode idle|walk [--seconds 60]

Start the body first, in buddy mode with Little Riya on. For --mode walk start it with the walk forced, so she walks the
whole time instead of waiting for you to be idle (gaming, away, quiet hours and reduced motion still keep her still):

  $env:RIYA_PET_FORCE_WALK = 1; npm --prefix body start

It reads the CPU seconds of the Electron processes that run out of this repo's body/node_modules (her main, renderer and
GPU processes; other Electron apps are left out) with one read-only PowerShell query, waits, reads them again and prints
the share of the whole machine they used. It starts, moves and stops nothing. Always exits 0; not part of pytest.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"idle": 1.0, "walk": 3.0}  # percent of the whole machine
PS_QUERY = "Get-Process electron -ErrorAction SilentlyContinue | Select-Object Id,CPU,Path | ConvertTo-Json"


def parse_ps_json(text: str) -> list[dict]:
    """ConvertTo-Json gives one object for one process and a list for several; empty or bad text gives []."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        return [data]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _under(path: str, root: Path) -> bool:
    base = os.path.normcase(os.path.normpath(str(root)))
    return os.path.normcase(os.path.normpath(path)).startswith(base + os.sep)


def body_cpu_seconds(run=subprocess.run) -> dict[int, float]:
    """pid -> total CPU seconds of the Electron processes under body/node_modules; {} when there are none or the shell fails."""
    try:
        out = run(["powershell", "-NoProfile", "-Command", PS_QUERY], capture_output=True, text=True, timeout=30)
    except Exception:
        return {}
    root = REPO_ROOT / "body" / "node_modules"
    found = {}
    for row in parse_ps_json(getattr(out, "stdout", "") or ""):
        path, cpu, pid = row.get("Path"), row.get("CPU"), row.get("Id")
        if isinstance(path, str) and isinstance(pid, int) and isinstance(cpu, (int, float)) and _under(path, root):
            found[pid] = float(cpu)
    return found


def cpu_percent(before: dict[int, float], after: dict[int, float], wall_s: float, cores: int) -> float:
    """Percent of the whole machine: the CPU seconds the processes in both samples spent, over wall time x cores."""
    used = sum(max(0.0, after[p] - before[p]) for p in before.keys() & after.keys())
    return 100 * used / (wall_s * cores)


def cpu_verdict(mode: str, pct: float) -> str:
    return f"VERDICT: {mode} {pct:.2f}% (budget {BUDGET[mode]}%) {'ok' if pct <= BUDGET[mode] else 'OVER'}"


def main(argv: list[str] | None = None, *, sample=None, sleep=time.sleep, cores: int | None = None,
         clock=time.monotonic) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.pet_cpu", description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(BUDGET), required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    sample = sample or body_cpu_seconds
    cores = cores or os.cpu_count() or 1
    seconds = max(1.0, args.seconds)
    try:
        print("Start the body first (buddy mode, Little Riya on)"
              + (", with $env:RIYA_PET_FORCE_WALK=1 so she keeps walking." if args.mode == "walk" else ", and leave her alone."))
        before = sample()
        t0 = clock()
        print(f"sampled {len(before)} body process(es); measuring {args.mode} for {seconds:.0f} s on {cores} cores...", flush=True)
        sleep(seconds)
        after = sample()
        wall = clock() - t0
        if not before.keys() & after.keys():
            print("VERDICT: not measured: no Riya body process found in both samples (is the body running from this repo?)")
        else:
            counted, seen = len(before.keys() & after.keys()), len(before.keys() | after.keys())
            # a process that started or ended mid-run is left out of the sum, so say how many were counted
            print(f"{cpu_verdict(args.mode, cpu_percent(before, after, max(wall, 1e-6), cores))} "
                  f"({counted} of {seen} pids counted)")
    except Exception as e:  # a check script: report, never fail the caller
        print(f"VERDICT: error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
