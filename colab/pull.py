"""Copy run artifacts from the Colab VM to this PC and into Google Drive (G:/My Drive/jarvis/runs).

  python colab/pull.py                 # logs + checkpoints for all known runs
  python colab/pull.py --logs-only     # just the small log files

Uses the Colab CLI inside WSL (Arch). Downloads land in scratch_runs/colab/ first, then are copied to Drive,
so a VM reclaimed by Colab never takes more than the progress since the last pull with it.
"""
import pathlib
import shutil
import subprocess
import sys

COLAB = "/root/.local/bin/colab"
FILES = {  # run: (session, files)
    "lm-small": ("jarvis", ["log.jsonl", "weights.pt"]),  # small fp16 copy; full ckpt.pt (530 MB) is too big to re-upload
    "voice": ("jarvis", ["log.jsonl", "weights.pt"]),
    "synth": ("jarvis", ["riya_synth.jsonl"]),  # teacher-written Riya dialogues (resumable)
    "riya-v1": ("jarvis", ["riya-v1-Q4_K_M.gguf"]),
}
root = pathlib.Path(__file__).resolve().parent.parent
local = root / "scratch_runs" / "colab"
drive = pathlib.Path("G:/My Drive/jarvis/runs")
logs_only = "--logs-only" in sys.argv


def wsl_path(p: pathlib.Path):
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


for run, (SESSION, names) in FILES.items():
    for name in names:
        if logs_only and not name.endswith((".jsonl", ".out", ".json")) or (logs_only and run == "synth"):
            continue
        dst = local / run / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        r = subprocess.run(["wsl", "-d", "Arch", "-e", COLAB, "download", "-s", SESSION,
                            f"/content/runs/{run}/{name}", wsl_path(tmp)], capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            continue  # run not started yet
        tmp.replace(dst)
        if drive.parent.exists():
            (drive / run).mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, drive / run / name)
        print(f"pulled {run}/{name} ({dst.stat().st_size / 1e6:.1f} MB)")
