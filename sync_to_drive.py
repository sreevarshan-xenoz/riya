"""Mirror the project code into Google Drive for desktop so Colab always runs the latest version.

  python sync_to_drive.py                 # -> G:/My Drive/jarvis/code
  python sync_to_drive.py "H:/My Drive"   # if Drive is mounted elsewhere

Only code is mirrored; data/ and runs/ on Drive (datasets, checkpoints, logs) are never touched.
"""
import pathlib
import shutil
import sys

SKIP = {".venv", ".claude", ".git", "data", "scratch_data", "scratch_runs", "__pycache__", ".pytest_cache"}
SKIP_SUFFIX = {".pt", ".bin", ".zip", ".pyc"}

src = pathlib.Path(__file__).resolve().parent
drive = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "G:/My Drive")
dst = drive / "jarvis" / "code"
if not drive.exists():
    sys.exit(f"{drive} not found: is Google Drive for desktop running?")

wanted = {p.relative_to(src) for p in src.rglob("*")
          if p.is_file() and not SKIP.intersection(p.relative_to(src).parts) and p.suffix not in SKIP_SUFFIX}
copied = 0
for rel in sorted(wanted):
    s, d = src / rel, dst / rel
    if not d.exists() or s.stat().st_mtime > d.stat().st_mtime or s.stat().st_size != d.stat().st_size:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        copied += 1
removed = 0
if dst.exists():  # drop files deleted locally, only inside jarvis/code
    for p in sorted(dst.rglob("*"), reverse=True):
        rel = p.relative_to(dst)
        if p.is_file() and rel not in wanted:
            p.unlink()
            removed += 1
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
for sub in ("data", "runs"):
    (drive / "jarvis" / sub).mkdir(parents=True, exist_ok=True)
shutil.copy2(src / "notebooks" / "colab_train.ipynb", drive / "jarvis" / "colab_train.ipynb")
print(f"synced to {dst}: {copied} updated, {removed} removed, {len(wanted)} files total")
