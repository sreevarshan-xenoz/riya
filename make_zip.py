"""Package the project as jarvis.zip for uploading to MyDrive/jarvis/ (the notebook's no-GitHub path)."""
import pathlib
import zipfile

SKIP = {".venv", ".claude", ".git", "data", "scratch_data", "scratch_runs", "__pycache__", ".pytest_cache"}
root = pathlib.Path(__file__).parent
with zipfile.ZipFile(root / "jarvis.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not SKIP.intersection(rel.parts) and p.suffix not in {".pt", ".bin", ".zip"}:
            z.write(p, rel.as_posix())
print("wrote jarvis.zip")
