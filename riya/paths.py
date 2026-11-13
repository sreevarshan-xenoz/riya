"""Where Riya keeps her own files: %LOCALAPPDATA%\\Riya on Windows (or wherever LOCALAPPDATA points, which is how
tests redirect it), $XDG_DATA_HOME/riya (by default ~/.local/share/riya) elsewhere."""
import os
import sys
from pathlib import Path


def data_dir() -> Path:
    if os.environ.get("LOCALAPPDATA") or sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "riya"
