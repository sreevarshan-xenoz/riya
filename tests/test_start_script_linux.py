"""The Linux launcher: valid bash, the model server capped and low priority, one GPU pinned, her adapter passed."""
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_riya.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_the_model_server_is_capped_pinned_and_follows_the_adapter():
    text = SCRIPT.read_text(encoding="utf-8")
    for needle in ("systemd-run --user --scope", "CPUQuota=", "MemoryMax=", "CPUWeight=", "nice -n 10",
                   "--threads 2", "--device", "--print-lora", "--lora", "--follow-settings", "RIYA_MODEL_DIR",
                   "trap stop_server EXIT"):
        assert needle in text, needle
    assert "riya.server" in text and "riya.core" in text
