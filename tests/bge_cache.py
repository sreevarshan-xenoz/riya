"""The opt-in bge tests (v0.11): they run only when RIYA_TEST_BGE_CACHE names a folder holding the bge snapshot cache.

The folder is never touched: live_bge copies it into tmp_path first, so even a fastembed that writes metadata can't
reach the original, then loads the copy offline through the real loader that tests/real_paths.py recorded."""
import os
import shutil
from pathlib import Path

import pytest

from riya.core import embed
from riya.core.embed import FastEmbedder
from tests import real_paths

ENV = "RIYA_TEST_BGE_CACHE"
REPO_DIR = "models--Qdrant--bge-small-en-v1.5-onnx-Q"


def live_bge(tmp_path: Path, monkeypatch) -> FastEmbedder:
    """The pinned bge model, loaded from a tmp copy of the cache the env var names (local files only, verified).
    Skips, with the reason, when the variable is unset or its folder holds no such repo."""
    src = os.environ.get(ENV)
    if not src:
        pytest.skip(f"{ENV} is not set: this test loads the real bge model from a copy of a local cache")
    if not (Path(src) / REPO_DIR).is_dir():
        pytest.skip(f"{ENV}={src} holds no {REPO_DIR}")
    copy = Path(tmp_path) / "bge-cache"
    shutil.copytree(src, copy)  # symlinks are followed: the copy is real files
    monkeypatch.setattr(embed, "_load_text_embedding", real_paths.REAL_LOAD_TEXT_EMBEDDING)
    with real_paths.allow_real_fastembed():
        return FastEmbedder(cache_dir=copy, verify=True, local_files_only=True)
