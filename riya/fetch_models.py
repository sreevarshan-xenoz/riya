"""Download Riya candidate GGUFs (one quant each) into C:/Users/sreev/models/riya.

  python -m riya.fetch_models [--quant Q4_K_M]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

CANDIDATES = [
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
    "mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF",
    "mradermacher/LFM2.5-1.2B-Instruct-Heretic-i1-GGUF",
    "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
]
DEST = Path("C:/Users/sreev/models/riya")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quant", default="Q4_K_M")
    a = p.parse_args()
    api = HfApi()
    DEST.mkdir(parents=True, exist_ok=True)
    for repo in CANDIDATES:
        files = [f for f in api.list_repo_files(repo) if f.endswith(".gguf") and a.quant.lower() in f.lower()
                 and "mmproj" not in f.lower()]
        if not files:
            print(f"!! {repo}: no {a.quant} file")
            continue
        path = hf_hub_download(repo, files[0], local_dir=DEST)
        print(f"{repo} -> {path} ({Path(path).stat().st_size / 1e6:.0f} MB)")
