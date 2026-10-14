"""Generates notebooks/colab_train.ipynb (keeps the notebook diff-friendly to edit as Python)."""
import json
import os

cells = [
    ("md", "# Jarvis — Colab T4 training\nRuntime → Change runtime type → **T4 GPU**. Everything is saved to Google Drive, so if Colab disconnects just re-run all cells: training resumes from the last checkpoint."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\nROOT = '/content/drive/MyDrive/jarvis'\n!mkdir -p {ROOT}/data {ROOT}/runs\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("md", "## Get the code\nDefault: the code folder synced to `MyDrive/jarvis/code` (by `sync_to_drive.py` on the PC). Alternatives: set `REPO` to a GitHub repo, or put `jarvis.zip` in `MyDrive/jarvis/`.\nThe code is copied to local disk so Python imports are fast; checkpoints still go to Drive."),
    ("code", "REPO = ''  # optional, e.g. 'https://github.com/<you>/jarvis.git'\nimport os\n%cd /content\n!rm -rf code\nif REPO:\n    !git clone -q {REPO} code\nelif os.path.isdir(f'{ROOT}/code'):\n    !cp -r \"{ROOT}/code\" code\nelse:\n    !mkdir code && unzip -q {ROOT}/jarvis.zip -d code\n%cd /content/code\n!pip -q install tiktoken datasets\n!python -m pytest -q"),
    ("md", "## 1. World-model core (JEPA + affect + global workspace)\nSmall and fast — start here. Watch `survival` go up, `poison` spike after each hidden rule shift and then drop, and `S1%` rise as the fast policy learns from planning."),
    ("code", "!python train_world.py --out_dir {ROOT}/runs/world --episodes 2000 --max_minutes 170"),
    ("code", "import pandas as pd\ndf = pd.read_json(f'{ROOT}/runs/world/log.jsonl', lines=True)\ndf[['survival','poison','food','S1%','mood']].rolling(20).mean().plot(subplots=True, figsize=(10,9));"),
    ("code", "# H1 control run: same agent, affect has no influence. Compare logs with summarize_world.py\n# !python train_world.py --out_dir {ROOT}/runs/world-noaffect --episodes 2000 --ablate_affect --max_minutes 170\n!python summarize_world.py {ROOT}/runs/world/log.jsonl 100"),
    ("md", "## 2. Language backbone (Gated DeltaNet + gated attention, 3:1 hybrid)\nTokenize once (saved to Drive), then train. `tiny` ≈ 1 h sanity run; `small` is the main run. `--max_minutes` stops cleanly before Colab's limit — re-run to continue."),
    ("code", "!python prepare_data.py tinystories {ROOT}/data/tinystories --max_tokens 470000000"),
    ("code", "!python train_lm.py --data_dir {ROOT}/data/tinystories --out_dir {ROOT}/runs/lm-small --preset small --batch_size 32 --block_size 512 --max_steps 20000 --max_minutes 170"),
    ("md", "### Ablations (same data, same steps) — the experiment from Notion page 15\nSwap the pattern: `AAAA` (pure attention) · `RRRA` (Mamba-2-style) · `DDDA` (Gated DeltaNet hybrid) · add `--bitnet`."),
    ("code", "# !python train_lm.py --data_dir {ROOT}/data/tinystories --out_dir {ROOT}/runs/lm-attn --preset small --pattern AAAAAAAA --max_steps 5000\n# !python train_lm.py --data_dir {ROOT}/data/tinystories --out_dir {ROOT}/runs/lm-bitnet --preset small --bitnet --max_steps 5000"),
    ("md", "## 3. Talk to it — affect-conditioned generation"),
    ("code", "!python talk.py --lm {ROOT}/runs/lm-small/ckpt.pt --world {ROOT}/runs/world/world.pt --prompt 'Once upon a time'"),
]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md"
        else {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/colab_train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/colab_train.ipynb")
