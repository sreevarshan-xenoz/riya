"""Tokenize a dataset into uint16 memmaps (run once on Colab; output goes to Drive and is reused).

  python prepare_data.py tinystories /content/drive/MyDrive/jarvis/data/tinystories --max_tokens 470000000
  python prepare_data.py fineweb-edu /content/drive/MyDrive/jarvis/data/fineweb-edu --max_tokens 2000000000
"""
import argparse

from jarvis.lm.data import DATASETS, prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DATASETS))
    p.add_argument("out_dir")
    p.add_argument("--max_tokens", type=int, default=500_000_000)
    a = p.parse_args()
    prepare(a.dataset, a.out_dir, a.max_tokens)
