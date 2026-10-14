"""Token data: stream a Hugging Face dataset, tokenize with GPT-2 BPE (tiktoken), store as uint16 memmaps."""
import os

import numpy as np
import torch

DATASETS = {
    # name: (hf path, config, text field)
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
}


def prepare(name: str, out_dir: str, max_tokens: int = 500_000_000, val_tokens: int = 2_000_000):
    """Write out_dir/train.bin and out_dir/val.bin. Safe to re-run: skips if files exist."""
    import tiktoken
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    train_path, val_path = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"{out_dir} already prepared")
        return
    path, config, field = DATASETS[name]
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(path, config, split="train", streaming=True)

    total = max_tokens + val_tokens
    arr = np.memmap(train_path + ".tmp", dtype=np.uint16, mode="w+", shape=(total,))
    pos, batch = 0, []

    def flush(texts):
        nonlocal pos
        for ids in enc.encode_ordinary_batch(texts):
            ids.append(enc.eot_token)
            n = min(len(ids), total - pos)
            arr[pos : pos + n] = ids[:n]
            pos += n

    for ex in ds:
        batch.append(ex[field])
        if len(batch) == 1024:
            flush(batch)
            batch = []
            print(f"\r{pos / 1e6:.1f}M / {total / 1e6:.0f}M tokens", end="")
            if pos >= total:
                break
    if batch and pos < total:
        flush(batch)
    arr.flush()
    print(f"\ntokenized {pos / 1e6:.1f}M tokens")

    val = np.array(arr[:val_tokens])
    train = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(pos - val_tokens,))
    train[:] = arr[val_tokens:pos]
    train.flush()
    val.tofile(val_path)
    del arr
    os.remove(train_path + ".tmp")


class TokenLoader:
    def __init__(self, data_dir: str, split: str, block_size: int, batch_size: int, device: str):
        self.data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.block_size, self.batch_size, self.device = block_size, batch_size, device

    def batch(self):
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, self.batch_size)
        x = np.stack([self.data[i : i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
