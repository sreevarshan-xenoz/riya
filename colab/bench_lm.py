"""Throughput benchmark of LM block patterns on the VM (fwd+bwd, fp16 autocast, batch 8 x 512)."""
import time

import torch

import sys

sys.path.insert(0, "/content/code")
from jarvis.lm import HybridLM, LMConfig  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
x = torch.randint(0, 50304, (8, 512), device="cuda")
for pattern, chunk in [("AAAAAAAA", 64), ("RRRARRRA", 64), ("DDDADDDA", 64), ("DDDADDDA", 128)]:
    model = HybridLM(LMConfig(pattern=pattern, chunk=chunk)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    for i in range(13):
        if i == 3:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 10
    print(f"{pattern} chunk {chunk}: {8 * 512 / dt / 1e3:.1f}k tok/s ({dt * 1000:.0f} ms/step)", flush=True)
    del model, opt
    torch.cuda.empty_cache()
