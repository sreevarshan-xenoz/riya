"""HybridLM: mostly gated-recurrent blocks with periodic attention, optional BitNet-style linears,
and an optional affect-conditioning input so the world-model core can colour how the LM speaks."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BitLinear, CausalAttention, GatedRecurrence, RMSNorm, SwiGLU


@dataclass
class LMConfig:
    vocab_size: int = 50304  # GPT-2 vocab (50257) padded to a multiple of 64
    dim: int = 512
    n_heads: int = 8
    # One block per letter. D = Gated DeltaNet, R = gated linear attention (Mamba-2-like),
    # A = gated softmax attention. Default is the 3:1 Qwen3.5-style hybrid.
    pattern: str = "DDDADDDA"
    chunk: int = 64
    bitnet: bool = False
    affect_dim: int = 8  # size of the affect vector from jarvis.world.affect (0 disables)
    block_size: int = 512

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # ~19M non-embedding params; fast sanity run (TinyStories in ~1h on a T4)
    "tiny": LMConfig(dim=384, n_heads=6, pattern="DDADDA"),
    "small": LMConfig(dim=512, n_heads=8, pattern="DDDADDDA"),
    "base": LMConfig(dim=768, n_heads=12, pattern="DDDADDDADDDA"),
}


class Block(nn.Module):
    def __init__(self, cfg: LMConfig, kind: str):
        super().__init__()
        linear = BitLinear if cfg.bitnet else nn.Linear
        self.norm1, self.norm2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        if kind in "DR":
            self.mixer = GatedRecurrence(cfg.dim, cfg.n_heads, cfg.chunk, linear=linear, delta=kind == "D")
        elif kind == "A":
            self.mixer = CausalAttention(cfg.dim, cfg.n_heads, linear=linear)
        else:
            raise ValueError(f"unknown block kind {kind!r}")
        self.mlp = SwiGLU(cfg.dim, linear=linear)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HybridLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        # Zero-init so pretraining is unaffected until affect-labelled data is used.
        self.affect_proj = nn.Linear(cfg.affect_dim, cfg.dim) if cfg.affect_dim else None
        self.apply(self._init)
        if self.affect_proj is not None:
            nn.init.zeros_(self.affect_proj.weight)
            nn.init.zeros_(self.affect_proj.bias)
        out_std = 0.02 / math.sqrt(2 * len(cfg.pattern))
        for name, p in self.named_parameters():
            if name.endswith(("out.weight", "w3.weight")):
                nn.init.normal_(p, std=out_std)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):  # biases (e.g. recurrence decay init) are left untouched
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, affect=None):
        x = self.embed(idx)
        if affect is not None and self.affect_proj is not None:
            x = x + self.affect_proj(affect.to(x.dtype)).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, affect=None):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size :]
            logits, _ = self(ctx, affect=affect)
            logits = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx
