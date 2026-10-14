"""Muon (Keller Jordan; Moonlight, Kimi K2) for 2-D hidden weights + AdamW for the rest, and a WSD schedule."""
import torch


def newton_schulz(G, steps: int = 5):
    """Approximately orthogonalise G (quintic Newton-Schulz). Runs in fp32: T4 has no bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transpose else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p.grad))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])  # Nesterov
                update = newton_schulz(g).to(p.dtype)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


def build_optimizers(model, lr_adam: float, lr_muon: float, use_muon: bool = True, weight_decay: float = 0.1):
    """Muon for 2-D matrices inside blocks; AdamW for embeddings/head, norms, biases, convs."""
    muon_params, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if use_muon and p.dim() == 2 and name.startswith("blocks."):
            muon_params.append(p)
        elif p.dim() >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    fused = torch.cuda.is_available()
    opts = [
        torch.optim.AdamW(
            [{"params": adam_decay, "weight_decay": weight_decay}, {"params": adam_no_decay, "weight_decay": 0.0}],
            lr=lr_adam, betas=(0.9, 0.95), fused=fused,
        )
    ]
    if muon_params:
        opts.append(Muon(muon_params, lr=lr_muon, weight_decay=weight_decay * 0.1))
    for opt in opts:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return opts


def wsd_lr_factor(step: int, max_steps: int, warmup: int, decay_frac: float = 0.2):
    """Warmup-Stable-Decay: constant LR for most of training, so any stable-phase checkpoint is a
    good resume point across Colab sessions; cosine-free linear cooldown at the end."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(max_steps * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    return max(0.0, (max_steps - step) / max(1, max_steps - decay_start))
