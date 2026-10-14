"""Building blocks for the hybrid LM. Pure PyTorch so it runs on a Colab T4 (fp16, no custom kernels)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class BitLinear(nn.Linear):
    """BitNet b1.58-style linear: ternary weights {-1,0,+1}*scale and 8-bit activations in the
    forward pass, full-precision shadow weights for the update (straight-through estimator)."""

    def forward(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        wq = (w / scale).round().clamp(-1, 1) * scale
        w = w + (wq - w).detach()
        xs = 127.0 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        xq = (x * xs).round().clamp(-128, 127) / xs
        x = x + (xq - x).detach()
        return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def chunk_gated_linear_attention(q, k, v, log_a, chunk: int = 64, state=None):
    """Gated linear attention with a scalar, data-dependent decay per head (Mamba-2 / GLA style).

    Recurrence:  S_t = a_t * S_{t-1} + k_t^T v_t,   o_t = q_t S_t
    computed chunkwise: parallel inside a chunk, recurrent across chunks.
    q, k, v: (B, H, T, D) float32; log_a: (B, H, T) float32 (<= 0).
    Returns (o, final_state) with o: (B, H, T, D), state: (B, H, D, D).
    """
    B, H, T, D = q.shape
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        log_a = F.pad(log_a, (0, pad))
    n = (T + pad) // chunk
    q = q.view(B, H, n, chunk, D)
    k = k.view(B, H, n, chunk, D)
    v = v.view(B, H, n, chunk, D)
    b = log_a.view(B, H, n, chunk).cumsum(-1)

    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    diff = (b.unsqueeze(-1) - b.unsqueeze(-2)).masked_fill(~causal, float("-inf"))
    intra = ((q @ k.transpose(-1, -2)) * diff.exp()) @ v

    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype) if state is None else state
    outs = []
    for i in range(n):
        bi = b[:, :, i]
        outs.append(intra[:, :, i] + (q[:, :, i] * bi.exp().unsqueeze(-1)) @ S)
        b_last = bi[..., -1:]
        kw = k[:, :, i] * (b_last - bi).exp().unsqueeze(-1)
        S = S * b_last.exp().unsqueeze(-1) + kw.transpose(-1, -2) @ v[:, :, i]
    o = torch.stack(outs, 2).view(B, H, n * chunk, D)[:, :, :T]
    return o, S


def unit_lower_inverse(N):
    """(I + N)^-1 for strictly lower-triangular N (..., C, C), C a power of two.
    N is nilpotent (N^C = 0), so (I + N)^-1 = (I - N)(I + N^2)(I + N^4)...(I + N^(C/2)): log2(C) batched GEMMs,
    much faster on GPU than a batched triangular solve and exact up to float rounding."""
    C = N.shape[-1]
    assert C & (C - 1) == 0, "chunk size must be a power of two"
    eye = torch.eye(C, device=N.device, dtype=N.dtype)
    out = eye - N
    P = N @ N
    k = 2
    while k < C:
        out = out + out @ P
        P = P @ P
        k *= 2
    return out


def chunk_gated_delta_rule(q, k, v, log_a, beta, chunk: int = 64, state=None):
    """Gated DeltaNet (Yang et al., ICLR 2025; used 3:1 with attention in Qwen3-Next / Qwen3.5).

    Recurrence:  S_t = a_t (I - b_t k_t^T k_t) S_{t-1} + b_t k_t^T v_t,   o_t = q_t S_t
    The delta rule overwrites the memory slot addressed by k_t instead of only adding to it.
    Chunkwise form: inside a chunk the corrected values U solve a unit-lower-triangular system
    (I + M) U = b*V - b*K*exp(g) S0, with M_tj = b_t (k_t.k_j) exp(g_t - g_j) for j < t.
    q, k (L2-normalised), v: (B, H, T, D) float32; log_a, beta: (B, H, T) float32.
    """
    B, H, T, D = q.shape
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        log_a, beta = F.pad(log_a, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (t.view(B, H, n, chunk, D) for t in (q, k, v))
    beta = beta.view(B, H, n, chunk, 1)
    g = log_a.view(B, H, n, chunk).cumsum(-1)

    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~lower, float("-inf")).exp()  # [t, j] = exp(g_t - g_j)
    kk = k @ k.transpose(-1, -2)
    M = (beta * kk * decay).masked_fill(lower.logical_not() | torch.eye(chunk, dtype=torch.bool, device=q.device), 0.0)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    # unit_lower_inverse(M) is an exact GEMM-only alternative; on a shared T4 it was not reliably faster.
    W = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)
    u_v = W @ (beta * v)  # value part of U
    u_k = W @ (beta * k * g.exp().unsqueeze(-1))  # multiplies the incoming state S0
    qk = (q @ k.transpose(-1, -2)) * decay

    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype) if state is None else state
    outs = []
    for i in range(n):
        gi = g[:, :, i]
        U = u_v[:, :, i] - u_k[:, :, i] @ S
        outs.append((q[:, :, i] * gi.exp().unsqueeze(-1)) @ S + qk[:, :, i] @ U)
        g_last = gi[..., -1:]
        kw = k[:, :, i] * (g_last - gi).exp().unsqueeze(-1)
        S = S * g_last.exp().unsqueeze(-1) + kw.transpose(-1, -2) @ U
    o = torch.stack(outs, 2).view(B, H, n * chunk, D)[:, :, :T]
    return o, S


class GatedRecurrence(nn.Module):
    """Recurrent token mixer: short causal conv -> q/k/v -> gated delta rule (or plain gated linear
    attention when delta=False) -> output gate. Constant-size state per head, linear cost in length."""

    def __init__(self, dim: int, n_heads: int, chunk: int = 64, conv: int = 4, linear=nn.Linear, delta: bool = True):
        super().__init__()
        self.h, self.dh, self.chunk, self.delta = n_heads, dim // n_heads, chunk, delta
        self.qkv = linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(3 * dim, 3 * dim, conv, groups=3 * dim, padding=conv - 1)
        self.decay = nn.Linear(dim, n_heads)
        self.beta = nn.Linear(dim, n_heads) if delta else None
        self.gate = linear(dim, dim, bias=False)
        self.out = linear(dim, dim, bias=False)
        self.head_norm = RMSNorm(self.dh)
        # Spread per-head memory horizons: decay a in ~[0.9, 0.999] => -log a in ~[0.001, 0.1].
        target = torch.logspace(math.log10(1e-3), math.log10(1e-1), n_heads)
        with torch.no_grad():
            self.decay.bias.copy_(torch.log(torch.expm1(target)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.conv(self.qkv(x).transpose(1, 2))[..., :T]
        qkv = F.silu(qkv).transpose(1, 2)
        q, k, v = (t.reshape(B, T, self.h, self.dh).transpose(1, 2).float() for t in qkv.split(C, -1))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        log_a = -F.softplus(self.decay(x).float()).transpose(1, 2)
        if self.delta:
            beta = torch.sigmoid(self.beta(x).float()).transpose(1, 2)
            o, _ = chunk_gated_delta_rule(q, k, v, log_a, beta, self.chunk)
        else:
            o, _ = chunk_gated_linear_attention(q, k, v, log_a, self.chunk)
        o = self.head_norm(o).transpose(1, 2).reshape(B, T, C).to(x.dtype)
        return self.out(o * F.silu(self.gate(x)))


def rope_cache(T: int, dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class CausalAttention(nn.Module):
    """Causal softmax attention with RoPE, QK-norm and a sigmoid output gate (Gated Attention,
    NeurIPS 2025): removes attention sinks / activation spikes, which matters for fp16 on a T4.
    Used sparsely for exact content lookup."""

    def __init__(self, dim: int, n_heads: int, linear=nn.Linear):
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.qkv = linear(dim, 3 * dim, bias=False)
        self.gate = linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.dh), RMSNorm(self.dh)
        self.out = linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.reshape(B, T, self.h, self.dh).transpose(1, 2) for t in self.qkv(x).split(C, -1))
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cache(T, self.dh, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        return self.out(o * torch.sigmoid(self.gate(x)))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, linear=nn.Linear):
        super().__init__()
        hidden = 64 * math.ceil(dim * 8 / 3 / 64)
        self.w12 = linear(dim, 2 * hidden, bias=False)
        self.w3 = linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)
