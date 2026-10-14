"""Micro-benchmark of the Gated DeltaNet chunk kernel on the T4: triangular solve vs power-product inverse."""
import sys
import time

import torch

sys.path.insert(0, "/content/code")
import jarvis.lm.layers as L  # noqa: E402


def solve_inverse(M):
    C = M.shape[-1]
    eye = torch.eye(C, device=M.device, dtype=M.dtype)
    return torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)


B, H, T, D = 8, 8, 512, 64
q, k = (torch.nn.functional.normalize(torch.randn(B, H, T, D, device="cuda"), dim=-1).requires_grad_() for _ in range(2))
v = torch.randn(B, H, T, D, device="cuda", requires_grad=True)
log_a = (-torch.rand(B, H, T, device="cuda") * 0.1).requires_grad_()
beta = torch.rand(B, H, T, device="cuda").requires_grad_()
fast = L.unit_lower_inverse
for name, inv in [("solve_triangular", solve_inverse), ("power product", fast), ("solve_triangular", solve_inverse), ("power product", fast)]:
    L.unit_lower_inverse = inv
    for i in range(8):
        if i == 3:
            torch.cuda.synchronize()
            t = time.time()
        o, _ = L.chunk_gated_delta_rule(q, k, v, log_a, beta, 64)
        o.sum().backward()
    torch.cuda.synchronize()
    print(f"{name}: {(time.time() - t) / 5 * 1000:.1f} ms per layer fwd+bwd", flush=True)
