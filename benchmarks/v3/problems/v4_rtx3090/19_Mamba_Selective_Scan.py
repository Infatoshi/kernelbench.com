"""Mamba-2 style selective scan: the core primitive of state-space models.

Pattern: an inherently sequential recurrence
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t * h_t
parallelized via associative scan. Tests whether the model can reason about
parallel prefix operations outside the attention paradigm entirely.

Hardware target: memory-bound at moderate state sizes. On RTX3090 this is
ALU+SMEM bound, not tensor-core bound.
"""
import torch
import torch.nn as nn


HARDWARE_REQUIRED = ['RTX3090', 'H100', 'B200']
OP_TYPE = "scan"
SUPPORTED_PRECISIONS = ["bf16", "fp32"]

# Open framework: Triton has native associative_scan; CUDA works with
# Blelloch scan. Both valid, let the model choose.
FRAMEWORK_GATE = None


class Model(nn.Module):
    """Simplified selective scan (Mamba-style) -- no convolution, no activation.

    y = scan(A_t, B_t * x_t) then y_t *= C_t. Operates per-channel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, A, B, C):
        """
        Args:
            x: (Bsz, L, D)      inputs
            A: (Bsz, L, D, N)   per-timestep state transition (diagonal over N)
            B: (Bsz, L, D, N)   per-timestep input projection
            C: (Bsz, L, D, N)   per-timestep output projection
        Returns:
            y: (Bsz, L, D)
        """
        Bsz, L, D = x.shape
        N = A.shape[-1]
        h = torch.zeros(Bsz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A[:, t] * h + B[:, t] * x[:, t].unsqueeze(-1)   # (Bsz, D, N)
            y_t = (C[:, t] * h).sum(dim=-1)                     # (Bsz, D)
            ys.append(y_t)
        return torch.stack(ys, dim=1)


# Shape anchor: Mamba-2-ish -- L=4096, D=256, N=16 (state dim)
BSZ = 2
L = 4096
D_DIM = 256
N_STATE = 16


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(BSZ, L, D_DIM, dtype=torch.bfloat16)
    # A in (0, 1) to keep recurrence stable for correctness check
    A = torch.sigmoid(torch.randn(BSZ, L, D_DIM, N_STATE, dtype=torch.bfloat16))
    B = torch.randn(BSZ, L, D_DIM, N_STATE, dtype=torch.bfloat16) * 0.1
    C = torch.randn(BSZ, L, D_DIM, N_STATE, dtype=torch.bfloat16) * 0.1
    return [x, A, B, C]


def get_init_inputs():
    return []
