"""Naive FP8 e4m3 GEMM reference (correctness only, NOT the SOTA baseline).

We cast inputs to bf16 and use torch.matmul. The agent's solution must match
this numerically within the fp8 tolerance declared in problem.yaml.
"""
import torch
import torch.nn as nn

OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp8_e4m3"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


class Model(nn.Module):
    """y = (x @ w.T).to(bf16), where x is fp8_e4m3 (M, K), w is fp8_e4m3 (N, K)."""

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        # Weights stored as parameters so state_dict is well-defined.
        # We initialize in bf16 then cast; the fp8 dtype is set by get_inputs.
        self.weight = nn.Parameter(torch.empty(N, K, dtype=torch.bfloat16))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to bf16 for the naive reference; the kernel equivalent would
        # use mma.sync f8f6f4 kind directly.
        x_bf = x.to(torch.bfloat16)
        w_bf = self.weight.to(torch.bfloat16)
        return x_bf @ w_bf.T  # (M, N) bf16


M = 4096
N = 4096
K = 4096


def get_inputs():
    # fp8_e4m3 input; random uniform in [-4, 4] then cast.
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]


def get_init_inputs():
    return [M, N, K]
