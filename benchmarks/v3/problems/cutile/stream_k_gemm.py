"""Stream-K GEMM workload for split-K / stream-k CuTile strategies.

Target CuTile API pattern:
- `import cuda.tile as ct`
- split-K tiles mapped from CTA ids
- partial accumulators per split
- reduction of split outputs into final C
"""

import torch
import torch.nn as nn

CUTILE_REFERENCE_SNIPPET = """
import cuda.tile as ct
ConstInt = ct.Constant[int]

@ct.kernel
def stream_k_kernel(A, B, partials, tm: ConstInt, tn: ConstInt, tk: ConstInt, split_k: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    split = ct.bid(2)
    num_k_tiles = ct.cdiv(A.shape[1], tk)
    k_start = split * ct.cdiv(num_k_tiles, split_k)
    k_end = min(num_k_tiles, (split + 1) * ct.cdiv(num_k_tiles, split_k))
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(k_start, k_end):
        a = ct.load(A, (pid_m, k), (tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, (k, pid_n), (tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(partials, (split, pid_m, pid_n), ct.astype(acc, partials.dtype))
"""


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp16", "bf16"]
HARDWARE_REQUIRED = ["B200"]
CUTILE_LEVEL = 1


def get_inputs():
    m = 4096
    n = 2048
    k = 8192
    return [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]


def get_init_inputs():
    return []
