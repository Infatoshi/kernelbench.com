"""Warp-specialized GEMM workload for CuTile scheduling.

Target CuTile API pattern:
- `import cuda.tile as ct`
- producer/consumer phases within one kernel
- cooperative tile loads and MMA by specialized warp groups
"""

import torch
import torch.nn as nn

CUTILE_REFERENCE_SNIPPET = """
import cuda.tile as ct
ConstInt = ct.Constant[int]

@ct.kernel
def warp_specialized_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(ct.cdiv(A.shape[1], tk)):
        # Producer warps conceptually stage tiles; consumer warps apply MMA.
        a = ct.load(A, (pid_m, k), (tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, (k, pid_n), (tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, (pid_m, pid_n), ct.astype(acc, C.dtype))
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
    n = 4096
    k = 2048
    return [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]


def get_init_inputs():
    return []
