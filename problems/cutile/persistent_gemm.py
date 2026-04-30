"""Persistent GEMM workload for CuTile-style scheduling.

Target CuTile API pattern:
- `import cuda.tile as ct`
- `@ct.kernel`
- persistent tile loop (`for tile_idx in range(...)`)
- launch via `ct.launch(torch.cuda.current_stream(), grid, kernel, args)`
"""

import torch
import torch.nn as nn

CUTILE_REFERENCE_SNIPPET = """
import cuda.tile as ct
ConstInt = ct.Constant[int]

@ct.kernel
def persistent_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt, num_tiles: ConstInt):
    tile_id = ct.bid(0)
    tile_stride = ct.num_blocks(0)
    while tile_id < num_tiles:
        pid_m = tile_id // ct.cdiv(B.shape[1], tn)
        pid_n = tile_id % ct.cdiv(B.shape[1], tn)
        acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
        for k in range(ct.cdiv(A.shape[1], tk)):
            a = ct.load(A, (pid_m, k), (tm, tk), padding_mode=ct.PaddingMode.ZERO)
            b = ct.load(B, (k, pid_n), (tk, tn), padding_mode=ct.PaddingMode.ZERO)
            acc = ct.mma(a, b, acc)
        ct.store(C, (pid_m, pid_n), ct.astype(acc, C.dtype))
        tile_id += tile_stride
"""


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Reference implementation; custom CuTile kernels should use persistent blocks.
        return a @ b


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp16", "bf16"]
HARDWARE_REQUIRED = ["B200"]
CUTILE_LEVEL = 1


def get_inputs():
    m = 3072
    n = 3072
    k = 3072
    return [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]


def get_init_inputs():
    return []
