"""FP8 e4m3 GEMM reference (correctness only, NOT the SOTA baseline).

Genuine fp8 x fp8: BOTH operands are fp8_e4m3. The weight is stored as fp8
(normalized into the e4m3 range) together with a per-output-channel scale, the
standard scaled-fp8 inference layout. The reference upcasts the fp8 operands to
bf16 and matmuls, then applies the per-channel scale — this DEFINES the fp8
target. A real fp8 x fp8 MMA kernel matches it (and can exceed the bf16 roofline
ceiling of ~0.5); a bf16-upcast kernel also matches but stays capped at ~0.5.
"""
import torch
import torch.nn as nn

OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp8_e4m3"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]
E4M3_MAX = 448.0


class Model(nn.Module):
    """y = ((x @ w.T) * weight_scale).to(bf16).

    x: fp8_e4m3 (M, K).  w: fp8_e4m3 (N, K) normalized to the e4m3 range.
    weight_scale: (N,) per-output-channel dequant scale.
    """

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        w = torch.empty(N, K, dtype=torch.bfloat16)
        nn.init.normal_(w, std=0.02)
        s = (w.float().abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)  # (N,1)
        w_fp8 = (w.float() / s).to(torch.float8_e4m3fn)
        self.register_buffer("weight", w_fp8)                         # (N, K) fp8
        self.register_buffer("weight_scale", s.squeeze(1).to(torch.float32))  # (N,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf = x.to(torch.bfloat16)
        w_bf = self.weight.to(torch.bfloat16)        # fp8 -> bf16 (exact)
        y = (x_bf @ w_bf.T).float()                  # (M, N)
        y = y * self.weight_scale[None, :]           # per-channel dequant
        return y.to(torch.bfloat16)


M = 4096
N = 4096
K = 4096


def get_inputs():
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]


def get_init_inputs():
    return [M, N, K]
