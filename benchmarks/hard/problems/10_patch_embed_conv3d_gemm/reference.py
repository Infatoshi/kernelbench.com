"""Naive Conv3d-as-GEMM patch embedding reference (correctness only).

Vision-Transformer / Qwen2-VL style patch embedding: a video tensor
(B, C, T, H, W) is split into non-overlapping (kT, kH, kW) patches and each
patch projected to embed_dim. This is mathematically a 3D convolution with
stride == kernel and equivalently a single GEMM after a strided reshape.

Reference uses nn.Conv3d (cuDNN dispatch) for clarity. The agent's solution is
forbidden from using Conv3d / conv3d / matmul / linear / einsum, forcing them
to write a fused patch-gather + tensor-core GEMM kernel.

Output layout: (B, embed_dim, T/kT, H/kH, W/kW). No bias.
"""
import torch
import torch.nn as nn

OP_TYPE = "patch_embed"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]


class Model(nn.Module):
    def __init__(self, B: int, C: int, T: int, H: int, W: int,
                 kT: int, kH: int, kW: int, embed_dim: int):
        super().__init__()
        assert T % kT == 0 and H % kH == 0 and W % kW == 0, \
            f"Input dims must be divisible by patch size: T={T} kT={kT} H={H} kH={kH} W={W} kW={kW}"
        self.B, self.C, self.T, self.H, self.W = B, C, T, H, W
        self.kT, self.kH, self.kW = kT, kH, kW
        self.embed_dim = embed_dim

        self.conv = nn.Conv3d(
            C, embed_dim,
            kernel_size=(kT, kH, kW),
            stride=(kT, kH, kW),
            bias=False,
            dtype=torch.bfloat16,
        )
        nn.init.normal_(self.conv.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) bf16  ->  (B, embed_dim, T/kT, H/kH, W/kW) bf16
        return self.conv(x)


# Module-level shape shims (overwritten per-shape by check.py / benchmark.py).
B = 1
C = 3
T = 2
H = 224
W = 224
kT = 2
kH = 14
kW = 14
embed_dim = 1280


def get_inputs():
    x = torch.randn(B, C, T, H, W, dtype=torch.bfloat16) * 0.5
    return [x]


def get_init_inputs():
    return [B, C, T, H, W, kT, kH, kW, embed_dim]
