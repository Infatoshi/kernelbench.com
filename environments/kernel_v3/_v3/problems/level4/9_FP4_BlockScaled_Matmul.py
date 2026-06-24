import torch
import torch.nn as nn

# FP4 Block-Scaled Matrix Multiplication (Blackwell tcgen05.mma)
# Reference: CUTLASS 3.x Blackwell narrow precision GEMMs
#
# Blackwell SM100 introduces FP4 (E2M1) tensor core instructions at 4x the
# throughput of Hopper FP8 tensor cores. Block-scaled FP4 GEMMs apply per-block
# scale factors along the K dimension, enabling high accuracy despite the
# narrow 4-bit representation.
#
# tcgen05.mma.kind::mxf4.block_scale runs at 4x Hopper FP8 throughput
# tcgen05.mma.kind::f8f6f4 supports mixed FP4/FP6/FP8 operands at 2x
#
# This problem uses a dequant-to-FP16 baseline: pack weights as FP4 with
# block-wise scales, dequantize to FP16, then matmul in FP16. This is the
# naive approach that wastes the FP4 tensor core throughput advantage.
#
# An optimized kernel should:
# 1. Keep operands in FP4 and use native tensor core instructions
# 2. Apply block scales inside the GEMM kernel (fused scaling)
# 3. Target tcgen05.mma.kind::mxf4.block_scale for peak throughput
#
# Data format:
# - Weights are packed 2 values per byte (4-bit symmetric quantization)
# - Scale factors: one FP16 scale per block of 32 elements along K
# - Dequantization: W_fp16 = scale * (W_fp4 - 8)  (zero-point = 8)
OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ['fp4', 'fp8', 'fp16', 'bf16', 'fp32']
HARDWARE_REQUIRED = ['B200']

BLOCK_SIZE = 32


class Model(nn.Module):
    """
    FP4 block-scaled GEMM via naive dequant-to-FP16 path.

    The baseline dequantizes FP4 weights to FP16 and does a standard matmul.
    An optimized kernel should use Blackwell's native FP4 tensor cores to avoid
    the dequantization overhead entirely.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        self.K = K
        self.N = N

        # Fixed-seed weight generation so ref and sol models get identical weights
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1337)
        w_fp16 = torch.randn(N, K) * 0.02
        torch.random.set_rng_state(rng_state)
        self.weight_packed, self.scales = self._quantize_fp4(w_fp16)

    def _quantize_fp4(self, w: torch.Tensor):
        """Symmetric FP4 quantization with block-wise scales."""
        N, K = w.shape
        n_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        w_padded = torch.zeros(N, n_blocks * BLOCK_SIZE, dtype=w.dtype)
        w_padded[:, :K] = w

        w_blocked = w_padded.view(N, n_blocks, BLOCK_SIZE)
        amax = w_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scales = amax / 7.0  # FP4 signed range: [-7, 7] with zero-point 8
        w_int = torch.round(w_blocked / scales).clamp(-7, 7).to(torch.int8) + 8  # shift to [1, 15]

        # Pack 2x4-bit values into uint8
        w_flat = w_int.view(N, -1)
        assert w_flat.shape[1] % 2 == 0
        lo = w_flat[:, 0::2].to(torch.uint8)
        hi = w_flat[:, 1::2].to(torch.uint8)
        packed = (hi << 4) | lo

        return nn.Parameter(packed, requires_grad=False), nn.Parameter(scales.squeeze(-1).to(torch.float16), requires_grad=False)

    def _dequantize_fp4(self) -> torch.Tensor:
        """Dequantize packed FP4 weights back to FP16."""
        lo = (self.weight_packed & 0x0F).to(torch.int8) - 8
        hi = ((self.weight_packed >> 4) & 0x0F).to(torch.int8) - 8
        # Interleave back
        N = self.weight_packed.shape[0]
        K_half = self.weight_packed.shape[1]
        w_int = torch.stack([lo, hi], dim=-1).view(N, K_half * 2)

        # Apply block scales
        n_blocks = self.scales.shape[1]
        w_blocked = w_int[:, :n_blocks * BLOCK_SIZE].view(N, n_blocks, BLOCK_SIZE).float()
        scales = self.scales.unsqueeze(-1).float()
        w_fp = (w_blocked * scales).view(N, -1)[:, :self.K]
        return w_fp.to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Naive FP4 GEMM: dequantize to FP16, then matmul.

        Input x: (batch, seq_len, K) in FP16
        Output: (batch, seq_len, N) in FP16

        An optimized kernel should avoid the dequantization and use
        Blackwell FP4 tensor cores directly.
        """
        w_fp16 = self._dequantize_fp4()
        return torch.matmul(x, w_fp16.t())


K = 4096
N = 4096
batch_size = 8
seq_len = 2048


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [K, N]
