"""Punica SGMV: Segmented Grouped Matrix-Vector multiplication for LoRA serving.

Real production pattern: multi-tenant LLM inference serves N concurrent requests,
each targeting a different LoRA adapter. Instead of N separate GEMV launches,
SGMV batches them into a single kernel that picks the right low-rank weight
per row.

    y[i] = x[i] @ B[adapter[i]] @ A[adapter[i]] + base_out[i]

This is distinct from grouped GEMM (which has variable M per group). Here,
each output row independently picks which (A, B) pair to use.
"""
import torch
import torch.nn as nn


HARDWARE_REQUIRED = ['RTX3090', 'H100', 'B200']
OP_TYPE = "gemv"
SUPPORTED_PRECISIONS = ["bf16", "fp16"]

FRAMEWORK_GATE = None


class Model(nn.Module):
    """LoRA SGMV: per-row adapter-indexed low-rank update.

    For each row i with adapter_id a = adapter[i]:
        y[i] = base_out[i] + (x[i] @ A[a]) @ B[a] * scale
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, A, B, adapter_ids, base_out, scale):
        """
        Args:
            x:            (N, D_in)          inputs (N concurrent requests, decode -> M=1)
            A:            (K, D_in, R)       per-adapter down-projection (K adapters, rank R)
            B:            (K, R, D_out)      per-adapter up-projection
            adapter_ids:  (N,) int64         which adapter each row uses
            base_out:     (N, D_out)         output from base (non-LoRA) layer
            scale:        float              LoRA alpha / rank
        Returns:
            y: (N, D_out)
        """
        # Gather per-row adapter weights, then apply: y = base + x @ A[a] @ B[a] * scale
        y = base_out.clone()
        for i in range(x.shape[0]):
            a = int(adapter_ids[i].item())
            lora = (x[i:i+1] @ A[a]) @ B[a]                  # (1, D_out)
            y[i:i+1] = y[i:i+1] + lora * scale
        return y


# Shape anchor: decode-time SGMV, 8 concurrent requests, 4 distinct adapters,
# rank=16, Llama FFN widths.
N = 8
D_IN = 8192
D_OUT = 8192
K = 4
RANK = 16
SCALE = 16.0 / RANK  # lora_alpha / rank convention


def get_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, D_IN, dtype=torch.bfloat16)
    A = torch.randn(K, D_IN, RANK, dtype=torch.bfloat16) * (1.0 / (D_IN ** 0.5))
    B = torch.randn(K, RANK, D_OUT, dtype=torch.bfloat16) * (1.0 / (RANK ** 0.5))
    adapter_ids = torch.tensor([i % K for i in range(N)], dtype=torch.int64)
    base_out = torch.randn(N, D_OUT, dtype=torch.bfloat16)
    return [x, A, B, adapter_ids, base_out, SCALE]


def get_init_inputs():
    return []
