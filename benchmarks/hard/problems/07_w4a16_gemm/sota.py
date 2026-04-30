"""SOTA reference for W4A16 GEMM.

Library survey on RTX PRO 6000 Blackwell (SM120, CC 12.0):

  - Marlin (IST-DASLab):         no SM120 kernels (Ampere/Hopper only). Skip.
  - GPTQ-Triton (fpgaminer):     unmaintained; pure Triton path works on SM120
                                 but is not faster than Marlin on its target HW
                                 and has no Blackwell tuning. Skip as primary.
  - AWQ (mit-han-lab/llm-awq):   CUDA kernels not built for SM120 in the wheel.
                                 Skip.
  - bitsandbytes >= 0.49.2:      CUDA kernels compile and run on SM120 (verified
                                 on this machine). Different quant scheme (NF4,
                                 symmetric, blocksize 64) than our reference's
                                 AWQ-style asymmetric INT4 with group_size 128,
                                 but it occupies the same memory regime and is
                                 the only tuned W4A16-class kernel that runs on
                                 SM120 today. Used here as an *informational*
                                 SOTA line, not as a numerical reference.

The benchmark calls `sota_forward(x, ref_model)` and times it; correctness is
NOT checked against this path (the quant scheme differs).
"""
from __future__ import annotations

import torch

_BNB_OK: bool | None = None


def is_available() -> bool:
    global _BNB_OK
    if _BNB_OK is not None:
        return _BNB_OK
    try:
        import bitsandbytes  # noqa: F401
        from bitsandbytes.functional import quantize_4bit  # noqa: F401
        _BNB_OK = torch.cuda.is_available()
    except Exception:
        _BNB_OK = False
    return _BNB_OK


_CACHE: dict[tuple[int, int, int], tuple] = {}


def _prepare(ref_model) -> tuple:
    """Quantize the reference's bf16-equivalent weight with bnb NF4 once."""
    key = (ref_model.M, ref_model.N, ref_model.K)
    if key in _CACHE:
        return _CACHE[key]
    from bitsandbytes.functional import quantize_4bit
    # Reconstruct the bf16 weight that the reference effectively uses.
    # We dequantize the int4 packed weights via the reference's own formula
    # so the SOTA line operates on the *same* underlying matrix.
    # (Numerics will still differ slightly because bnb re-quantizes to NF4.)
    K, N = ref_model.K, ref_model.N
    w_packed = ref_model.w_q  # (K//2, N) uint8
    scales = ref_model.scales  # (K/group, N) bf16
    zeros = ref_model.zeros    # (K/group, N) bf16
    g = ref_model.group_size

    w_unpacked = torch.empty((K, N), dtype=torch.uint8, device=w_packed.device)
    w_unpacked[0::2] = w_packed & 0xF
    w_unpacked[1::2] = (w_packed >> 4) & 0xF
    s_full = scales.repeat_interleave(g, dim=0)  # (K, N)
    z_full = zeros.repeat_interleave(g, dim=0)
    w_bf = (w_unpacked.to(torch.bfloat16) - z_full) * s_full  # (K, N) bf16

    # bnb expects (out_features, in_features) = (N, K)
    w_for_bnb = w_bf.t().contiguous()
    qw, qstate = quantize_4bit(w_for_bnb, blocksize=64, quant_type="nf4")
    _CACHE[key] = (qw, qstate, w_bf)
    return _CACHE[key]


def sota_forward(x: torch.Tensor, ref_model) -> torch.Tensor:
    """W4A16 GEMM via bitsandbytes NF4. x: (M, K) bf16, returns (M, N) bf16."""
    from bitsandbytes.functional import dequantize_4bit, gemv_4bit
    qw, qstate, _ = _prepare(ref_model)
    M = x.shape[0]
    if M == 1:
        # Decode path: bnb gemv_4bit. Wants (1, 1, K).
        out = gemv_4bit(x.view(1, 1, -1).contiguous(), qw.t(), state=qstate)
        return out.view(1, -1)
    # Prefill: dequant then matmul (bnb has no batched W4A16 GEMM kernel).
    w_deq = dequantize_4bit(qw, qstate, blocksize=64, quant_type="nf4")  # (N, K)
    return x @ w_deq.t()
