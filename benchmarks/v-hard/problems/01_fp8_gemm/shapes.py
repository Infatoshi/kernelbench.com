"""Canonical shape sweep for FP8 GEMM.

Mix of:
  - square aligned (the easy case)
  - off-alignment K (common real-world failure mode for tile-quantized kernels)
  - skinny (decode-like, memory-bound)
  - rectangular (prefill with grouped attention)
"""

SHAPES = [
    {"M": 4096, "N": 4096, "K": 4096},        # square aligned
    {"M": 4096, "N": 4096, "K": 4127},        # K not multiple of 128 -> forces predicated tails
    {"M": 32,   "N": 8192, "K": 8192},        # skinny M (decode)
    {"M": 4096, "N": 14336, "K": 4096},       # Llama3 up-proj shape
]
