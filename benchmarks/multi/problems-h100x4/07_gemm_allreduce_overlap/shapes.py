"""Canonical shapes for TP GEMM + all-reduce overlap.

The axis that matters here is the ratio of GEMM time to all-reduce time, since
that is what decides how much of the comm a perfect overlap can hide:

  gemm_flops = 2 * tokens * k_local * hidden
  ar_bytes   = 2*(w-1)/w * tokens * hidden * 2

  shape 0: balanced      (~0.20 ms gemm vs ~0.29 ms comm at bf16 peak / 350 GB/s)
  shape 1: compute-heavy (4x K per rank — comm should disappear entirely)
  shape 2: small         (latency-bound comm, GEMM too short to hide it)
  shape 3: off-alignment hidden (non-power-of-two epilogue width)
"""

SHAPES = [
    {"tokens": 4096, "k_local": 2048, "hidden": 8192},
    {"tokens": 4096, "k_local": 8192, "hidden": 8192},
    {"tokens": 512, "k_local": 2048, "hidden": 8192},
    {"tokens": 4096, "k_local": 2048, "hidden": 8191},
]
