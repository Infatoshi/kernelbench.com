"""Canonical shape sweep for TopK.

Mix of:
  - decoder vocab top-k (single sequence, very large n, moderate k) — pure
    bandwidth test; the input read dominates everything.
  - prefill / batched attention top-k (many rows, moderate n, small k) — tests
    per-row parallelism and shared-memory bitonic networks.
  - non-power-of-2 n stress case — bitonic sort networks naturally want
    powers of two; this forces the agent to handle padding or partial sorts.
  - small-k limit — k=1 (argmax) is a degenerate but useful sanity case.
"""

SHAPES = [
    {"batch": 1,   "n": 131072, "k": 64},   # decoder vocab top-k (Llama vocab ~128k)
    {"batch": 64,  "n": 8192,   "k": 8},    # prefill / attention top-k
    {"batch": 32,  "n": 16384,  "k": 32},   # mid-size batched
    {"batch": 16,  "n": 12000,  "k": 16},   # non-power-of-2 n stress
    {"batch": 128, "n": 4096,   "k": 1},    # batched argmax (k=1 corner case)
]
