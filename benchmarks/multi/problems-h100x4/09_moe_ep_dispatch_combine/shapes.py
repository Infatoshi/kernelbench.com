"""Canonical shapes for MoE expert-parallel dispatch + combine.

Counts per destination are data-dependent (the gate decides), so these fix only
the token budget and width. What varies is how much the per-call routing metadata
and the permutation cost matter relative to the payload:

  shape 0: prefill-sized batch, wide hidden
  shape 1: 4x the tokens — permutation and count exchange amortize
  shape 2: decode-sized batch — latency-bound, metadata round trip dominates
  shape 3: off-alignment hidden
"""

SHAPES = [
    {"tokens": 2048, "hidden": 4096},
    {"tokens": 8192, "hidden": 4096},
    {"tokens": 128, "hidden": 4096},
    {"tokens": 2048, "hidden": 4095},
]
