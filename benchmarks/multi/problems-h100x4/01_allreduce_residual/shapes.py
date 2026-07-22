"""Canonical shapes for TP all-reduce + residual.

Mix of message sizes: small (latency-bound, one-shot territory), medium, large
(bandwidth-bound, two-shot/ring territory), and an off-alignment hidden.
"""

SHAPES = [
    {"tokens": 4096, "hidden": 8192},    # ~64 MB bf16 — bandwidth-bound
    {"tokens": 512, "hidden": 8192},     # ~8 MB — transitional
    {"tokens": 64, "hidden": 8192},      # ~1 MB — latency-bound (one-shot wins)
    {"tokens": 4096, "hidden": 8191},    # off-alignment hidden
]
