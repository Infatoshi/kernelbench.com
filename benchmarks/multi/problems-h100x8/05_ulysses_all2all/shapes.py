"""Canonical shapes for Ulysses seq<->head all-to-all.

heads must be divisible by world_size (heads are scattered across ranks). seq is
the per-rank sequence shard. Includes an off-alignment head_dim.
"""

SHAPES = [
    {"seq": 2048, "heads": 32, "head_dim": 128},   # large, bandwidth-bound
    {"seq": 512, "heads": 32, "head_dim": 128},    # transitional
    {"seq": 64, "heads": 32, "head_dim": 128},     # small, latency-bound
    {"seq": 2048, "heads": 32, "head_dim": 127},   # off-alignment head_dim
]
