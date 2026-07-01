"""Canonical shapes for SP reduce-scatter + RMSNorm.

tokens must be divisible by world_size (rows are scattered across ranks).
Includes an off-alignment hidden.
"""

SHAPES = [
    {"tokens": 4096, "hidden": 8192},    # large, bandwidth-bound
    {"tokens": 512, "hidden": 8192},     # transitional
    {"tokens": 64, "hidden": 8192},      # small, latency-bound
    {"tokens": 4096, "hidden": 8191},    # off-alignment hidden
]
