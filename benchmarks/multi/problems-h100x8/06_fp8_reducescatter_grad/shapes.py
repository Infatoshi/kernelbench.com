"""Canonical shapes for fp8 gradient reduce-scatter. tokens must be divisible by
world_size (the summed gradient rows are scattered). Off-alignment hidden last."""

SHAPES = [
    {"tokens": 4096, "hidden": 8192},    # large, bandwidth-bound
    {"tokens": 512, "hidden": 8192},     # transitional
    {"tokens": 64, "hidden": 8192},      # small, latency-bound
    {"tokens": 4096, "hidden": 8191},    # off-alignment hidden
]
