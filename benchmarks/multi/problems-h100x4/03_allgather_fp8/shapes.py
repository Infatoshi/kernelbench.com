"""Canonical shapes for fp8 all-gather. tokens is the per-rank shard size; the
gathered output is world_size x larger. Includes an off-alignment hidden."""

SHAPES = [
    {"tokens": 2048, "hidden": 8192},    # large shard, bandwidth-bound
    {"tokens": 256, "hidden": 8192},     # transitional
    {"tokens": 32, "hidden": 8192},      # small, latency-bound
    {"tokens": 2048, "hidden": 8191},    # off-alignment hidden
]
