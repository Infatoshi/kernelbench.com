"""Canonical shapes for MoE all-to-all. `capacity` is rows per (src,dst) pair;
each rank's buffer is world_size*capacity rows. Includes off-alignment hidden."""

SHAPES = [
    {"capacity": 2048, "hidden": 4096},   # large, bandwidth-bound
    {"capacity": 512, "hidden": 4096},    # transitional
    {"capacity": 64, "hidden": 4096},     # small, latency-bound
    {"capacity": 2048, "hidden": 4095},   # off-alignment hidden
]
