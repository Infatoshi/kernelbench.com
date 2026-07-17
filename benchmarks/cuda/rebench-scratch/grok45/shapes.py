"""Env-count / horizon sweep for grid+MinGRU SPS.

Larger env counts amortize launch overhead and favor fused kernels. Short
horizons stress per-step dispatch; longer horizons favor persistent fusion.
"""

SHAPES = [
    {"num_envs": 4096, "horizon": 32},
    {"num_envs": 16384, "horizon": 32},
    {"num_envs": 65536, "horizon": 16},
    {"num_envs": 8192, "horizon": 64},
]
