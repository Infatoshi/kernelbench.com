"""Informational SOTA pointer for vectorized GPU RL throughput.

The peak_sps ceiling in problem.yaml stands in for a hand-fused on-GPU
training loop in the spirit of PufferLib / Brax / Isaac Gym. It is PROVISIONAL
and must be calibrated on an idle GPU before any published sweep.
"""


def is_available() -> bool:
    return False
