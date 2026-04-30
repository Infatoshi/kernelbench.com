"""Roofline math: achieved TFLOPS / GB/s, peak fraction.

FLOPS and bytes formulas come from the problem's `problem.yaml`. They are the
dense-equivalent algorithmic work; kernels that skip work (sparsity, early
exit) are still measured against the dense cost.
"""
from __future__ import annotations


def compute_tflops(flops: float, time_ms: float) -> float:
    """flops per invocation and wall time in ms -> TFLOPS."""
    if time_ms <= 0:
        return 0.0
    return flops / (time_ms * 1e-3) / 1e12


def compute_gbps(bytes_moved: float, time_ms: float) -> float:
    """bytes per invocation and wall time in ms -> GB/s."""
    if time_ms <= 0:
        return 0.0
    return bytes_moved / (time_ms * 1e-3) / 1e9


def peak_fraction(
    achieved: float,
    peak: float,
) -> float:
    """Return achieved / peak, clamped to [0, infinity)."""
    if peak <= 0:
        return 0.0
    return max(0.0, achieved / peak)
