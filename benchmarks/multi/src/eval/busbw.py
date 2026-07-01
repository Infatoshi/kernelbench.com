"""Bus-bandwidth (busbw) roofline math for KernelBench-Multi.

`busbw_bytes` is the NCCL-convention effective bytes for a collective: the
message size times the collective's bandwidth factor (e.g. 2*(n-1)/n for
all-reduce). Each problem.yaml declares the formula directly so the math here is
trivial and uniform.
"""
from __future__ import annotations

from math import exp, log


def busbw_gb_s(busbw_bytes: float, ms: float) -> float:
    """Achieved bus bandwidth in GB/s given effective bytes and wall time (ms)."""
    sec = ms / 1e3
    if sec <= 0:
        return 0.0
    return busbw_bytes / sec / 1e9


def peak_fraction(achieved_gb_s: float, peak_gb_s: float) -> float:
    if peak_gb_s <= 0:
        return 0.0
    return achieved_gb_s / peak_gb_s


def geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return exp(sum(log(max(v, 1e-12)) for v in values) / len(values))


def eval_formula(expr: str, variables: dict) -> float:
    """Tiny sandboxed eval: only names from `variables` are visible."""
    return float(eval(expr, {"__builtins__": {}}, dict(variables)))
