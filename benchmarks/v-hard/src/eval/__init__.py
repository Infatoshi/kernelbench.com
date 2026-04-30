"""Evaluation helpers: correctness, roofline math, shape handling, reporting."""
from src.eval.correctness import check_correctness, tolerance_for_dtype
from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction

__all__ = [
    "tolerance_for_dtype",
    "check_correctness",
    "compute_tflops",
    "compute_gbps",
    "peak_fraction",
]
