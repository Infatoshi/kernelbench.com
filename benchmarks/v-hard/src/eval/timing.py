"""Centralized GPU timing for benchmark.py files.

All problems' benchmark.py should call `time_fn` from here so we have one
implementation of warmup, L2 flush, and CUDA event capture to maintain.

Methodology:
  - 10 warmup calls absorb Triton autotune (typical ~7 configs) and
    torch.compile reduce-overhead CUDA-graph capture.
  - Between each timed call, `_l2_flush()` writes 128 MB to evict L2
    (Blackwell L2 is 96 MB; we want any prior L2 contents flushed so
    we measure HBM-load bandwidth, not L2-cached bandwidth).
  - GPU timing via cuda.Event with synchronize() AFTER record() but
    BEFORE elapsed_time() — the canonical NVIDIA pattern.
  - Reported value is the median of `iters` trials, robust to outliers.

Notes / known biases not addressed here:
  - torch.compile(mode="reduce-overhead") gets CUDA graphs which eliminate
    launch overhead. Custom Triton/CUDA kernels do NOT get this treatment.
    On small shapes where launch overhead matters, this gives compile an
    artificial advantage. We accept this as the cost of using the agreed
    "compiled" baseline policy.
  - cuBLAS / cuDNN allocate workspaces on first call. 10 warmup absorbs.
  - Median over a fairly small number of trials (default 30) is fine for
    headline numbers but won't catch bimodal distributions. Use --extra
    instrumentation if you ever care.
"""
from __future__ import annotations

import statistics

import torch

# Blackwell consumer (RTX PRO 6000) L2 is 96 MB. Allocate strictly larger.
_L2_FLUSH_BYTES = 128 * 1024 * 1024
_l2_scratch: torch.Tensor | None = None


def _l2_flush() -> None:
    """Evict L2 by writing 128 MB on the GPU."""
    global _l2_scratch
    if _l2_scratch is None:
        _l2_scratch = torch.empty(
            _L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda"
        )
    _l2_scratch.zero_()


def time_fn(fn, inputs, iters: int = 30, warmup: int = 10) -> float:
    """Time `fn(*inputs)` and return the median wall time in milliseconds.

    Each timed call is preceded by an L2 flush.
    """
    for _ in range(warmup):
        with torch.no_grad():
            fn(*inputs)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        _l2_flush()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            fn(*inputs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)
