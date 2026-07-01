"""SOTA ceiling for TP all-reduce + residual: NCCL.

The reference IS the NCCL path (`dist.all_reduce` + add), so NCCL's busbw is the
ceiling the agent is chasing. There is no separate library to call. This stub
exists for harness convention; benchmark.py grades the solution's absolute busbw
against the NVLink peak, with NCCL as the human-readable reference line.
"""
from __future__ import annotations


def is_available() -> bool:
    return False  # no separate SOTA lib; NCCL == reference
