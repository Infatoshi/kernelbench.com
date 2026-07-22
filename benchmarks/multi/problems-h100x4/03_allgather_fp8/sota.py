"""SOTA ceiling: NCCL all_gather of fp8 bytes + dequant. Reference IS the NCCL
path; no separate library. Stub for harness convention."""
from __future__ import annotations


def is_available() -> bool:
    return False
