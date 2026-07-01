"""SOTA ceiling: NCCL reduce_scatter over fp8 bytes. Reference IS the NCCL path
(modeling the fp8 compression); no separate library. Stub for harness convention."""
from __future__ import annotations


def is_available() -> bool:
    return False
