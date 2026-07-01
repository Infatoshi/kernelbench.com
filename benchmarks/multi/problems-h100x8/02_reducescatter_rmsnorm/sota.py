"""SOTA ceiling: NCCL reduce_scatter + a fused RMSNorm. The reference IS the NCCL
path; NCCL busbw is the ceiling the agent is chasing. No separate library to
call. Stub for harness convention."""
from __future__ import annotations


def is_available() -> bool:
    return False
