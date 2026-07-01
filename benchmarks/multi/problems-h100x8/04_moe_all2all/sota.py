"""SOTA ceiling: NCCL all_to_all_single (dispatch + combine). Reference IS the
NCCL path; no separate library. Stub for harness convention."""
from __future__ import annotations


def is_available() -> bool:
    return False
