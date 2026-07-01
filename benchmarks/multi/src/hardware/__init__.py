"""Hardware peak lookup for KernelBench-Multi.

Only one SKU is graded: 8xH100 SXM (NVLink4). See SPEC.md for why a single SKU
suffices (the bench targets the NVLink fabric, not single-GPU compute).
"""
from __future__ import annotations

from . import h100x8

_TABLE = {
    "H100x8": h100x8,
    "h100x8": h100x8,
    "8xH100": h100x8,
}


def get(name: str):
    if name not in _TABLE:
        raise KeyError(f"unknown hardware '{name}'; known: {sorted(_TABLE)}")
    return _TABLE[name]
