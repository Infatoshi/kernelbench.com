"""Informational SOTA pointer for Kimi-Linear decode.

The aspirational ceiling is Moonshot's shipped decode path (FlashMLA-style MLA,
a fused KDA recurrence, and grouped-expert MoE). Not installed here; the score
is speedup over baseline.py, with this as the direction of travel.
"""


def is_available() -> bool:
    return False
