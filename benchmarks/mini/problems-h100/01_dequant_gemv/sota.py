"""No SOTA library line for group-96 W4A16.

Marlin / machete / bitsandbytes all assume group sizes of 32/64/128; none
support a ragged group-96 layout. The ceiling is the DRAM bandwidth roofline.
"""


def is_available() -> bool:
    return False


def sota_forward(*args, **kwargs):
    raise NotImplementedError("no library implements group-96 asymmetric int4")
