"""No SOTA library line for the segmented decay scan.

Selective-scan kernels (mamba-ssm, flash-linear-attention) implement related
recurrences but not this reset-mask semantics. The ceiling is the DRAM roofline.
"""


def is_available() -> bool:
    return False


def sota_forward(*args, **kwargs):
    raise NotImplementedError("no library implements this reset-mask scan")
