"""No separate SOTA line — the eager sort-based reference is the anchor."""


def is_available() -> bool:
    return False


def sota_forward(*args, **kwargs):
    raise NotImplementedError("anchor is the eager reference")
