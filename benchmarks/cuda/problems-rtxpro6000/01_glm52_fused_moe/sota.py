"""Optional ceiling: real vLLM fused_moe if installed (diagnostic only)."""
from __future__ import annotations


def is_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False
