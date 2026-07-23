"""Per-rank tensor comparison with per-dtype tolerance.

Tighter than KernelBench-Hard on purpose: loose tolerance is what lets a
multi-GPU kernel skip a rank or return a local-only result and still "pass".
"""
from __future__ import annotations

import torch

# (atol, rtol) per dtype. fp8 is noise-dominated; everything else is tight.
_DEFAULT_TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float64: (1e-6, 1e-6),
    torch.bfloat16: (5e-3, 5e-3),
    torch.float16: (5e-3, 5e-3),
}
_FP8_TOL = (5e-2, 5e-2)


def _tol_for(dtype: torch.dtype, override: dict | None):
    if override:
        key = str(dtype).replace("torch.", "")
        if key in override:
            v = override[key]
            return (float(v), float(v)) if not isinstance(v, (list, tuple)) else tuple(v)
    if dtype in _DEFAULT_TOL:
        return _DEFAULT_TOL[dtype]
    # fp8 variants
    if "float8" in str(dtype):
        return _FP8_TOL
    return (1e-3, 1e-3)


def compare(ref: torch.Tensor, sol: torch.Tensor, override: dict | None = None) -> tuple[bool, str]:
    """Return (ok, message). Compares dtype-cast to float for error reporting."""
    if sol is None:
        return False, "solution returned None"
    if ref.shape != sol.shape:
        return False, f"shape mismatch: ref {tuple(ref.shape)} vs sol {tuple(sol.shape)}"
    atol, rtol = _tol_for(ref.dtype, override)
    r = ref.detach().float()
    s = sol.detach().float()
    if torch.isnan(s).any() or torch.isinf(s).any():
        return False, "solution contains NaN/Inf"
    diff = (r - s).abs()
    # Scale-aware atol: honest low-precision rounding in a reduction is
    # proportional to the magnitude of the summed operands (ulps of the
    # intermediates), not to any fixed constant — and cancellation elements
    # (|r| ~ 0 while intermediates are O(scale)) are exactly where a fixed
    # atol misfires in both directions: it fails honest kernels at large input
    # scale and waves through skip-a-rank cheats at small input scale. rms(ref)
    # tracks the operand scale, making the gate invariant under the
    # numeric-stress rescales. Calibration: scripts/numerics_probe.py.
    rms = float(r.pow(2).mean().sqrt().item())
    tol = atol * max(rms, 1e-30) + rtol * r.abs()
    bad = diff > tol
    n_bad = int(bad.sum().item())
    if n_bad == 0:
        return True, "ok"
    worst = int(diff.argmax().item())
    max_abs = float(diff.max().item())
    denom = r.abs().flatten()[worst].item()
    max_rel = max_abs / denom if denom > 0 else float("inf")
    return False, (
        f"{n_bad} elem(s) exceed tol (atol={atol}, rtol={rtol}); "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} at flat idx {worst}"
    )
