"""Correctness runner for Sonic-MoE up-projection (grouped GEMM + fused SwiGLU).

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3 seeds
each, with per-dtype atol/rtol. Also rejects forbidden ops by grep.
"""
import re
import sys
from pathlib import Path

import torch
import yaml

# Make the repo's src/ importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.correctness import check_correctness  # noqa: E402


def main():
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    problem_yaml = Path("problem.yaml")
    meta = yaml.safe_load(problem_yaml.read_text()) if problem_yaml.exists() else {}

    # --- Forbidden-op check ------------------------------------------------
    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    for forbidden in meta.get("forbidden", []):
        pat = re.escape(forbidden)
        if re.search(pat, sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    device = torch.device("cuda:0")
    tol_override = meta.get("tolerance") or None

    # --- Per-shape correctness --------------------------------------------
    all_shapes = shapes.SHAPES
    for shape_idx, shape in enumerate(all_shapes):
        # Rebuild reference module's module-level shape shims.
        reference.T_total = shape["T_total"]
        reference.H = shape["H"]
        reference.I = shape["I"]
        reference.E = shape["E"]
        reference.K = shape["K"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()

        sd = ref_model.state_dict()
        try:
            sol_model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch at shape {shape_idx} ({shape}): {e}")
            sys.exit(1)

        for seed in (42, 123, 456):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = [t.to(device) for t in reference.get_inputs()]

            with torch.no_grad():
                ref_out = ref_model(*inputs)
                sol_out = sol_model(*inputs)

            ok, msg = check_correctness(
                ref_out, sol_out,
                dtype=ref_out.dtype,
                override=tol_override,
            )
            if not ok:
                print(f"FAIL: shape {shape_idx} {shape} seed {seed}: {msg}")
                sys.exit(1)

    # --- Framework label (for stats) --------------------------------------
    _emit_framework_label()
    print("PASS")


def _emit_framework_label():
    """Write framework.txt with the detected kernel framework."""
    patterns = [
        ("ptx",       r"asm\s+volatile|asm\s*\(|mma\.sync|tcgen05\."),
        ("cutlass3",  r"\bcute::|cutlass/gemm/collective|cutlass::arch::Sm(9|10|12)"),
        ("cutlass2",  r"cutlass/gemm/device/gemm|cutlass::gemm::device"),
        ("cuda_wmma", r"\bnvcuda::wmma\b|wmma::fragment"),
        ("triton",    r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
        ("cuda_raw",  r"torch\.utils\.cpp_extension\.load_inline|__global__\s+void"),
    ]
    sol = Path("solution.py")
    if not sol.exists():
        return
    code = sol.read_text()
    label = "unknown"
    for name, pat in patterns:
        if re.search(pat, code):
            label = name
            break
    Path("framework.txt").write_text(label + "\n")


if __name__ == "__main__":
    main()
