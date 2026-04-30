"""Correctness runner for TopK.

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3 seeds
each. Top-k correctness has two parts:

  1. VALUES: sol_values must match ref_values within fp32 tol. Both are
     returned sorted descending, so positional comparison is well-defined.
  2. INDICES: lenient — we do NOT require sol_indices == ref_indices because
     ties in x can yield multiple valid index sets. Instead we gather x at
     sol_indices and check those values match ref_values within tol. This
     catches "wrong indices" without false-failing on legitimate tie-breaks.

Also rejects forbidden ops by grep.
"""
import re
import sys
from pathlib import Path

import torch
import yaml

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

    all_shapes = shapes.SHAPES
    for shape_idx, shape in enumerate(all_shapes):
        reference.batch = shape["batch"]
        reference.n = shape["n"]
        reference.k = shape["k"]

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
                ref_values, ref_indices = ref_model(*inputs)
                sol_out = sol_model(*inputs)

            if not (isinstance(sol_out, (tuple, list)) and len(sol_out) == 2):
                print(f"FAIL: shape {shape_idx} {shape} seed {seed}: "
                      f"solution must return (values, indices); got {type(sol_out)}")
                sys.exit(1)
            sol_values, sol_indices = sol_out

            # Shape checks
            expected_shape = (shape["batch"], shape["k"])
            if tuple(sol_values.shape) != expected_shape:
                print(f"FAIL: shape {shape_idx} values shape {tuple(sol_values.shape)} "
                      f"!= expected {expected_shape}")
                sys.exit(1)
            if tuple(sol_indices.shape) != expected_shape:
                print(f"FAIL: shape {shape_idx} indices shape {tuple(sol_indices.shape)} "
                      f"!= expected {expected_shape}")
                sys.exit(1)

            # 1. Strict-ish values check (positional, both are sorted desc)
            ok, msg = check_correctness(
                ref_values.float(), sol_values.float(),
                dtype=torch.float32,
                override=tol_override,
            )
            if not ok:
                print(f"FAIL: shape {shape_idx} {shape} seed {seed} values: {msg}")
                sys.exit(1)

            # 2. Lenient indices check: gather x at sol_indices, compare to ref_values.
            # This handles ties without false negatives.
            x = inputs[0]
            sol_idx_long = sol_indices.to(torch.int64)
            if sol_idx_long.min() < 0 or sol_idx_long.max() >= shape["n"]:
                print(f"FAIL: shape {shape_idx} indices out of range "
                      f"[{int(sol_idx_long.min())}, {int(sol_idx_long.max())}]")
                sys.exit(1)
            gathered = torch.gather(x, dim=-1, index=sol_idx_long)
            ok, msg = check_correctness(
                ref_values.float(), gathered.float(),
                dtype=torch.float32,
                override=tol_override,
            )
            if not ok:
                print(f"FAIL: shape {shape_idx} {shape} seed {seed} indices "
                      f"(gather mismatch): {msg}")
                sys.exit(1)

    _emit_framework_label()
    print("PASS")


def _emit_framework_label():
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
