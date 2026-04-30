"""Correctness runner for Kahan-corrected softmax.

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3
seeds each, with the tight (1e-5) fp32 tolerance from problem.yaml. Also
rejects forbidden ops via grep.
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


def _make_inputs(batch: int, vocab: int, extreme: bool, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if extreme:
        # Adversarial: most logits are mild but a handful per row are huge.
        # If the kernel forgets to subtract the row-max before exp, this
        # row overflows fp32 and produces NaN/Inf. If it accumulates in
        # fp16, the long tail of small exp() values is lost beneath the
        # tolerance threshold.
        x = torch.randn(batch, vocab, generator=g) * 2.0
        # Spike: 4 very large positive logits per row.
        idx = torch.randint(0, vocab, (batch, 4), generator=g)
        x.scatter_(1, idx, 30.0)
    else:
        x = torch.randn(batch, vocab, generator=g) * 4.0
    return x.to(torch.float32)


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
    for shape_idx, shape in enumerate(shapes.SHAPES):
        batch = shape["batch"]
        vocab = shape["vocab"]
        extreme = shape.get("extreme", False)

        reference.BATCH = batch
        reference.VOCAB = vocab

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
            x = _make_inputs(batch, vocab, extreme, seed).to(device)

            with torch.no_grad():
                ref_out = ref_model(x)
                sol_out = sol_model(x)

            ok, msg = check_correctness(
                ref_out, sol_out,
                dtype=torch.float32,
                override=tol_override,
            )
            if not ok:
                print(f"FAIL: shape {shape_idx} {shape} seed {seed}: {msg}")
                sys.exit(1)

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
