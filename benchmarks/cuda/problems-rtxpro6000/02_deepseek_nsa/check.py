"""Correctness + CUDA language gate for NSA-style sparse attention.

Uses smaller S for check wall-clock (reference is O(B H S^2) Python).
"""
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval import cuda_language as cl  # noqa: E402
from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.cuda_language import collect_solution_sources  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)


def main():
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    meta = yaml.safe_load(Path("problem.yaml").read_text()) if Path("problem.yaml").exists() else {}
    sol_src = collect_solution_sources(Path("."))
    for forbidden in meta.get("forbidden", []):
        if re.search(re.escape(forbidden), sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    ok, messages, report = cl.check_cuda_language(sol_src, meta)
    Path("cuda_language.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("framework.txt").write_text(report["framework"] + "\n")
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    print(
        f"cuda_language: ok framework={report['framework']} "
        f"evidence={','.join(report['cuda_evidence']) or 'none'}"
    )

    device = torch.device("cuda:0")
    tol = meta.get("tolerance") or {"bfloat16": 0.1}

    # Check at the GRADED shapes, not a cheap subset. These two used to be
    # hardcoded at S=256/384 with D=64 only, which meant: (a) D=128 was never
    # verified anywhere, and (b) every block was selected at both sizes, so the
    # block-selection path that IS the op at long context was never exercised.
    # No benchmark.py compares outputs, so the graded number rested on shapes
    # the benchmark never times. Use each graded shape at one seed; the
    # reference is fp32 and cheap relative to the 1800s check budget.
    check_shapes = [dict(s) for s in shapes.SHAPES]
    check_seeds = (42,)
    for shape_idx, shape in enumerate(check_shapes):
        reference.B = shape["B"]
        reference.H = shape["H"]
        reference.S = shape["S"]
        reference.D = shape["D"]
        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch: {e}")
            sys.exit(1)

        for seed in check_seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_inputs = [t.to(device) for t in reference.get_inputs()]
            # large_qkv (q,k,v x8) is skipped here: x8 on q and k makes logits 64x,
            # the softmax a hardmax, and top-8 block selection a sub-ULP tie
            # detector. At (1,8,8191,128) row (h=6,t=4835) the 8th/9th block
            # importances differ by 0.47 fp32 ULP, so the winner depends on
            # accumulation order, which the spec does not fix; an independent
            # oracle agreed with the four "failing" kernels (DEVLOG 2026-09-22).
            # small_qkv and nominal still run at every graded shape.
            cases = [c for c in numeric_stress_cases(meta.get("name", "")) if c.name != "large_qkv"]
            for case in cases:
                with numeric_stress_context(ref_model, sol_model, base_inputs, case) as inputs:
                    with torch.no_grad():
                        ref_out = ref_model(*inputs)
                        sol_out = sol_model(*inputs)
                ok, msg = check_correctness(
                    ref_out.float(),
                    sol_out.float(),
                    dtype=torch.bfloat16,
                    override=tolerance_for_case(tol, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} seed {seed} case {case.name}: {msg}")
                    sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
