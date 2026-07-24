"""Correctness + CUDA language gate for the sort-free top-p mask.

Exact-integer grading with a float64 oracle band: for each token, compute the
exclusive cumulative mass (fp64 softmax, stable descending order, ties by
lower index). Tokens with excl < p - TAU MUST be kept; tokens with
excl >= p + TAU MUST be dropped; tokens inside the ±TAU band are free — that
band absorbs legitimate fp32 summation-order rounding in sort-free solutions.
There is no float tolerance anywhere else to game.
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
from src.eval.cuda_language import collect_solution_sources  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
)

TAU = 1e-3  # nucleus-boundary band, in cumulative probability mass


def oracle_bands(logits: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (must_keep, must_drop) bool masks from the fp64 oracle."""
    probs = torch.softmax(logits.double(), dim=-1)
    sp, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    excl = sp.cumsum(dim=-1) - sp  # exclusive cumulative mass, sorted order
    keep_sorted = excl < (p - TAU)
    drop_sorted = excl >= (p + TAU)
    must_keep = torch.zeros_like(keep_sorted)
    must_keep.scatter_(-1, idx, keep_sorted)
    must_drop = torch.zeros_like(drop_sorted)
    must_drop.scatter_(-1, idx, drop_sorted)
    return must_keep, must_drop


def main():
    try:
        import reference
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

    check_shapes = [
        {"B": 1, "V": 151936, "P": 0.9},
        {"B": 4, "V": 32000, "P": 0.95},
        {"B": 2, "V": 100003, "P": 0.92},
    ]
    for shape_idx, shape in enumerate(check_shapes):
        reference.B = shape["B"]
        reference.V = shape["V"]
        reference.P = shape["P"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch at shape {shape_idx} ({shape}): {e}")
            sys.exit(1)

        for seed in (42, 123, 456):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_inputs = [t.to(device) for t in reference.get_inputs()]

            for case in numeric_stress_cases(meta.get("name", "")):
                with numeric_stress_context(ref_model, sol_model, base_inputs, case) as inputs:
                    with torch.no_grad():
                        sol_out = sol_model(*inputs)

                if sol_out.dtype != torch.bool or sol_out.shape != inputs[0].shape:
                    print(
                        f"FAIL: shape {shape_idx} seed {seed} case {case.name}: "
                        f"output must be bool of shape {tuple(inputs[0].shape)}, "
                        f"got {sol_out.dtype} {tuple(sol_out.shape)}"
                    )
                    sys.exit(1)

                must_keep, must_drop = oracle_bands(inputs[0], shape["P"])
                missing = (must_keep & ~sol_out).sum().item()
                extra = (must_drop & sol_out).sum().item()
                if missing or extra:
                    print(
                        f"FAIL: shape {shape_idx} {shape} seed {seed} case {case.name}: "
                        f"{missing} must-keep tokens dropped, {extra} must-drop tokens kept "
                        f"(band tau={TAU})"
                    )
                    sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
