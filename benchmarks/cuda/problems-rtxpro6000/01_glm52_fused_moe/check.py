"""Correctness + CUDA language gate for GLM-5.2 fused MoE."""
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.cuda_language import collect_solution_sources  # noqa: E402
from src.eval import cuda_language as cl  # noqa: E402
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
    tol_override = meta.get("tolerance") or None

    for shape_idx, shape in enumerate(shapes.SHAPES):
        # Cap T for check wall-clock / memory; full T in benchmark.
        check_shape = dict(shape)
        check_shape["T"] = min(int(shape["T"]), 256)

        reference.T = check_shape["T"]
        reference.E = check_shape["E"]
        reference.top_k = check_shape["top_k"]
        reference.n_shared = check_shape["n_shared"]
        reference.H = check_shape["H"]
        reference.I = check_shape["I"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch at shape {shape_idx}: {e}")
            sys.exit(1)

        for seed in (42, 123, 456):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_inputs = [t.to(device) for t in reference.get_inputs()]
            for case in numeric_stress_cases(meta.get("name", "")):
                with numeric_stress_context(ref_model, sol_model, base_inputs, case) as inputs:
                    with torch.no_grad():
                        ref_out = ref_model(*inputs)
                        sol_out = sol_model(*inputs)
                ok, msg = check_correctness(
                    ref_out.float(),
                    sol_out.float(),
                    dtype=torch.bfloat16,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} seed {seed} case {case.name}: {msg}")
                    sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
