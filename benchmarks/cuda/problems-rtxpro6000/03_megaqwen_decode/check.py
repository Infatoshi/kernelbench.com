"""Correctness + CUDA language gate for MegaQwen decode.

Uses CHECK_SHAPES only (short ctx). Full 2k/8k/32k/128k is for benchmark.
Compares last_hidden after prefill+decode — no tokens.
"""
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
import reference as ref  # noqa: E402
import shapes as shape_mod  # noqa: E402


def _reinit(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02, generator=g)
        else:
            torch.nn.init.ones_(p)


def main():
    try:
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

    if not hasattr(solution, "Model"):
        print("FAIL: solution.py must define Model")
        sys.exit(1)
    if not hasattr(solution, "run"):
        print(
            "FAIL: solution.py must define "
            "run(ctx_len, decode_steps, seed, model=None) -> dict(last_hidden=...)"
        )
        sys.exit(1)

    device = torch.device("cuda:0")
    tol = meta.get("tolerance") or {"bfloat16": 0.08}
    check_shapes = getattr(shape_mod, "CHECK_SHAPES", shape_mod.SHAPES[:1])

    for seed in (42, 123, 456):
        for shape in check_shapes:
            ctx = int(shape["ctx_len"])
            dec = int(shape["decode_steps"])
            max_seq = ctx + dec + 8
            ref_model = ref.Model(ref.NUM_LAYERS, max_seq).to(device).eval()
            _reinit(ref_model, seed)
            sol_model = solution.Model(ref.NUM_LAYERS, max_seq).to(device).eval()
            try:
                sol_model.load_state_dict(ref_model.state_dict(), strict=True)
            except RuntimeError as e:
                print(f"FAIL: state_dict mismatch: {e}")
                sys.exit(1)

            ref_out = ref.run(ctx, dec, seed, model=ref_model, max_seq=max_seq)
            try:
                sol_out = solution.run(ctx, dec, seed, model=sol_model)
            except TypeError:
                sol_out = solution.run(ctx, dec, seed)

            if "last_hidden" not in sol_out:
                print("FAIL: solution.run must return dict with last_hidden")
                sys.exit(1)

            ok, msg = check_correctness(
                ref_out["last_hidden"].float().cpu(),
                sol_out["last_hidden"].float().cpu(),
                dtype=torch.bfloat16,
                override=tol,
            )
            if not ok:
                print(f"FAIL: seed={seed} ctx={ctx} decode_steps={dec}: {msg}")
                sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
