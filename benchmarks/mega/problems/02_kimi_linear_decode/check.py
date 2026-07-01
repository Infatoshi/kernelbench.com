"""Correctness runner for the Kimi-Linear hybrid decode unit.

Per-step correctness, not a chained trajectory: for each seed and a couple of
context lengths, the reference and the solution are given an IDENTICAL fed
state and one new token, each runs ONE decode step, and we compare the
next-token hidden plus the updated decode state (KDA recurrent state, MLA latent
cache) by cosine similarity. Cosine sim is robust to bf16/fp8 precision but
still rejects wrong answers. Chaining steps would just measure how fast two
precisions diverge, which is not a kernel-correctness signal.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main() -> None:
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    # Scan solution.py AND every local module it imports (recursively), so
    # forbidden ops / the megakernel escape hatches (CUDA graphs, torch.compile)
    # can't hide in a sidecar kernels.py. Harness files are excluded.
    import ast as _ast
    _HARNESS = {"reference", "baseline", "shapes", "check", "benchmark", "problem"}

    def _local_imports(_src):
        _mods = set()
        try:
            _tree = _ast.parse(_src)
        except Exception:
            return _mods
        for _node in _ast.walk(_tree):
            if isinstance(_node, _ast.Import):
                for _n in _node.names:
                    _mods.add(_n.name.split(".")[0])
            elif isinstance(_node, _ast.ImportFrom):
                if _node.module and _node.level == 0:
                    _mods.add(_node.module.split(".")[0])
        return _mods

    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    _seen = {"solution"}
    _queue = [
        _m for _m in _local_imports(sol_src)
        if _m not in _HARNESS and Path(_m + ".py").exists()
    ]
    while _queue:
        _m = _queue.pop()
        if _m in _seen:
            continue
        _seen.add(_m)
        try:
            _s = Path(_m + ".py").read_text()
        except Exception:
            continue
        sol_src += "\n" + _s
        for _mm in _local_imports(_s):
            if _mm not in _seen and _mm not in _HARNESS and Path(_mm + ".py").exists():
                _queue.append(_mm)
    for forbidden in meta.get("forbidden", []):
        if re.search(re.escape(forbidden), sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    thr = float(meta["tolerance"]["cos_sim"])
    cfg = reference.build_config(shapes.SHAPES[0])
    device = torch.device("cuda:0")
    ref_model = reference.Model(cfg).to(device).eval()
    try:
        sol_model = solution.Model(cfg).to(device).eval()
        sol_model.load_state_dict(ref_model.state_dict(), strict=True)
    except Exception as e:
        print(f"FAIL: solution.Model could not load reference weights: {e}")
        sys.exit(1)

    kda_layer = cfg.pattern.index("K")
    mla_layer = cfg.pattern.index("M")

    for seed in shapes.GRADING_SEEDS:
        for ctx in (2048, 8192):
            h = reference.init_token(cfg, seed)
            st_ref = reference.init_state(cfg, ctx, seed)
            st_sol = reference.init_state(cfg, ctx, seed)
            try:
                with torch.no_grad():
                    o_ref, st_ref = ref_model.step(h.clone(), st_ref)
                    o_sol, st_sol = sol_model.step(h.clone(), st_sol)
            except Exception as e:
                print(f"FAIL: seed {seed} ctx {ctx}: solution.step raised {type(e).__name__}: {e}")
                sys.exit(1)

            if o_ref.shape != o_sol.shape or not torch.isfinite(o_sol).all():
                print(f"FAIL: seed {seed} ctx {ctx}: bad output shape/finite")
                sys.exit(1)

            checks = {
                "out": _cos(o_ref, o_sol),
                "kda_state": _cos(st_ref[kda_layer]["S"], st_sol[kda_layer]["S"]),
                "mla_cache": _cos(st_ref[mla_layer]["c_kv"], st_sol[mla_layer]["c_kv"]),
            }
            for name, c in checks.items():
                if c < thr:
                    print(f"FAIL: seed {seed} ctx {ctx}: {name} cosine {c:.4f} < {thr}")
                    sys.exit(1)
            print(f"seed {seed} ctx {ctx}: out={checks['out']:.4f} "
                  f"S={checks['kda_state']:.4f} cache={checks['mla_cache']:.4f} OK", flush=True)

    _emit_framework_label(sol_src)
    print("PASS")


def _emit_framework_label(code: str) -> None:
    patterns = [
        ("ptx", r"asm\s+volatile|mma\.sync|tcgen05\."),
        ("cuda_raw", r"load_inline|__global__\s+void"),
        ("triton", r"import\s+triton\b|@triton\.jit"),
        ("cudagraph", r"CUDAGraph|make_graphed_callables|capture_begin"),
        ("compile", r"torch\.compile"),
    ]
    label = "eager"
    for name, pat in patterns:
        if re.search(pat, code):
            label = name
            break
    Path("framework.txt").write_text(label + "\n")


if __name__ == "__main__":
    main()
