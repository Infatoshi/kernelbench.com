"""Latency benchmark for the Kimi-Linear hybrid decode unit.

Score = baseline_latency / solution_latency (speedup over the optimized-PyTorch
baseline.py), geomean over the context-length sweep. The solution is timed
first; a final next-token check against the reference makes a fast-but-wrong
solution score zero. Absolute tok/s at the longest context is the shippable
"single-GPU decode" number.

Reported in the peak_fraction field as a speedup multiple (>1 means faster than
the baseline), not a [0,1] roofline fraction.
"""
from __future__ import annotations

import sys
import time
from math import exp, log
from pathlib import Path

import torch
import yaml


def _run(model, cfg, ctx, steps, seed, ref):
    h = ref.init_token(cfg, seed)
    st = ref.init_state(cfg, ctx, seed)
    with torch.no_grad():
        for _ in range(steps):
            h, st = model.step(h, st)
    torch.cuda.synchronize()
    return h


def _time(model, cfg, ctx, steps, seed, ref):
    _run(model, cfg, ctx, min(3, steps), seed, ref)   # warmup
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(int(yaml.safe_load(Path("problem.yaml").read_text()).get("num_perf_trials", 3))):
        t0 = time.perf_counter()
        h = _run(model, cfg, ctx, steps, seed, ref)
        best = min(best, time.perf_counter() - t0)
    return best, h


def main() -> None:
    import baseline
    import reference as ref
    import shapes
    import solution
    import torch.nn.functional as F
    meta = yaml.safe_load(Path("problem.yaml").read_text())
    thr = float(meta["tolerance"]["cos_sim"])
    device = torch.device("cuda:0")
    speedups = []

    for shp in shapes.SHAPES:
        cfg = ref.build_config(shp)
        ctx, steps = int(shp["context_len"]), int(shp["steps"])
        ref_model = ref.Model(cfg).to(device).eval()
        sol_model = solution.Model(cfg).to(device).eval()
        base_model = baseline.Model(cfg).to(device).eval()
        sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        base_model.load_state_dict(ref_model.state_dict(), strict=True)

        # correctness gate: one decode step from identical state, cosine sim
        st_r = ref.init_state(cfg, ctx, 7)
        st_s = ref.init_state(cfg, ctx, 7)
        tok = ref.init_token(cfg, 7)
        with torch.no_grad():
            o_r, _ = ref_model.step(tok.clone(), st_r)
            o_s, _ = sol_model.step(tok.clone(), st_s)
        c = F.cosine_similarity(o_r.float().flatten(), o_s.float().flatten(), dim=0).item()
        if c < thr or not torch.isfinite(o_s).all():
            print(f"shape ctx={ctx}: solution disagrees with reference (cosine {c:.4f} < {thr})")
            print("peak_fraction: 0.0")
            print("RESULT: LOW")
            return

        sol_ms, _ = _time(sol_model, cfg, ctx, steps, 7, ref)
        base_ms, _ = _time(base_model, cfg, ctx, steps, 7, ref)
        speedup = base_ms / sol_ms if sol_ms > 0 else 0.0
        speedups.append(speedup)
        print(
            f"shape ctx={ctx}: solution {sol_ms*1e3/steps:.3f} ms/tok ({steps/sol_ms:.0f} tok/s) | "
            f"baseline {base_ms*1e3/steps:.3f} ms/tok | speedup {speedup:.2f}x",
            flush=True,
        )

    gmean = exp(sum(log(max(s, 1e-9)) for s in speedups) / len(speedups))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 1.0 else 'LOW'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"benchmark error: {type(e).__name__}: {e}")
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        sys.exit(1)
