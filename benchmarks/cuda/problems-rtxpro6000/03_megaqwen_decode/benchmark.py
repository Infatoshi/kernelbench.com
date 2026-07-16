"""Decode-only throughput at fixed context lengths (2k/8k/32k/128k).

Prefill is performed but NOT timed. Only decode_steps after KV is warm count
toward tok/s. Score = geomean(tok_s / peak_tok_s) over shapes.
"""
import sys
import time
from math import exp, log
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    peak = float(meta.get("peak_tok_s", 5e4))
    num_perf_trials = int(meta.get("num_perf_trials", 3))

    if not hasattr(solution, "run") and not (
        hasattr(solution, "prefill") and hasattr(solution, "decode_steps")
    ):
        # Prefer explicit prefill + decode_steps for fair timing; fall back to run().
        if not hasattr(solution, "run"):
            print(
                "FAIL: solution must define prefill+decode_steps or run(ctx_len, decode_steps, seed)"
            )
            sys.exit(1)

    fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        ctx = int(shape["ctx_len"])
        dec = int(shape["decode_steps"])
        max_seq = ctx + dec + 8
        seed = 2026 + shape_idx

        # Build model once per shape
        if hasattr(solution, "Model"):
            sol_model = solution.Model(reference.NUM_LAYERS, max_seq).cuda().eval()
            ref_model = reference.Model(reference.NUM_LAYERS, max_seq).cuda().eval()
            sol_model.load_state_dict(ref_model.state_dict(), strict=False)
        else:
            sol_model = None

        # --- untimed prefill ---
        torch.cuda.synchronize()
        t_pre0 = time.perf_counter()
        if hasattr(solution, "prefill") and sol_model is not None:
            h, k_caches, v_caches = solution.prefill(sol_model, ctx, seed)
        else:
            # Fall back: full run path (prefill internal); still only time decode below via second path
            h = k_caches = v_caches = None
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_pre0

        # Warmup decode
        if h is not None and hasattr(solution, "decode_steps"):
            solution.decode_steps(sol_model, h, k_caches, v_caches, ctx, min(4, dec), seed)
            # re-prefill clean state
            h, k_caches, v_caches = solution.prefill(sol_model, ctx, seed)
        torch.cuda.synchronize()

        times = []
        for trial in range(num_perf_trials):
            if hasattr(solution, "prefill") and hasattr(solution, "decode_steps"):
                h, k_caches, v_caches = solution.prefill(sol_model, ctx, seed + trial)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                solution.decode_steps(
                    sol_model, h, k_caches, v_caches, ctx, dec, seed + trial
                )
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            else:
                # run() includes prefill — time whole thing is wrong. Approximate by
                # subtracting a measured prefill-only if solution only has run().
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                solution.run(ctx, dec, seed + trial, model=sol_model)
                torch.cuda.synchronize()
                full = time.perf_counter() - t0
                # one prefill-only estimate via reference (not ideal)
                t1 = time.perf_counter()
                reference.prefill(
                    reference.Model(reference.NUM_LAYERS, max_seq).cuda().eval(),
                    ctx,
                    seed + trial,
                )
                torch.cuda.synchronize()
                # Don't subtract ref prefill from sol full — just report full with warning once
                times.append(full)
                if trial == 0 and shape_idx == 0:
                    print(
                        "WARN: solution lacks prefill/decode_steps split; "
                        "timing includes prefill. Implement prefill+decode_steps for fair score.",
                        flush=True,
                    )

        times.sort()
        wall = times[len(times) // 2]
        tok_s = dec / max(wall, 1e-9)
        frac = tok_s / peak
        fractions.append(frac)
        print(
            f"shape={shape_idx} variant=solution "
            f"ctx_len={ctx} decode_steps={dec} "
            f"tok_s={tok_s:.3f} decode_wall_s={wall:.4f} "
            f"prefill_wall_s={prefill_s:.4f} peak_fraction={frac:.4f}",
            flush=True,
        )

        # Eager reference floor on shortest shape only (short ctx)
        if shape_idx == 0:
            rmodel = reference.Model(reference.NUM_LAYERS, max_seq).cuda().eval()
            h, k_c, v_c = reference.prefill(rmodel, min(ctx, 256), seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reference.decode_steps(rmodel, h, k_c, v_c, min(ctx, 256), min(dec, 16), seed)
            torch.cuda.synchronize()
            rw = time.perf_counter() - t0
            print(
                f"shape={shape_idx} variant=eager_ref "
                f"tok_s={min(dec, 16) / max(rw, 1e-9):.3f} decode_wall_s={rw:.4f}",
                flush=True,
            )

    gmean = exp(sum(log(max(f, 1e-12)) for f in fractions) / len(fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.001 else 'LOW'}")


if __name__ == "__main__":
    main()
