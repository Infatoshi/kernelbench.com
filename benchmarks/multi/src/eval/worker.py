"""Per-rank torchrun entrypoint for KernelBench-Multi.

Launched by check.py / benchmark.py via the launcher as:

    torchrun --nproc_per_node=N <repo>/src/eval/worker.py \
        --mode {check,benchmark} --problem-dir <abs path>

Each rank initializes the process group, imports the problem's reference / shapes
/ solution from --problem-dir, builds RANK-DISTINCT inputs (defeats the
rank-symmetry hack), and either compares per-rank outputs (check) or times the
solution and reports busbw (benchmark).

Backend / device are env-controlled so the whole thing validates on a single-GPU
box with gloo+cpu before any NVLink node is rented:
    KBM_BACKEND=gloo KBM_DEVICE=cpu   # local correctness smoke
    KBM_BACKEND=nccl (default)        # real 4xH100 run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

_PRIME = 1_000_003  # rank stride for distinct seeds


def _setup_paths(problem_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]  # benchmarks/multi
    sys.path.insert(0, str(problem_dir))             # reference/shapes/solution
    sys.path.insert(0, str(repo_root))               # src.*
    return repo_root


def _device(backend: str) -> torch.device:
    if backend == "gloo" or os.environ.get("KBM_DEVICE") == "cpu":
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{local_rank}")


def _seed(base: int, rank: int) -> None:
    torch.manual_seed(base + rank * _PRIME)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base + rank * _PRIME)


def _max_reduce(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _set_shape(reference, shape: dict) -> None:
    for k, v in shape.items():
        setattr(reference, k, v)


# The c10d surface the reference oracle may call. If a solution rebinds any of
# these (or any function/class defined in reference.py), the oracle itself is
# compromised — caught in the wild on the first 4xH100 smoke run, where the
# solution monkey-patched dist.all_reduce so the reference computed the same
# fp32-accumulated sum as the solution ("_install_exact_reference_reduce").
_GUARDED_DIST = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
    "reduce_scatter_tensor", "all_to_all", "all_to_all_single", "broadcast",
    "reduce", "gather", "scatter", "send", "recv", "barrier",
)


def oracle_snapshot(reference) -> dict:
    """Capture identities of everything the reference oracle depends on.

    Must be called after importing reference but BEFORE importing solution.
    """
    snap = {f"dist.{n}": getattr(dist, n) for n in _GUARDED_DIST if hasattr(dist, n)}
    for k, v in vars(reference).items():
        if callable(v):
            snap[f"reference.{k}"] = v
            for ak, av in vars(v).items():
                if callable(av):
                    snap[f"reference.{k}.{ak}"] = av
    return snap


def oracle_tampered(reference, snap: dict) -> str | None:
    """Return the first rebound name, or None if the oracle is intact."""
    for key, orig in snap.items():
        parts = key.split(".")
        obj = dist if parts[0] == "dist" else reference
        cur = obj
        for attr in parts[1:]:
            cur = getattr(cur, attr, None) if not isinstance(cur, dict) else None
        if cur is not orig:
            return key
    return None


def run_check(reference, shapes, solution, meta, device, rank, world, snap=None):
    from src.eval.compare import compare
    from src.eval.stress import apply_scale, stress_cases

    tol_override = meta.get("tolerance")
    trials = int(os.environ.get("KBM_TRIALS", meta.get("num_correct_trials", 5)))
    fail_msg = ""
    failed = 0

    for shape_idx, shape in enumerate(shapes.SHAPES):
        _set_shape(reference, shape)
        init_args = reference.get_init_inputs()
        torch.manual_seed(2026)  # params identical between ref and sol on this rank
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError as e:
            if not failed:
                fail_msg = f"shape {shape_idx} state_dict mismatch: {e}"
            failed = 1
            # keep collectives in lockstep by still running forwards below

        for trial in range(trials):
            for case in stress_cases():
                _seed(1000 + trial, rank)
                base_inputs = [t.to(device) for t in reference.get_inputs()]
                inputs = apply_scale(base_inputs, case)
                try:
                    with torch.no_grad():
                        ref_out = ref_model(*[t.clone() for t in inputs])
                        sol_out = sol_model(*[t.clone() for t in inputs])
                    ok, msg = compare(ref_out, sol_out, tol_override)
                except Exception as e:  # noqa: BLE001
                    ok, msg = False, f"exception in forward: {type(e).__name__}: {e}"
                if not ok and not failed:
                    fail_msg = (
                        f"shape {shape_idx} {shape} trial {trial} case {case.name} "
                        f"rank {rank}: {msg}"
                    )
                    failed = 1

    tampered = oracle_tampered(reference, snap) if snap else None
    if tampered and not failed:
        fail_msg = f"reference oracle tampered: {tampered} was rebound by the solution"
        failed = 1

    if failed:
        print(f"RANK {rank} FAIL: {fail_msg}", flush=True)
    any_fail = _max_reduce(float(failed), device)
    if rank == 0:
        print("FAIL" if any_fail > 0 else "PASS", flush=True)
    return int(any_fail)


def run_benchmark(reference, shapes, solution, meta, device, rank, world):
    from src.eval.busbw import busbw_gb_s, eval_formula, geomean, peak_fraction
    from src.hardware import get as get_hw

    hw = get_hw(meta.get("hardware", ["H100x4"])[0])
    peak_busbw = hw.peak_nvlink_busbw_gb_s
    dtype_bytes = int(meta.get("dtype_bytes", 2))
    busbw_formula = meta["busbw_bytes_formula"]
    warmup = int(os.environ.get("KBM_WARMUP", meta.get("num_warmup", 500)))
    iters = int(os.environ.get("KBM_ITERS", meta.get("num_perf_trials", 100)))
    use_cuda = device.type == "cuda"

    fractions: list[float] = []
    for shape_idx, shape in enumerate(shapes.SHAPES):
        _set_shape(reference, shape)
        init_args = reference.get_init_inputs()
        torch.manual_seed(2026)
        sol_model = solution.Model(*init_args).to(device).eval()
        _seed(7, rank)
        inputs = [t.to(device) for t in reference.get_inputs()]

        with torch.no_grad():
            for _ in range(warmup):
                sol_model(*[t.clone() for t in inputs])
            if use_cuda:
                torch.cuda.synchronize()
            dist.barrier()

            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    sol_model(*[t.clone() for t in inputs])
                end.record()
                torch.cuda.synchronize()
                ms_local = start.elapsed_time(end) / iters
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    sol_model(*[t.clone() for t in inputs])
                ms_local = (time.perf_counter() - t0) * 1e3 / iters

        ms = _max_reduce(ms_local, device)  # slowest rank gates the collective
        if rank == 0:
            fvars = dict(shape)
            fvars.update(world_size=world, dtype_bytes=dtype_bytes)
            busbw_bytes = eval_formula(busbw_formula, fvars)
            achieved = busbw_gb_s(busbw_bytes, ms)
            frac = peak_fraction(achieved, peak_busbw)
            fractions.append(frac)
            print(
                f"shape={shape_idx} variant=solution busbw={achieved:.2f} "
                f"ms={ms:.4f} peak_fraction={frac:.4f}",
                flush=True,
            )

    if rank == 0:
        g = geomean(fractions)
        print(f"peak_fraction: {g:.4f}", flush=True)
        print(f"RESULT: {'OK' if g >= 0.1 else 'LOW'}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "benchmark"])
    ap.add_argument("--problem-dir", required=True)
    args = ap.parse_args()

    problem_dir = Path(args.problem_dir).resolve()
    _setup_paths(problem_dir)

    import reference  # noqa: E402
    import shapes  # noqa: E402

    snap = oracle_snapshot(reference)  # BEFORE solution import: it can patch
    import solution  # noqa: E402

    meta = yaml.safe_load((problem_dir / "problem.yaml").read_text())

    backend = os.environ.get("KBM_BACKEND", "nccl")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = _device(backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    try:
        if args.mode == "check":
            rc = run_check(reference, shapes, solution, meta, device, rank, world, snap)
        else:
            rc = run_benchmark(reference, shapes, solution, meta, device, rank, world)
    finally:
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
