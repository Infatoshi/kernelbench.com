"""Latency benchmark for the Kimi-Linear hybrid decode unit.

Score = baseline_latency / solution_latency (speedup over the optimized-PyTorch
baseline.py), geomean over the context-length sweep. Reported in the
peak_fraction field as a speedup multiple (>1 means faster than the baseline),
not a [0,1] roofline fraction. Absolute tok/s at the longest context is the
shippable "single-GPU decode" number.

The two sides are timed in separate processes, solution first. This parent
never touches CUDA; it runs
  1. a solution child: per shape, a one-step correctness gate against
     reference.py (on the graded cache and on a cache where MLA attention
     carries the output), then the timed decode; a wrong solution scores zero
     and the baseline is never run;
  2. a baseline child, started only after the solution child and everything
     it spawned have exited, so no CUDA context that ran solution code is alive
     (device-level state a solution sets, such as L2 persistence, cannot leak
     into the baseline's timing). It runs isolated (`python -I`) and loads
     reference.py / baseline.py / shapes.py by path; it never imports solution.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from math import exp, log
from pathlib import Path

SOLUTION_TIMEOUT_S = 1200
BASELINE_TIMEOUT_S = 600
WARMUP_STEPS = 3


# --------------------------------------------------------------------------- #
# children (CUDA lives only here)
# --------------------------------------------------------------------------- #
def _run(model, cfg, ctx, steps, seed, ref):
    import torch
    h = ref.init_token(cfg, seed)
    st = ref.init_state(cfg, ctx, seed)
    with torch.no_grad():
        for _ in range(steps):
            h, st = model.step(h, st)
    torch.cuda.synchronize()
    return h


def _time(model, cfg, ctx, steps, seed, ref, trials):
    import torch
    _run(model, cfg, ctx, min(WARMUP_STEPS, steps), seed, ref)   # warmup
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        _run(model, cfg, ctx, steps, seed, ref)
        best = min(best, time.perf_counter() - t0)
    return best


def _load_by_path(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _child(variant: str, out_path: str, pdir: Path) -> None:
    import torch
    import torch.nn.functional as F
    import yaml

    meta = yaml.safe_load((pdir / "problem.yaml").read_text())
    thr = float(meta["tolerance"]["cos_sim"])
    trials = int(meta.get("num_perf_trials", 3))
    device = torch.device("cuda:0")
    if variant == "solution":
        import reference as ref
        import shapes
        import solution as impl
    else:   # baseline: isolated interpreter, graded files loaded by path
        ref = _load_by_path("reference", pdir / "reference.py")
        shapes = _load_by_path("shapes", pdir / "shapes.py")
        impl = _load_by_path("baseline", pdir / "baseline.py")

    res = {"variant": variant, "shapes": []}
    ref_models = {}
    for shp in shapes.SHAPES:
        cfg = ref.build_config(shp)
        ctx, steps = int(shp["context_len"]), int(shp["steps"])
        if cfg not in ref_models:
            ref_models[cfg] = ref.Model(cfg).to(device).eval()
        ref_model = ref_models[cfg]
        model = impl.Model(cfg).to(device).eval()
        model.load_state_dict(ref_model.state_dict(), strict=True)

        if variant == "solution":
            # correctness gate: one decode step from identical state, on the
            # graded cache and on an attention-sensitive cache
            for label, mk, seed in (("", ref.init_state, 7),
                                    ("attention case, ", ref.init_state_attn, shapes.ATTENTION_SEEDS[0])):
                tok = ref.init_token(cfg, seed)
                with torch.no_grad():
                    o_r, _ = ref_model.step(tok.clone(), mk(cfg, ctx, seed))
                    o_s, _ = model.step(tok.clone(), mk(cfg, ctx, seed))
                c = F.cosine_similarity(o_r.float().flatten(), o_s.float().flatten(), dim=0).item()
                if c < thr or not torch.isfinite(o_s).all():
                    res["gate_fail"] = (f"shape ctx={ctx}: solution disagrees with reference "
                                        f"({label}cosine {c:.4f} < {thr})")
                    Path(out_path).write_text(json.dumps(res))
                    return

        secs = _time(model, cfg, ctx, steps, 7, ref, trials)
        res["shapes"].append({"ctx": ctx, "steps": steps, "secs": secs})
        del model
        torch.cuda.empty_cache()
    res["complete"] = True
    Path(out_path).write_text(json.dumps(res))


# --------------------------------------------------------------------------- #
# parent (no CUDA)
# --------------------------------------------------------------------------- #
_ACTIVE: list[subprocess.Popen] = []


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _on_term(signum, _frame):
    for p in _ACTIVE:
        _kill_group(p)
    raise SystemExit(128 + signum)


def _reap_stragglers(nonce: str) -> list[int]:
    """SIGKILL any process still carrying this child's nonce (Linux /proc)."""
    hits = []
    me = os.getpid()
    for d in Path("/proc").iterdir() if Path("/proc").is_dir() else ():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            env = (d / "environ").read_bytes()
        except OSError:
            continue
        if f"KB_BENCH_CHILD={nonce}".encode() in env.split(b"\0"):
            hits.append(int(d.name))
            try:
                os.kill(int(d.name), signal.SIGKILL)
            except OSError:
                pass
    return hits


def _spawn(variant: str, pdir: Path, timeout: int) -> dict:
    tmpdir = tempfile.mkdtemp(prefix=f"kb_bench_{variant}_")
    out = Path(tmpdir) / "result.json"
    nonce = os.urandom(8).hex()
    env = dict(os.environ, KB_BENCH_CHILD=nonce)
    py = [sys.executable] + (["-I"] if variant == "baseline" else [])
    cmd = py + [str(Path(__file__).resolve()), "--child", variant, str(out), str(pdir)]
    p = subprocess.Popen(cmd, cwd=pdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    _ACTIVE.append(p)

    def relay():
        for line in p.stdout:
            sys.stderr.write(f"[{variant}] " + line.decode(errors="replace"))
            sys.stderr.flush()

    t = threading.Thread(target=relay, daemon=True)
    t.start()
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = None
    _kill_group(p)           # nothing the child spawned outlives it
    p.wait()
    t.join(timeout=5)
    _ACTIVE.remove(p)
    stray = _reap_stragglers(nonce)
    if stray:
        print(f"[{variant}] killed {len(stray)} leftover process(es) of the {variant} child",
              file=sys.stderr, flush=True)
    if rc is None:
        raise RuntimeError(f"{variant} child timed out after {timeout}s")
    try:
        res = json.loads(out.read_text())
    except (OSError, ValueError):
        raise RuntimeError(f"{variant} child exited {rc} without a result") from None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if rc != 0 or not (res.get("complete") or res.get("gate_fail")):
        raise RuntimeError(f"{variant} child exited {rc} with an incomplete result")
    return res


def main() -> None:
    pdir = Path(__file__).resolve().parent
    signal.signal(signal.SIGTERM, _on_term)
    sol = _spawn("solution", pdir, SOLUTION_TIMEOUT_S)
    if sol.get("gate_fail"):
        print(sol["gate_fail"])
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        return
    base = _spawn("baseline", pdir, BASELINE_TIMEOUT_S)
    if [s["ctx"] for s in base["shapes"]] != [s["ctx"] for s in sol["shapes"]]:
        raise RuntimeError("solution and baseline children timed different shapes")

    speedups = []
    for s, b in zip(sol["shapes"], base["shapes"]):
        steps, sol_s, base_s = s["steps"], s["secs"], b["secs"]
        speedup = base_s / sol_s if sol_s > 0 else 0.0
        speedups.append(speedup)
        print(
            f"shape ctx={s['ctx']}: solution {sol_s*1e3/steps:.3f} ms/tok ({steps/sol_s:.0f} tok/s) | "
            f"baseline {base_s*1e3/steps:.3f} ms/tok | speedup {speedup:.2f}x",
            flush=True,
        )
    gmean = exp(sum(log(max(s, 1e-9)) for s in speedups) / len(speedups))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 1.0 else 'LOW'}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        sys.exit(0)
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"benchmark error: {type(e).__name__}: {e}")
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        sys.exit(1)
