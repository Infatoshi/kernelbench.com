"""Set up a sandboxed workspace for a single KernelBench problem.

Creates a directory with:
  - reference.py        (the problem to optimize)
  - CLAUDE.md           (instructions for any agent harness)
  - check.py            (correctness check script)
  - benchmark.py        (timing script, run after check passes)

Usage:
  uv run python scripts/setup_workspace.py <hardware> <problem_path> [--out-dir DIR]

Example:
  uv run python scripts/setup_workspace.py rtx3090 problems/level1/1_Square_matrix_multiplication_.py
  # Creates: workspaces/rtx3090/1_Square_matrix_multiplication_/
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import ARCH_B200, ARCH_H100, ARCH_RTX3090, ARCH_RTX_PRO_6000, PROFILING_TOOLS  # noqa: E402

ARCH_SECTIONS = {
    "rtx3090": ARCH_RTX3090,
    "rtx_pro_6000": ARCH_RTX_PRO_6000,
    "h100": ARCH_H100,
    "b200": ARCH_B200,
}

HARDWARE_INFO = {
    "rtx3090": ("RTX 3090", 24, "Ampere SM86"),
    "rtx_pro_6000": ("RTX PRO 6000 Blackwell Workstation", 96, "Blackwell SM120"),
    "h100": ("H100", 80, "Hopper SM90"),
    "b200": ("B200", 192, "Blackwell SM100"),
}


def infer_level(problem_path: Path) -> int:
    """Infer problem level from directory name."""
    parent = problem_path.parent.name
    for i in range(1, 5):
        if str(i) in parent:
            return i
    if parent in ("graphics",):
        return 1
    if parent in ("tile_specialized",):
        return 3
    if parent in ("cutile",):
        return 4
    return 2


GATE_DESCRIPTIONS = {
    "triton":    "Your solution MUST use Triton (@triton.jit). Raw CUDA / CUTLASS / PTX not accepted for this problem.",
    "cutlass2":  "Your solution MUST use CUTLASS 2.x (device GEMM or CuTe layouts) -- `cutlass/gemm/device/gemm.h` etc. Triton not accepted.",
    "cutlass3":  "Your solution MUST use CUTLASS 3.x or CuTe DSL (`cute::` namespace, collective builders). Triton not accepted.",
    "cuda_wmma": "Your solution MUST use raw CUDA with nvcuda::wmma tensor-core APIs (`<mma.h>`). Triton not accepted.",
    "ptx":       "Your solution MUST include inline PTX (`asm volatile` with mma.sync / wgmma / tcgen05). Triton not accepted.",
    "cutile":    "Your solution MUST use cuTile (CUDA 13.x `cutile::` namespace).",
    "no_triton": "Your solution MUST NOT use Triton. Any CUDA dialect (raw CUDA, CUTLASS, PTX, WMMA) is fine.",
}


def build_claude_md(hardware: str, problem_name: str, level: int, reference_code: str, framework_gate: str | None = None) -> str:
    """Build the CLAUDE.md that any agent harness will read."""
    gpu_name, vram, arch = HARDWARE_INFO[hardware]
    arch_section = ARCH_SECTIONS.get(hardware, "")

    gate_text = ""
    if framework_gate:
        gate_rule = GATE_DESCRIPTIONS.get(framework_gate, f"Framework gated to: {framework_gate}")
        gate_text = f"\n## Framework Requirement\n{gate_rule}\n"

    return f"""# KernelBench Task

## Objective
Write an optimized CUDA kernel that is faster than the PyTorch reference implementation.

## Hardware
- GPU: NVIDIA {gpu_name} ({vram}GB VRAM) -- {arch}
- CUDA toolkit available, CUTLASS headers at `/opt/cutlass/include`

## Files
- `reference.py` -- the PyTorch reference to beat. Do NOT modify this file.
- `solution.py` -- write your optimized solution here. Must be interface-compatible with reference.py.
- `check.py` -- run this to verify correctness: `python check.py`
- `benchmark.py` -- run this after check passes to measure speedup: `python benchmark.py`

## Rules
1. Your `solution.py` must define `Model`, `get_inputs`, `get_init_inputs` matching the reference
2. You MUST write at least one custom kernel using one of:
   - CUDA C++ via `torch.utils.cpp_extension.load_inline`
   - Triton via `@triton.jit`
   - Inline PTX assembly
   - CUTLASS templates (headers at `/opt/cutlass/include`, compile with `-I/opt/cutlass/include -std=c++17`)
3. FORBIDDEN -- your solution will be rejected if it uses any of these:
   - `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.conv2d`, `F.linear`, `F.conv2d`
   - `torch.compile`, `@torch.jit.script`
   - External compute libraries: `flash_attn`, `xformers`, `fla.ops`
   - Simply wrapping an nn.Module with a dtype cast is not optimization
4. Correctness is mandatory. Run `python check.py` and get PASS before submitting.
5. After correctness, run `python benchmark.py` to measure your speedup vs the reference.

## Workflow
1. Read `reference.py` to understand the operation
2. Analyze: is this memory-bound or compute-bound? (see hardware capabilities below)
3. Write `solution.py` with your optimized kernel
4. Run `python check.py` -- fix until PASS
5. Run `python benchmark.py` -- iterate to improve speedup
6. When satisfied, your `solution.py` is the final submission

## Difficulty
Level {level} (1=basic ops, 2=fused ops, 3=architecture blocks, 4=novel/advanced)
{gate_text}
{arch_section}
{PROFILING_TOOLS}
"""


CHECK_SCRIPT = '''"""Correctness check -- run this to verify your solution matches the reference."""
import sys
import torch

def main():
    try:
        import reference
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    device = torch.device("cuda:0")
    ref_model = reference.Model(*reference.get_init_inputs()).to(device).eval()
    sol_model = solution.Model(*reference.get_init_inputs()).to(device).eval()

    try:
        sol_model.load_state_dict(ref_model.state_dict(), strict=False)
    except Exception:
        pass

    seeds = [42, 123, 456]
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]

        with torch.no_grad():
            ref_out = ref_model(*inputs)
            sol_out = sol_model(*inputs)

        if ref_out.shape != sol_out.shape:
            print(f"FAIL: shape mismatch: ref={tuple(ref_out.shape)} sol={tuple(sol_out.shape)}")
            sys.exit(1)

        if torch.isnan(sol_out).any():
            print("FAIL: solution output contains NaN")
            sys.exit(1)

        max_diff = torch.max(torch.abs(ref_out.float() - sol_out.float())).item()
        if not torch.allclose(ref_out, sol_out, atol=1e-2, rtol=1e-2):
            print(f"FAIL: max_diff={max_diff:.6f} (seed={seed})")
            sys.exit(1)

    _emit_framework_label()

    print("PASS")


def _emit_framework_label():
    """Write framework.txt with the detected kernel framework used."""
    import re
    from pathlib import Path
    patterns = [
        ("ptx",       r"asm\\s+volatile|asm\\s*\\(|\\.ptx\\b|wgmma\\.mma_async|mma\\.sync|tcgen05\\."),
        ("cutlass3",  r"\\bcute::|cutlass/gemm/collective|cutlass/gemm/kernel/sm(9|10)|cutlass::arch::Sm(9|10)"),
        ("cutlass2",  r"cutlass/gemm/device/gemm|cutlass::gemm::device|cutlass::epilogue::thread"),
        ("cuda_wmma", r"\\bnvcuda::wmma\\b|#include\\s*<mma\\.h>|wmma::fragment|wmma::mma_sync"),
        ("cutile",    r"\\bcutile::|#include\\s*<cutile"),
        ("triton",    r"import\\s+triton\\b|@triton\\.jit|triton\\.language\\b|\\btl\\.dot\\b"),
        ("mlx",       r"import\\s+mlx\\b|mlx\\.core\\b|mx\\.fast\\."),
        ("metal",     r"#include\\s*<metal_stdlib>|using\\s+namespace\\s+metal\\b|simdgroup_"),
        ("cuda_raw",  r"torch\\.utils\\.cpp_extension\\.load_inline|__global__\\s+void|<<<[^>]+>>>"),
    ]
    try:
        sol = Path("solution.py")
        if not sol.exists():
            return
        code = sol.read_text()
        label = "unknown"
        for name, pat in patterns:
            if re.search(pat, code):
                label = name
                break
        Path("framework.txt").write_text(label + "\\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
'''


BENCHMARK_SCRIPT = '''"""Benchmark -- measures speedup of solution vs adaptive reference baseline.

Graded baseline: torch.compile(mode="reduce-overhead"), falling back to eager if
compile raises or is <5% faster than eager. Eager timing is always printed as a
supplementary number so regressions against raw PyTorch are visible.
"""
import statistics
import sys
import torch

def _time_fn(fn, inputs, iters=30):
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    import reference
    import solution

    device = torch.device("cuda:0")
    ref_model = reference.Model(*reference.get_init_inputs()).to(device).eval()
    sol_model = solution.Model(*reference.get_init_inputs()).to(device).eval()

    try:
        sol_model.load_state_dict(ref_model.state_dict(), strict=False)
    except Exception:
        pass

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]

    # Warmup eager
    for _ in range(5):
        with torch.no_grad():
            ref_model(*inputs)
            sol_model(*inputs)
    torch.cuda.synchronize()

    # Eager baseline (always measured)
    eager_ms = _time_fn(ref_model, inputs)

    # Compiled baseline (adaptive): use it if it runs and is >=5% faster than eager
    compiled_ms = None
    baseline_mode = "eager"
    try:
        compiled_ref = torch.compile(ref_model, mode="reduce-overhead")
        # Warmup / capture
        for _ in range(5):
            with torch.no_grad():
                compiled_ref(*inputs)
        torch.cuda.synchronize()
        compiled_ms = _time_fn(compiled_ref, inputs)
        if compiled_ms < eager_ms * 0.95:
            baseline_mode = "compiled"
    except Exception as e:
        print(f"torch.compile fallback: {type(e).__name__}: {e}")
        compiled_ms = None

    ref_ms = compiled_ms if baseline_mode == "compiled" else eager_ms

    # Solution
    sol_ms = _time_fn(sol_model, inputs)
    speedup = ref_ms / sol_ms

    print(f"Baseline:  {baseline_mode} ({ref_ms:.3f}ms)")
    print(f"Eager:     {eager_ms:.3f}ms (median)")
    if compiled_ms is not None:
        print(f"Compiled:  {compiled_ms:.3f}ms (median)")
    print(f"Solution:  {sol_ms:.3f}ms (median)")
    print(f"Speedup:   {speedup:.2f}x (vs {baseline_mode})")

    if speedup >= 1.0:
        print("RESULT: FASTER")
    else:
        print("RESULT: SLOWER")

if __name__ == "__main__":
    main()
'''


def setup_workspace(hardware: str, problem_path: Path, out_dir: Path | None = None) -> Path:
    """Create a sandboxed workspace directory for one problem."""
    problem_name = problem_path.stem
    level = infer_level(problem_path)

    if out_dir is None:
        out_dir = ROOT / "workspaces" / hardware

    workspace = out_dir / problem_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # Copy reference
    shutil.copy2(problem_path, workspace / "reference.py")

    # Read reference code for context
    reference_code = problem_path.read_text()

    # Extract optional FRAMEWORK_GATE from the problem file
    framework_gate = None
    m = re.search(r'^FRAMEWORK_GATE\s*=\s*["\']([\w_]+)["\']', reference_code, re.MULTILINE)
    if m:
        framework_gate = m.group(1)

    # Write CLAUDE.md
    claude_md = build_claude_md(hardware, problem_name, level, reference_code, framework_gate)
    (workspace / "CLAUDE.md").write_text(claude_md)

    # Write check and benchmark scripts
    (workspace / "check.py").write_text(CHECK_SCRIPT)
    (workspace / "benchmark.py").write_text(BENCHMARK_SCRIPT)

    return workspace


def setup_all_workspaces(hardware: str, levels: list[int], out_dir: Path | None = None) -> list[Path]:
    """Set up workspaces for all problems on a hardware target."""
    from src.hardware import get_target

    target = get_target(hardware)
    problems = target.find_problems(ROOT)
    level_set = set(levels)
    workspaces = []

    for lv, problem_path in problems:
        if lv not in level_set:
            continue
        ws = setup_workspace(hardware, problem_path, out_dir)
        workspaces.append(ws)
        print(f"  L{lv} {problem_path.stem} -> {ws}")

    return workspaces


def main():
    parser = argparse.ArgumentParser(description="Set up KernelBench workspace(s)")
    parser.add_argument("hardware", choices=sorted(HARDWARE_INFO.keys()))
    parser.add_argument("problem", nargs="?", help="Path to a single problem .py file")
    parser.add_argument("--all", action="store_true", help="Set up all problems for this hardware")
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels (default: 1,2,3,4)")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: workspaces/<hardware>)")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    if args.all:
        print(f"Setting up all {args.hardware} workspaces (levels {levels})...")
        workspaces = setup_all_workspaces(args.hardware, levels, args.out_dir)
        print(f"\nCreated {len(workspaces)} workspaces")
    elif args.problem:
        problem_path = Path(args.problem)
        if not problem_path.exists():
            print(f"Error: {problem_path} not found")
            sys.exit(1)
        ws = setup_workspace(args.hardware, problem_path, args.out_dir)
        print(f"Created workspace: {ws}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
