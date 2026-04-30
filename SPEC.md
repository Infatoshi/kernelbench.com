# KernelBench v3.1 Specification

Version: 3.1-draft
Date: 2026-03-18

## Problem: Why v3.0 Is Not Enough

### 1. Correctness is saturated
GPT-5.4 scores 86% on B200, 78% on H100, 77% on RTX 3090. The ceiling is near.
Models that produce a correct-but-slow kernel get the same credit as models that
deeply optimize. Correctness has become a binary gate, not a differentiator.

### 2. 139 of 1350 solutions had no real kernel
Across all v3.0 runs, 139 accepted solutions contained no custom kernel work --
just bf16 casts of the reference, trivial wrappers around PyTorch ops that slipped
past string-based guardrails, or solutions that compiled but delegated all compute
to PyTorch internals. The `_has_custom_kernel()` check in `src/eval/guardrails.py`
catches literal indicators (`triton.jit`, `load_inline`, `__global__`) but cannot
verify that a custom kernel *actually executed at runtime*.

### 3. PyTorch is the wrong baseline
The current benchmark compares solutions against PyTorch (eager or `torch.compile`).
This creates skewed difficulty:
- **Fused ops (level 2)**: PyTorch already fuses well with `torch.compile`, making
  it hard to beat. A model that writes a correct fused Triton kernel gets speedup
  < 1.0x and appears to have failed.
- **Simple ops (level 1)**: PyTorch calls cuBLAS/cuDNN, which are near-optimal. A
  model's handwritten GEMM rarely beats cuBLAS, so the benchmark punishes ambition.
- **Novel ops (level 4)**: PyTorch has no optimized path, so even naive kernels
  show large speedups. Inflated signal.

### 4. No framework-specific evaluation
v3.0 allows any approach (Triton, CUDA, CUTLASS, mixed). This hides whether a model
can actually write Triton vs. raw CUDA. A model that only knows Triton and falls back
to it for CUTLASS problems gets the same evaluation as one that uses architecture-
specific instructions (WGMMA, tcgen05).

### 5. Models that solve fast get same credit as models that deeply optimize
A model that submits on turn 2 with a naive correct kernel scores identically to one
that iterates for 12 turns and achieves 4x speedup. The metric is boolean correctness;
performance is recorded but not scored.

---

## v3.1 Design

### Two-Stage Evaluation

**Stage 1: Correctness Gate** (pass/fail)
- Multi-seed correctness check: 5 seeds (42, 123, 456, 789, 1337) -- same as v3.0
- Multi-shape correctness check: 3-5 shape configurations per problem (NEW)
- Precision-aware tolerances: same `PRECISION_TOLERANCES` dict from v3.0
- Determinism check: 2 repeat runs must produce bitwise-identical output
- NaN/Inf rejection, shape mismatch rejection -- same as v3.0
- A solution that fails Stage 1 scores 0 on that problem. No partial credit.

**Stage 2: Performance Score** (speedup multiplier)
- Only solutions that pass Stage 1 enter Stage 2
- Speedup measured against the **naive framework baseline** (NOT PyTorch)
- Timing: 5 warmup + 30 timed iterations, median of per-iteration CUDA events
- Speedup = baseline_median_ms / solution_median_ms
- Measured at every shape configuration; final speedup = geometric mean across shapes

### Framework Tracks

Each problem ships with naive baselines for each applicable framework. The model is
told which track it is solving and must use that framework.

| Track | Constraint | Baseline file | Required indicator |
|-------|-----------|---------------|--------------------|
| **Triton** | Must use `@triton.jit` | `baseline_triton.py` | `triton.jit` in source |
| **CUDA** | Must use `load_inline` | `baseline_cuda.py` | `load_inline` or `__global__` in source |
| **CUTLASS** | Must use CUTLASS headers | `baseline_cutlass.py` | `cutlass/` include in CUDA source |
| **Open** | Any framework allowed | fastest naive baseline | any custom kernel indicator |

Track enforcement:
- `src/eval/guardrails.py` gains per-track validation functions
- Triton track: `validate_triton(code)` checks for `@triton.jit`, forbids `load_inline`
- CUDA track: `validate_cuda(code)` checks for `load_inline` or `__global__`, forbids `@triton.jit`
- CUTLASS track: `validate_cutlass(code)` checks for CUTLASS header includes
- Open track: uses existing `validate_nvidia(code)` -- any custom kernel is fine

Leaderboards:
- Per-track leaderboard (Triton, CUDA, CUTLASS)
- Overall leaderboard uses Open track scores
- Models evaluated on all tracks; total eval = problems * tracks

### Naive Baselines

Each problem gets a correct-but-unoptimized implementation per framework. These are
human-authored, checked into the repo, and version-pinned.

**Directory structure change:**
```
problems/
  level1/
    1_Square_matrix_multiplication_/        # directory, not a file
      reference.py                          # PyTorch reference (unchanged from v3.0)
      baseline_triton.py                    # naive Triton: simple grid, no tiling tricks
      baseline_cuda.py                      # naive CUDA: one thread per output element
      baseline_cutlass.py                   # naive CUTLASS: default template params
      shapes.py                             # shape configurations for multi-shape testing
    23_Softmax/
      reference.py
      baseline_triton.py
      baseline_cuda.py
      shapes.py                             # CUTLASS not applicable for softmax
    ...
```

**Baseline quality requirements:**
- Must be correct (pass Stage 1 against reference.py)
- Must be naive: no shared memory tiling, no vectorized loads, no software pipelining
- Must actually execute a custom kernel (not wrap PyTorch)
- Triton baselines: simple 1D/2D grid, `tl.load`/`tl.store`, no `tl.dot` tricks
- CUDA baselines: one thread per output element, global memory only, no `__shared__`
- CUTLASS baselines: default epilogue, no swizzle, no pipeline stages

**PyTorch reference kept for:**
- Generating ground truth outputs for correctness checking
- Diagnostic timing (reported but not used for scoring)
- Weight initialization (via `load_state_dict`)

### Scoring Function

**Per-problem score (for a given track):**
```
if not correct:
    score_i = 0.0
else:
    speedup = geomean(baseline_ms_shape_j / solution_ms_shape_j for j in shapes)
    raw = log2(speedup)                    # 1x -> 0, 2x -> 1, 4x -> 2, 0.5x -> -1
    score_i = clamp(raw, -1.0, +3.0)      # floor at -1, cap at +3 (8x)
```

Rationale for bounds:
- Floor at -1: a correct solution slower than the naive baseline still gets partial
  credit (score_i + 1 = 0, mapping to 0 points). Prevents negative total scores.
- Cap at +3: diminishing returns above 8x. Prevents a single outlier problem from
  dominating the total. Also limits incentive to game a single easy problem.

**Model score (per track, per hardware):**
```
model_score = 100 * mean(score_i + 1) / 4    for all problems i
```

Scale: 0 to 100.
- A model that gets everything correct at exactly 1x speedup scores 25.
- A model that gets everything correct at 2x geomean speedup scores 50.
- A model that gets everything correct at 4x geomean speedup scores 75.
- A model that gets everything correct at 8x+ geomean speedup scores 100.
- A model that gets nothing correct scores 0.

**Side metrics (not part of the score, but reported):**
- `correct_rate`: fraction of problems passing Stage 1
- `geomean_speedup_on_correct`: geometric mean of speedup across correct solutions
- `cost_per_point`: estimated_cost_usd / model_score
- `turns_used`: mean turns across all problems
- `median_sol_ms`: median solution runtime across correct solutions

### Runtime Kernel Verification

String-based guardrails (`_has_custom_kernel()`) are demoted to a fast pre-check.
The primary verification is runtime profiling.

**Implementation in `src/eval/benchmark.py`:**

After Stage 1 passes and before Stage 2 timing:

```python
# Already exists in v3.0 benchmark template:
def count_kernels(model, model_inputs):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            model(*model_inputs)
    torch.cuda.synchronize()
    return sum(1 for e in prof.key_averages()
               if e.device_type == torch.profiler.DeviceType.CUDA)

# NEW: kernel name inspection
def has_custom_kernel_runtime(model, model_inputs):
    """Check that at least one non-PyTorch kernel executed."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            model(*model_inputs)
    torch.cuda.synchronize()

    PYTORCH_KERNEL_PREFIXES = [
        "aten::", "at::native::", "void at::", "cutlass_",  # PyTorch's internal CUTLASS
        "void (anonymous namespace)::",
        "ampere_", "volta_", "sm80_", "sm90_",             # cuBLAS/cuDNN internal
        "cudnn::", "cublasLt", "cublas",
        "void cunn_", "void thrust::",
    ]

    for event in prof.key_averages():
        if event.device_type != torch.profiler.DeviceType.CUDA:
            continue
        name = event.key
        is_pytorch_internal = any(name.startswith(p) for p in PYTORCH_KERNEL_PREFIXES)
        if not is_pytorch_internal:
            return True  # Found a custom kernel

    return False
```

If `has_custom_kernel_runtime()` returns False, the solution is rejected:
```json
{"compiled": true, "correct": false, "speedup": null,
 "error": "No custom kernel detected at runtime (solution delegates to PyTorch internals)"}
```

String-based check remains as a fast-fail before compilation (avoids wasting GPU time).

### Multi-Shape Testing

Each problem includes a `shapes.py` file defining 3-5 shape configurations.

**`shapes.py` format:**
```python
# problems/level1/1_Square_matrix_multiplication_/shapes.py
SHAPES = [
    {"N": 256},       # small: latency-bound
    {"N": 1024},      # medium
    {"N": 2048},      # large: default from v3.0
    {"N": 4096},      # xlarge: throughput-bound
    {"N": 8192},      # xxlarge: memory-pressure test
]
```

**How shapes integrate with `get_inputs()`:**

The reference.py `get_inputs()` signature is extended with an optional `shape_config` parameter:
```python
def get_inputs(shape_config=None):
    cfg = shape_config or {}
    n = cfg.get("N", 2048)  # default matches v3.0 behavior
    A = torch.randn(n, n)
    B = torch.randn(n, n)
    return [A, B]
```

Solutions must also accept `shape_config` in `get_inputs()`. The benchmark template
iterates over all shapes for both correctness and performance.

**Benchmark template changes:**

```python
import shapes  # loaded from shapes.py in problem directory

# Stage 1: correctness across seeds AND shapes
for shape_config in shapes.SHAPES:
    for seed in CORRECTNESS_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                  for x in get_inputs(shape_config)]
        # ... same correctness checks as v3.0 ...

# Stage 2: performance per shape
shape_speedups = []
for shape_config in shapes.SHAPES:
    bench_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in get_inputs(shape_config)]
    baseline_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                       for x in baseline_get_inputs(shape_config)]
    baseline_ms = summarize_runtime_ms(baseline_model, baseline_inputs)["median"]
    sol_ms = summarize_runtime_ms(sol_model, bench_inputs)["median"]
    shape_speedups.append(baseline_ms / sol_ms)

# Final speedup = geometric mean across shapes
import math
geomean_speedup = math.exp(sum(math.log(s) for s in shape_speedups) / len(shape_speedups))
```

### Problem Taxonomy

Problems are categorized by compute characteristic. This taxonomy determines which
shapes are relevant and which baselines are applicable.

| Category | Problems | Key characteristic | Baseline difficulty |
|----------|----------|--------------------|---------------------|
| **Compute-bound** | GEMM variants, convolution, attention | Arithmetic intensity > memory bandwidth | Hardest to beat (cuBLAS is good) |
| **Memory-bound** | Elementwise, reductions, softmax, norms | Low arithmetic intensity, bandwidth-limited | Easier to match (memory access is the bottleneck) |
| **Fused** | Matmul+activation, conv+BN+relu, multi-op pipelines | Multiple ops that benefit from kernel fusion | Naive baseline runs each op separately |
| **Architecture** | Transformer blocks, MoE, MLA | Complex control flow + multiple kernels | Opportunity for algorithmic optimization |
| **Quantized** | FP8, FP4, INT4, block-scaled GEMM | Low-precision tensor core utilization | Framework-specific (CUTLASS vs Triton FP8) |

Each problem's `reference.py` already declares `OP_TYPE`. v3.1 adds a `CATEGORY` field:
```python
OP_TYPE = "gemm"
CATEGORY = "compute_bound"  # one of: compute_bound, memory_bound, fused, architecture, quantized
```

Leaderboard breakdowns show per-category scores so models that excel at memory-bound
ops vs. compute-bound ops are distinguished.

### What Stays From v3.0

**Unchanged:**
- Multi-turn agent loop (10-15 turns, hardware-level dependent)
- Sandbox execution: local (`src/agent/local_sandbox.py`) for RTX 3090, Modal
  (`src/agent/modal_sandbox.py`) for H100/B200
- Per-architecture system prompts in `src/prompts.py` (WMMA, WGMMA, tcgen05 guidance)
- Guardrails: forbidden PyTorch fallbacks, forbidden external libraries (`flash_attn`, etc.)
- Blocked commands in `src/tools.py` (pkill, rm -rf /, benchmark tampering)
- Hardware targets: RTX 3090, H100, B200 (M4 Max via MetalBench, separate repo)
- Tool schema: read_file, write_file, edit_file, bash, submit
- Provider routing: OpenRouter for most models, OpenAI direct for GPT-5.x
- EvalResult dataclass with token counts, cost estimation, fingerprinting
- Precision tolerance matrix (`src/config/precision_matrix.py`)
- Weight sharing via `load_state_dict(strict=False)`

**Modified:**
- `src/eval/benchmark.py`: new benchmark template with multi-shape, baseline loading,
  runtime kernel verification, and per-shape timing
- `src/eval/guardrails.py`: per-track validation functions, runtime kernel check
- `src/eval/results.py`: new fields for per-shape speedups, track, geomean_speedup,
  performance_score
- `src/prompts.py`: track-aware system prompts (tell model which framework to use,
  provide naive baseline code)
- `src/eval/context.py`: load shapes.py and baseline code into workspace context
- `bench.py`: new CLI flags for `--track triton|cuda|cutlass|open`
- Problem directory structure: files become directories with baselines + shapes

---

## Implementation Plan

### Phase 1: Problem restructuring (no code changes)

Convert each problem file to a directory:
```bash
# For each problem file:
# problems/level1/1_Square_matrix_multiplication_.py
#   -> problems/level1/1_Square_matrix_multiplication_/reference.py
#   -> problems/level1/1_Square_matrix_multiplication_/shapes.py
```

Script: `scripts/restructure_problems.py`
- For each `.py` in `problems/level*/`:
  1. Create directory (strip `.py` suffix)
  2. Move file to `directory/reference.py`
  3. Generate `shapes.py` from `get_inputs()` parameter analysis
- Preserve git history with `git mv`

Deliverable: all 43 RTX 3090 problems restructured, shapes.py for each.

### Phase 2: Naive baseline authoring

For each problem, write naive baselines. Prioritize by frequency:
- Triton baselines for all problems (Triton is most commonly used by models)
- CUDA baselines for all problems
- CUTLASS baselines for compute-bound problems only (GEMM, conv, attention)

Script: `scripts/validate_baselines.py`
- Load each baseline, run against reference with 5 seeds
- Verify it's actually naive (no shared memory in CUDA, no `tl.dot` in Triton for non-GEMM)
- Print timing vs. PyTorch reference

Estimated effort: ~60 baselines for RTX 3090 (43 problems * ~1.5 tracks average).

### Phase 3: Benchmark template rewrite

File: `src/eval/benchmark.py`

New template `CUDA_BENCHMARK_TEMPLATE_V31`:
1. Load reference.py, solution.py, shapes.py, and baseline_{track}.py
2. Stage 1: correctness gate (multi-seed * multi-shape)
3. Runtime kernel verification via `torch.profiler`
4. Stage 2: per-shape timing against baseline, compute geomean speedup
5. Compute performance score: `clamp(log2(geomean_speedup), -1, +3)`
6. Output JSON with all per-shape details

The v3.0 template remains available as `CUDA_BENCHMARK_TEMPLATE_V30` for backward
compatibility and re-running old evaluations.

### Phase 4: Guardrails and track enforcement

File: `src/eval/guardrails.py`

New functions:
```python
def validate_triton_track(code: str) -> Optional[str]:
    """Triton track: must have @triton.jit, must NOT have load_inline."""
    if "@triton.jit" not in code and "triton.jit" not in code:
        return "Triton track requires @triton.jit kernel"
    if "load_inline" in code:
        return "Triton track forbids CUDA via load_inline"
    return validate_nvidia(code)  # also run base checks

def validate_cuda_track(code: str) -> Optional[str]:
    """CUDA track: must have load_inline or __global__, must NOT have @triton.jit."""
    if "load_inline" not in code and "__global__" not in code:
        return "CUDA track requires load_inline or __global__ kernel"
    if "@triton.jit" in code:
        return "CUDA track forbids Triton"
    return validate_nvidia(code)

def validate_cutlass_track(code: str) -> Optional[str]:
    """CUTLASS track: must include CUTLASS headers."""
    if "cutlass/" not in code:
        return "CUTLASS track requires CUTLASS header includes"
    return validate_cuda_track(code)  # CUTLASS is a subset of CUDA track

def validate_solution_v31(code: str, track: str, is_metal: bool = False) -> Optional[str]:
    if is_metal:
        return validate_metal(code)
    validators = {
        "triton": validate_triton_track,
        "cuda": validate_cuda_track,
        "cutlass": validate_cutlass_track,
        "open": validate_nvidia,
    }
    validator = validators.get(track, validate_nvidia)
    return validator(code)
```

### Phase 5: EvalResult and scoring

File: `src/eval/results.py`

New fields on `EvalResult`:
```python
@dataclass
class EvalResult:
    # ... existing fields ...
    track: str = "open"                                # triton, cuda, cutlass, open
    shape_speedups: List[float] = field(default_factory=list)  # per-shape speedups
    geomean_speedup: Optional[float] = None            # geometric mean across shapes
    performance_score: Optional[float] = None           # clamp(log2(geomean_speedup), -1, 3)
    baseline_ms_per_shape: List[float] = field(default_factory=list)
    solution_ms_per_shape: List[float] = field(default_factory=list)
    shapes_tested: int = 0
    custom_kernel_verified: bool = False                # runtime profiler confirmed
```

New scoring function in `src/eval/scoring.py`:
```python
import math
from typing import List, Optional

def compute_performance_score(speedup: float) -> float:
    """Per-problem performance score from speedup multiplier."""
    raw = math.log2(speedup)
    return max(-1.0, min(3.0, raw))

def compute_model_score(scores: List[Optional[float]], n_problems: int) -> float:
    """Aggregate model score on 0-100 scale.

    scores: list of per-problem performance scores (None if incorrect).
    n_problems: total problems in the benchmark.
    """
    adjusted = []
    for s in scores:
        if s is None:
            adjusted.append(0.0)   # incorrect -> 0 contribution
        else:
            adjusted.append(s + 1.0)  # shift so 1x speedup = 1.0, floor 0.0

    # Pad with zeros for unattempted problems
    while len(adjusted) < n_problems:
        adjusted.append(0.0)

    return 100.0 * sum(adjusted) / (4.0 * n_problems)

def compute_geomean_speedup(speedups: List[float]) -> float:
    """Geometric mean of speedups (only for correct solutions)."""
    if not speedups:
        return 0.0
    log_sum = sum(math.log(s) for s in speedups if s > 0)
    return math.exp(log_sum / len(speedups))
```

### Phase 6: Prompt updates

File: `src/prompts.py`

The system prompt gains track-specific instructions:
```python
TRACK_INSTRUCTIONS = {
    "triton": """
TRACK: Triton
You MUST write your kernel using @triton.jit. Do NOT use CUDA C++ (load_inline).
A naive Triton baseline is provided in /workspace/baseline_triton.py. Your goal
is to beat its performance while maintaining correctness.
""",
    "cuda": """
TRACK: Raw CUDA
You MUST write your kernel using CUDA C++ via torch.utils.cpp_extension.load_inline.
Do NOT use Triton. A naive CUDA baseline is provided in /workspace/baseline_cuda.py.
Your goal is to beat its performance while maintaining correctness.
""",
    "cutlass": """
TRACK: CUTLASS
You MUST write your kernel using CUTLASS templates via load_inline with headers
at /opt/cutlass/include. A naive CUTLASS baseline is provided in
/workspace/baseline_cutlass.py. Your goal is to beat its performance.
""",
    "open": """
TRACK: Open (any framework)
Use whichever framework you prefer: Triton, CUDA C++, CUTLASS, or any combination.
Your solution will be benchmarked against the fastest naive baseline across all
frameworks. Focus on maximum performance.
""",
}
```

The naive baseline code is injected into the workspace alongside reference.py, so
the model can read it and understand what it needs to beat.

### Phase 7: CLI and batch runner

File: `bench.py`

New flags:
```
uv run python bench.py run rtx3090 --models gpt-5.4 --track triton --levels 1,2,3,4
uv run python bench.py run rtx3090 --models gpt-5.4 --track all    # runs all 4 tracks
uv run python bench.py run rtx3090 --models gpt-5.4                # defaults to open track
```

Summary output includes performance score:
```
uv run python bench.py summary outputs/batch_eval/run_XXXXXXXX
# Model Score: 62.3 / 100
# Correct: 38/43 (88%)
# Geomean Speedup (correct): 3.2x
# Track: triton
# Cost: $4.28 ($0.069/point)
```

---

## Output Schema (v3.1 benchmark JSON)

```json
{
  "version": "3.1",
  "compiled": true,
  "correct": true,
  "custom_kernel_verified": true,
  "track": "triton",

  "shapes_tested": 5,
  "shape_results": [
    {"shape": {"N": 256},  "baseline_ms": 0.042, "sol_ms": 0.038, "speedup": 1.11},
    {"shape": {"N": 1024}, "baseline_ms": 0.31,  "sol_ms": 0.14,  "speedup": 2.21},
    {"shape": {"N": 2048}, "baseline_ms": 1.82,  "sol_ms": 0.61,  "speedup": 2.98},
    {"shape": {"N": 4096}, "baseline_ms": 13.4,  "sol_ms": 3.9,   "speedup": 3.44},
    {"shape": {"N": 8192}, "baseline_ms": 104.2, "sol_ms": 28.7,  "speedup": 3.63}
  ],

  "geomean_speedup": 2.54,
  "performance_score": 1.34,

  "ref_ms": 1.12,
  "ref_ms_note": "PyTorch reference (diagnostic only, not used for scoring)",

  "correctness_seeds": [42, 123, 456, 789, 1337],
  "precision_used": "fp32",
  "tolerance_atol": 0.001,
  "tolerance_rtol": 0.001,
  "has_nan": false,
  "has_inf": false,
  "is_deterministic": true,

  "baseline_type": "naive_triton",
  "op_type": "gemm",
  "category": "compute_bound",
  "achieved_tflops": 45.2,
  "pct_of_peak": 63.7,

  "sol_kernels": 1,
  "baseline_kernels": 1,
  "ref_kernels": 3
}
```

---

## Migration Path

### Backward compatibility
- v3.0 results remain valid and comparable to each other
- v3.1 results are NOT comparable to v3.0 results (different baseline, different scoring)
- The v3.0 benchmark template is preserved as `CUDA_BENCHMARK_TEMPLATE_V30`
- `bench.py run` defaults to v3.1; `bench.py run --legacy` uses v3.0

### Re-running v3.0 top models on v3.1
Priority models for v3.1 baseline establishment:
1. GPT-5.4 (current leader)
2. Gemini 3 Flash (strong and cheap)
3. GPT-5.3 (solid mid-tier)
4. Claude Opus 4.6 (comparison point)

Run all 4 on RTX 3090 across all tracks first (4 models * 4 tracks * 43 problems = 688 evals).
Then expand to H100 and B200.

### Problem compatibility
- Existing `reference.py` files are unchanged
- `get_inputs()` gains optional `shape_config` parameter (backward-compatible default)
- Old solutions that don't accept `shape_config` are tested only at the default shape

---

## Open Questions

1. **CUTLASS track applicability**: Should CUTLASS track exist for memory-bound ops
   (softmax, layernorm) where CUTLASS provides no benefit? Current plan: CUTLASS
   track only for compute-bound + quantized categories.

2. **Score cap at 3 (8x)**: Is 8x the right cap? For memory-bound ops, 8x over a
   naive baseline is achievable with basic optimizations. For GEMM, 8x over naive is
   very hard. Consider per-category caps.

3. **Baseline versioning**: Naive baselines are human-authored and could contain bugs.
   Need a validation pipeline that runs baselines against references nightly. Consider
   `scripts/validate_baselines.py` as a CI check.

4. **Metal/M4 Max**: This spec covers CUDA targets only. Metal track (MLX) continues
   in separate MetalBench repo. Unify later if framework tracks prove useful.

5. **Cost normalization**: Should cost_per_point use input+output token cost, or
   include infrastructure cost (Modal GPU-seconds)?
