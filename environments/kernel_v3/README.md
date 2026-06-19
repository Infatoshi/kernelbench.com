# kernel_v3

Verifiers / Prime-RL environment that **wraps the existing hardened v3 kernel-benchmark
harness** (`anvil:~/kernelbench.com/benchmarks/v3`). It does **not** reimplement eval logic:
the reward writes the model's solution to a v3 `LocalSandbox` and calls v3's `run_benchmark`
verbatim (multi-seed correctness + determinism/race check + guardrails + roofline %).

## Reward design (mechanical-primary)

- **Mechanical primary:** `reward = pct_of_peak / 100` (v3 `pct_of_peak` is in percent, 0-100)
  if the solution is **correct**, else `0.0`.
- **Correctness is a HARD GATE** (v3 multi-seed + determinism + guardrails). The PyTorch
  reference is correctness-only; speedup-vs-eager is **not** the reward.
- **Judge is an opt-in VETO, OFF by default.** When `enable_judge=True`, it only fires on a
  correct, `pct_of_peak>0` solution and can only *zero* the reward (never adds). Composed
  multiplicatively (`final = mechanical * judge_veto`). When disabled it is a no-op.
- `correct` and `pct_of_peak` are surfaced as **weight-0 metric reader funcs**.

## Solution contract (CUDA/NVIDIA)

`solution.py` must define `class Model(nn.Module)` (same interface as the reference),
plus `get_inputs()` / `get_init_inputs()`. CUDA C++ (`load_inline`) or Triton; no PyTorch
operator fallbacks. (The Metal `def solution(*inputs)` contract is a separate v3 path and is
not used here.) The model returns ONE ```python``` block; the parser takes the last one.

## load_environment kwargs

| kwarg | default | meaning |
|---|---|---|
| `levels` | `"1,2,3,4"` | which v3 problem levels (str/list/int) |
| `hardware` | `"RTX_PRO_6000"` | v3 gpu_sku passed to `run_benchmark` (anvil/sm_120) |
| `sandbox` | `"local"` | `"local"` implemented; `"modal"` is a TODO |
| `op_types` | `("gemm","attention")` | restrict to ops with a defined roofline; `None` = all ops |
| `eval_frac` | `0.2` | deterministic hash-based held-out eval split (disjoint from train) |
| `enable_judge` | `False` | opt-in judge veto |
| `judge_model` / `judge_base_url` / `judge_api_key` | | judge config (v3 `judge_solution`) |
| `max_concurrent` | `1` | cap concurrent GPU benchmarks |

## Dataset rows

`{question: v3-style task prompt (reference + template + task context), answer: reference
source, info: {problem, problem_id, level, op_type, hardware}}`.

## IMPORTANT: roofline coverage (29/42 problems score 0 when correct)

v3's `compute_tflops` only has a roofline for **`gemm` and `attention`** op_types. Every other
op (`softmax`, `layernorm`, `reduction`, `conv`, `elementwise`, `fused`, `model` -> 29 of 42
problems) returns `pct_of_peak=None`, i.e. a **correct** solution scores `0`. Therefore the
dataset **defaults to gemm+attention only** (13 problems across levels). Pass `op_types=None`
to include everything (most will give flat-0 reward). Extending `compute_tflops` to more ops
would be reimplementing v3 eval logic (out of scope per the brief) -- flagged for the user.

## RTX_PRO_6000 peak-TFLOPS patch (required)

v3's `src/config/precision_matrix.py` ships **no** peak row for `RTX_PRO_6000` (anvil's gpu_sku),
so `pct_of_peak` would be `None` for everything. This env **additively injects** a peak row at
import (`HARDWARE_PEAK_TFLOPS.setdefault("RTX_PRO_6000", ...)`) using NVIDIA RTX PRO 6000
Blackwell Workstation datasheet dense-tensor TFLOPS (`tf32=252, fp16/bf16=504, fp8=1008,
fp4=2016`). It never mutates existing B200/H100/... rows. The `fp32` entry is required because
the default level-1 matmuls run at fp32.

**Calibration note (anti-saturation, important for perf-learning).** A torch fp32 matmul on
Blackwell actually executes on **tf32 tensor cores** (the known-good Triton kernels set
`input_precision="tf32"`), measured ~122 TFLOPS. v3 keys the peak lookup by `precision="fp32"`.
If you set the fp32 peak to **125** (the fp32 CUDA-core FMA ceiling), a *stock* matmul already
reads **~97.6% of roofline** (reward 0.976) and the 122->~250 TFLOPS optimization range clips
flat into [0.97, 1.0] -- a **saturated** reward with no perf gradient. So we set the fp32-key
peak to the **tf32 tensor ceiling (252)**: a stock matmul reads **~0.48**, and a fully optimized
tf32 matmul approaches 1.0 -- real gradient. **Flip `fp32` back to `125.0`** in
`RTX_PRO_6000_PEAK_TFLOPS` if you want literal fp32-CUDA-core roofline semantics (saturated).

## How v3 is located

Set `KERNEL_V3_ROOT` (default `~/kernelbench.com/benchmarks/v3`). The env adds it to `sys.path`
and imports v3's `eval`, `agent` (LocalSandbox), `config`. **TODO(vendoring):** Hub publishing
must vendor v3's `src/eval`, `src/agent`, `src/config` and `problems/` into this dir and list
them in `[tool.hatch.build].include`. **TODO(modal):** wire `src/agent/modal_sandbox.py` for the
remote-GPU (`sandbox="modal"`) path; today only `local` is implemented.

## Validation (run on anvil, RTX PRO 6000 / sm_120)

Validated with verifiers 0.1.14 + torch 2.12.0+cu130:
- GATE 1: `load_environment(levels="1", sandbox="local")` builds; dataset = 6 gemm problems.
- GATE 2: known-good B200 `1_Square_matrix_multiplication_` solution -> `correct=True`,
  **pct_of_peak=48.79%** (tf32 ceiling), reward=0.4879 (headroom preserved).
- GATE 3: `def solution(*inputs): return inputs[0]` -> reward=0.0 (correctness gate).

Reward validated via direct rubric call (`rubric.score_rollout` on a hand-built state), not a
full `vf-eval` rollout. The judge-veto path was smoke-tested with a stub returning
`{legitimate: False}` to confirm it zeros a correct reward.

CPU-import-safe: torch / v3 eval are imported lazily inside the reward, never at module top.
