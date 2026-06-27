# kernel_hard

Verifiers / Prime-RL environment that **wraps the native KernelBench-Hard agentic harness**
(`anvil:~/kernelbench.com/benchmarks/hard`). It does **not** reimplement eval: the reward
replicates the exact scoring contract of the native `scripts/run_hard.sh` — lay out a
workspace (`src/` at `parents[2]`, the immutable problem template files, the model's
`solution.py`), run the problem's own `check.py` (correctness HARD GATE → `^PASS`) and
`benchmark.py` (prints `peak_fraction:` via `src/eval/roofline.peak_fraction`).

Sibling of **kernel_mega**: identical native-harness machinery (`src/eval`, `check.py`,
`benchmark.py`), only the problem deck differs. Both wrap `kernel_native_harness.py`.

## Task / deck

6 operation-level hard CUDA kernels: `01_fp8_gemm`, `02_kda_cutlass`, `03_paged_attention`,
`05_topk_bitonic`, `06_sonic_moe_swiglu`, `07_w4a16_gemm` (link-don't-spoil briefs;
forbidden-op lists enforced by the native `check.py`).

## Multi-turn (agentic) structure

A verifiers `StatefulToolEnv`. The model iterates the native flywheel via three tools backed
by **one persistent per-rollout workspace** (`state["_ws"]`):

| tool | effect |
|---|---|
| `write_solution(code)` | overwrite `solution.py` (full file) |
| `run_check()` | run native `check.py` (correctness gate; PASS/FAIL over all shapes) |
| `run_benchmark()` | run native `benchmark.py` (prints `peak_fraction`) |

`StatefulToolEnv.update_tool_args` injects the rollout `state` (hidden from the agent schema)
so all tools share the workspace. The rollout ends when the model stops calling tools
(`no_tools_called` stop condition) or `max_turns` is hit. The reward is computed at the end
from that workspace's final `solution.py` — the same native `check.py`/`benchmark.py`, never an
opaque black-box runner. (If the model never used the tools, the reward falls back to scoring
the last fenced code block in a fresh workspace.)

## Reward (mechanical-primary)

- `reward = peak_fraction` (clamped to `[0,1]`) if `check.py` **PASS**, else `0.0`.
- **Correctness is a HARD GATE** (native multi-seed + multi-shape + numeric-stress + forbidden-op
  grep). The PyTorch `reference.py` is correctness-only; speedup-vs-eager is **not** the reward.
- **Judge is an opt-in VETO, OFF by default.** `enable_judge=True` only fires on a correct,
  `peak_fraction>0` solution and can only *zero* the reward (multiplicative; default model
  `z-ai/glm-5.2` via OpenRouter). No-op when disabled.
- Metrics surfaced as weight-0 reader funcs: `correct`, `peak_fraction`, `raw_peak_fraction`.

## load_environment kwargs

| kwarg | default | meaning |
|---|---|---|
| `bench_root` | `$KERNEL_HARD_ROOT` or repo-relative `benchmarks/hard` | KernelBench-Hard checkout |
| `hardware` | `"RTX_PRO_6000"` | gpu_sku for the roofline (sm_120) |
| `deck` | `None` (all) | explicit list of problem dir names |
| `eval_frac` | `0.2` | deterministic hash-based held-out eval split (disjoint from train) |
| `max_turns` | `12` | max agentic turns |
| `check_timeout_s` / `bench_timeout_s` | `600` / `900` | per-script subprocess timeouts |
| `enable_judge` | `False` | opt-in judge veto |
| `judge_model` / `judge_base_url` / `judge_api_key` | `z-ai/glm-5.2` / OpenRouter | judge config |
| `max_concurrent` | `1` | cap concurrent GPU scoring subprocesses |

## GPU / isolation

The reward shells out to the native `check.py`/`benchmark.py` via the benchmark's **own `uv`
env** (torch 2.11+cu130, CUDA 13), so this env module is **CPU-import-safe** (no torch import).
On a cluster, run the reward GPU on a device **not** used by vLLM/FSDP (timing under training
contention is meaningless) — point `CUDA_VISIBLE_DEVICES` at a held-out GPU for the orchestrator
process, and cap `max_concurrent`.

## Dataset rows

`{question: native PROMPT.txt + tool contract + reference.py, answer: problem_id, info:
{problem, hardware, bench_root}}`.

## Validation (anvil, RTX PRO 6000 / sm_120)

- GATE 1: `load_environment()` builds; dataset = the hard deck (problem count printed at validation).
- GATE 2: a known-good `03_paged_attention` solution from `benchmarks/hard/outputs` → **correct=True,
  peak_fraction≈0.67** via the env's reward path (native check.py PASS + benchmark.py peak_fraction).
- GATE 3: the multi-turn loop is wired — `StatefulToolEnv` exposes `write_solution` / `run_check` /
  `run_benchmark` over one persistent workspace; reward reads the workspace's final solution.py.

## TODO

- **Vendoring** for standalone Hub publishing: vendor `benchmarks/hard`'s `src/` + `problems/`
  (+ `pyproject.toml`/`uv.lock`) into this dir and add to `[tool.hatch.build].include`. Today the
  env points at the anvil checkout via `KERNEL_HARD_ROOT`.
- Richer tool set (clone-repo / read-file / profile) is possible; the current minimal faithful loop
  is write → check → benchmark.
