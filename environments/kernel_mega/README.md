# kernel_mega

Verifiers / Prime-RL environment that **wraps the native KernelBench-Mega agentic harness**
(`anvil:~/kernelbench.com/benchmarks/mega`). Per the Mega README it uses the **same
native-harness, transcript, and roofline machinery as KernelBench-Hard**; only the `problems/`
deck differs. So this env is the **sibling of kernel_hard** and shares `kernel_native_harness.py`
verbatim — it just binds `bench_root` to the mega checkout.

It does **not** reimplement eval: the reward replicates the native `scripts/run_hard.sh` scoring
contract — workspace (`src/` at `parents[2]` + template files + `solution.py`), native `check.py`
(correctness HARD GATE → `^PASS`), native `benchmark.py` (prints `peak_fraction:`).

## Task / deck

Full **megakernel** deck: `01_qwen3_decode_block` (Qwen3-0.6B B=1 decode-block megakernel — whole-block
fusion: GQA attention + RoPE + RMSNorm + SwiGLU MLP), `02_rl_grid_ppo`, `03_kimi_linear_decode`,
plus operation-level problems shared with Hard. Pass `deck=["01_qwen3_decode_block"]` for just the
active megakernel.

Megakernel problems are `regime: memory` (peak_fraction off the HBM-bandwidth roofline) and can
report **raw peak_fraction > 1** (the dense-FLOPS/bytes formula overcounts work the kernel
legitimately skips). The **reward clamps to `[0,1]`**; the native value is preserved as the
weight-0 metric `raw_peak_fraction`.

## Multi-turn (agentic) structure

A verifiers `StatefulToolEnv`. The model iterates the native flywheel via three tools backed by
**one persistent per-rollout workspace** (`state["_ws"]`):

| tool | effect |
|---|---|
| `write_solution(code)` | overwrite `solution.py` (full file) |
| `run_check()` | run native `check.py` (correctness gate) |
| `run_benchmark()` | run native `benchmark.py` (prints `peak_fraction`) |

`update_tool_args` injects the rollout `state` (hidden from the agent) so all tools share the
workspace; the rollout ends when the model stops calling tools or `max_turns` is hit. Reward is
computed at the end from the workspace's final `solution.py` via the same native scripts.

## Reward (mechanical-primary)

- `reward = peak_fraction` (clamped to `[0,1]`) if `check.py` **PASS**, else `0.0`.
- **Correctness HARD GATE** (native multi-seed + multi-shape + numeric-stress + forbidden-op grep);
  the PyTorch `reference.py` is correctness-only.
- **Judge = opt-in VETO, OFF by default** (`enable_judge=True`): only zeros a correct, `peak_fraction>0`
  reward; default model `z-ai/glm-5.2` via OpenRouter; multiplicative; no-op when disabled.
- Weight-0 metrics: `correct`, `peak_fraction`, `raw_peak_fraction`.

## load_environment kwargs

| kwarg | default | meaning |
|---|---|---|
| `bench_root` | `$KERNEL_MEGA_ROOT` or repo-relative `benchmarks/mega` | KernelBench-Mega checkout |
| `hardware` | `"RTX_PRO_6000"` | gpu_sku for the roofline (sm_120) |
| `deck` | `None` (all) | explicit list of problem dir names (e.g. `["01_qwen3_decode_block"]`) |
| `eval_frac` | `0.2` | deterministic hash-based held-out eval split |
| `max_turns` | `12` | max agentic turns |
| `check_timeout_s` / `bench_timeout_s` | `900` / `1200` | per-script subprocess timeouts (megakernels are slow) |
| `enable_judge` | `False` | opt-in judge veto |
| `judge_model` / `judge_base_url` / `judge_api_key` | `z-ai/glm-5.2` / OpenRouter | judge config |
| `max_concurrent` | `1` | cap concurrent GPU scoring subprocesses |

## GPU / isolation

Reward shells out to the native `check.py`/`benchmark.py` in the benchmark's **own `uv` env**
(torch 2.11+cu130, CUDA 13); this module is **CPU-import-safe**. Run the reward GPU on a device
not used for training, cap `max_concurrent`.

## Validation (anvil, RTX PRO 6000 / sm_120)

- GATE 1: `load_environment()` builds; dataset = the mega deck (problem count printed at validation).
- GATE 2: a known-good `02_rl_grid_ppo` solution from `benchmarks/mega/outputs` → **correct=True,
  peak_fraction≈0.32** via the env's reward path. (Megakernel `03_kimi_linear_decode` solutions score
  correct=True but raw_peak_fraction > 1 → reward clamps to 1.0.)
- GATE 3: multi-turn loop wired — `StatefulToolEnv` exposes `write_solution`/`run_check`/`run_benchmark`
  over one persistent workspace.

## TODO

- **Vendoring** for standalone Hub publishing: vendor `benchmarks/mega`'s `src/` + `problems/`
  into this dir. Today points at the anvil checkout via `KERNEL_MEGA_ROOT`.
