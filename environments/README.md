# KernelBench RL environments

Verifiers/Prime-compatible reinforcement-learning environments for **GPU-kernel optimization**, built
on this repo's own benchmarks (`benchmarks/v3`, `benchmarks/hard`, `benchmarks/mega`). Each env is a
verifiable kernel-writing task with a **mechanical reward** — compile, check numerical correctness
against a PyTorch reference, measure performance — so they plug into `prime-rl` / `verifiers` / `veRL`
and someone can RL-train a model against them on a GPU cluster, unchanged.

## Reward design (shared across envs)
- **Correctness is a hard gate**; the reward signal is **% of roofline** (achieved vs hardware peak),
  which is hardware-absolute and arch-portable (sm_120 → B200 unchanged). The PyTorch reference is
  used for correctness only, never as the speed denominator.
- **Mechanical-primary**: the gradient comes from the mechanical compile/correctness/roofline path.
- **Judge is an opt-in veto, off by default**: when enabled, an LLM judge (default `z-ai/glm-5.2` via
  OpenRouter) only fires on already-correct solutions and can only zero the reward (reward-hack
  veto) — never adds reward. Keeps the live RL signal mechanical (no judge in the gradient).
- **No eval contamination**: agentic envs run in an isolated sandbox so the agent can't read prior
  winning solutions (the cross-run contamination hole in the old harness; see repo CLAUDE.md).

## Environments
| dir | task | turns | status |
|---|---|---|---|
| `kernel_v3/` | 42 curated/hardened single-op problems (levels 1-4), `class Model` contract, v3's `run_benchmark` (multi-seed correctness + determinism + guardrails + % of peak) | single-turn | **validated** on RTX PRO 6000 (matmul 48% of peak; wrong→0; judge-veto works) |
| `kernel_hard/` | ~10 operation-level hard CUDA kernels, native agentic harness (write/check/benchmark flywheel) + roofline | multi-turn | **built** — validated on RTX PRO 6000 (`03_paged_attention` known-good → correct, peak_fraction 0.67) |
| `kernel_mega/` | full megakernel deck (Qwen3 decode block, Kimi-Linear decode, …), same native harness as Hard; owns the decode + RL-sim + megakernel deck. Prime-sandbox (contamination-fix) execution = PRIME_SANDBOX_TODO.md | multi-turn | **built** — validated on RTX PRO 6000 (`02_rl_grid_ppo` known-good → correct, peak_fraction 0.32) |

## Run an env (eval or as an RL environment)
Each env is a standard verifiers environment package (`load_environment(**kwargs)` + `pyproject.toml`).

```bash
# install into a verifiers/prime workspace
prime env install <env> -p ./environments        # or: vf-install <env>

# quick eval against an API/served model
prime eval run kernel_v3 --env-dir-path ./environments -m <model> -n 5 -r 4 --save-results
```

For **RL on a cluster** (prime-rl, 3 disaggregated processes — vLLM inference + orchestrator + FSDP2
trainer), reference the env in the orchestrator TOML; `args` are passed straight to `load_environment`:
```toml
[[orchestrator.train.env]]
id   = "kernel-v3"
name = "kernel_v3"
args = { levels = "1,2,3,4", hardware = "RTX_PRO_6000", enable_judge = false }
```
Split GPUs in `[deployment]` (e.g. `num_train_gpus`/`num_infer_gpus`); run the kernel **reward on a
GPU that is NOT a training GPU** (the spec's rule — timing under vLLM/FSDP contention is meaningless).
Per-env kwargs (hardware, sandbox, judge, concurrency) are documented in each env's README.

## GPU / contract notes
- `kernel_v3` reward needs CUDA + the repo's `benchmarks/v3` (imported via a repo-relative path; set
  `KERNEL_V3_ROOT` to override). Hub-publishing an env standalone must vendor the benchmark eval into
  the env dir — see each env's TODO.
- Roofline today covers gemm+attention (compute roofline); memory-bound ops use a peak-bandwidth
  roofline. Calibration + per-arch peak tables live with the eval.

## Provenance
Built on KernelBench-Hard/Mega/v3 (this repo). See `docs/env-interface-spec.md` (in the env scaffold)
for the pinned verifiers/prime-rl API the envs target.
