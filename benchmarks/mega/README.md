# KernelBench-Mega

Full fused **megakernel** bench (same harness / archive / roofline machinery
as Hard). Agents write one whole-block kernel, not a per-op microbench.
Unlimited wall-clock. Live board: [/mega](https://kernelbench.com/mega).

| NN | problem | note |
| -- | ------- | ---- |
| 02 | `02_kimi_linear_decode` | Kimi-Linear W4A16 hybrid decode megakernel |

Problem 01 (RL grid PPO) was removed 2026-07-21; skill lives on the CUDA
bench now — do not re-add.

```bash
cd benchmarks/mega && ./scripts/run_hard.sh claude claude-opus-4-7 problems/02_kimi_linear_decode
```

See `SPEC.md`, `DEVLOG.md`, and repo-root `AGENTS.md`.
