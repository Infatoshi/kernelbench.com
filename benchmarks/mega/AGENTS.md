# KernelBench-Mega

Full fused **megakernel** bench (same harness, archive, and roofline machinery as Hard). Agents write one whole-block kernel, not a per-op microbench. Unlimited wall-clock. Live board: `/mega` (three GPUs: RTX PRO 6000, H100, B200; the June H100 and B200 rows ran under a 3-hour cap, everything since is unlimited).

| NN | problem | note |
| -- | ------- | ---- |
| 02 | `02_kimi_linear_decode` | Kimi-Linear W4A16 hybrid decode megakernel; score is speedup vs `baseline.py` (optimized PyTorch) |

Problem 01 (RL grid PPO) was removed 2026-07-21; the skill lives on the CUDA bench now. Do not re-add.

```bash
cd benchmarks/mega && ./scripts/run_hard.sh claude claude-opus-4-7 problems/02_kimi_linear_decode
./scripts/sweep.sh      # the matrix; mega is not driven by kb/kbh
```

## Mega deltas from the shared single-GPU runner

- `scripts/run_hard.sh` is a deliberate fork of `scripts/lib/run_harness.sh` (recorded in `DEVLOG.md` 2026-07-31): it carries the `bwrap` filesystem-hiding sandbox (`KBH_SANDBOX=1` default; `0` exposes the host view) and has no container path (`KBH_AGENT_CONTAINER` does nothing here). Fold it back only when the shared lib grows `KBH_SANDBOX` and gpu-lock-exec.
- Budget: the direct runner reads unprefixed `BUDGET_SECONDS`; the sweep launchers read `KBH_BUDGET_SECONDS` and export it per run.
- Lock file is `outputs/gpu.lock` (not a lock dir). Harness alias collision: `inkling` means direct Tinker on Hard but OpenRouter on Mega (`kbtool/AGENTS.md` harness table).
- `kb publish mega` writes `public/data/mega/results.csv` (not `leaderboard.json`); kernels publish as `/data/mega/code/{run_id}.solution.py.txt`. `build_mega_leaderboard.py` requires a per-run `gpu` marker.
- Megakernel authenticity: a scored path must be one genuinely fused launch. The post-run judge gate plus advisory tripwires (`megakernel_judged` in `results.csv`) are specified in `SPEC.md`; a substring ban on `torch.compile` / `CUDAGraph` was tried and punished honest disclaimers, so do not reintroduce it.
- Audit YAML: same schema as Hard (`benchmarks/hard/AGENTS.md`), `contamination` verdict included. Regrades with `scripts/regrade_sequential.sh`; `KBH_REGRADE_DECK` swaps in the canonical deck.

Layout, adding a problem, correctness, results, torch policy, and the `KBH_` variables: `benchmarks/hard/AGENTS.md`. Harness routes and rented workers: `kbtool/AGENTS.md`. Methodology: `SPEC.md`. History: `DEVLOG.md`. Publish gates: root `AGENTS.md`.
