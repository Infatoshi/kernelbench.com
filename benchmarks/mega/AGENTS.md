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
- Sandbox v2 is built by the shared helper `scripts/lib/sandbox.sh` (`kbh_sandbox_init`); do not re-fork the hide list in `run_hard.sh`. It tmpfs's the whole monorepo and binds back only `$RUN_DIR`, the lock dir, the grading interpreter's prefix, and an `agent_home` overlay for session history. Env vars it reads or sets: `KBH_SANDBOX` (1 default, 0 disables), `KBH_SANDBOX_ACTIVE` (set to 1 once the canary passes), `KBH_SANDBOX_BEACON` (honeytoken beacon string in `sandbox.json`), `KBH_SANDBOX_LIB` (helper path override), `KBH_SANDBOX_HIDE` and `$REPO_ROOT/.kbh_sandbox_hide` (extra paths to hide). The launch canary runs inside the sandbox before the model starts and exits 4 if any foreign archive is visible.
- Graded-surface stamp (all benches): every published cell records `graded_surface_sha` in `result.json`, the digest of the deck problem files plus `src/` that produced its number (`scripts/lib/graded_surface.py`, shared by `run_harness.sh`, `regrade_sequential.sh`, and publish gate E in `scripts/check_publish_gates.py`). Gate E refuses any cell whose stamp is not the current deck: re-grade it or withdraw it, never edit generated files. `KBH_GATE_GRADED_SURFACE` (set `0`) disables the gate and is legal ONLY while a backfill is in flight. The backfill is `KBH_REGRADE_CHECK_ONLY` (set `1`; all four `regrade_sequential.sh` copies): `check.py` replays and the stamp is written, `benchmark.py` is skipped, the published `peak_fraction` and `benchmark.log` are kept (voided if the check now fails), provenance lands in `recheck`. Timing never moves to other hardware or thermal state. `scripts/check_publish_gates.py --list-stale <bench>` prints the run dirs to feed it.
- Lock file is `outputs/gpu.lock` (not a lock dir). Harness alias collision: `inkling` means direct Tinker on Hard but OpenRouter on Mega (`kbtool/AGENTS.md` harness table).
- `kb publish mega` writes `public/data/mega/results.csv` (not `leaderboard.json`); kernels publish as `/data/mega/code/{run_id}.solution.py.txt`. `build_mega_leaderboard.py` requires a per-run `gpu` marker file (`outputs/runs/<run_id>/gpu` holding the GPU label) and silently drops rows without it; `kb publish` fails via `scripts/check_publish_gates.py` when a clean annotation has no marker.
- Megakernel authenticity: a scored path must be one genuinely fused launch. The post-run judge gate plus advisory tripwires (`megakernel_judged` in `results.csv`) are specified in `SPEC.md`; a substring ban on `torch.compile` / `CUDAGraph` was tried and punished honest disclaimers, so do not reintroduce it.
- Audit YAML: same schema as Hard (`benchmarks/hard/AGENTS.md`), `contamination` verdict included. Regrades with `scripts/regrade_sequential.sh`; `KBH_REGRADE_DECK` swaps in the canonical deck.

Codex pin: `scripts/cloud_bootstrap.sh` (and `scripts/brev_worker.sh`, `scripts/lambda_worker.sh` at the repo root) install `@openai/codex@0.140.0`; override with `KB_CODEX_VERSION` in cloud_bootstrap only. Codex 0.141 through 0.153.4 sends no inline `tools` array to third-party Responses providers (OpenRouter): its exec tool became a server-side namespaced tool, so GPT-6 through OpenRouter emits `exec` calls with `{}` arguments and every call dies with "tool exec invoked with incompatible payload" (upstream openai/codex #37380 and #33405 still open). 0.140.0 sends the 10 inline function tools. Also set `model_supports_reasoning_summaries = true` in `~/.codex/config.toml` or the `reasoning` field, and with it the effort, is dropped for unrecognized model ids. The 0.141-0.153.4 changelogs carry no `codex exec` quality change (reconnect through provider outages, exec-server disconnect survival, tool-heavy latency, Astra catalog entries), so there is no score reason to move. Raise the pin only after a local logging proxy set as `model_providers.openrouter.base_url` shows `tools` in the request body. Native `api.openai.com` transport is unaffected but needs `codex login --with-api-key` and org API credits.

Layout, adding a problem, correctness, results, torch policy, and the `KBH_` variables: `benchmarks/hard/AGENTS.md`. Harness routes and rented workers: `kbtool/AGENTS.md`. Methodology: `SPEC.md`. History: `DEVLOG.md`. Publish gates: root `AGENTS.md`.
