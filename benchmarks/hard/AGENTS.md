# KernelBench-Hard — Developer Instructions (codex / droid)

This is the codex / droid / cursor-agent equivalent of `CLAUDE.md`. Content is identical; format is plain markdown for any CLI.

See [`CLAUDE.md`](./CLAUDE.md) for the canonical version. All rules there apply.

Summary of the non-negotiables:

- **uv only.** `uv run ...`, `uv add ...`, `uv pip install ...`. Never `pip` or bare `python`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Do not edit `problems/*/solution.py`** — those are agent output.
- **Do not modify `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** of an already-published problem; `scripts/run_hard.sh` invalidates and restores runs that change them.
- **Apply the torch 2.11 inductor CSE hotfix** via `./scripts/patch_torch.sh` after any `uv sync`.
- **Z.ai GLM-5.1 Claude Code reruns:** use `zai-claude`; `scripts/run_hard.sh` sets `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `CLAUDE_CODE_MAX_RETRIES=1000000`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, and routes all Claude Code aliases, including Haiku / Explore / subagents, to `glm-5.1`.
- **GPU work must go through `scripts/run_hard.sh`.** It creates archive-local workspaces, isolated CUDA/Triton/Torch caches, and a shared GPU lock so concurrent agent sweeps can edit in parallel while compile/check/benchmark work queues cleanly.

## Quick actions

```bash
uv sync
./scripts/patch_torch.sh
./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm
```

## Repo layout and adding a new problem

See `CLAUDE.md` — everything there is authoritative.

## Current sweep context

Use `scripts/run_hard.sh` for all model evals. It stages each problem under
`outputs/runs/<run_id>/repo/problems/<problem_name>/`, copies immutable problem
files, symlinks `src/`, copies project metadata, and sets per-run
`TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, and temp dirs.
It also wraps `uv`, `python`, `python3`, `nvidia-smi`, `ncu`, `nsys`, and
`nvcc` behind `outputs/gpu.lock`. The wrapper is reentrant through
`KBH_GPU_LOCK_HELD=1`; preserve that guard or `uv run python benchmark.py` can
deadlock when child `nvcc` tries to acquire the same lock.
Run metadata includes agent wall time, total/check/benchmark wall time,
check/benchmark exit codes, parsed token/cache/reasoning usage, and GPU lock
wait/active totals via `scripts/summarize_runs.py`.
Each `result.json` also includes `failure_reason` and
`retryable_infra_failure`; the website should show these instead of flattening
every non-pass into the same red cell. No-solution rows with fewer than 5,000
output tokens are treated as `provider_early_stop` and retryable by default.
Provider-credit detection must stay credit-specific; do not match plain
`overage`, because ordinary model text like `coverage` will false-positive.
Provider credit/rate classifications should only apply to rows without a
solution; successful sessions may quote old run logs or result JSON in the
transcript, and those quotes are not provider failures.
Provider credit/rate detection must read explicit CLI/API error events and
stderr only, not arbitrary assistant text or tool outputs; models can read
AGENTS.md, `run_hard.sh`, and old artifacts containing those trigger words.
When `KBH_DISABLE_AGENT_CUDA=1`, agent-phase `uv`/`python`/`python3` probes
bypass the lock because CUDA is hidden and guarded; harness-owned
`check.py`/`benchmark.py` still lock normally.
The `check.py`/`benchmark.py` execution timeout must start after the GPU lock is
acquired. Use `run_gpu_locked_timeout`; do not wrap `timeout` outside `uv run`
or queued rows can fail while merely waiting for `outputs/gpu.lock`.
Transcript usage extraction also bypasses the lock; it is CPU-only post-processing.

Before expensive sweeps, check:

```bash
overnight-compute status
nvidia-smi
```

The lock only protects KernelBench children launched through this harness. It
does not stop unrelated CUDA compiles or benchmark jobs elsewhere on Anvil.

For parallel sweeps, prefer:

```bash
KBH_DISABLE_AGENT_CUDA=1 ./scripts/launch_parallel_sweep.sh
```

`scripts/launch_parallel_sweep.sh` defaults to `KBH_HARNESS_CONCURRENCY=2`,
so each harness/provider path gets at most two active agent sessions at once.
Raise it only after a preflight proves quota and rate limits are healthy.
The launcher must use per-harness workers. A problem-major loop causes
head-of-line blocking: if Codex or Claude holds its two slots, freed Cursor,
Gemini, or OpenCode slots do not backfill.
Run `./scripts/preflight_harnesses.sh` before expensive sweeps; it sends tiny
text prompts through the current matrix and fails fast on auth/quota/model-route
problems. After a sweep, `./scripts/launch_infra_retries.sh <run_group>` reruns
only rows where `result.json` has `retryable_infra_failure=true`, such as
provider rate limits, early stops, or no-solution timeouts.
Retry rows must preserve empty effort fields and pass full `problems/<name>`
paths to `run_hard.sh`; otherwise the problem can slide into the effort column
and produce blank-problem manifest rows.
If OpenRouter is depleted, use
`KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1` to run the non-OpenRouter rows
plus Gemini through the Gemini API key; Qwen remains pending until OpenRouter is
topped up or a direct Alibaba/Qwen key is added.
If aborting a sweep, kill the launcher process group and then verify by cwd:
some harness CLIs spawn their own orphaned timeout groups under
KernelBench-Hard. Do not kill unrelated higher-priority jobs such as IVA just
to make the GPU table empty; note them in the report and keep KernelBench at
lower priority.

`KBH_DISABLE_AGENT_CUDA=1` hides CUDA from OpenCode/Cursor agent phases and
lets only the harness-owned `check.py` / `benchmark.py` path touch the GPU
under `outputs/gpu.lock`. PATH wrappers alone are not sufficient because
agents can call absolute interpreters such as `.venv/bin/python3`; the current
harness also injects an agent-phase `sitecustomize.py` guard and wrapper
recursion fallback. If you see `REAL_UV=$(which uv)` inside a model command, it
must not resolve back to the per-run wrapper or the lock owner can hang. Run
archives now include `run_group`, total/check/benchmark wall time, queue mode,
and token metadata where the harness exposes it. Summaries:

Claude-family harnesses must launch from the archive-local `$PROBLEM_DIR`, not
from repo root with only `--add-dir`; otherwise models can write
`problems/<name>/solution.py` in the source tree and the archived run records
`no_solution`.

```bash
uv run python scripts/summarize_runs.py --run-group <name>
```

High-priority rows:

```bash
./scripts/run_hard.sh minimax-claude MiniMax-M3 problems/01_fp8_gemm
./scripts/run_hard.sh opencode openrouter-alibaba/qwen/qwen3.7-max problems/01_fp8_gemm
./scripts/run_hard.sh opencode openrouter-google-ai-studio/google/gemini-3.5-flash problems/01_fp8_gemm
./scripts/run_hard.sh cursor composer-2.5 problems/01_fp8_gemm
./scripts/run_hard.sh cursor composer-2.5-fast problems/01_fp8_gemm
./scripts/run_hard.sh grok grok-build problems/01_fp8_gemm max
```

Claude Code runs explicitly pass `--settings
'{"fastMode":false,"alwaysThinkingEnabled":true}'`. Opus comparability also
requires `--effort max`.
MiniMax M3 through Claude Code uses harness `minimax-claude`, model
`MiniMax-M3`, and MiniMax's Anthropic-compatible endpoint
`https://api.minimax.io/anthropic`. Put `export MINIMAX_API_KEY=...` in
Anvil's `~/.env_vars`; do not put it in repo files. Enable it in broad
preflight/sweeps with `KBH_USE_MINIMAX_M3_CLAUDE=1`.
Do not pass provider API keys with `timeout env KEY=... claude`; that puts the
key in process argv. Export secrets inside the subshell before invoking
`timeout claude` instead.

As of 2026-05-22, Qwen 3.7 Max passed a 300 second `01_fp8_gemm` smoke via
OpenCode/Alibaba (`correct=true`, `peak_fraction=0.4257`). Gemini 3.5 Flash and
Composer 2.5 wrote solutions in 300 second smokes but did not pass correctness;
Composer's sidecar-CUDA behavior makes that smoke harness-sensitive, so rerun it
with the current isolation before judging the model. Cursor CLI on Anvil is
`agent`, not `cursor`; available Cursor model slugs include `composer-2.5` and
`composer-2.5-fast`.
Grok CLI is installed on Anvil as `/home/infatoshi/.local/bin/grok`, not on the
Mac control plane. Use harness `grok` with model `grok-build`; the working
headless route is top-level `grok --cwd <workspace> --output-format
streaming-json -p <prompt>`, not `grok agent`.
