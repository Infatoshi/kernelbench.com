# KernelBench-Hard — Developer Instructions

Last updated: 2026-06-02.

This file is for **coding agents editing the repo** (you, via Claude Code). Do not confuse with `problems/<X>/PROMPT.txt` — those are the human-voice queries fed to agents _under test_.

For the journey behind the current design, read [DEVLOG.md](./DEVLOG.md).

## What this repo is

Small kernel benchmark. Frontier coding agents are given URLs to SOTA implementations (sonic-moe, flashinfer, marlin) and asked to write a competitive kernel on RTX PRO 6000 Blackwell (SM120) with unlimited time (one autonomous session, runs until the model decides it is done, under a large wall-clock ceiling). Roofline-graded. Published artifact is the best kernel per (problem × model × harness), plus the agent trace.

See [SPEC.md](./SPEC.md) for methodology. See [README.md](./README.md) for the model matrix and quick start.

## Non-negotiable rules

- **uv only.** No bare `python`, no `pip`. Use `uv run ...`, `uv add ...`, `uv pip install ...`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Never edit `problems/*/solution.py`**. Those files are agent output; they're gitignored for a reason. If you need to inspect one, read it from `outputs/runs/<run>/<problem>/solution.py`.
- **Never modify `problems/*/reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** once a sweep has been published unless you are intentionally versioning the benchmark/validation surface. Those files define the benchmark. `scripts/run_hard.sh` snapshots them and marks the run invalid if an agent under test changes them.
- **torch.compile fix.** torch 2.11.0+cu130 has a broken inductor CSE typing annotation that breaks the compile baseline. Run `./scripts/patch_torch.sh` after every `uv sync`.
- **GPU work must go through `uv run kbh run`.** It creates archive-local workspaces, isolated CUDA/Triton/Torch caches, and a shared GPU lock so concurrent agent sweeps can edit in parallel while compile/check/benchmark work queues cleanly.
- **Correctness now includes numeric stress.** `check.py` reruns canonical shapes/seeds under problem-specific small/large activation or weight scales via `src/eval/numeric_stress.py`. This hardens correctness against zero-output, cached-nominal, and loose-tolerance cheats. `benchmark.py` still measures only the canonical performance deck.

## Repo layout

```
KernelBench-Hard/
├── README.md, SPEC.md         project docs
├── CLAUDE.md, AGENTS.md, .cursorrules   dev-facing (this file)
├── pyproject.toml             uv project
├── problems/                  the deck (append-only after release)
│   └── NN_name/
│       ├── reference.py       naive PyTorch, for correctness
│       ├── sota.py            library call for the ceiling number
│       ├── shapes.py          canonical shape list (read by check.py / benchmark.py)
│       ├── problem.yaml       metadata (flops, bytes, tolerance, forbidden ops)
│       ├── check.py           correctness runner (per-dtype atol)
│       ├── benchmark.py       roofline measurement: solution; optional eager/compiled/sota
│       ├── PROMPT.txt         human-voice query sent to the agent under test
│       └── solution.py        agent output (gitignored)
├── src/
│   ├── harness/               claude.py, codex.py, kimi.py, ccr_router.py
│   ├── eval/                  correctness.py, numeric_stress.py, roofline.py, shapes.py, report.py
│   ├── hardware/              rtx_pro_6000.py, m4_max.py — peak lookup
│   └── sandbox/               local.py, metal.py
├── scripts/
│   ├── run_hard.sh            fire one (harness, model, problem)
│   ├── sweep.sh               full active matrix
│   ├── setup_problem.py       install SOTA deps for a problem
│   ├── roofline_plot.py       post-hoc plot from run artifacts
│   └── patch_torch.sh         torch 2.11 inductor CSE typing hotfix
├── outputs/runs/              per-run archival (gitignored)
└── docs/                      design notes, reward-hack case studies
```

## Adding a new problem

1. Pick the next NN (zero-padded). Don't reuse numbers.
2. Create `problems/NN_name/`.
3. Required files (order matters — write them in this order so you can sanity-check each):
   - `reference.py` — shortest naive PyTorch that produces the right answer. No optimization tricks. This is the correctness oracle.
   - `shapes.py` — 3 to 5 canonical shapes as a list of dicts. Include at least one "off-alignment" shape (e.g., K not multiple of 128 for GEMM).
   - `problem.yaml` — metadata. See `problems/01_fp8_gemm/problem.yaml` as the canonical example.
   - `sota.py` — wrap the library function that defines the ceiling. If no library supports SM120 yet, leave a stub and document the H100 paper number in a comment.
   - `check.py` — copy from 01_fp8_gemm, change the import line for `reference` and `shapes`.
   - `benchmark.py` — copy from 01_fp8_gemm, change the throughput formula to match `problem.yaml.flops_formula` / `bytes_formula`.
   - `PROMPT.txt` — single cohesive human-voice query. Match the structure of the existing seven: hardware in parenthetical on first line, file roles + "make a mess" allowance, op semantics + tolerance + every shape inlined as prose, custom-kernel mandate + forbidden ops list spelled out + suggested implementation paths + "look it up yourself" directive, flywheel sentence ending with "Take as long as you need to actually push the number up." Do not include peak throughput numbers, optimization recipes, or "you are being evaluated" framing.
4. Smoke-test: `uv run kbh run claude claude-opus-4-7 problems/NN_name` on a cheap model first. Verify `check.py` runs, `benchmark.py` runs, result.json is sane.
5. Once you're happy, run the full model matrix sweep.

## Running a sweep

```bash
# Single (harness, model, problem)
uv run kbh run claude claude-opus-4-7 problems/01_fp8_gemm

# Full active matrix on one problem
for model_harness in "claude claude-opus-4-7" "codex gpt-5.5 xhigh" "kimi kimi-k2.6"; do
    read -r HARNESS MODEL <<< "$model_harness"
    uv run kbh run "$HARNESS" "$MODEL" problems/01_fp8_gemm
done

# Everything (this is what sweep.sh does)
./scripts/sweep.sh
```

### Correctness validation

The correctness gate is stricter than the performance gate. `check.py` first
validates nominal canonical shapes/seeds, then reruns the same shapes/seeds
under problem-specific numeric stress cases from `src/eval/numeric_stress.py`.
Stress cases only rescale existing floating inputs or model state; they do not
add hidden shapes. Integer/discrete outputs are exact, while floating outputs
use explicit per-dtype tolerances and report max absolute/relative error, bad
element count, worst index, and tolerance on failure.

`KBH_NUMERIC_STRESS=0` disables stress cases for local debugging only. Do not
use it for official checks, sweeps, or published result backfills.

The performance score remains comparable across the canonical deck:
`benchmark.py` does not import numeric stress and still times the submitted
solution on the normal benchmark inputs.

### Concurrent sweeps and GPU isolation

`scripts/run_hard.sh` gives every run a repo-shaped workspace under
`outputs/runs/<run_id>/repo/problems/<problem_name>/`. Agents write there, not
directly in `problems/<problem>/`. The workspace has immutable problem files
copied from the source tree, `src/` symlinked from the repo, and local copies of
`pyproject.toml`, `uv.lock`, and `.python-version` so an agent can mutate
dependencies inside the run archive without touching repo metadata.

Each run gets isolated build/cache state:

```bash
TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
TMPDIR="$RUN_DIR/tmp"
TEMP="$RUN_DIR/tmp"
TMP="$RUN_DIR/tmp"
```

The harness prepends `$RUN_DIR/bin` to `PATH` and wraps `uv`, `python`,
`python3`, `nvidia-smi`, `ncu`, `nsys`, and `nvcc`. These wrappers acquire the
shared `outputs/gpu.lock`, log timing to `$RUN_DIR/gpu_lock.log`, and then run
the real binary. The lock wrapper is intentionally reentrant:
`KBH_GPU_LOCK_HELD=1` lets child tools such as `nvcc` run under the parent
`uv run python benchmark.py` lock instead of deadlocking.
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
When `KBH_DISABLE_AGENT_CUDA=1`, agent-phase `uv`/`python`/`python3`,
`nvidia-smi`, and `nvcc` probes bypass the lock because CUDA is hidden and
guarded or the probe is harmless; `ncu` and `nsys` fail fast. Harness-owned
`check.py`/`benchmark.py` still lock normally.
Transcript usage extraction also bypasses the lock; it is CPU-only post-processing.
`benchmark.py` must score `variant=solution` first. Eager / compiled / SOTA
reference diagnostics are opt-in via `KBH_BENCHMARK_BASELINES=1` (or a
per-problem alias) and emit `benchmark_event` start/end/error lines for audits.

For broad sweeps, `scripts/launch_parallel_sweep.sh` defaults to
`KBH_HARNESS_CONCURRENCY=2`, meaning each harness/provider path can have at
most two active agent sessions at once. Raise it only after a preflight proves
quota and rate limits are healthy.
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

Before expensive work, still check external machine state:

```bash
overnight-compute status
nvidia-smi
```

The lock only governs KernelBench children launched through `run_hard.sh`; it
cannot serialize unrelated CUDA compiles or benchmark jobs elsewhere on Anvil.

### Current model/harness candidates

Smoke-tested candidates and useful commands:

```bash
uv run kbh run opencode openrouter-alibaba/qwen/qwen3.7-max problems/01_fp8_gemm
uv run kbh run opencode openrouter-google-ai-studio/google/gemini-3.5-flash problems/01_fp8_gemm
uv run kbh run opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b problems/01_fp8_gemm
uv run kbh run cursor composer-2.5 problems/01_fp8_gemm
uv run kbh run cursor composer-2.5-fast problems/01_fp8_gemm
uv run kbh run grok grok-build problems/01_fp8_gemm max
```

Other serious rows to keep in the matrix if their auth/config is healthy:
`codex gpt-5.5 xhigh`, `claude claude-opus-4-7 max`, `claude
claude-opus-4-8 max`, `zai-claude glm-5.1`, `opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b`, Factory/Droid GLM-5.1 if the auth
bundle is present, and Kimi only after auth is fixed. Enable Nemotron in broad preflight/sweeps with `KBH_USE_OPENROUTER_NEMOTRON=1`; target only that row with `KBH_USE_OPENROUTER_NEMOTRON=1 KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra ./scripts/preflight_harnesses.sh`.

Nemotron 3 Ultra should be scored through `opencode-nemotron`, not Claude Code or Droid. OpenCode speaks OpenAI-compatible APIs directly and the harness pins OpenRouter to DeepInfra with `allow_fallbacks=false`. Claude Code through CCR smoked once, but it adds an Anthropic-router translation layer; Droid is not the native endpoint for this provider. The NVCF route is diagnostic only because Ultra was observed degrading/504ing.

For Claude Code runs, `scripts/run_hard.sh` passes `--settings
'{"fastMode":false,"alwaysThinkingEnabled":true}'` by default. The Opus matrix
also passes `--effort max`; do not count a Claude rerun as comparable if either
fast mode or a lower effort tier slipped in.

Current smoke notes:

- Qwen 3.7 Max via OpenCode/Alibaba passed a 300 second `01_fp8_gemm` smoke:
  `correct=true`, `peak_fraction=0.4257`, `template_mutated=false`.
- Gemini 3.5 Flash via OpenCode/Google AI Studio wrote a solution in a 300
  second smoke but did not pass correctness.
- Composer 2.5 via Cursor Agent wrote a solution in a 300 second smoke but did
  not pass correctness. That run was harness-sensitive because it generated a
  sidecar CUDA file, so rerun it with the current workspace/cache/lock isolation
  before judging capability.
- Cursor CLI on Anvil is `/home/infatoshi/.local/bin/agent`, not `cursor`.
  Available Cursor model slugs include `composer-2.5` and `composer-2.5-fast`.
- Grok CLI is installed on Anvil as `/home/infatoshi/.local/bin/grok`, not on
  the Mac control plane. `grok models` currently exposes `grok-build` as the
  default model. The working headless route is top-level `grok --cwd
  <workspace> --output-format streaming-json -p <prompt>`; `grok agent` does
  not accept the same cwd/output flags.

### Z.ai GLM-5.1 via Claude Code

For a canonical GLM-5.1 rerun through Claude Code, use the `zai-claude`
harness, not OpenCode. Z.ai's Anthropic-compatible Claude Code endpoint is
`https://api.z.ai/api/anthropic`; the OpenAI-compatible coding endpoint is for
Droid/Factory.

`scripts/run_hard.sh` sets the Z.ai-recommended Claude Code defaults in the
`zai-claude` branch:

```bash
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
CLAUDE_CODE_MAX_RETRIES=1000000
CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
ZAI_CLAUDE_HAIKU_MODEL=glm-5.1
```

All Claude Code aliases should map to `glm-5.1`, including Haiku / Explore /
subagent calls. The harness also passes `--disallowedTools ExitPlanMode
EnterPlanMode AskUserQuestion` for this route.

## Interpreting results

- `outputs/runs/<ts>_<harness>_<model>_<problem>/result.json` — scalar metrics (correct, achieved_tflops, peak_fraction, shape-by-shape times)
- `outputs/runs/<ts>_..../transcript.jsonl` — full agent trace
- `outputs/runs/<ts>_..../solution.py` — final agent-written kernel
- `outputs/runs/<ts>_..../roofline.png` — visual: peak line, eager/compile/SOTA/solution points per shape

Run `uv run python scripts/roofline_plot.py outputs/runs/<ts>_...` to (re)generate the plot.

## Testing

Keep `tests/` minimal. We test:
- `src/hardware/` peak-value lookup
- `src/eval/roofline.py` throughput math
- `src/eval/correctness.py` per-dtype tolerance enforcement
- `src/eval/numeric_stress.py` against classic zero-output, cached-nominal, and
  parameter-scale restoration cases

We do **not** test full problem files directly — those are validated by running
a real agent or a disposable smoke workspace against them.

```bash
uv run pytest
```

## When a sweep fails

Most likely causes:
1. **torch.compile CSE crash** — run `./scripts/patch_torch.sh`.
2. **CUDA_HOME pointing at 12.8** — harness script already sets `CUDA_HOME=/usr/local/cuda-13`; make sure you sourced it.
3. **`sota.py` import fails** — the SOTA dep isn't installed. Check `problem.yaml` for the pinned version; install with `uv pip install <spec>`.
4. **Agent CLI not authenticated** — `claude`, `codex`, `kimi` each need their own auth. Check `~/.env_vars` and each CLI's `info` / `whoami` command.
5. **Agent stopped before writing anything** — runs are unlimited-time (no wall-clock cap; `BUDGET_SECONDS=0` in `run_hard.sh`), so a no-solution row is a real failure mode (early stop / provider error), not a budget cutoff. Record it.
