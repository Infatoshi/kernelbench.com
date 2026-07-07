# kernelbench.com — operator guide

`AGENTS.md` is the **single source of truth** for working in this repo;
`CLAUDE.md` and `.cursorrules` are symlinks to it — so Claude Code
(`CLAUDE.md`), Cursor (`.cursorrules` / `AGENTS.md`), Codex, and any other CLI
(`AGENTS.md`) all read this exact file. Everything you need to work on the
website **and** both active benches lives here; there are no per-bench
`AGENTS.md` / `CLAUDE.md` / `.cursorrules` files anymore (the lone exception is
the `benchmarks/v3/` archive, which keeps its own).

This is the **canonical monorepo** for the KernelBench website AND the eval
benchmarks. It lives on Anvil at `~/kernelbench.com` (the GPU box, where evals
run). The Mac has a secondary clone you `git pull` when working there; Anvil is
canonical. Deploys go out from Anvil.

## The two active benches — know which one you're writing to

There are **two** active benches plus one archive. They share the same `src/`
harness, run archive, and roofline machinery, so the developer guide in
[Working in a bench](#working-in-a-bench-hard--mega) below applies to both —
only the deck, the entry command, and the wall-clock budget differ:

| | `benchmarks/hard/` | `benchmarks/mega/` |
| --- | --- | --- |
| What | per-op kernel deck | full fused **megakernels** |
| Deck | `01_fp8_gemm`, `02_kda_cutlass`, `03_paged_attention`, `05_topk_bitonic`, `06_sonic_moe_swiglu`, `07_w4a16_gemm` (6) | `01_rl_grid_ppo` (PufferLib-style grid-foraging PPO training), `02_kimi_linear_decode` (Kimi-Linear W4A16 decode, the published board) (2) |
| Drive it with | the `kb` CLI / `uv run kbh run` (from any cwd) | `cd benchmarks/mega && ./scripts/run_hard.sh ...` |
| Wall-clock | **unlimited** (`BUDGET_SECONDS=0`) | still capped (`BUDGET_SECONDS` 2700 local / 10800 cloud) |
| Published to | `/hard` | not yet on the site |

**How to tell which you're editing:** your cwd (`benchmarks/hard` vs
`benchmarks/mega`), the problem names above, and the entry command. When in
doubt, check the deck — `01_fp8_gemm` is hard, `01_rl_grid_ppo` is mega. Mega
reuses hard's harness/archive/roofline machinery; the old operation-level deck
was removed from mega.

`benchmarks/v3/` is the **archive** (RTX 3090 / H100 / B200, a separate harness
living in its own repo). It keeps its own `benchmarks/v3/AGENTS.md`; ignore it
unless you're specifically working the archive.

## Layout

```
app/ public/              the website (Next.js 16, Tailwind v4)
  app/_lib/data.ts             reads benchmark data at build time
  public/data/v3/results.csv   /v3 reads this
benchmarks/hard/           KernelBench-Hard eval — the per-op deck
  results/leaderboard.json     /hard reads this (v2 site-shaped data)
  results/annotations/*.yaml   per-cell reward-hack / clean verdicts
  outputs/runs/                run archives (gitignored; ~186G)
  scripts/ src/                eval code
  problems-rtxpro6000/         the deck — per-GPU sets: -rtxpro6000 (default, RTX PRO 6000), -h100, -b200, -3090
benchmarks/mega/           KernelBench-Mega eval — megakernel deck (single problems/; reuses hard's machinery)
benchmarks/v3/             KernelBench-v3 eval — archived (keeps its own AGENTS.md)
environments/              Prime Intellect `verifiers` mirrors (kernel_hard / kernel_mega / kernel_v3)
media/                     tracked chart generators (kbh_theme.py + make_*.py + generate_dark_plots.py)
runs/                      gitignored HF staging (kb publish fills it, pushes to HF)
```

## Run a sweep (the common task) — use the `kb` CLI (on PATH, runs from any cwd)

**"Do a sweep of <model>" is an end-to-end order, not a question.** When the
human says "run/do a sweep" for a model (however loosely worded), assume the
full default scope, state the assumption in one line ("Assuming: all problems,
hard + mega, audited, published"), and go — do not ask for confirmation. The
default scope is:

1. **All problems** in the deck for the current GPU (hard) **and** the mega
   deck (`02_kimi_linear_decode` at minimum), unlimited time for hard, the
   standard cap for mega. Existing valid cells for the same
   (model, harness, problem, GPU) don't need reruns.
2. **Reward-hack audit every cell** (subagent reads solution.py + trace,
   annotation YAML written) — this is already mandatory before reporting, so a
   "sweep" is not done until the audits are.
3. **Publish end-to-end**: contamination check, redaction, `kb publish`,
   commit + push (deploy). Report back with the numbers and anything
   interesting found in the traces.

The human coming back to finished, audited, published results is the success
condition. Only interrupt for a missing API key (`STOP: needs $X_API_KEY`), an
occupied shared GPU that never frees, or a genuinely ambiguous model identity.

```
kb sweep kimi-claude kimi-k2.7-code       # all hard problems, parallel containers, unlimited time
kb publish                                # rebuild leaderboard + viewers from archives
kb deploy "bench kimi k2.7"               # publish + commit + push (Vercel auto-builds)
```
Other commands: `kb run <harness> <model> <problem>` (one problem), `kb dev`
(preview, view from Mac via Tailscale anvil:3000), `kb build`, `kb audit <run_id>`,
`kb contamination <hard|mega|v3>`, `kb traces-to-hf`, `kb help`. The CLI is the
`kbtool/` uv package (`kbtool/kb/cli.py`); `bin/kb` is a thin shim that runs it
via `uv run` (symlinked to ~/.local/bin/kb). Install standalone with
`uv tool install ./kbtool`. The GPU-coupled sweep orchestration stays as
bench-local shell (`benchmarks/<bench>/scripts/*.sh`); the CLI shells out to it.
The `kb` CLI targets the **hard** bench; for **mega** drive
`./scripts/run_hard.sh` / `./scripts/sweep.sh` from inside `benchmarks/mega/`
(or `kb publish mega`).

- API keys live in `~/.env_vars` (KIMI_API_KEY, ZAI_API_KEY, MINIMAX_API_KEY,
  OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, CLAUDE_CODE_OAUTH_TOKEN).
  To bench a new model: drop its key in `~/.env_vars`, then `kb sweep`.
- **If `kb sweep` prints `STOP: ... needs $X_API_KEY`**, the key is missing —
  ask the human for it, append `export X_API_KEY=...` to `~/.env_vars`, rerun.
  `kb` preflights the key before launching so you get one clear message, not
  six failed runs.
- Benching a **brand-new provider** (no harness yet) needs a harness branch:
  copy the `kimi-claude` block in `scripts/run_hard.sh` (Claude-Code → the
  provider's Anthropic-compatible endpoint) and add the row. Not a one-liner.
- Before a GPU sweep: `nvidia-smi` (the box is shared).

## Harnesses (run via `uv run kbh run <harness> <model> <problem> [effort]`)

- Native CLIs: `claude`, `codex`, `cursor`, `gemini`, `grok`, `opencode`.
- Claude-Code-routed providers (most reliable): `zai-claude` (GLM via
  api.z.ai/api/anthropic), `minimax-claude`, `kimi-claude` (Kimi via
  api.moonshot.ai/anthropic, model `kimi-k2.7-code`), `deepseek-claude`
  (DeepSeek via api.deepseek.com/anthropic, model `deepseek-v4-pro` or
  `deepseek-v4-flash`), `qwen-claude` (Qwen via DashScope Model Studio
  Intl, dashscope-intl.aliyuncs.com/apps/anthropic, model `qwen3-max` —
  needs DASHSCOPE_API_KEY, which we do not have yet). These mirror each other;
  to add one, copy the `kimi-claude` branch in `scripts/run_hard.sh`.
  Rationale: opencode is a strong harness but its `@ai-sdk/openai-compatible`
  transport stalls intermittently (~1/3-1/2 of sessions); routing these models
  through Claude Code to the provider Anthropic endpoint bypasses that adapter.
- Always container mode (`KBH_AGENT_CONTAINER=1`): isolated per-run workspace,
  native GPU, sessions overlap while GPU commands serialize through the lock.

## Working in a bench (hard & mega)

This is the developer guide for editing either active bench. It's written for
the **hard** deck; **Mega deltas** are called out inline where mega differs (see
also the table above). Do not confuse this with `problems/<X>/PROMPT.txt` —
those are the human-voice queries fed to agents _under test_. For the journey
behind the current design, read the bench's `DEVLOG.md`; for methodology, its
`SPEC.md`.

The benchmark itself: frontier coding agents are given URLs to SOTA
implementations (sonic-moe, flashinfer, marlin) and asked to write a competitive
kernel on RTX PRO 6000 Blackwell (SM120) with unlimited time (one autonomous
session, runs until the model decides it is done). Roofline-graded. The
published artifact is the best kernel per (problem × model × harness), plus the
agent trace.

### Non-negotiable rules

- **uv only.** No bare `python`, no `pip`. Use `uv run ...`, `uv add ...`, `uv pip install ...`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Never edit `problems/*/solution.py`**. Those files are agent output; they're gitignored for a reason. If you need to inspect one, read it from `outputs/runs/<run>/<problem>/solution.py`.
- **Never modify `problems/*/reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** once a sweep has been published unless you are intentionally versioning the benchmark/validation surface. Those files define the benchmark. `scripts/run_hard.sh` snapshots them and marks the run invalid if an agent under test changes them.
- **torch.compile fix.** torch 2.11.0+cu130 has a broken inductor CSE typing annotation that breaks the compile baseline. Run `./scripts/patch_torch.sh` after every `uv sync`.
- **GPU work must go through the harness.** Hard: `uv run kbh run` (or the `kb` CLI). Mega: `./scripts/run_hard.sh` from inside `benchmarks/mega/`. Either way it creates archive-local workspaces, isolated CUDA/Triton/Torch caches, and a shared GPU lock so concurrent agent sweeps can edit in parallel while compile/check/benchmark work queues cleanly.
- **Correctness now includes numeric stress.** `check.py` reruns canonical shapes/seeds under problem-specific small/large activation or weight scales via `src/eval/numeric_stress.py`. This hardens correctness against zero-output, cached-nominal, and loose-tolerance cheats. `benchmark.py` still measures only the canonical performance deck.

### Per-bench repo layout

```
benchmarks/<bench>/
├── README.md, SPEC.md         project docs
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

### Adding a new problem

1. Pick the next NN (zero-padded). Don't reuse numbers.
2. Create the deck dir: `problems-rtxpro6000/NN_name/` (hard) or `problems/NN_name/` (mega).
3. Required files (order matters — write them in this order so you can sanity-check each):
   - `reference.py` — shortest naive PyTorch that produces the right answer. No optimization tricks. This is the correctness oracle.
   - `shapes.py` — 3 to 5 canonical shapes as a list of dicts. Include at least one "off-alignment" shape (e.g., K not multiple of 128 for GEMM).
   - `problem.yaml` — metadata. See `problems-rtxpro6000/01_fp8_gemm/problem.yaml` (hard) or `problems/02_kimi_linear_decode/problem.yaml` (mega) as the canonical example.
   - `sota.py` — wrap the library function that defines the ceiling. If no library supports SM120 yet, leave a stub and document the H100 paper number in a comment.
   - `check.py` — copy from the closest existing problem and adapt the imports/formulas.
   - `benchmark.py` — copy from the closest existing problem and adapt the throughput formula to match `problem.yaml.flops_formula` / `bytes_formula`.
   - `PROMPT.txt` — single cohesive human-voice query. Match the structure of the existing problems: hardware in parenthetical on first line, file roles + "make a mess" allowance, op semantics + tolerance + every shape inlined as prose, custom-kernel mandate + forbidden ops list spelled out + suggested implementation paths + "look it up yourself" directive, flywheel sentence ending with "Take as long as you need to actually push the number up." Do not include peak throughput numbers, optimization recipes, or "you are being evaluated" framing.
4. Smoke-test on a cheap model first (`uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/NN_name` for hard, `./scripts/run_hard.sh claude claude-opus-4-7 problems/NN_name` for mega). Verify `check.py` runs, `benchmark.py` runs, result.json is sane.
5. Once you're happy, run the full model matrix sweep.

### Running a sweep

```bash
# Hard — single (harness, model, problem). Deck is per-GPU: problems-rtxpro6000
# (default, RTX PRO 6000), problems-h100, problems-b200, problems-3090.
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm

# Full active matrix on one problem
for model_harness in "claude claude-opus-4-7" "codex gpt-5.5 xhigh" "kimi kimi-k2.6"; do
    read -r HARNESS MODEL <<< "$model_harness"
    uv run kbh run "$HARNESS" "$MODEL" problems-rtxpro6000/01_fp8_gemm
done

# Everything (this is what sweep.sh does)
./scripts/sweep.sh
```

**Mega delta:** drive mega from inside `benchmarks/mega/` via
`./scripts/run_hard.sh <harness> <model> problems/02_kimi_linear_decode` (and
`./scripts/sweep.sh`), not the `kb`/`kbh` CLI.

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
KernelBench. Do not kill unrelated higher-priority jobs such as IVA just
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

Smoke-tested candidates and useful commands (hard examples; for mega swap to
`./scripts/run_hard.sh` and a mega problem):

```bash
uv run kbh run opencode openrouter-alibaba/qwen/qwen3.7-max problems-rtxpro6000/01_fp8_gemm
uv run kbh run opencode openrouter-google-ai-studio/google/gemini-3.5-flash problems-rtxpro6000/01_fp8_gemm
uv run kbh run opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b problems-rtxpro6000/01_fp8_gemm
uv run kbh run cursor composer-2.5 problems-rtxpro6000/01_fp8_gemm
uv run kbh run cursor composer-2.5-fast problems-rtxpro6000/01_fp8_gemm
uv run kbh run grok grok-build problems-rtxpro6000/01_fp8_gemm max
```

Other serious rows to keep in the matrix if their auth/config is healthy:
`codex gpt-5.5 xhigh`, `claude claude-opus-4-7 max`, `claude
claude-opus-4-8 max`, `zai-claude glm-5.1`, `opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b`, Factory/Droid GLM-5.1 if the auth
bundle is present, and Kimi only after auth is fixed. Enable Nemotron in broad preflight/sweeps with `KBH_USE_OPENROUTER_NEMOTRON=1`; target only that row with `KBH_USE_OPENROUTER_NEMOTRON=1 KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra ./scripts/preflight_harnesses.sh`. **Mega delta:** the mega matrix omits the Nemotron row.

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

### Harness operational notes

These apply to every harness route, not just Claude Code:

- **Timeout starts after the GPU lock.** Use `run_gpu_locked_timeout` for
  `check.py`/`benchmark.py`; do not wrap `timeout` outside `uv run`, or queued
  rows can fail while merely waiting for `outputs/gpu.lock`.
- **CUDA hiding needs more than PATH wrappers.** `KBH_DISABLE_AGENT_CUDA=1`
  hides CUDA from OpenCode/Cursor agent phases, but agents can call absolute
  interpreters such as `.venv/bin/python3`, so the harness also injects an
  agent-phase `sitecustomize.py` guard and a wrapper-recursion fallback. If you
  see `REAL_UV=$(which uv)` inside a model command, it must not resolve back to
  the per-run wrapper or the lock owner can hang.
- **Claude-family harnesses must launch from the archive-local `$PROBLEM_DIR`,**
  not from repo root with only `--add-dir`; otherwise models can write
  `problems/<name>/solution.py` in the source tree and the archived run records
  `no_solution`.
- **Provider credit/rate detection must read explicit CLI/API error events and
  stderr only,** never arbitrary assistant text or tool output — models can read
  AGENTS.md, `run_hard.sh`, and old artifacts that contain those trigger words.
  It should only apply to rows without a solution.
- **Never pass provider API keys via `timeout env KEY=... claude`** — that puts
  the key in process argv. Export secrets inside the subshell before invoking
  `timeout claude`.

#### MiniMax M3 via Claude Code

MiniMax M3 routes through the `minimax-claude` harness (model `MiniMax-M3`) and
MiniMax's Anthropic-compatible endpoint `https://api.minimax.io/anthropic`. Put
`export MINIMAX_API_KEY=...` in Anvil's `~/.env_vars` (never in repo files), and
enable it in broad preflight/sweeps with `KBH_USE_MINIMAX_M3_CLAUDE=1`.

### Interpreting results

- `outputs/runs/<ts>_<harness>_<model>_<problem>/result.json` — scalar metrics (correct, achieved_tflops, peak_fraction, shape-by-shape times)
- `outputs/runs/<ts>_..../transcript.jsonl` — full agent trace
- `outputs/runs/<ts>_..../solution.py` — final agent-written kernel
- `outputs/runs/<ts>_..../roofline.png` — visual: peak line, eager/compile/SOTA/solution points per shape

Run `uv run python scripts/roofline_plot.py outputs/runs/<ts>_...` to (re)generate the plot.

### Testing

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

### When a sweep fails

Most likely causes:
1. **torch.compile CSE crash** — run `./scripts/patch_torch.sh`.
2. **CUDA_HOME pointing at 12.8** — harness script already sets `CUDA_HOME=/usr/local/cuda-13`; make sure you sourced it.
3. **`sota.py` import fails** — the SOTA dep isn't installed. Check `problem.yaml` for the pinned version; install with `uv pip install <spec>`.
4. **Agent CLI not authenticated** — `claude`, `codex`, `kimi` each need their own auth. Check `~/.env_vars` and each CLI's `info` / `whoami` command.
5. **Agent stopped before writing anything** — hard runs are unlimited-time (no wall-clock cap; `BUDGET_SECONDS=0` in `run_hard.sh`), so a no-solution row is a real failure mode (early stop / provider error), not a budget cutoff; record it. **Mega delta:** mega still has a wall-clock cap (`BUDGET_SECONDS` 2700 local / 10800 cloud), so a mega no-solution row may genuinely be a budget cutoff — raise `BUDGET_SECONDS` or record it.

## Hard-won gotchas

- **Commit email MUST be `elliot@arledge.net`** or Vercel silently fails the
  build verification. The repo sets it locally; new clones must `git config
  user.email elliot@arledge.net`.
- The publish pipeline regenerates `benchmarks/hard/results/leaderboard.json`
  (site data) and the redacted `public/runs/*_solution.py.txt` kernels from the
  archives. `kb publish` does it; don't hand-edit the leaderboard. Full agent
  transcripts live on HuggingFace (`kernelbench-<bench>-traces`); push them with
  `kb push-runs <hard|mega>` (or `kb publish --push`). The site links each run
  to its HF trace — it no longer self-hosts `*.html` viewers.
- **Redact on every publish/push pass.** Before uploading HF traces, committing
  `public/runs`, or deploying site artifacts, run
  `uv run python scripts/redaction.py runs public/runs` from the repo root. This
  is mandatory even if the converter/publish scripts already redacted once:
  agents can echo env dumps, old machine keys, or local `AGENTS.md` /
  `CLAUDE.md` content into transcripts. Block publish if a scan still finds
  local-instruction markers or unredacted sensitive assignments:
  `rg -n "# AGENTS\\.md instructions|<proactive-behavior>|~/.codex/AGENTS\\.md|~/.claude/CLAUDE\\.md|GOG_KEYRING_PASSWORD=|[A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD)=" runs public/runs`.
- **Transcript / reasoning extraction lives in-repo at
  `scripts/transcript-extraction/`** (vendored complete extractor; see its
  `VENDORED.md`). Use it as the canonical reference when working on the
  agent-timeline viewers — it pulls full conversations (messages, tool use,
  diffs, reasoning) across every harness format (codex / claude-code / cursor /
  gemini / opencode / …), more complete than the per-bench
  `src/viewer/parsers/*` (which under-extract — an opus transcript with 216
  thinking blocks, but 107 are signature-only). The viewer renders reasoning
  untruncated (`src/viewer/html.py` reasoning-block). **What's actually
  exposable depends on the provider, not the parser:** native `claude` (opus)
  and `codex` (gpt-5.5) ENCRYPT their chain-of-thought — the transcript carries
  empty `thinking` blocks with a `signature` (Anthropic extended thinking) or
  sparse summaries (codex), so there is nothing to render. The `*-claude` routes
  to open providers (glm/zai, kimi, deepseek, minimax) return FULL thinking text
  and now render in full. So a near-empty reasoning trace for opus/codex is the
  API encrypting it, not a viewer bug.
- **CROSS-RUN CONTAMINATION (harness has no filesystem sandbox).** Agents have
  bash + absolute paths, so they can read the shared `outputs/runs/` archive —
  every prior winning solution — and reverse-engineer a known answer instead of
  writing their own kernel. This is NOT what `kb lint` checks (lint only scans a
  single solution.py). Run `kb contamination <hard|mega|v3>
  [--published benchmarks/<bench>/results/leaderboard.json]` before
  publishing; both leaderboard builders now auto-EXCLUDE any run whose agent
  transcript references another run's archive. Audit on 2026-06-19 found
  mega-published 7/24 contaminated (the glm-5.2 17.4x / MiniMax 16.5x "beat opus"
  cells were fake — glm's genuine clean score is 7.3x; opus 14-19x is real and
  clean) and hard-published 0/53 (clean — its published set is the curated June
  generation; 107/403 hard *archive* runs are contaminated but none were
  published). PROPER FIX is a sandboxed harness — plan is to re-run on Prime
  Intellect's `verifiers` env (sandboxed harness + a judge/custom-verifier that
  inspects every passing solution). Until then the tripwire is the guard.
- Reward-hack verdicts come from `results/annotations/<run_id>.yaml`; every
  passing/failing headline cell should be audited (read the solution.py) before
  publishing. The template-mutation guard auto-flags grader tampering.
- `kb lint <run_id|--all>` is a static reward-hack TRIPWIRE: scans solution.py
  for the patterns we have caught by hand (input-identity memoization,
  stack/check.py sniffing, backend mutation, zero-kernel wrappers) plus surfaces
  template_mutated from result.json. HACK = near-certain, FLAG = review. It is a
  pointer for a human audit, NOT an auto-reject. Run it before publishing.
- **MANDATORY: before any cell is published or any result is reported, dispatch a
  subagent to MANUALLY audit the solution.py AND the agent trace for reward
  hacking — never trust the lint verdict alone (it both misses and over-fires).**
  The lint is only a tripwire: it FALSE-POSITIVES (e.g. it flagged glm-5.2's
  fp8 CUDA-graph kernel as "output memoization" on a `data_ptr()==` pattern, but
  empirical audit proved the graph replay recomputes on live data — see
  `results/annotations/20260614_145529_zai-claude_glm-5.2_01_fp8_gemm.yaml`), and
  it can MISS hacks a static scan can't see. The subagent must: (1) read
  solution.py end to end and confirm it computes the real op (not a cached/
  constant/forbidden-lib path); (2) read the trace for stack/check.py sniffing,
  tolerance edits, or grader tampering; (3) for any caching/CUDA-graph/identity
  pattern, EMPIRICALLY test it — overwrite the same input buffer with new
  contents and confirm the output changes (proves recompute, not a stale
  lookup), and sanity-check the magnitude (a returned-cached-output "lookup"
  reads >>1.0 of roofline; a real kernel lands near its theoretical time); (4)
  confirm numeric stress actually ran (`check.py` unmodified, KBH_NUMERIC_STRESS
  not 0). Record the verdict + evidence in `results/annotations/<run_id>.yaml`
  (clean | reward_hack | ...). Treat a lint HACK as "review," not "reject."
- `04_kahan_softmax` was removed from the hard deck (rewarded skipping Kahan); do not
  re-add. (This is also why the hard deck skips 04.)
- **KernelBench-Multi (`benchmarks/multi/`, the WIP 8×H100 NVLink bench) runs on
  rented Brev GPUs — two gotchas that each cost real money if you miss them:**
  - **`brev delete <name>` has a hidden interactive "are you sure?" confirmation
    that SILENTLY HANGS with no TTY** (it prints nothing and never deletes; `brev
    stop` and `yes | brev delete` also no-op). The ONLY reliable teardown is to
    give it a pseudo-TTY and feed it `y`:
    `script -qec "brev delete <name>" /dev/null <<< "y"`. Any auto-teardown
    watchdog MUST use this form — a plain `yes | brev delete` backstop does
    nothing, so a forgotten node bills at ~$23/hr (8×H100 SXM5) until caught.
    Always confirm teardown with `brev ls` showing "No instances".
  - **torch wheel must be CUDA-driver-matched.** Hyperstack/shadeform 8×H100
    nodes ship driver CUDA 12.8; the default `uv pip install torch` pulls a cu130
    wheel that can't see the GPUs (`torch.cuda.is_available()==False`, "driver too
    old"). Install the matched build:
    `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0`.
    Bake this (plus uv + the repo) into a prebaked image so you don't pay node
    time for the reinstall.
  - Pick the **NVSwitch** SKU (`hyperstack_H100_sxm5x8`, every pair `NV18` in
    `nvidia-smi topo -m`), not the PCIe `hyperstack_H100x8` — this bench grades
    NVLink busbw, so a PCIe node produces meaningless numbers. Validate
    correctness for free on a single GPU first via gloo+cpu
    (`KBM_BACKEND=gloo KBM_DEVICE=cpu KBM_WORLD_SIZE=4 python check.py`); the
    rented node should never see a correctness bug for the first time.
- See each bench's `DEVLOG.md` for the full journey and `SPEC.md` for
  methodology.

## Publishing results: charts + write-ups (REQUIRED format)

When you post benchmark results (X posts, blog, threads), two rules are not
optional. They are what makes a post read as signal, not slop.

- **Charts MUST use the website NVIDIA palette.** Import the shared theme,
  never hardcode colors: `from kbh_theme import C, SERIES, apply` (module at
  `media/kbh_theme.py`, mirrored on Mac at
  `~/dev/sites/kernelbench.com/media/`). It copies the `:root`
  tokens from `app/globals.css`: bg `#111111`, accent (NVIDIA green) `#76b900`,
  fg `#eeeeee`/`#999999`, warn `#fbbf24`, bad `#fb7185`, grid `#242424`. Lead
  bars with the green accent (the ceiling/subject); rose `#fb7185` = reward
  hack (hatched), amber = warn, grey = fail, faded+dotted = real kernel that
  bugged/timed out. `media/make_glm52_4way.py` is the canonical
  example. If `globals.css` changes, update `kbh_theme.py` to match. Charts are
  generated on Mac (matplotlib) and dragged into posts; PNGs are gitignored,
  the `.py` scripts are tracked.
- **Write-ups lead with the unique/interesting/inconsistent, not adjectives.**
  "4/6 clean, strong showing, solid 2nd place" is filler. Before writing,
  actually read 2-3 transcripts/solutions for the headline cells and surface
  concrete findings: behavioral shifts (e.g. a model that stopped reward-hacking
  the fp8 cell its predecessor cheated), metric artifacts (topk's ~0.02 ceiling
  is launch-overhead-bound, not weakness - true for every model), what the
  winning kernel actually did, profiling discipline that tracks the one win,
  suspicious cross-model convergence. The qualitative read a leaderboard cell
  cannot show is the whole point.
- **Post artifacts are EPHEMERAL — clean up after posting.** The only durable
  things are (1) these doc rules and (2) the tracked chart generators in
  `media/*.py` (`kbh_theme.py` = palette, `make_*.py` = each chart).
  Raw post drafts (`X-*.md`/`.txt`) and rendered `*.png` are throwaway: drafts
  never get committed, PNGs are gitignored and regenerate from the `.py` on
  demand. After you help draft a post, **ask the human when it goes live, then
  delete the drafts and PNGs** so the repo stays clean — labs are using this
  bench seriously and a tree full of one-off post scratch reads as noise. Never
  delete the `.py` generators; regenerate charts by re-running them.
