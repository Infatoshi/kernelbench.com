# KernelBench-Hard — and the single-GPU bench internals (hard, cuda, mini; mega deltas noted)

Per-op GPU kernel bench: frontier coding agents write competitive CUDA or Triton kernels, unlimited wall-clock, roofline-graded, reward-hack audited. Default deck is `problems-rtxpro6000/` (RTX PRO 6000); variants exist as `problems-h100/` and `problems-b200/`. Live board: `/hard`.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_fp8_gemm` | FP8 e4m3 GEMM, off-alignment shapes |
| 02 | `02_kda_cutlass` | Kimi Delta Attention via CUTLASS CuTe |
| 03 | `03_paged_attention` | paged attention decode |
| 05 | `05_topk_bitonic` | top-k via bitonic sort (ms-anchored) |
| 06 | `06_sonic_moe_swiglu` | Sonic-MoE grouped GEMM + SwiGLU |
| 07 | `07_w4a16_gemm` | W4A16 weight-only quantized GEMM |

04 (`kahan_softmax`) is retired and stays retired. Drive it with `kb sweep <harness> <model>` or `uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm` on the GPU box. Harness routes, the shared runner, rented workers, and `KB_` variables: `kbtool/AGENTS.md`. Methodology: `SPEC.md`. History: `DEVLOG.md`. The rules that gate a published number: root `AGENTS.md`.

The benchmark: frontier coding agents get URLs to SOTA implementations (sonic-moe, flashinfer, marlin) and are asked to write a competitive kernel on the deck's GPU in one autonomous session with unlimited time. Roofline-graded. The published artifact is the best kernel per (problem x model x harness) plus the agent trace. `problems/<X>/PROMPT.txt` is the human-voice query fed to the agent under test, not operator documentation.

## Layout

```
benchmarks/<bench>/
├── AGENTS.md, SPEC.md, DEVLOG.md
├── pyproject.toml             uv project (the venv exists only on the GPU box)
├── problems*/                 the deck (append-only after release); hard has problems-rtxpro6000 (default), -h100, -b200
│   └── NN_name/
│       ├── reference.py       naive PyTorch, the correctness oracle
│       ├── sota.py            library call for the ceiling number
│       ├── shapes.py          canonical shape list (read by check.py / benchmark.py)
│       ├── problem.yaml       metadata (flops, bytes, tolerance, forbidden ops, hardware key)
│       ├── check.py           correctness runner (per-dtype atol + numeric stress)
│       ├── benchmark.py       roofline measurement: solution first; optional eager/compiled/sota
│       ├── PROMPT.txt         query sent to the agent under test
│       └── solution.py        agent output (gitignored; never edit)
├── src/
│   ├── kbh/                   the `kbh` CLI
│   ├── harness/               classification.py (failure/usage classification)
│   ├── eval/                  correctness.py, numeric_stress.py, roofline.py, shapes.py, timing.py, report.py, cuda_language.py (cuda bench)
│   ├── hardware/              peak lookup per GPU + identify.py (nvidia-smi name -> key)
│   └── viewer/                transcript parsers + HTML rendering
├── scripts/
│   ├── run_hard.sh            thin identity wrapper over repo-root scripts/lib/run_harness.sh (edit the lib; mega keeps a deliberate fork with its bwrap sandbox)
│   ├── sweep.sh               full active matrix
│   ├── setup_problem.py       install SOTA deps for a problem
│   ├── roofline_plot.py       post-hoc plot from run artifacts
│   └── patch_torch.sh         torch 2.11 inductor CSE typing hotfix
├── results/annotations/       audit YAML per run (schema below)
└── outputs/runs/              per-run archives (gitignored)
```

Shared files listed in `kbtool/tests/test_repo_consistency.py` must stay byte-identical across hard/cuda/mini/mega; a deliberate fork needs a DEVLOG entry and removal from that list.

## Adding a problem

1. Next NN, zero-padded; never reuse a number (04 is retired on hard, 01 on mega).
2. Create `problems-rtxpro6000/NN_name/` (hard) or `problems/NN_name/` (mega).
3. Write in this order so each file can be sanity-checked: `reference.py` (shortest naive PyTorch, no tricks), `shapes.py` (3 to 5 shapes, at least one off-alignment, e.g. K not a multiple of 128), `problem.yaml` (copy `01_fp8_gemm` or mega `02_kimi_linear_decode`), `sota.py` (library ceiling; stub with the H100 paper number in a comment if nothing supports the GPU yet), `check.py` and `benchmark.py` (copy the closest problem; throughput formula must match `problem.yaml` flops/bytes formulas), `PROMPT.txt` (one cohesive human-voice query matching the existing structure: hardware in a parenthetical on line one, file roles and the "make a mess" allowance, op semantics, tolerance, every shape inlined as prose, custom-kernel mandate with the forbidden ops spelled out, suggested paths, "look it up yourself", and the closing flywheel sentence ending "Take as long as you need to actually push the number up." No peak numbers, recipes, or "you are being evaluated" framing).
4. Smoke on a cheap model: `uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/NN_name` (mega: `./scripts/run_hard.sh claude claude-opus-4-7 problems/NN_name`). Confirm `check.py` and `benchmark.py` run and `result.json` is sane.
5. Then the full matrix.

## Running

```bash
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm    # one cell
for mh in "claude claude-opus-4-7" "codex gpt-5.5 xhigh"; do              # matrix on one problem
    read -r HARNESS MODEL EFFORT <<< "$mh"
    uv run kbh run "$HARNESS" "$MODEL" problems-rtxpro6000/01_fp8_gemm $EFFORT
done
./scripts/sweep.sh                                                        # everything
```

Mega is driven from inside `benchmarks/mega/` with `./scripts/run_hard.sh <harness> <model> problems/02_kimi_linear_decode` and `./scripts/sweep.sh`, not `kb`/`kbh`. Before a sweep: `nvidia-smi`, and `./scripts/patch_torch.sh` after every `uv sync` (torch policy below).

## Correctness

`check.py` validates nominal canonical shapes and seeds, then reruns them under problem-specific numeric stress from `src/eval/numeric_stress.py` (rescaled activations or weights; no hidden shapes). Integer outputs are exact; floating outputs use explicit per-dtype tolerances and report max abs/rel error, bad element count, worst index, and tolerance on failure. This catches zero-output, cached-nominal, and loose-tolerance cheats. `KBH_NUMERIC_STRESS=0` is for local debugging only, never official checks or backfills. `benchmark.py` does not import numeric stress and times the canonical deck only, so scores stay comparable.

## Results

`outputs/runs/<ts>_<harness>_<model>_<problem>/` holds `result.json` (correct, achieved_tflops, peak_fraction, per-shape times, gpu stamp, failure_reason), `transcript.jsonl`, `solution.py`, `roofline.png`. Regenerate the plot with `uv run python scripts/roofline_plot.py outputs/runs/<run>`. Archives are thin: the per-run `repo/.venv` is stripped after scoring (`scripts/lib/strip_run_venv.sh`; `KBH_KEEP_RUN_VENV=1` to keep; `--tree benchmarks/<bench>/outputs` reclaims old fat archives). Recreate an env with `uv run` from the archived `uv.lock`.

## Tests

On the GPU box, `uv run pytest` covers `src/hardware/` peak lookup, `src/eval/roofline.py`, `src/eval/correctness.py`, and `src/eval/numeric_stress.py` against the classic cheats (the bench venv exists only there; the Mac keeps none). Problem files are validated by running a real agent or a disposable smoke workspace, not by unit tests. Repo-wide guards run on the Mac: `uv run --project kbtool pytest kbtool/tests/`.

## When a sweep fails

1. torch.compile CSE crash: `./scripts/patch_torch.sh`.
2. `CUDA_HOME` at 12.8: the runner sets `/usr/local/cuda-13`; make sure it was sourced.
3. `sota.py` import fails: install the pinned dep from `problem.yaml` with `uv pip install <spec>`.
4. Agent CLI not authenticated: each CLI has its own login; check `~/.env_vars` and the CLI's `whoami`.
5. No solution written: runs are unlimited-time, so this is a real failure (early stop, provider error), not a budget cutoff. Record it; `retryable_infra_failure` marks the retryable ones.

## Torch version policy

Locked torch versions differ across benches (as of 2026-07-31: hard and mega lock 2.11.0; cuda, mini, and multi lock 2.13.0; all specify `torch>=2.11`). This is DELIBERATELY not unified, and the split is documented instead of "fixed", because torch is not the scored surface:

- **Correctness**: `reference.py` in eager torch is the oracle, and per-dtype tolerances absorb minor cross-version numeric drift. Any working torch produces the same pass/fail verdicts.
- **Performance**: the published number is the agent kernel's measured time against a hardware roofline (peak TFLOPS / bandwidth from `src/hardware/`), which torch's version cannot move. Even a slow torch reference would not change a roofline grade.
- **ms-anchored problems** (cuda bench): graded against the eager anchor FROZEN at deck publication. Upgrading torch later cannot re-grade historical cells; anchors are never re-measured on a new torch.
- **Provenance**: `environment_notes` in each leaderboard build records the live torch version via `importlib.metadata`, so every published wave self-describes its instrument.

What DOES matter about torch, and is enforced elsewhere:

- **Functionality**: the wheel must match the node's CUDA driver (cu128 vs cu130; see the bootstrap order in `kbtool/AGENTS.md`). Launch gates probe `torch.cuda.is_available()`, not `nvcc`.
- **torch.compile baselines**: opt-in diagnostics only (`KBH_BENCHMARK_BASELINES=1`), never the score. torch 2.11 needs `./scripts/patch_torch.sh` for the inductor CSE bug.
- **Mid-generation stability**: do not bump a bench's lock in the middle of a published wave; cells within one comparison set should share an instrument. Bump between waves freely; the version lands in `environment_notes`.

## Audit YAML schema (hard, cuda, mini, mega)

Annotations attach the audit verdict and human commentary to one run. They live in `results/annotations/<run_id>.yaml`, where `<run_id>` matches the directory name of a run under `outputs/runs/` (e.g. `20260428_040539_claude_claude-opus-4-7_01_fp8_gemm`). The site (`app/_lib/data.ts`) reads them alongside `leaderboard.json` and the per-run `result.json` to add side-margin notes, callouts, and pull quotes; `kb publish` only joins annotations that git tracks.

```yaml
run_id: <matches outputs/runs/ directory name>
model: <model name only — display label>
harness: <claude | codex | kimi | opencode>
effort: <max | xhigh | "" if default>
problem: <problem dir name, e.g. 01_fp8_gemm>
peak_fraction: <number, or null if not correct>

verdict: <one of: clean | rubric_leak | reward_hack | contamination | interesting | bug>

summary: >
  One-paragraph plain-language description of what's happening in this run,
  written for a leaderboard reader, not a model auditor. Surfaces the headline,
  not the procedural detail.

quotes:
  - file: solution.py | transcript.jsonl | check.log | stderr.log
    lines: [<start>, <end>]   # inclusive; or [N] for a single line
    label: <short caption shown above the pull quote>
    text: |
      <verbatim content from the file at those lines, for the website to render
      directly without re-fetching solution.py>

implication: >
  What this means for how the leaderboard cell should be read. The "so what"
  paragraph. For rubric_leak verdicts, this should explain how the cell's
  number diverges from the skill the problem name implies.
```

Verdicts:

- **clean** — solution does the work the problem name promises; peak_fraction reflects real kernel-engineering skill on the targeted algorithmic challenge. Use this to highlight exemplary implementations, not just to confirm absence-of-cheating.
- **rubric_leak** — solution passes the correctness check but takes a shortcut the rubric was meant to forbid (the rubric just didn't enforce it tightly enough). The peak_fraction is real, but it measures the wrong skill.
- **reward_hack** — solution actively games the eval (precision downcast to inflate "speedup", PyTorch wrapper masquerading as a custom kernel, hardcoded test-input handling, baseline gaming). Cell number should be treated as invalid.
- **contamination** — the agent read or copied another run's archive (`outputs/runs`, `outputs/runs-remote-*`). The peak is not this model's work. Exclude from the board. A same-buffer overwrite pass does not clear this.
- **interesting** — neither leak nor hack, but worth surfacing: novel algorithm choice, surprising failure mode, unique approach, etc.
- **bug** — harness/infra issue distorting the result (timeout, sandbox failure, library missing on test rig). Cell number is unreliable.

Conventions: quotes are VERBATIM from the source file at the specified lines, including indentation, 1-indexed; `summary` is one paragraph, ideally under 60 words; `implication` is optional for `clean`, required for everything else; one YAML document per run, multiple aspects go under `quotes`. Write the YAML by hand; if a tool gets built it should generate this schema, not replace it.

Optional `trajectory` (highlight chart checkpoints): `media/trajectory.py <run_dir>` draws the annotated optimization trajectory for one run (the Fable 5 Mega 18.7x chart). It pulls every in-session `benchmark.py` result, the baseline timing, and wall clock from the transcript by itself. The moves between those points are what the audit already read; list them here so the chart can label them:

```yaml
trajectory:
  - {t: 43.6, score: 1.0, kind: baseline, label: "baseline timed:\n5.47 ms/tok floor"}
  - {t: 98.8, score: 14.38, label: "single cooperative megakernel v1\npasses check, first benchmark"}
  - {t: 144.2, score: 15.94, kind: regress, label: "finer split-K regresses\n-> measured, reverted"}
  - {t: 152.7, score: 18.70, kind: final, label: "final: MLA barrier folds,\n14 barriers/step -> 18.7x"}
```

`t` is minutes since session start (from the transcript timestamps), `score` is what the agent measured at that point (Mega: speedup; Hard/CUDA: roofline fraction), `label` is at most two short lines naming the kernel move, `kind` is `baseline | bench | regress | final` (default `bench`). A checkpoint within 0.6 min of an automatic benchmark point labels that point. Every entry must point at something in the trace; do not invent a move.

## `KBH_` environment variables (single-GPU harness)

`kbtool/tests` fails on a variable read by code that no AGENTS.md documents. The Danger list (which of these change comparability or cost) is in `kbtool/AGENTS.md`. Paths use brace notation to collapse identical copies.

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KBH_AGENT_CONTAINER` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh`; `kbtool/kb/cli.py` | `0`; `kb` forces `1` | Runs supported agents inside the configured Docker image instead of on the host. | Mega has no container path. |
| `KBH_AGENT_CONTAINER_CLAUDE_BIN` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | newest `~/.local/share/claude/versions/*` | Selects the host Claude binary bind-mounted into agent containers. | Container mode only. |
| `KBH_AGENT_CONTAINER_CODEX_NODE` | same | `~/.local/node-v22.14.0-linux-x64` | Selects the Node installation bind-mounted for Codex. | Container mode only. |
| `KBH_AGENT_CONTAINER_CUDA_HOME` | same | `/usr/local/cuda-13.2` | Selects the host CUDA toolkit bind-mounted into containers. | Runner refuses container mode if absent. |
| `KBH_AGENT_CONTAINER_CURSOR_DIR` | same | `~/.local/share/cursor-agent/versions/2026.05.27-fe9a6e2` | Selects the Cursor Agent installation mounted into containers. | Container mode only. |
| `KBH_AGENT_CONTAINER_DROID_BIN` | same | `~/.local/bin/droid` | Selects the Droid binary mounted into containers. | Container mode only. |
| `KBH_AGENT_CONTAINER_GEMINI_DIR` | same | `/usr/lib/node_modules/@google/gemini-cli` | Selects the Gemini CLI installation mounted into containers. | Container mode only. |
| `KBH_AGENT_CONTAINER_GROK_DIR` | same | `~/.grok` | Selects the Grok installation/config tree mounted into containers. | Container mode only. |
| `KBH_AGENT_CONTAINER_IMAGE` | same | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` | Selects the Docker image for agent sessions. | Changes the agent toolchain environment. |
| `KBH_AGENT_CONTAINER_NETWORK` | same | `bridge` | Selects Docker networking for agent sessions. | Included in the generated agent instructions. |
| `KBH_AGENT_CONTAINER_OPENCODE_BIN` | `benchmarks/{hard,cuda,mini}/scripts/{run_hard,warm_opencode_home}.sh` | `~/.opencode/bin/opencode` | Selects the OpenCode binary mounted into containers or used to warm the home template. | Container mode / warm-up only. |
| `KBH_AGENT_CONTAINER_SESSION_LOCK` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `0` | Holds the GPU lock for an entire container session instead of individual GPU-facing commands. | Value `1`; reduces concurrency and avoids wrapper bypass. |
| `KBH_AGENT_CONTAINER_UV_CACHE` | same | `<bench>/outputs/container_uv_cache` | Selects the shared uv cache mounted into agent containers. | Operational; shared across runs in a bench. |
| `KBH_BASELINE_OUT` | `benchmarks/{hard,cuda,mini}/scripts/run_baselines.sh` | `results/problem_baselines.json` | Selects the baseline JSON output file. | Pair with the correct hardware label. |
| `KBH_BENCHMARK_BASELINES` | `benchmarks/{hard,cuda,mini,mega}/src/eval/timing.py` | unset / `0` | Enables opt-in eager, compiled, and SOTA timing variants in addition to `solution`. | Baseline scripts set it to `1`; official solution timing is still emitted first. |
| `KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `KBH_BENCHMARK_TIMEOUT_SECONDS`, else `7200` | Overrides the post-agent benchmark timeout for `02_kda_cutlass`. | The letter-only scan reports this as `KBH_BENCHMARK_TIMEOUT_`; see exclusions. |
| `KBH_BENCHMARK_TIMEOUT_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `1800`; runner uses `7200` for KDA absent its specific override | Sets the post-agent benchmark/regrade timeout. | Timeout affects whether a cell receives a score, not the agent budget. |
| `KBH_BUDGET_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,launch_infra_retries}.sh` | `0` | Sets the sweep/retry budget that launchers export as unprefixed `BUDGET_SECONDS`. | Direct hard/cuda/mini runners do not read it; use `KBH_BUDGET_SECONDS_OVERRIDE` there. |
| `KBH_BUDGET_SECONDS_OVERRIDE` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `0` hard/cuda; `1800` mini | Sets the direct agent-session wall-clock budget. | Changes benchmark protocol; Mega reads unprefixed `BUDGET_SECONDS` instead. |
| `KBH_CHECK_TIMEOUT_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `1800` | Sets the correctness-check timeout. | A timeout prevents successful grading. |
| `KBH_CLAUDE_AUTH` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | inherited environment | When `keychain`, unsets `CLAUDE_CODE_OAUTH_TOKEN` so Claude uses its local login/keychain. | Auth and billing route control. |
| `KBH_CONTAINER_GPUS` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `all` | Supplies Docker's `--gpus` selector for agent containers. | Container mode only. |
| `KBH_CUDA_HOME` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `/usr/local/cuda-13` | Selects the host CUDA toolkit and exports it as `CUDA_HOME` when present. | Changes compiler/toolkit selection. |
| `KBH_GPU` | `scripts/lib/run_harness.sh`; `benchmarks/mega/scripts/run_hard.sh` | `0` | Pins `CUDA_VISIBLE_DEVICES` to one physical index and stamps `gpu_index` / `gpu_name` / `gpu_uuid` into `result.json`. | Must match the deck hardware key. Empty hide is illegal. |
| `KBH_GPU_LOCK` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `<lock-dir>/gpu.lock`; Mega uses `<bench>/outputs/gpu.lock` | Selects the lock file used by GPU-facing wrappers. | Normally derive it via `KBH_GPU_LOCK_DIR`. |
| `KBH_GPU_LOCK_DIR` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh`; `scripts/guarded_sweep.sh`; `benchmarks/mini/scripts/launch_matrix.sh` | `<bench>/outputs/gpu_lock` | Selects the lock domain and therefore which sessions serialize. | A wrong domain permits benchmark contention. |
| `KBH_GPU_LOCK_HELD` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `0` | Makes GPU wrappers reentrant and bypasses reacquiring the lock. | Set internally while a parent owns the lock; manual `1` bypasses serialization. |
| `KBH_GPU_LOCK_LOG` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `<run>/gpu_lock.log` | Tells generated wrappers where to record lock wait/active events. | Set by the runner, not a supported caller override. |
| `KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS` | same | `7200` | Limits how long a wrapper waits for the GPU lock; empty means no wrapper deadline before the runner sets its default. | Lock wait is separate from check/benchmark timeouts. |
| `KBH_HARDWARE` | `benchmarks/{hard,cuda,mini}/scripts/build_v2_leaderboard.py`; `benchmarks/{hard,cuda,mini}/scripts/resweep_deck.sh`; `scripts/{brev_worker,lambda_worker}.sh` | `RTX_PRO_6000`; remote regrade defaults `H100` | Selects hardware peaks/metadata for leaderboard construction or remote regrade. | Must match the physical GPU and deck. |
| `KBH_HARDWARE_LABEL` | `benchmarks/{hard,cuda,mini}/scripts/run_baselines.sh` | `RTX_PRO_6000_BLACKWELL_SM120` | Labels generated baseline records with a hardware identity. | Label only; it does not move work to that GPU. |
| `KBH_HARNESS_CONCURRENCY` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,launch_infra_retries}.sh` | `2` | Caps concurrent sessions per harness/provider worker. | Provider-load control; GPU commands still use the lock. |
| `KBH_INKLING_CONTINUES` | `benchmarks/mega/scripts/run_hard.sh` | `KBH_TINKER_CONTINUES`, else `30` | Caps automatic same-session continuation turns for Mega's OpenRouter Inkling route. | Agent-protocol setting. |
| `KBH_KEEP_RUN_VENV` | `scripts/lib/strip_run_venv.sh` (sourced by `scripts/lib/run_harness.sh`, `benchmarks/mega/scripts/run_hard.sh`, and `benchmarks/{hard,cuda,mini,mega}/scripts/regrade_sequential.sh`) | unset / `0` | When `1`, skips deleting the per-run `repo/.venv` after scoring/regrade. | Default strips venvs (reproducible from `uv.lock`). Debug only; leaving them on can fill local disk. |
| `KBH_MIN_USEFUL_OUTPUT_TOKENS` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `5000` | Sets the token threshold below which a no-solution run is classified as provider early-stop/retryable. | Classification only; does not cap output. |
| `KBH_NUMERIC_STRESS` | `benchmarks/{hard,cuda,mini,mega}/src/eval/numeric_stress.py` | `1` | Disables extra numeric-stress correctness cases when `0`, `false`, or `no`. | Never disable for official runs. |
| `KBH_OPENCODE_BIN` | `benchmarks/{hard,cuda,mini}/scripts/probe_opencode_multistep.sh` | `~/.opencode/bin/opencode` | Selects the OpenCode executable for the multistep probe. | Probe only. |
| `KBH_OPENCODE_CONFIG_FILE` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | unset | Replaces the container's OpenCode config with a supplied file. | Used internally for archive-local provider routes; changes endpoint/model mapping. |
| `KBH_OPENCODE_HOME_TEMPLATE` | `benchmarks/{hard,cuda,mini}/scripts/{run_hard,warm_opencode_home}.sh` | `<bench>/outputs/opencode_home_template` | Selects the prewarmed OpenCode home copied into each run. | Can change installed provider/plugin state. |
| `KBH_OPENCODE_STALL_RETRIES` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `2` retries | Sets how many times stalled OpenCode/Hy3 sessions are resumed or retried. | Agent-protocol setting. |
| `KBH_OPENCODE_STALL_SECONDS` | same | `900`; Hy3 defaults `1500` | Sets the no-log-growth interval before the stall watchdog kills an OpenCode-family attempt. | Affects session completion behavior. |
| `KBH_PREFLIGHT_CLAUDE_MAX_BUDGET_USD` | `benchmarks/{hard,cuda,mini,mega}/scripts/preflight_harnesses.sh` | `0.25` | Caps spend for each Claude-family preflight prompt. | Preflight cost control. |
| `KBH_PREFLIGHT_DIR` | same | timestamped `<bench>/outputs/preflight/...` | Selects the preflight output directory. | Operational only. |
| `KBH_PREFLIGHT_MULTISTEP` | `benchmarks/{hard,cuda,mini}/scripts/preflight_harnesses.sh` | `1` | Enables the OpenCode multistep/tool-result probe after basic preflight. | Value `0` skips it; Mega has no multistep phase. |
| `KBH_PREFLIGHT_MULTISTEP_TIMEOUT_SECONDS` | same | `420` | Sets the multistep probe timeout. | Preflight only. |
| `KBH_PREFLIGHT_ONLY` | same | unset | Filters preflight rows by row name, harness, or model. | Mega's preflight script does not implement this filter. |
| `KBH_PREFLIGHT_PROMPT` | `benchmarks/{hard,cuda,mini,mega}/scripts/preflight_harnesses.sh` | exact sentinel-reply prompt | Replaces the tiny prompt sent to every preflight route. | The result still must contain the sentinel. |
| `KBH_PREFLIGHT_TIMEOUT_SECONDS` | same | `120` | Sets the basic per-route preflight timeout. | Preflight only. |
| `KBH_PROBE_PROBLEM` | `benchmarks/{hard,cuda,mini}/scripts/probe_opencode_multistep.sh` | `05_topk_bitonic` | Selects the problem used by the OpenCode multistep probe. | Probe only. |
| `KBH_OR_PROVIDER` | `scripts/lib/run_harness.sh` | unset | or-fable only: pins the OpenRouter provider (e.g. `novita`) for the whole session via a local body-rewriting proxy (`scripts/lib/or_provider_proxy.py`, upstream override `OR_PROXY_UPSTREAM`). | Provider identity affects comparability — record it with the run. Pinning a non-DeepSeek host also bypasses BYOK, so billing moves to OpenRouter credits. Host mode only (refuses `KBH_AGENT_CONTAINER=1`). |
| `KBH_PROBLEMS` | `benchmarks/{hard,cuda,mini,mega}/scripts/launch_parallel_sweep.sh` | bench-specific problem list | Replaces the problem list for a parallel sweep. | Mega defaults to `problems/02_kimi_linear_decode`; copied single-GPU launchers carry their own literal defaults. |
| `KBH_PROBLEMS_ROOT` | `kbtool/kb/cli.py`; `benchmarks/{hard,cuda,mini}/scripts/{sweep_deck,resweep_deck}.sh` | `problems-rtxpro6000` | Selects the deck root prefixed by CLI/deck sweep commands. | Must match the intended GPU; callers passing a full path can bypass this helper. |
| `KBH_PROPERTY_SEED` | `benchmarks/hard/src/eval/property_stress.py` | random 64-bit integer | Replays the fixed-plus-generated structural correctness plan from a prior `PROPERTY_SEED` log line. | Accepts decimal or `0x` notation. Leave unset for a fresh official check; set only to reproduce a failure. |
| `KBH_PUBLISHED_MANIFEST` | `benchmarks/{hard,cuda,mini}/scripts/build_v2_leaderboard.py` | `results/published_runs.json` | Selects the allowlist of run IDs used for leaderboard construction; empty disables it. | Changes which cells can be published. |
| `KBH_REGRADE_ALLOW_BUSY` | `benchmarks/{hard,cuda,mini,mega}/scripts/regrade_sequential.sh` | `0` | Skips the idle-GPU precondition when `1`. | Debug only; contaminated timing is not publishable. |
| `KBH_REGRADE_DECK` | `benchmarks/{hard,cuda,mega}/scripts/regrade_sequential.sh` | unset | Selects a canonical deck root whose immutable files, `src/`, and locked project environment replace the archived grading surface before grading. | Changes the validation surface and dependencies; fails closed if any canonical component is missing. Mini's regrader does not read it. |
| `KBH_REGRADE_DRY_RUN` | `benchmarks/{hard,cuda,mini,mega}/scripts/regrade_sequential.sh` | `0` | Prints planned regrades without running checks, benchmarks, or writes when `1`. | Operational safety control. |
| `KBH_REGRADE_GPU` | same | `0` | Selects the physical GPU for sequential regrading and idle checks. | Must match the run's hardware/deck. |
| `KBH_RETRY_LABEL` | `benchmarks/{hard,cuda,mini,mega}/scripts/launch_infra_retries.sh` | `retry1` | Sets the suffix/label for an infrastructure retry wave. | Classification/organization only. |
| `KBH_RUNS_DIR` | `benchmarks/{hard,cuda,mini}/scripts/build_v2_leaderboard.py` | `<bench>/outputs/runs` | Selects the run archive scanned to build a leaderboard. | Changes the publication input set. |
| `KBH_RUN_GROUP` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,launch_parallel_sweep,launch_infra_retries}.sh` | empty in runner; timestamped `sweep_*` in launcher | Groups run IDs and sweep artifacts under a common campaign label. | Metadata/organization only. |
| `KBH_SANDBOX` | `benchmarks/mega/scripts/run_hard.sh` | `1` | Enables Mega's `bwrap` filesystem-hiding sandbox when available. | Value `0` exposes the normal host view to the agent. |
| `KBH_SKIP_OPENROUTER` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,preflight_harnesses}.sh` | `0` | Removes OpenRouter-backed rows from sweep/preflight matrices. | Changes matrix coverage. |
| `KBH_STALL_SECONDS` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `0` unless set by a route | Sets the generic container/host no-growth watchdog interval. | Active only with `KBH_STALL_WATCH_LOG`. |
| `KBH_STALL_WATCH_LOG` | same | unset; runner supplies route log | Selects the file whose mtime drives the generic stall watchdog. | Primarily an internal runner channel. |
| `KBH_TIMEOUT_KILL_AFTER_SECONDS` | same | `30` | Sets GNU `timeout --kill-after` grace for agent/check/benchmark processes. | Process-cleanup control. |
| `KBH_TINKER_CONTINUES` | `benchmarks/{hard,mega}/scripts/run_hard.sh` | `30` | Caps automatic same-session continuation turns for Tinker/Inkling routes. | Agent-protocol setting. |
| `KBH_USE_DIRECT_GEMINI` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,preflight_harnesses}.sh` | `0` | Adds the native Gemini CLI row to the generated matrix. | Changes matrix coverage. |
| `KBH_USE_MINIMAX_M3_CLAUDE` | same | `0` | Adds the MiniMax M3 Claude-routed row. | The letter-only scan reports `KBH_USE_MINIMAX_M`; see exclusions. |
| `KBH_USE_NVCF_NEMOTRON` | `benchmarks/{hard,cuda,mini}/scripts/{sweep,launch_parallel_sweep,preflight_harnesses}.sh` | `0` | Adds the NVIDIA NVCF Nemotron route. | Changes matrix coverage and provider billing. |
| `KBH_USE_OPENCODE_ZAI` | `benchmarks/{hard,cuda,mini}/scripts/{launch_parallel_sweep,preflight_harnesses}.sh` | `0` | Adds the diagnostic OpenCode-to-Z.ai row. | Disabled because that adapter has stalled on reasoning streams. |
| `KBH_USE_OPENROUTER_NEMOTRON` | `benchmarks/{hard,cuda,mini}/scripts/{sweep,launch_parallel_sweep,preflight_harnesses}.sh` | `0` | Adds the OpenRouter/DeepInfra-pinned Nemotron row. | Changes matrix coverage and provider billing. |

Scan exclusions (tokens the `kbtool/tests` scan finds that are not caller-facing variables): `KBH_EMPTY` and `KBH_SBX` are Mega runner shell locals used to assemble the `bwrap` command. `KBH_SETTINGS` is a substring of the separate variable `CLAUDE_KBH_SETTINGS`. `KBH_PREFLIGHT_OK` is a response sentinel string, not an environment setting. `KBH_BENCHMARK_TIMEOUT_` and `KBH_USE_MINIMAX_M` are the letter-only regex's truncated matches for `KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS` and `KBH_USE_MINIMAX_M3_CLAUDE`.
