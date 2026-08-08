# KernelBench environment variables

This is the reference for `KB_`, `KBH_`, `KBM_`, and `KBMINI_` variables read by the orchestration and benchmark code. Paths use brace notation to collapse identical copies, for example `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh`.

## Danger

The following variables can change the meaning, comparability, publishability, or cost of a run. Record non-default values with the run and do not publish an experimental result as a canonical cell.

- Deck and hardware identity: `KBH_PROBLEMS_ROOT`, `KBH_HARDWARE`, and `KBH_HARDWARE_LABEL` select problem material or the hardware/roofline label. A mismatch can produce a plausible but meaningless score.
- Time budget: `KBH_BUDGET_SECONDS_OVERRIDE` changes the direct hard/cuda/mini agent budget. Mega's direct runner instead reads unprefixed `BUDGET_SECONDS`; its sweep launchers read `KBH_BUDGET_SECONDS` and export `BUDGET_SECONDS` to each run.
- Regrading: the `KBH_REGRADE_ALLOW_BUSY`, `KBH_REGRADE_DECK`, `KBH_REGRADE_DRY_RUN`, and `KBH_REGRADE_GPU` family controls which GPU and deck are used and whether contention checks or writes occur. `KBH_REGRADE_ALLOW_BUSY=1` can make timings unpublishable.
- Correctness: `KBH_NUMERIC_STRESS=0` removes the single-GPU numeric-stress cases. `KBH_PROPERTY_SEED` pins Hard's generated structural cases instead of drawing a fresh plan. `KBM_NUMERIC_STRESS=0`, `KBM_SKIP_FORBIDDEN=1`, and `KBM_SKIP_GRADE=1` similarly weaken or omit Multi validation. These are debugging/replay controls, never default official-run settings.
- Cloud cost and target: `KB_LAMBDA_BENCH` changes which bench is copied to and run on a Lambda instance. `KB_LAMBDA_TYPE` and `KB_LAMBDA_REGION` affect the billed instance. Always terminate the instance after use.
- Provider identity: `KBH_OR_PROVIDER` changes which OpenRouter host serves an or-fable session (and moves billing off BYOK). Different hosts can serve different quantizations; record the pin with the run.
- Local execution: `KB_ALLOW_LOCAL=1` bypasses the `kb` CLI's remote-worker safety gate.
- Multi hardware: `KBM_ALLOW_DEVICE_MISMATCH=1` permits grading on a heterogeneous or wrong-SKU fabric. `KBM_BACKEND`, `KBM_DEVICE`, and `KBM_WORLD_SIZE` also change the execution topology.
- Multi measurement: `KBM_ALLOW_BUSY`, `KBM_TRIALS`, `KBM_WARMUP`, `KBM_ITERS`, and `KBM_ANCHOR_REPEATS` change contention safeguards or sampling. `KBM_ALLOW_OFF_ROSTER=1` creates an exploratory, non-publishable cell.

## `KB_` orchestration

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KB_ALLOW_LOCAL` | `kbtool/kb/cli.py` | unset / `0` | Lets `kb run` and `kb sweep` proceed on the local host when the normal remote-worker guard would refuse. | Safety override; value must be `1`. |
| `KB_BENCH` | `kbtool/kb/cli.py` | `hard` | Default bench for `kb` commands when no `-b/--bench` flag is passed. | Flag wins over env. Benches: hard, cuda, mini, mega, multi. |
| `KB_BENCH_BANNER` | `scripts/lib/run_harness.sh` | `KERNELBENCH RUN` | Banner line printed at session start. | Pinned by each bench's `run_hard.sh` wrapper; not user-set. |
| `KB_BENCH_DIR` | `scripts/lib/run_harness.sh` | required | Bench root (outputs/, problems, src/) the shared runner operates in. | Pinned by each bench's `run_hard.sh` wrapper; the lib refuses to run without it. |
| `KB_BUDGET_SECONDS_DEFAULT` | `scripts/lib/run_harness.sh` | `0` (unlimited) | Bench-identity wall-clock cap default (mini pins `1800`). | Pinned by the wrapper. Per-run override remains `KBH_BUDGET_SECONDS_OVERRIDE` (see Danger). |
| `KB_BREV_BENCH` | `scripts/brev_worker.sh` | `hard` | Selects the bench directory, remote directory, runner, and sync/pull payload for a Brev worker. | Bench identity changes the problem deck and publication destination. |
| `KB_BREV_PROBLEMS_ROOT` | `scripts/brev_worker.sh` | `problems-h100` | Selects the problem tree synced to and run on a Brev worker. | Deck identity affects comparability. |
| `KB_BREV_RUN_ENV` | `scripts/brev_worker.sh` | empty | Extra `VAR=VALUE` pairs injected into the detached remote run environment (for example, a shared GPU-lock path). | Injected values can change provider, grading, or isolation. Never put secrets here because the string lands in remote argv. |
| `KB_BREV_TYPE` | `scripts/brev_worker.sh` | `hyperstack_H100` | Selects the Brev instance type for `up` when no positional type is supplied. | Can change cost and hardware. |
| `KB_GUARD_RESERVE` | `scripts/openrouter_guard.sh` | `130` USD | Sets the minimum OpenRouter balance required before starting another guarded cell. | Cost-control threshold. |
| `KB_GUARD_SH` | `scripts/guarded_sweep.sh` | `scripts/openrouter_guard.sh` in the repo | Selects the guard executable used between cells. | Override only with a compatible `check` interface. |
| `KB_GUARD_STATE` | `scripts/openrouter_guard.sh` | `~/.kb_openrouter_guard` | Selects the directory containing the guard balance, log, and stop marker. | Persistent operator state. |
| `KB_LAMBDA_BENCH` | `scripts/lambda_worker.sh` | `hard` | Selects the bench directory, remote directory, runner, and sync payload. | `multi` also changes the default problem root and ships `.kbm_env`. |
| `KB_LAMBDA_PROBLEMS_ROOT` | `scripts/lambda_worker.sh` | `problems-h100`; `problems-h100x4` for Multi | Selects the problem tree used by the Lambda worker. | Deck identity affects comparability. |
| `KB_LAMBDA_REGION` | `scripts/lambda_worker.sh` | empty, auto-pick available region | Pins the launch region instead of capacity-based selection. | May affect availability; instance cost follows Lambda pricing. |
| `KB_LAMBDA_RUN_ENV` | `scripts/lambda_worker.sh` | empty | Extra `VAR=VALUE` pairs injected into the remote run's environment by `kb lambda run` (e.g. `KBH_OR_PROVIDER=novita KBH_BUDGET_SECONDS_OVERRIDE=900`). | Whatever it injects can change budget, provider identity, or grading — the injected vars carry their own danger flags. Never put secrets here (lands in remote argv). |
| `KB_LAMBDA_SSH_KEYS` | `scripts/lambda_worker.sh` | current host key name (`macbook` or `anvil`) | Selects the Lambda account SSH key attached at launch. | Lambda's launch API accepts exactly one key here. |
| `KB_LAMBDA_SSH_USER` | `scripts/lambda_worker.sh` | `ubuntu` | Selects the remote SSH/rsync user. | Operational only. |
| `KB_LAMBDA_TORCH_INDEX` | `scripts/lambda_worker.sh` | `https://download.pytorch.org/whl/cu128` | Selects the PyTorch wheel index used during worker bootstrap. | Can change the CUDA/PyTorch runtime. |
| `KB_LAMBDA_TYPE` | `scripts/lambda_worker.sh` | `gpu_1x_h100_sxm5` | Selects the Lambda instance type when no positional type is supplied. | Direct cost and hardware control. |
| `KB_REPO_ROOT` | `kbtool/kb/cli.py` | walk upward from cwd / installed package | Overrides how the `kb` CLI locates the monorepo. | Used only if the path contains `benchmarks/`. |
| `KB_SWEEP_EFFORT` | `scripts/guarded_sweep.sh` | `max` | Sets the reasoning effort passed to every guarded sweep cell. | Changes the agent configuration. |
| `KB_SWEEP_HARNESS` | `scripts/guarded_sweep.sh` | `or-opus` | Selects the harness used by the guarded sweep. | Must be a runner case from [HARNESSES.md](HARNESSES.md). |
| `KB_SWEEP_LOG` | `scripts/guarded_sweep.sh` | `~/guarded_sweep.log` | Selects the guarded sweep's aggregate log file. | Operational only. |
| `KB_SWEEP_MODEL` | `scripts/guarded_sweep.sh` | `anthropic/claude-opus-5` | Selects the model used by the guarded sweep. | Changes the evaluated model. |

## `KBH_` single-GPU harness

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
| `KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `KBH_BENCHMARK_TIMEOUT_SECONDS`, else `7200` | Overrides the post-agent benchmark timeout for `02_kda_cutlass`. | The requested letter-only scan reports this as `KBH_BENCHMARK_TIMEOUT_`; see exclusions. |
| `KBH_BENCHMARK_TIMEOUT_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `1800`; runner uses `7200` for KDA absent its specific override | Sets the post-agent benchmark/regrade timeout. | Timeout affects whether a cell receives a score, not the agent budget. |
| `KBH_BUDGET_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,launch_infra_retries}.sh` | `0` | Sets the sweep/retry budget that launchers export as unprefixed `BUDGET_SECONDS`. | Direct hard/cuda/mini runners do not read it; use `KBH_BUDGET_SECONDS_OVERRIDE` there. |
| `KBH_BUDGET_SECONDS_OVERRIDE` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `0` hard/cuda; `1800` mini | Sets the direct agent-session wall-clock budget. | Changes benchmark protocol; Mega reads unprefixed `BUDGET_SECONDS` instead. |
| `KBH_CHECK_TIMEOUT_SECONDS` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `1800` | Sets the correctness-check timeout. | A timeout prevents successful grading. |
| `KBH_CLAUDE_AUTH` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | inherited environment | When `keychain`, unsets `CLAUDE_CODE_OAUTH_TOKEN` so Claude uses its local login/keychain. | Auth and billing route control. |
| `KBH_CONTAINER_GPUS` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh` | `all` | Supplies Docker's `--gpus` selector for agent containers. | Container mode only. |
| `KBH_CUDA_HOME` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `/usr/local/cuda-13` | Selects the host CUDA toolkit and exports it as `CUDA_HOME` when present. | Changes compiler/toolkit selection. |
| `KBH_GPU_LOCK` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `<lock-dir>/gpu.lock`; Mega uses `<bench>/outputs/gpu.lock` | Selects the lock file used by GPU-facing wrappers. | Normally derive it via `KBH_GPU_LOCK_DIR`. |
| `KBH_GPU_LOCK_DIR` | `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh`; `scripts/guarded_sweep.sh`; `benchmarks/mini/scripts/launch_matrix.sh` | `<bench>/outputs/gpu_lock` | Selects the lock domain and therefore which sessions serialize. | A wrong domain permits benchmark contention. |
| `KBH_GPU_LOCK_HELD` | `benchmarks/{hard,cuda,mini,mega}/scripts/{run_hard,regrade_sequential}.sh` | `0` | Makes GPU wrappers reentrant and bypasses reacquiring the lock. | Set internally while a parent owns the lock; manual `1` bypasses serialization. |
| `KBH_GPU_LOCK_LOG` | `benchmarks/{hard,cuda,mini,mega}/scripts/run_hard.sh` | `<run>/gpu_lock.log` | Tells generated wrappers where to record lock wait/active events. | Set by the runner, not a supported caller override. |
| `KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS` | same | `7200` | Limits how long a wrapper waits for the GPU lock; empty means no wrapper deadline before the runner sets its default. | Lock wait is separate from check/benchmark timeouts. |
| `KBH_HARDWARE` | `benchmarks/{hard,cuda,mini}/scripts/build_v2_leaderboard.py`; `benchmarks/{hard,cuda,mini}/scripts/resweep_deck.sh`; `scripts/{brev_worker,lambda_worker}.sh` | `RTX_PRO_6000`; remote regrade defaults `H100` | Selects hardware peaks/metadata for leaderboard construction or remote regrade. | Must match the physical GPU and deck. |
| `KBH_HARDWARE_LABEL` | `benchmarks/{hard,cuda,mini}/scripts/run_baselines.sh` | `RTX_PRO_6000_BLACKWELL_SM120` | Labels generated baseline records with a hardware identity. | Label only; it does not move work to that GPU. |
| `KBH_HARNESS_CONCURRENCY` | `benchmarks/{hard,cuda,mini,mega}/scripts/{launch_parallel_sweep,launch_infra_retries}.sh` | `2` | Caps concurrent sessions per harness/provider worker. | Provider-load control; GPU commands still use the lock. |
| `KBH_INKLING_CONTINUES` | `benchmarks/mega/scripts/run_hard.sh` | `KBH_TINKER_CONTINUES`, else `30` | Caps automatic same-session continuation turns for Mega's OpenRouter Inkling route. | Agent-protocol setting. |
| `KBH_KEEP_RUN_VENV` | `scripts/lib/strip_run_venv.sh` (sourced by `scripts/lib/run_harness.sh`, `benchmarks/mega/scripts/run_hard.sh`, and `benchmarks/{hard,cuda,mini,mega}/scripts/regrade_sequential.sh`) | unset / `0` | When `1`, skips deleting per-run `repo/.venv` after scoring/regrade. | Default strips venvs (reproducible from `uv.lock`). Debug only; leaving them on can fill local disk. |
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
| `KBH_PROPERTY_SEED` | `benchmarks/hard/src/eval/property_stress.py`; `environments/kernel_hard/_bench/src/eval/property_stress.py` | random 64-bit integer | Replays the fixed-plus-generated structural correctness plan from a prior `PROPERTY_SEED` log line. | Accepts decimal or `0x` notation. Leave unset for a fresh official check; set only to reproduce a failure. |
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


## `KBM_` Multi

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KBM_ALLOW_BUSY` | `benchmarks/multi/scripts/{regrade,measure_anchors}.py` | `0` | Skips the quiet-node precondition for regrades or frozen anchor measurements. | Can contaminate published timings; value must be `1`. |
| `KBM_ALLOW_DEVICE_MISMATCH` | `benchmarks/multi/src/eval/worker.py` | `0` | Allows heterogeneous ranks or a GPU name that does not match the problem hardware key. | Deliberate off-SKU experiments only. |
| `KBM_ALLOW_OFF_ROSTER` | `benchmarks/multi/scripts/sweep_wave.sh` | `0` | Bypasses the frontier roster launch gate. | Result is exploratory and not publishable until rostered. |
| `KBM_ANCHOR_REPEATS` | `benchmarks/multi/scripts/measure_anchors.py` | `5` | Sets the number of frozen-anchor measurements. | Changes anchor statistics used by speedup scores. |
| `KBM_BACKEND` | `benchmarks/multi/src/eval/{launcher,worker}.py`; `benchmarks/multi/scripts/numerics_probe.py` | `nccl` | Selects the distributed backend. | `gloo` is for local CPU validation, not official GPU scores. |
| `KBM_CU` | `benchmarks/multi/scripts/remote_ceiling.sh` | `cu128` | Selects the CUDA-tagged PyTorch wheel for a remote ceiling run. | Runtime/toolchain control. |
| `KBM_DEVICE` | `benchmarks/multi/src/eval/worker.py` | CUDA local rank | Forces CPU when set to `cpu`. | Local correctness smoke only with `gloo`. |
| `KBM_GPU_LOCK_DIR` | `benchmarks/multi/scripts/run_agent.sh` | `<multi>/outputs/gpu_lock` | Selects the node-wide lock domain for all GPU-facing agent commands. | A wrong domain permits 4-GPU fabric contention. |
| `KBM_GPU_LOCK_HELD` | generated wrappers in `benchmarks/multi/scripts/run_agent.sh` | `0` | Makes GPU wrappers reentrant when a parent already owns the lock. | Set internally; manual `1` bypasses serialization. |
| `KBM_ITERS` | `benchmarks/multi/src/eval/worker.py`; `benchmarks/multi/scripts/{nccl_ceiling.py,remote_ceiling.sh}` | problem `num_perf_trials` or `100`; ceiling `50`/remote `50` | Sets measured performance iterations. | Changes timing statistics. |
| `KBM_MASTER_PORT` | `benchmarks/multi/src/eval/launcher.py`; set by `scripts/run_agent.sh` | `29571`; runner assigns a per-run port in `29600..29999` | Selects the local torchrun rendezvous port. | Runner-generated to prevent sibling collisions. |
| `KBM_NUMERIC_STRESS` | `benchmarks/multi/src/eval/stress.py` | `1` | Disables scaled-input stress cases when `0`. | Never disable for official runs. |
| `KBM_OR_CONTEXT` | `benchmarks/multi/scripts/run_agent.sh` | `262144` | Sets the advertised OpenRouter model context limit in the generated OpenCode config. | `opencode-or` only. |
| `KBM_OR_PROVIDER` | same | `Moonshot AI` | Pins the OpenRouter serving provider with fallbacks disabled. | `opencode-or` only; changes serving stack. |
| `KBM_PROTECTED_PROCS` | same | `vllm\|sglang\|trtllm\|nanbeige\|laguna\|dspark\|demon/harness` | Replaces the regex used by wrapped `pkill`/`killall` to protect other tenants. | Safety boundary on shared nodes. |
| `KBM_QUIET_MB` | `benchmarks/multi/scripts/wait_quiet.sh` | `2048` MiB per GPU | Sets the memory threshold below which a GPU counts as quiet. | Used before long waves. |
| `KBM_QUIET_MINUTES` | same | `10` | Sets the required sustained quiet interval. | Samples once per minute. |
| `KBM_QUIET_TIMEOUT_MINUTES` | same | `0` (wait forever) | Sets how long the quiet watcher waits before failing. | Operational only. |
| `KBM_SKIP_FORBIDDEN` | `benchmarks/multi/src/eval/launcher.py` | `0` | Skips the forbidden-import/source tripwire when `1`. | Debug only; changes correctness policy. |
| `KBM_SKIP_GRADE` | `benchmarks/multi/scripts/run_agent.sh` | `0` | Runs the agent but omits post-session check and benchmark when `1`. | Launch-only mode; no publishable score until separately graded. |
| `KBM_TRIALS` | `benchmarks/multi/src/eval/worker.py` | problem `num_correct_trials` or `5` | Sets the number of correctness trials. | Changes validation strength. |
| `KBM_WARMUP` | `benchmarks/multi/src/eval/worker.py`; `benchmarks/multi/scripts/{nccl_ceiling.py,remote_ceiling.sh}` | problem `num_warmup` or `500`; ceiling `200`/remote `200` | Sets warm-up iterations before timing. | Changes timing methodology. |
| `KBM_WORLD_SIZE` | `benchmarks/multi/src/eval/launcher.py` | problem `world_size` or `4` | Overrides torchrun process count. | Official problems expect their declared fabric size. |

## `KBMINI_` Mini

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KBMINI_API_KEY` | `benchmarks/mini/scripts/run_hard.sh` | `local` | Supplies the placeholder/auth key to local LFM OpenAI-compatible routes. | Used by `lfm-opencode`, `lfm-claude`, `hermes`, `pi`, and `lfm-grok`. |
| `KBMINI_BASE_URL` | same | `http://127.0.0.1:8765/v1` | Selects the local OpenAI-compatible inference server. | On a remote eval node this commonly relies on a reverse SSH tunnel to Anvil. |
| `KBMINI_GPUS` | `benchmarks/mini/scripts/launch_matrix.sh` | `0` | Supplies the space-separated GPU IDs used to place matrix workers round-robin. | Each GPU receives its own lock directory. |
| `KBMINI_HERMES_MAX_TURNS` | `benchmarks/mini/scripts/run_hard.sh` | `1000` | Caps Hermes agent turns. | `hermes` only. |
| `KBMINI_HERMES_MODEL` | same | runner model argument | Overrides the model passed to Hermes. | `hermes` only. |
| `KBMINI_HERMES_PROVIDER` | same | `lfm` | Overrides the Hermes provider name. | `hermes` only; endpoint still comes from `KBMINI_BASE_URL`. |
| `KBMINI_PI_API_KEY` | inline Python in `benchmarks/mini/scripts/run_hard.sh` | forced from `KBMINI_API_KEY` | Transfers the local API key into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PI_BASE_URL` | same | forced from `KBMINI_BASE_URL` | Transfers the local endpoint into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PI_MODEL` | same | forced from the runner model argument | Transfers the model ID into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PROBLEMS` | `benchmarks/mini/scripts/{sweep_mini,launch_matrix}.sh` | four `problems-h100/*` Mini problems | Replaces the Mini problem list. | Changes deck coverage. |
| `KBMINI_REPEATS` | `benchmarks/mini/scripts/sweep_mini.sh` | `5` | Sets repeats per model/harness/problem cell. | Changes pass-rate and best-of-N methodology. |
| `KBMINI_SPLIT_BY_PROBLEM` | `benchmarks/mini/scripts/launch_matrix.sh` | `0` | Launches one worker per `(harness, problem)` instead of one per harness when `1`. | Changes concurrency, not the intended result set. |

## Scan exclusions and internal names

The acceptance scan also finds the following tokens, but they are not caller-facing environment variables read from the process environment:

- `KB_BREV_GPU` appears only in a stale comment; `scripts/brev_worker.sh` does not read it.
- `KB_LAMBDA_DEFAULT_KEY` is a substring of the shell-local `_KB_LAMBDA_DEFAULT_KEY`, which is computed from `hostname`.
- `KBH_EMPTY` and `KBH_SBX` are Mega runner shell locals used to assemble the `bwrap` command.
- `KBH_SETTINGS` is a substring of the separate variable `CLAUDE_KBH_SETTINGS`.
- `KBH_PREFLIGHT_OK` is a response sentinel string, not an environment setting.
- `KBH_BENCHMARK_TIMEOUT_` is the letter-only regex's truncated match for the documented `KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS`.
- `KBH_USE_MINIMAX_M` is the letter-only regex's truncated match for the documented `KBH_USE_MINIMAX_M3_CLAUDE`.
- `KBMINI_PI_API_KEY`, `KBMINI_PI_BASE_URL`, and `KBMINI_PI_MODEL` are documented above because the inline Python really reads them, but they are private, one-command pass-through names populated by `run_hard.sh`; callers configure `KBMINI_API_KEY`, `KBMINI_BASE_URL`, and the runner model argument instead.
