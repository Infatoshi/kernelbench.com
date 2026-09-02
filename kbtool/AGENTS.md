# kbtool — driving runs: the `kb` CLI, harness routes, rented GPU workers

`kb` is this uv package: `uv tool install -e ./kbtool` once (editable), or `uv run --project kbtool python -m kb ...` on a fresh box. Repo root: walk up from cwd, or `KB_REPO_ROOT`. Harness routes, runner behaviour, rented nodes, and every `KB_` variable live here. Bench internals and `KBH_`: `benchmarks/hard/AGENTS.md`. Publish gates: root `AGENTS.md`.

```
kb sweep <harness> <model>      # all hard problems, parallel containers, unlimited time
kb publish [hard|cuda|mini|mega] # rebuild leaderboard, viewers, models.json from archives
kb deploy "<msg>"               # publish + commit + push
kb -b <cuda|mini> run|sweep|audit|lint|traces-to-hf ...   # other benches; mega and multi keep their own drivers
kb lambda ... | kb brev ... | kb contamination <bench> | kb push-runs <bench> | kb help
```

Tests: `uv run --project kbtool pytest kbtool/tests/` (the Mac gate; enforces this file against the code). The Mac keeps only this package's `.venv`.

## Harness routes

Rows come from the harness `case` statements in `scripts/lib/run_harness.sh` (hard, cuda, mini; their `run_hard.sh` are thin identity wrappers), `benchmarks/mega/scripts/run_hard.sh` (deliberate fork), and `benchmarks/multi/scripts/run_agent.sh`. A new provider needs a harness branch (copy `kimi-claude`) and a row here; `kbtool/tests` fails on a branch with no row.

“CLI login” = the branch enforces no key; the CLI uses its own login or a noted optional key. Single-GPU runners source `~/.env_vars`; Multi provider routes load the worker's `~/.kbm_env` where noted.

| Harness | Endpoint/transport | Required env key(s) | Benches that have it | Notes/quirks |
| --- | --- | --- | --- | --- |
| `claude` | Native Claude Code to Anthropic | CLI login; optionally `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` | hard, cuda, mini, mega, multi | The branch enforces no key. Reasoning effort is forwarded; single-GPU settings disable fast mode and enable thinking. |
| `ccr-claude` | Claude Code to local ccr-rust Anthropic-compatible router at `CCR_BASE_URL` or `http://127.0.0.1:3456` | None in runner; ccr-rust must already have upstream auth | hard, cuda, mini, mega | The model argument is the upstream provider's model ID. |
| `zai-claude` | Claude Code to Z.ai Anthropic API at `https://api.z.ai/api/anthropic` | `ZAI_API_KEY` | hard, cuda, mini, mega, multi | Claude aliases are remapped to the requested model. Multi reads the key from `~/.kbm_env`. |
| `minimax-claude` | Claude Code to `MINIMAX_ANTHROPIC_BASE_URL` or `https://api.minimax.io/anthropic` | `MINIMAX_API_KEY` | hard, cuda, mini, mega | Keeps MiniMax routing separate from native Claude defaults and remaps Claude aliases. |
| `kimi-claude` | Claude Code to `KIMI_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `KIMI_API_KEY` | hard, cuda, mini, mega, multi | Kimi's coding route expects thinking mode. Multi reads the key from `~/.kbm_env`. |
| `kinetic-claude` | Claude Code to `KINETIC_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `MOONSHOT_API_KEY` | hard, cuda, mini, mega | For `kinetic-0715`; `KIMI_API_KEY` is not interchangeable. Pins max effort, disables tool search, and pins the subagent model. |
| `or-fable` / `openrouter-fable` / `or-opus` / `openrouter-opus` | Claude Code to OpenRouter's Anthropic API at `OR_FABLE_BASE_URL` or `https://openrouter.ai/api` | `OPENROUTER_API_KEY` | hard, cuda, mini (shared runner), mega | Maps bare Fable/Opus aliases to Anthropic slugs and requests one-hour prompt caching. Other OpenRouter slugs pass through unchanged; Qwen 3.8 Max is `qwen/qwen3.8-max` and supports `xhigh` effort. |
| `longcat-claude` | Claude Code to `LONGCAT_ANTHROPIC_BASE_URL` or `https://api.longcat.chat/anthropic` | `LONGCAT_API_KEY` | hard, cuda, mini, mega | Claude aliases map to LongCat-2.0; the route raises the default output limit. |
| `hy3` / `hy3-claude` | OpenCode to Tencent TokenHub's OpenAI-compatible API at `HY3_TOKENHUB_BASE_URL` or `https://tokenhub.tencentmaas.com/v1` | `TENCENT_API_KEY` | hard, cuda, mini, mega | `hy3-claude` is a legacy name, not Claude Code. Only model `hy3` is accepted; preview/OpenRouter slugs are rejected; effort maps to `high` or `no_think`. |
| `tinker` / `inkling` (shared-runner branch) | OpenCode to Tinker's OpenAI-compatible API at `TINKER_BASE_URL` or `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | hard, cuda, mini (shared runner) | Both labels serve `thinkingmachines/Inkling` directly and auto-continue the same session when it asks to proceed. |
| `tinker` (Mega branch) | OpenCode to Tinker, same endpoint as above | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | mega | Direct Tinker route with bounded same-session auto-continuation. |
| `inkling` / `opencode-inkling` (Mega branch) | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | mega | Important alias collision: `inkling` means direct Tinker on Hard but OpenRouter on Mega. Mega defaults to high reasoning and bounded auto-continuation. |
| `deepseek-claude` | Claude Code to `DEEPSEEK_ANTHROPIC_BASE_URL` or `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | hard, cuda, mini, mega, multi | Intended for DeepSeek V4 Pro/Flash. Multi uses the fixed URL and loads the key from `~/.kbm_env`. |
| `qwen-claude` | Claude Code to `QWEN_ANTHROPIC_BASE_URL` or the token-plan endpoint `https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic` | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` | hard, cuda, mini, mega | The token-plan route serves `qwen3.8-max`; a DashScope key requires the Model Studio URL override. Tool search is disabled, effort is `xhigh`, context is 983,616 tokens, and the subagent model is pinned. Mega's post-run completeness selector omits this label. |
| `codex` | Native Codex CLI to its configured OpenAI transport | CLI login or `OPENAI_API_KEY` | hard, cuda, mini, mega, multi | Forwards reasoning effort and archives the rich session JSONL by parsed session ID. |
| `kimi` | Native Kimi CLI | Kimi CLI login/config | hard, cuda, mini, mega | The branch does not pass the runner's model argument; it invokes `kimi -w ... --print`. |
| `droid` | Factory Droid CLI to its configured provider | Droid login; container may receive `FACTORY_API_KEY` and `DROID_API_KEY` | hard, cuda, mini, mega | Forwards reasoning effort. Provider endpoint depends on Droid configuration. |
| `gemini` | Native Gemini CLI | Gemini CLI login or `GEMINI_API_KEY` | hard, cuda, mini, mega | Runs from the problem directory with yolo approval. |
| `cursor` | Cursor Agent CLI (`agent`) | Cursor CLI login; container may receive `CURSOR_API_KEY` | hard, cuda, mini, mega | The executable is `agent`, not `cursor`. |
| `grok` | Native Grok CLI | Grok CLI login; optionally `XAI_API_KEY` or `GROK_API_KEY` | hard, cuda, mini, mega, multi | Uses the top-level headless command, not `grok agent`; reasoning effort is forwarded. |
| `opencode` | OpenCode to the provider/model encoded in the model argument | Provider-dependent; container forwards `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GEMINI_API_KEY`, and `SAKANA_API_KEY` | hard, cuda, mini, mega | Generic OpenAI-shaped route; model syntax is `provider/model`. Container mode has a stall watchdog. |
| `opencode-nemotron` | OpenCode to OpenRouter `/api/v1`, pinned to DeepInfra with fallbacks disabled | `OPENROUTER_API_KEY` | hard, cuda, mini | Preferred Nemotron route; uses an archive-local OpenCode config so the serving stack cannot drift. |
| `nvcf-nemotron` | OpenCode to a per-run localhost OpenAI adapter, then NVIDIA NVCF | One of `NGC_API_KEY`, `NVIDIA_API_KEY`, or `NVCF_API_KEY` | hard, cuda, mini | NVCF is not OpenAI-compatible directly; the runner starts the adapter. Diagnostic route. |
| `lfm-opencode` | OpenCode to local vLLM at `KBMINI_BASE_URL` or `http://127.0.0.1:8765/v1` | `KBMINI_API_KEY` (defaults to `local`) | mini (dispatchable on hard/cuda via the shared runner, but meaningful only with mini's local serving) | Uses an archive-local OpenCode config. A real secret is not normally required for the local server. |
| `lfm-claude` | Claude Code to local ccr-rust at `CCR_BASE_URL` or `http://127.0.0.1:3456`, then local vLLM | `KBMINI_API_KEY` (defaults to `local`); ccr-rust must already be running | mini | Passes the local key as `ANTHROPIC_API_KEY`. |
| `hermes` | Nous Hermes Agent to local vLLM through `OPENAI_BASE_URL=KBMINI_BASE_URL` | `KBMINI_API_KEY` (defaults to `local`) | mini | `KBMINI_HERMES_PROVIDER`, `KBMINI_HERMES_MODEL`, and `KBMINI_HERMES_MAX_TURNS` override its invocation. |
| `pi` | badlogic pi through a generated `lfm` OpenAI-completions provider to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | mini | Additively updates `~/.pi/agent/models.json`; `--no-session` avoids a headless hang. |
| `lfm-grok` | Grok CLI custom `chat_completions` model to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | mini | Additively appends a model block to `~/.grok/config.toml`. |
| `opencode-or` | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` from `~/.kbm_env` | multi | Pins `KBM_OR_PROVIDER` with fallbacks disabled. The adapter has stalled intermittently, and the branch ignores the reasoning-effort argument. |

### Route notes

- Always container mode (`KBH_AGENT_CONTAINER=1`): isolated per-run workspace, native GPU, sessions overlap while GPU commands serialize through the lock.
- Claude-Code-routed providers (`zai-claude`, `minimax-claude`, `kimi-claude`, `deepseek-claude`, `qwen-claude`, `longcat-claude`) mirror each other; to add one, copy the `kimi-claude` branch in `scripts/lib/run_harness.sh` and add a row above. Rationale: opencode is a strong harness but its `@ai-sdk/openai-compatible` transport stalls intermittently (about a third to a half of sessions); routing through Claude Code to the provider's Anthropic endpoint bypasses that adapter.
- Qwen: the token-plan MaaS endpoint with `QWEN_API_KEY` is the paid coding-plan route and serves production `qwen3.8-max` (verified 2026-08-03). `DASHSCOPE_API_KEY` plus `QWEN_ANTHROPIC_BASE_URL=https://dashscope-intl.aliyuncs.com/apps/anthropic` reaches Model Studio pay-as-you-go. Qwen 3.8 Max is also on OpenRouter as `qwen/qwen3.8-max` via `or-fable` at `xhigh`.
- Tencent Hy3 (`hy3`): official TokenHub route, OpenAI-compatible only, not Claude Code and not OpenRouter. Model `hy3` only (`hy3-preview` / `tencent/hy3-preview` are retired). Defaults `reasoning_effort=high` (`no_think` / `low` for fast mode); output ceiling up to 262k per Tencent's eval guide. `uv run kbh run hy3 hy3 problems-rtxpro6000/01_fp8_gemm`.
- Nemotron 3 Ultra is scored through `opencode-nemotron` (OpenRouter pinned to DeepInfra, `allow_fallbacks=false`), not Claude Code via CCR (adds a translation layer) or Droid (not the native endpoint). NVCF is diagnostic only; Ultra was observed degrading and 504ing there. Enable in broad preflight/sweeps with `KBH_USE_OPENROUTER_NEMOTRON=1`; target only that row with `KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra ./scripts/preflight_harnesses.sh`. The mega matrix omits the Nemotron row.
- Z.ai GLM via Claude Code (`zai-claude`): endpoint `https://api.z.ai/api/anthropic` (the OpenAI-compatible coding endpoint is for Droid/Factory). The branch sets `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `CLAUDE_CODE_MAX_RETRIES=1000000`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, `ZAI_CLAUDE_HAIKU_MODEL=<model>` so Haiku / Explore / subagent calls map to the same model, and passes `--disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion`.
- MiniMax (`minimax-claude`, model `MiniMax-M3`): key in `~/.env_vars`, enable in broad sweeps with `KBH_USE_MINIMAX_M3_CLAUDE=1`.
- Native Claude Code runs pass `--settings '{"fastMode":false,"alwaysThinkingEnabled":true}'`; the Opus matrix also passes `--effort max`. A Claude rerun with fast mode or a lower effort tier is not comparable.
- Grok headless route is top-level `grok --cwd <workspace> --output-format streaming-json -p <prompt>`; `grok agent` does not accept those flags. Cursor's CLI binary is `agent`. The harness needs the node Codex binary at `~/.local/node-*/bin/codex`; the `codex` shell alias does not expand in non-interactive scripts.
- zsh: quote model strings containing `[]` or `:` or the shell globs them.

### Runner behaviour every route shares

- Timeout starts after the GPU lock. Use `run_gpu_locked_timeout` for `check.py`/`benchmark.py`; wrapping `timeout` outside `uv run` fails queued rows that were only waiting for `outputs/gpu.lock`.
- Never hide CUDA from the agent and never append instructions that prohibit checking, benchmarking, or profiling. If `REAL_UV=$(which uv)` appears in a model command it must not resolve back to the per-run wrapper or the lock owner hangs.
- Claude-family harnesses launch from the archive-local `$PROBLEM_DIR`, not repo root with `--add-dir`; otherwise the model writes `problems/<name>/solution.py` in the source tree and the archive records `no_solution`.
- Provider credit/rate detection reads explicit CLI/API error events and stderr only, never assistant text or tool output (models read AGENTS.md, `run_harness.sh`, and old artifacts containing the trigger words), and applies only to rows without a solution. Match credit-specific strings; plain `overage` false-positives on `coverage`.
- Never pass provider keys via `timeout env KEY=... claude`; that puts the key in argv. Export inside the subshell.
- `benchmark.py` scores `variant=solution` first. Eager / compiled / SOTA diagnostics are opt-in via `KBH_BENCHMARK_BASELINES=1` and emit `benchmark_event` lines for audits.

### Workspace, caches, GPU lock

`scripts/lib/run_harness.sh` gives every run a repo-shaped workspace under `outputs/runs/<run_id>/repo/problems/<problem>/`: immutable problem files copied from the source tree, `src/` symlinked, local copies of `pyproject.toml`, `uv.lock`, `.python-version` so the agent can mutate deps inside the archive. Per-run caches: `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `TMPDIR`/`TEMP`/`TMP` all under `$RUN_DIR`. The harness prepends `$RUN_DIR/bin` to `PATH` and wraps `uv`, `python`, `python3`, `nvidia-smi`, `ncu`, `nsys`, `nvcc`; the wrappers take `outputs/gpu.lock` (bounded `flock -w 5` retry), log to `$RUN_DIR/gpu_lock.log`, then exec the real binary. `KBH_GPU_LOCK_HELD=1` makes the wrapper reentrant so `nvcc` under `benchmark.py` does not deadlock.

The default lock is per bench (`benchmarks/{hard,cuda,mega}/outputs/gpu_lock/gpu.lock`), not machine-wide, so hard + cuda + mega agents can hold the GPU at once, and absolute-path `python`/`nvcc` bypass the wrapper. Parallel sessions are fine for the development flywheel; mid-session `result.json` numbers are never publish-grade (see the regrade gate in the root `AGENTS.md`). Use a machine-wide `KBH_GPU_LOCK_DIR` for a quiet regrade box. The lock only governs children launched through the runner. Transcript usage extraction is CPU-only and bypasses the lock.

`result.json` carries agent wall time, check/benchmark wall time and exit codes, token/cache/reasoning usage, GPU lock wait/active totals (`scripts/summarize_runs.py`), `failure_reason`, and `retryable_infra_failure`. No-solution rows under 5,000 output tokens are `provider_early_stop` and retryable. The site shows these instead of one red cell.

### Broad sweeps

`scripts/launch_parallel_sweep.sh` defaults to `KBH_HARNESS_CONCURRENCY=2` per harness; raise only after preflight proves quota. Workers are per harness: a problem-major loop head-of-line blocks (Codex holding its two slots keeps freed Cursor/Gemini/OpenCode slots idle). `./scripts/preflight_harnesses.sh` sends tiny prompts through the matrix and fails fast on auth/quota/route problems. After a sweep, `./scripts/launch_infra_retries.sh <run_group>` reruns only `retryable_infra_failure=true` rows; retry rows must keep empty effort fields and pass full `problems/<name>` paths or the problem slides into the effort column. `KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1` runs the non-OpenRouter rows plus Gemini direct when OpenRouter is depleted. Aborting: kill the launcher process group, then verify by cwd; some CLIs spawn orphaned timeout groups.

## Rented GPU workers (Lambda, Brev, Verda)

GPU eval sessions run on rented workers, not on anvil. Bring a node up, make it able to run torch and ncu, run cells, pull archives back, tear it down. Multi's 4xH100 node specifics are in `benchmarks/multi/AGENTS.md`.

### Lambda Cloud

Zach / Lambda sponsored $10k of Cloud credits (2026-07) for Hard / Mega / CUDA / Multi (RTX PRO 6000, H100, B200). Credits show only in the console: Settings -> Billing -> Credits (account `elliot@arledge.net`). Tag Lambda on X when posting runs from it.

- Auth: `LAMBDA_API_KEY` in `~/.env_vars` (keep the legacy typo `LAMDBA_API_KEY` in sync; kimi-sweep reads it). Mint at https://cloud.lambda.ai/api-keys/cloud-api. Keep Mac and anvil `~/.env_vars` in sync.
- SSH keys on the account: `macbook` and `anvil` (each host's `~/.ssh/id_ed25519.pub`). `lambda_worker.sh up` attaches ONE key, the current host's, because the launch API rejects more than one (observed 2026-07-21). Override with `KB_LAMBDA_SSH_KEYS`.
- The worker scripts use the Cloud API via curl; nothing needs brew. Optional Mac-only community CLI: `brew install strand-ai/tap/lambda-cli`.

```
kb lambda list                         # capacity by type
kb lambda ls                           # running instances
kb lambda up <name> [type] [region]    # default type gpu_1x_h100_sxm5
kb lambda sync <name>                  # thin bench + allowlisted keys (preserves the node's torch-index patch)
kb lambda bootstrap <name> [--agents]  # uv + torch; --agents = agent CLIs; ncu on PATH + NVreg_RestrictProfilingToAdminUsers=0
kb lambda run <name> <harness> <model> <problem> [effort]
kb lambda pull <name>                  # -> benchmarks/hard/outputs/runs-lambda-<name>/ (excludes the node's venv)
kb lambda regrade <name> <run_id> [runs_dir]   # sequential isolated re-grade on the node
kb lambda down <name>                  # terminate + poll until gone
kb lambda ssh <name> [cmd...]
```

Or `./scripts/lambda_worker.sh ...` from the repo root. Env overrides: `KB_LAMBDA_TYPE`, `KB_LAMBDA_REGION`, `KB_LAMBDA_SSH_KEYS`, `KB_LAMBDA_PROBLEMS_ROOT` (default `problems-h100`; `problems-h100x4` when `KB_LAMBDA_BENCH=multi`), `KB_LAMBDA_BENCH` (default `hard`; `multi`, `cuda`, `mega` point the worker at another bench).

Multi-GPU / NVLink work can use Lambda `gpu_8x_h100_sxm5` / `gpu_8x_b200_sxm6` when `kb lambda list` shows capacity, or Brev.

### Bootstrap order on a fresh node

1. A node can pass `nvcc` checks and still not run torch. Lambda's stock image ships driver 570 and no NVIDIA CUDA apt repo, so `apt-get install cuda-toolkit-13-0` returns rc=100 while the driver install succeeds from Lambda's own archive. The node then has a driver but no `/usr/local/cuda-13.0`, or a toolkit with a driver too old for the cu130 wheel. Both failures are silent. Add the repo first (`cuda-keyring_1.1-1_all.deb` from `developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/`), install `cuda-toolkit-13-0` and `nvidia-driver-595`, reboot.
2. Gate every launch on `torch.cuda.is_available()` after every `uv` command. `nvcc --version` is not a CUDA probe. `nvcc` must compile `#include <cuda_runtime.h>` with `cudafe++` next to it.
3. Hyperstack / Shadeform 8xH100 nodes ship driver CUDA 12.8; the default `uv pip install torch` pulls a cu130 wheel that cannot see the GPUs ("driver too old"). Install the matched build: `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0`. Bake uv, the repo, and this wheel into a prebaked image so node time is not spent on reinstalls. `kb lambda sync` preserves the node's patched pyproject/uv.lock (a re-sync once shipped the Mac's cu130 lock over the node's cu128 one and every later graded env died at check time, 2026-08-01).
4. Stock Lambda SXM5 image has `nvcc` and no `ncu` until `apt-get install nsight-compute`.

### ncu on rented VMs (closed 2026-08-24)

`ERR_NVGPUCTRPERM` is an admin gate, not missing bare metal. Lambda `gpu_1x_h100_pcie`, Lambda `gpu_1x_h100_sxm5`, and Verda `4RTXPRO6000.120V` are all KVM VMs with `RmProfilingAdminOnly: 1`. Root `ncu` wrote a real `smsp__cycles_elapsed.avg` on all three; the Lambda `ubuntu` user has passwordless `sudo -n`. Lambda DeepTalk (Hayden, 2025-10-01) saying Nsight is unsupported on Cloud VMs is stale for these SKUs; bare metal is not required. Brev -> Lambda should match; Brev -> AWS/GCP can block counters even as root, so probe before trusting.

Do not give the agent blanket sudo. Default remote sessions use `KBH_AGENT_CONTAINER=1`: `ncu` runs inside Docker as uid 1000 with `--cap-add CAP_PERFMON`, `--user $(id -u):$(id -g)`, and `--security-opt no-new-privileges`. `--cap-add CAP_PERFMON` alone does not clear the error. On every rented box, before the first agent session: (1) `ncu` on PATH, (2) `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-ncu.conf`, then reboot or reload `nvidia` with zero GPU users. `kb lambda bootstrap` does both. Host mode (`KBH_AGENT_CONTAINER=0`, some hy3, mega bwrap) may wrap only `ncu`/`nsys` with `sudo -n` via a NOPASSWD sudoers line limited to those binaries, never `/bin/bash`. Anvil and gamer are local compute; do not change their driver policy for this.

### Running and pulling back

- Bootstrap long work with `nohup` or `systemd-run` on the node; the laptop SSH is a probe. A live tmux, an SSH 255, or a nohup that inits then exits is not "launched". Wave complete means a `result.json` on the Mac for every cell.
- Incremental pullback every 10 minutes, excluding the node's venv. Before any `rsync`/`scp`, `du -sb` the remote paths and print full vs tiny. Tiny set = `result.json`, `solution.py`, `gpu`, `check.log`, `benchmark.log`, sidecar `*.cu`/`*.cuh`/`kernels.py`; that is the audit, draft, and `kb publish` input. Do not pull `transcript.jsonl`, `agent_home/`, `repo/`, or `cache/` until the run is being converted for HF. Over 20 MB, state the size and wait; over 1 GB, refuse until the user has seen the number. `ssh HOST cat .../result.json` when one file answers. (August 2026 wave: 21 full dirs = 12.7 GiB, tiny set = 0.5 MB, tiny + transcripts = 1.0 GiB, one DeepSeek TopK jsonl was 241 MB.)
- Point every worker's lock, log, and archive paths at in-repo locations and `kb lambda pull` its archives back into `outputs/runs/` before teardown. An archive stranded outside the repo is invisible to `kb publish`, `kb contamination`, and re-grades.
- Quiet GPU for regrade means zero compute PIDs, not "0% util with leftover VRAM". An `nvidia-smi` timeout means a wedged driver: stop.
- Contamination scan must rebuild grok streaming-json and read `~/.grok/sessions/*/chat_history.jsonl`. An empty `outputs/runs` is not a clean box while a CLI session store remains.

### Teardown

- Always `kb lambda down <name>` when done; idle nodes bill the credits. Confirm `kb lambda ls` no longer lists it.
- Kill by pidfile, never `pkill -f` (it matches its own ssh argv and kills the session with exit 255).
- Brev: `brev delete <name>` has a hidden interactive confirmation that silently hangs with no TTY, and `brev stop` / `yes | brev delete` no-op. Teardown goes through `scripts/brev_teardown.sh <name>`, which gives brev a pseudo-TTY, feeds it `y`, and polls `brev ls` until the instance is gone (it branches to `expect` on macOS because `script -qec ... <<< y` silently does nothing there). A forgotten 8xH100 node bills about $23/hr.

## Environment variables

Reference for the `KB_` orchestration variables. `KBH_` (single-GPU harness) is in `benchmarks/hard/AGENTS.md`, `KBM_` in `benchmarks/multi/AGENTS.md`, `KBMINI_` in `benchmarks/mini/AGENTS.md`. `kbtool/tests` fails on a variable read by code that no AGENTS.md documents. Paths use brace notation to collapse identical copies, for example `benchmarks/{hard,cuda,mini}/scripts/run_hard.sh`.

### Danger

The following variables can change the meaning, comparability, publishability, or cost of a run. Record non-default values with the run and do not publish an experimental result as a canonical cell.

- Deck and hardware identity: `KBH_PROBLEMS_ROOT`, `KBH_HARDWARE`, `KBH_HARDWARE_LABEL`, and `KBH_GPU` select problem material or the physical GPU / roofline label. A mismatch can produce a plausible but meaningless score. `KBH_GPU` also exports `CUDA_VISIBLE_DEVICES`; empty hide is illegal.
- Time budget: `KBH_BUDGET_SECONDS_OVERRIDE` changes the direct hard/cuda/mini agent budget. Mega's direct runner instead reads unprefixed `BUDGET_SECONDS`; its sweep launchers read `KBH_BUDGET_SECONDS` and export `BUDGET_SECONDS` to each run.
- Regrading: the `KBH_REGRADE_ALLOW_BUSY`, `KBH_REGRADE_DECK`, `KBH_REGRADE_DRY_RUN`, and `KBH_REGRADE_GPU` family controls which GPU and deck are used and whether contention checks or writes occur. `KBH_REGRADE_ALLOW_BUSY=1` can make timings unpublishable.
- Correctness: `KBH_NUMERIC_STRESS=0` removes the single-GPU numeric-stress cases. `KBH_PROPERTY_SEED` pins Hard's generated structural cases instead of drawing a fresh plan. `KBM_NUMERIC_STRESS=0`, `KBM_SKIP_FORBIDDEN=1`, and `KBM_SKIP_GRADE=1` similarly weaken or omit Multi validation. These are debugging/replay controls, never default official-run settings.
- Cloud cost and target: `KB_LAMBDA_BENCH` changes which bench is copied to and run on a Lambda instance. `KB_LAMBDA_TYPE` and `KB_LAMBDA_REGION` affect the billed instance. Always terminate the instance after use.
- Provider identity: `KBH_OR_PROVIDER` changes which OpenRouter host serves an or-fable session (and moves billing off BYOK). Different hosts can serve different quantizations; record the pin with the run.
- Local execution: `KB_ALLOW_LOCAL=1` bypasses the `kb` CLI's remote-worker safety gate.
- Multi hardware: `KBM_ALLOW_DEVICE_MISMATCH=1` permits grading on a heterogeneous or wrong-SKU fabric. `KBM_BACKEND`, `KBM_DEVICE`, and `KBM_WORLD_SIZE` also change the execution topology.
- Multi measurement: `KBM_ALLOW_BUSY`, `KBM_TRIALS`, `KBM_WARMUP`, `KBM_ITERS`, and `KBM_ANCHOR_REPEATS` change contention safeguards or sampling. `KBM_ALLOW_OFF_ROSTER=1` creates an exploratory, non-publishable cell.

### `KB_` orchestration

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
| `KB_REPO_ROOT` | `kbtool/kb/cli.py` | walk up from cwd | Overrides where `kb` finds the monorepo. | Honoured only if it contains `benchmarks/`. |
| `KB_SWEEP_EFFORT` | `scripts/guarded_sweep.sh` | `max` | Sets the reasoning effort passed to every guarded sweep cell. | Changes the agent configuration. |
| `KB_SWEEP_HARNESS` | `scripts/guarded_sweep.sh` | `or-opus` | Selects the harness used by the guarded sweep. | Must be a runner case from the harness table above. |
| `KB_SWEEP_LOG` | `scripts/guarded_sweep.sh` | `~/guarded_sweep.log` | Selects the guarded sweep's aggregate log file. | Operational only. |
| `KB_SWEEP_MODEL` | `scripts/guarded_sweep.sh` | `anthropic/claude-opus-5` | Selects the model used by the guarded sweep. | Changes the evaluated model. |

Scan exclusions (tokens the `kbtool/tests` scan finds that are not caller-facing variables): `KB_BREV_GPU` appears only in a stale comment; `scripts/brev_worker.sh` does not read it. `KB_LAMBDA_DEFAULT_KEY` is a substring of the shell-local `_KB_LAMBDA_DEFAULT_KEY`, which is computed from `hostname`.
