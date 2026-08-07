# KernelBench harnesses

These tables are the single source of truth for runner harness names, transports, and credentials; `AGENTS.md` links here instead of duplicating branch details. Rows come from the primary harness `case` statements in the shared single-GPU runner `scripts/lib/run_harness.sh` (dispatch for hard, cuda, and mini — their `run_hard.sh` files are thin identity wrappers), `benchmarks/mega/scripts/run_hard.sh` (deliberate fork), and `benchmarks/multi/scripts/run_agent.sh`.

“CLI login” means the branch does not enforce one environment key: the named CLI may use its existing login/config or one of the noted optional keys. Single-GPU runners source `~/.env_vars`; Multi provider routes load the worker's `~/.kbm_env` where noted.

## Immutable submission replay

Hard, CUDA, and Mini freeze agent-authored `solution.py` plus every regular
sidecar into `submission_bundle/` immediately after the agent exits and before
importing any submission code. The canonical manifest records each relative
path, mode, size, and SHA-256 digest; `submission_bundle_sha256` in
`result.json` binds later verification to the exact manifest captured at that
point. Root benchmark templates, virtual environments, and generated cache
directories are never submission artifacts.

Correctness and performance run in separate, newly extracted repo-shaped
workspaces. Each extraction is verified against the captured digest, receives
fresh compiler/JIT caches, and gets trusted templates, `src/`,
`pyproject.toml`, and `uv.lock` from the bench rather than the agent workspace.
The runner hashes that trusted surface before and after each phase and records
only one standalone `PASS` marker and one complete final score line. An
in-process completion guard rejects ordinary exceptions and `SystemExit` after
a forged marker. It is advisory, not completion provenance: plain Python in
`solution.py` can walk the worker frames, send the one exposed receipt, print a
forged marker or metric, and call `os._exit(0)`. A mutation, duplicate marker,
nonzero benchmark exit, bundle mismatch, or unavailable isolation stack still
fails the replay, but replay success alone does not authorize publication.

Replay fails closed unless util-linux `unshare` can create user, mount, PID,
network, IPC, and UTS namespaces together and enter a private-root backend
identified as `unshare-user-mount-pid-net-private-root-v1`. The runner builds an
empty tmpfs root, mounts only `/usr`, minimal loader/NSS files, read-only sysfs,
the selected GPU devices, the exact clean replay stage, CUDA/Python runtime
roots, and pinned tools, then uses `pivot_root` and detaches the old host root.
The root and trusted surfaces are read-only; only the replay stage, bounded
private `/tmp`, and bounded private `/dev/shm` are writable. This prevents
alternate hard links from recovering host Unix sockets and leaves unrelated
archives, operator state, `/var`, and `/opt` absent. Replay also drops
capabilities, closes inherited descriptors, bounds output size, fixes the
hostname, and uses a clean `env -i` environment with a new HOME and empty
private caches. Dependency sync completes before entry, and grading runs
frozen/offline. Submission code still executes inside the checker process and
inherits its stdout/stderr descriptors, so neither the receipt nor captured
log text proves the checker's internal control flow. A true completion
authority would require a trusted coordinator that never imports submission
code and independently validates data returned by an untrusted executor; this
runner does not claim that boundary.

The archive still exposes `solution.py` and `scratch/` for older audit tools,
but those are compatibility projections made from a final verified extraction,
not files copied after executing the submission. `result.json` is the run's
commit marker: it is fsynced and atomically replaced only after the bundle,
logs, projections, cleanup, and venv stripping are complete. The
`submission_replay` object records status, digest, fresh-extraction/cache flags,
stage count, normal-return supervision, and the network/mount/PID/environment
controls used for replay.

Hard, CUDA, and Mini agent sessions themselves are container-only. Each run
gets a private writable uv cache, and containers receive only the exact shared
GPU lock inode; the lock directory, other run archives, source checkout, and
host runtime sockets are not mounted. Routes that only have a localhost/host
implementation (`ccr-claude`, native `kimi`, the `lfm-*` routes, `hermes`,
`pi`, and `nvcf-nemotron`) fail closed in the shared runner until they gain a
container implementation.

Sequential regrades apply the same rules for bundle-aware runs and bind the
active top-level metrics to the same digest in `regrade`. Historical archives
before `20260808_000000` are explicitly grandfathered because they cannot be
retroactively given trustworthy capture-time provenance. Post-cutover shared-
runner archives cannot become legacy by deleting metadata. Mega's deliberate
runner fork is outside this rollout and retains its existing legacy behavior.
Hard, CUDA, and Mini publication also requires a bounded, regular, nonempty
curation manifest. Every selected bundle-era solution must have an exact
`<run_id>.yaml` annotation whose internal ID matches and which explicitly sets
`verdict: clean` and boolean `publish_grade: true`; `board_eligible: false`
always vetoes publication. Publication then scans the exact digest-bound
`solution.py` again and automatically rejects every high-confidence `HACK`
tripwire, including frame walking and `os._exit`; lower-confidence `FLAG`
findings remain manual-review prompts. These static checks can miss obfuscated
or sidecar behavior, so the independent audit is still mandatory. Missing,
linked, empty, contradictory, or malformed curation inputs abort before public
artifacts are replaced. Pre-cutover legacy selections retain their historical
annotation policy.

| Harness | Endpoint/transport | Required env key(s) | Benches that have it | Notes/quirks |
| --- | --- | --- | --- | --- |
| `claude` | Native Claude Code to Anthropic | CLI login; optionally `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` | hard, cuda, mini, mega, multi | The branch enforces no key. Reasoning effort is forwarded; single-GPU settings disable fast mode and enable thinking. |
| `ccr-claude` | Claude Code to local ccr-rust Anthropic-compatible router at `CCR_BASE_URL` or `http://127.0.0.1:3456` | None in runner; ccr-rust must already have upstream auth | mega | The shared Hard/CUDA/Mini branch is disabled because it has no container route. |
| `zai-claude` | Claude Code to Z.ai Anthropic API at `https://api.z.ai/api/anthropic` | `ZAI_API_KEY` | hard, cuda, mini, mega, multi | Claude aliases are remapped to the requested model. Multi reads the key from `~/.kbm_env`. |
| `minimax-claude` | Claude Code to `MINIMAX_ANTHROPIC_BASE_URL` or `https://api.minimax.io/anthropic` | `MINIMAX_API_KEY` | hard, cuda, mini, mega | Keeps MiniMax routing separate from native Claude defaults and remaps Claude aliases. |
| `kimi-claude` | Claude Code to `KIMI_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `KIMI_API_KEY` | hard, cuda, mini, mega, multi | Kimi's coding route expects thinking mode. Multi reads the key from `~/.kbm_env`. |
| `kinetic-claude` | Claude Code to `KINETIC_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `MOONSHOT_API_KEY` | hard, cuda, mini, mega | For `kinetic-0715`; `KIMI_API_KEY` is not interchangeable. Pins max effort, disables tool search, and pins the subagent model. |
| `or-fable` / `openrouter-fable` / `or-opus` / `openrouter-opus` | Claude Code to OpenRouter's Anthropic API at `OR_FABLE_BASE_URL` or `https://openrouter.ai/api` | `OPENROUTER_API_KEY` | hard, cuda, mini (shared runner), mega | Maps bare Fable/Opus aliases to Anthropic slugs and requests one-hour prompt caching. Other OpenRouter slugs pass through unchanged; Qwen 3.8 Max is `qwen/qwen3.8-max` and supports `xhigh` effort. |
| `longcat-claude` | Claude Code to `LONGCAT_ANTHROPIC_BASE_URL` or `https://api.longcat.chat/anthropic` | `LONGCAT_API_KEY` | hard, cuda, mini, mega | Claude aliases map to LongCat-2.0; the route raises the default output limit. |
| `hy3` / `hy3-claude` | OpenCode to Tencent TokenHub's OpenAI-compatible API at `HY3_TOKENHUB_BASE_URL` or `https://tokenhub.tencentmaas.com/v1` | `TENCENT_API_KEY` | hard, cuda, mini, mega | `hy3-claude` is a legacy name, not Claude Code. Only model `hy3` is accepted; preview/OpenRouter slugs are rejected; effort maps to `high` or `no_think`. |
| `tinker` / `inkling` (shared-runner branch) | OpenCode to Tinker's OpenAI-compatible API at `TINKER_BASE_URL` or `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | hard, cuda, mini (shared runner) | Both labels serve `thinkingmachines/Inkling` through the containerized OpenCode route. |
| `tinker` (Mega branch) | OpenCode to Tinker's OpenAI-compatible API at `TINKER_BASE_URL` or `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | mega | Direct Tinker route with bounded same-session auto-continuation. |
| `inkling` / `opencode-inkling` (Mega branch) | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | mega | Important alias collision: `inkling` means direct Tinker on Hard but OpenRouter on Mega. Mega defaults to high reasoning and bounded auto-continuation. |
| `deepseek-claude` | Claude Code to `DEEPSEEK_ANTHROPIC_BASE_URL` or `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | hard, cuda, mini, mega, multi | Intended for DeepSeek V4 Pro/Flash. Multi uses the fixed URL and loads the key from `~/.kbm_env`. |
| `qwen-claude` | Claude Code to `QWEN_ANTHROPIC_BASE_URL` or the token-plan endpoint `https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic` | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` | hard, cuda, mini, mega | The token-plan route serves `qwen3.8-max`; a DashScope key requires the Model Studio URL override. Tool search is disabled, effort is `xhigh`, context is 983,616 tokens, and the subagent model is pinned. Mega's post-run completeness selector omits this label. |
| `codex` | Native Codex CLI to its configured OpenAI transport | CLI login or `OPENAI_API_KEY` | hard, cuda, mini, mega, multi | Forwards reasoning effort and archives the rich session JSONL by parsed session ID. |
| `kimi` | Native Kimi CLI | Kimi CLI login/config | mega | The shared Hard/CUDA/Mini branch is disabled because it has no container route. |
| `droid` | Factory Droid CLI to its configured provider | Droid login; container may receive `FACTORY_API_KEY` and `DROID_API_KEY` | hard, cuda, mini, mega | Forwards reasoning effort. Provider endpoint depends on Droid configuration. |
| `gemini` | Native Gemini CLI | Gemini CLI login or `GEMINI_API_KEY` | hard, cuda, mini, mega | Runs from the problem directory with yolo approval. |
| `cursor` | Cursor Agent CLI (`agent`) | Cursor CLI login; container may receive `CURSOR_API_KEY` | hard, cuda, mini, mega | The executable is `agent`, not `cursor`. |
| `grok` | Native Grok CLI | Grok CLI login; optionally `XAI_API_KEY` or `GROK_API_KEY` | hard, cuda, mini, mega, multi | Uses the top-level headless command, not `grok agent`; reasoning effort is forwarded. |
| `opencode` | OpenCode to the provider/model encoded in the model argument | Provider-dependent; container forwards `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GEMINI_API_KEY`, and `SAKANA_API_KEY` | hard, cuda, mini, mega | Generic OpenAI-shaped route; model syntax is `provider/model`. Container mode has a stall watchdog. |
| `opencode-nemotron` | OpenCode to OpenRouter `/api/v1`, pinned to DeepInfra with fallbacks disabled | `OPENROUTER_API_KEY` | hard, cuda, mini | Preferred Nemotron route; uses an archive-local OpenCode config so the serving stack cannot drift. |
| `nvcf-nemotron` | OpenCode to a per-run localhost OpenAI adapter, then NVIDIA NVCF | One of `NGC_API_KEY`, `NVIDIA_API_KEY`, or `NVCF_API_KEY` | disabled in shared runner | No isolated container proxy route exists yet. |
| `lfm-opencode` | OpenCode to local vLLM at `KBMINI_BASE_URL` or `http://127.0.0.1:8765/v1` | `KBMINI_API_KEY` (defaults to `local`) | disabled in shared runner | No isolated container route to the local service exists yet. |
| `lfm-claude` | Claude Code to local ccr-rust at `CCR_BASE_URL` or `http://127.0.0.1:3456`, then local vLLM | `KBMINI_API_KEY` (defaults to `local`); ccr-rust must already be running | disabled in shared runner | No isolated container route to the local service exists yet. |
| `hermes` | Nous Hermes Agent to local vLLM through `OPENAI_BASE_URL=KBMINI_BASE_URL` | `KBMINI_API_KEY` (defaults to `local`) | disabled in shared runner | No container implementation exists yet. |
| `pi` | badlogic pi through a generated `lfm` OpenAI-completions provider to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | disabled in shared runner | No container implementation exists yet. |
| `lfm-grok` | Grok CLI custom `chat_completions` model to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | disabled in shared runner | No container implementation exists yet. |
| `opencode-or` | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` from `~/.kbm_env` | multi | Pins `KBM_OR_PROVIDER` with fallbacks disabled. The adapter has stalled intermittently, and the branch ignores the reasoning-effort argument. |
