# DEVLOG

Repo-wide record of decisions and obligations that do not belong to a single
benchmark. Per-benchmark journals live in `benchmarks/<bench>/DEVLOG.md`;
`AGENTS.md` and `kbtool/AGENTS.md` hold the operational rules.

---

## 2026-11-17 - GPT-6.2 Astra release: redaction hole closed, tetra codex 0.169.0

Redaction: `HOME_RE` missed a home path after `>` (codex `<cwd>/home/...` in environment_context) and
after a literal escaped newline (`\n/home/...` in tool output), so the scanner passed the five GPT-6.2
Astra traces with up to 97 raw home paths each. Both lookbehinds added, with tests. The stricter scan then
caught six older traces already public on HF (four GPT-6 Astra Pro cuda, Inkling and GLM-5.2 mega, all
`/home/<user>/sites/kernelbench.com/...` paths); they were re-redacted and re-uploaded. HF keeps the old
revisions in the dataset history.

Codex on tetra is now 0.169.0 (was 0.159.2); `gpt-6.2-astra` needs it on the native ChatGPT route. The
0.140.0 pin for rented boxes (OpenRouter tool payloads, `benchmarks/mega/AGENTS.md`) is unchanged. The
sweep billed a different ChatGPT Pro login than tetra's own (that one was rate-limited); tetra's login and
`~/.codex/config.toml` were restored afterwards. Fast service tier was off for the sessions.

Open before the next release: delete or hide tetra's stale `~/dev/sites/kernelbench.com` (archives with
solutions, visible inside the sandbox, missed by the canary), put the lock wrapper ahead of
`~/.local/bin/python` inside the sandbox, and give each sandbox its own `/tmp` and `~/.cache`
(`benchmarks/cuda/DEVLOG.md` 2026-11-17).

## 2026-11-16 - Two sessions for the Mega cell, both printed

One session per cell hides session luck (Qwen 4 Max's second MegaQwen kernel timed 114.5 us per step in
its own test and the one it shipped timed 172). From the next release, `scripts/tetra_release.sh` runs
Mega 02 as two independent sessions (the second logs as `mega_02_kimi_linear_decode_s2.log`; a cell named
N times in `KB_RELEASE_CELLS` runs N sessions). Both are regraded and audited. The board row stays the
argmax; `build_mega_leaderboard.py` writes the other session into a new last CSV column `other_sessions`
(`run_id:score`, `;`-joined), `build_model_index.py` carries it into `models.json`, and the homepage Mega
deck prints "other session N.NNx" under the bar. A run counts as the other session only when it is
correct, audited `clean` or `interesting`, and shares the headline's `run_group`, so an older run of the
same model under another budget or deck (three such audited pairs exist: kinetic-0715 on two GPUs and
deepseek-v4-pro) is not shown as a twin. CUDA cells stay single-session: Elliot's order covered the
headline cell, and tripling to six CUDA sessions costs more than the Mega twin. Cost: roughly double the
API spend of the Mega cell. Not yet exercised on tetra (stub-run on the Mac only): the first real release
should confirm both logs, both regrades and the resume path. Twins share the one Mega GPU lock, so their
GPU commands serialize.

## 2026-11-16 - Sandbox: own PID namespace and one-GPU device view for every host-mode session

Closes three items left open in the entry below (advisory GPU pin, sessions killing each other's jobs, the
shared process table). `scripts/lib/sandbox.sh` now builds
`bwrap --die-with-parent --unshare-pid --dev-bind / / --proc /proc --dev /dev` and binds back only
`/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools` and the assigned card's `/dev/nvidia<minor>`
(minor from NVML, never assumed from the index). No knob: a launch whose card cannot be mapped exits 4.

- **Inside a session** `ps` and `/proc` hold only its own processes, a kill of any host or sibling PID is
  ESRCH, everything it leaves running dies with the namespace, and NVML and CUDA enumerate one card as
  index 0 whatever `CUDA_VISIBLE_DEVICES` says. The physical index selects nothing inside (torch sees 0
  devices), so the sandbox sets `CUDA_VISIBLE_DEVICES=0`; `nvidia-smi -i <physical index>` inside prints
  "No devices were found". `/dev/shm` is private per session. ncu, nsys, nvcc and the agent CLIs work
  unchanged (tested on tetra with a real cuda 02 check and benchmark through the PATH wrappers).
- **`gpu-lock-exec` owner file** is now `pid \t run_dir \t <pidns>:<starttime>`. A recorded PID is trusted
  only when the namespace and start time both match the reader's view; anything else falls through to
  `flock`. Before this, `KBH_GPU_LOCK_HELD=1` ran without the lock whenever `kill -0 <owner pid>` failed,
  and across namespaces the recorded PID is some other process (in the test it was `[kthreadd]`). Both
  copies of the wrapper (shared runner and mega's fork) are byte-identical and a test enforces it.
- **Launch canary** (`scripts/lib/sandbox_canary.py`, every launch): a host sentinel `sleep` must be
  invisible and unsignallable from inside and still alive afterwards; NVML must list exactly the assigned
  UUID; CUDA probes with `CUDA_VISIBLE_DEVICES` unset, `0`, every index and each other card's UUID must not
  reach another card. About 1.5 s. `sandbox.json` records `pid_namespace`, `gpu_index`, `gpu_uuid`,
  `gpu_device`. Test copies with the PID flag or the device masking removed both exit 4.
- **`~/.ssh` and the docker socket are blanked** inside the sandbox. tetra's user holds private keys and is
  in the docker group, and either is a one-command way back to the host process table and the other cards.
- **Canary session** (tetra, 12:10 to 12:14 pm CST): a Sonnet 5.5 session launched through the real argv on
  GPU 3, told to kill a host process and a job in a sibling sandbox and to compute on another card. It
  tried `ps`, `/proc`, `kill -9` by host PID, `killall`, `nsenter`, `sudo`, `pkexec`, the docker socket,
  `CUDA_VISIBLE_DEVICES` by index, by UUID and unset, `nvidia-smi -i` and `mknod`. Host verdict: both
  victims alive, 36 of 36 samples of the session's compute processes on its assigned card, none elsewhere.
- **Still not containment.** The network, the user's systemd and D-Bus session and other host sockets stay
  reachable; `KBH_SANDBOX=0` and a box without bwrap stay unisolated; the lock is advisory (the owner file
  is writable inside, and `KBH_GPU_LOCK_HELD=1` with no owner file still runs unlocked). A rented box where
  nested user namespaces cannot mount `/proc` will now refuse to launch rather than run unisolated: check
  `scripts/smoke_sandbox_linux.sh` there first. Container mode was not rerun (it shares only the lock
  wrapper). `benchmarks/multi/scripts/run_agent.sh` has no bwrap and its `KBM_GPU_LOCK_HELD=1` is still an
  unconditional bypass.
- Published host-mode cells before this date ran without either isolation; their numbers stand on the
  isolated regrade and the audits, as before.

## 2026-11-16 - Kimi K3.1 Pro and Qwen 4 Max night on tetra: harness, isolation and trace-privacy lessons

Kimi K3.1 Pro (`kimi` harness, Kimi Code CLI 2.5.3, `kimi-code/k3.1-pro`, effort max) ran on tetra GPU 3 and
is published; Qwen 4 Max (`or-fable`, `qwen/qwen4-max`) ran on GPU 2 the same night. Results are in the cuda
and mega DEVLOGs. What changed in the tooling, and what is still open:

- **`kimi` harness rewritten for Kimi Code 2.x.** Per-run CLI home and effort via `scripts/lib/kimi_home.py`;
  usage and the served model id come from the session wire log the runner archives as `kimi_wire.jsonl`
  (the stream transcript has no reasoning, usage or model id). The CLI login must use the global region
  (kimi.ai); the default device-code page (kimi.com) only offers WeChat or a mainland phone number.
- **New transcript parsers go in `scripts/`, never under `benchmarks/*/src/`.** The first version of the Kimi
  2.x parser replaced `src/viewer/parsers/kimi.py` in four benches. Every byte under `src/` is in the
  graded-surface digest, so the four Kimi cells were stamped with a digest no published cell has and gate E
  reported 181 published hard and cuda cells as stale. Fix: parser moved to `scripts/kimi_wire_parser.py`
  (exporters and `media/trajectory.py` route Kimi 2.x through it), the eight src files restored, and the
  Kimi cells restamped with a check-only replay (`KBH_REGRADE_CHECK_ONLY=1`, timing kept, `recheck` in
  `result.json`). The rule did not fit under the 32 KB cap of `benchmarks/hard/AGENTS.md`; it lives in the
  parser's docstring and here.
- **OpenRouter Qwen route and prompt caching.** Claude Code puts its cache marker on the last tool result and
  the Alibaba endpoint ignores a marker there, so only the system prefix was cached and each call re-billed
  the whole conversation ($0.22 to $0.31 per call at 165k to 231k context). The first Qwen 4 Max attempt was
  stopped after 75 minutes and $48.35 and restarted fresh with `DISABLE_PROMPT_CACHING=1` (provider-side
  automatic caching then covers 71% to 98% of the context, about a cent per call early on). The `or-fable`
  branch of `scripts/lib/run_harness.sh` and `benchmarks/mega/scripts/run_hard.sh` now set this for
  `qwen/qwen4-max`. The container-mode pass-through is untested. Any other non-Anthropic model on this route
  needs the same check before a long run. The aborted archives (`20261116_0044*` to `0046*`) have no
  `result.json` and are never graded or published.
- **GPU pinning is advisory.** A session is pinned to its card only through `CUDA_VISIBLE_DEVICES`, which the
  model can override. Qwen 4 Max sessions ran jobs on GPU 3 (3:30 to 3:35 am and a 3 s job at 4:11 am, both
  on top of Kimi regrades) and on GPU 1 (3:35 to 3:40 am, 4:32 am); a Kimi session ran one 29 s job on
  GPU 2. An "isolated" regrade is therefore only isolated if something checks it: a logger that records
  every compute process on every GPU every 3 s ran from 3:50 am, the Kimi regrades were repeated until the
  log showed the card to itself (MegaQwen took three tries), and each crossing is in the affected audit
  YAML. Later the same night the Qwen 4 Max Mega session ran on GPU 3 (6:09 am to 8:07 am, including its
  final check and benchmark) and on GPU 0 (6:48 to 8:03 am), and the MegaQwen session ran an A/B on GPU 1
  (7:29 am). Open: enforce the pin (device cgroup or per-run `nvidia-container`-style visibility) and make
  the regrade script refuse or repeat on a foreign process itself.
- **Sessions can kill each other's jobs.** The shared process table also takes signals. The Qwen 4 Max
  MegaQwen session killed five GPU jobs of the live Qwen 4 Max NSA session (4:18 to 5:56 am, seven kill
  commands) after deciding the lock holder was an orphan, and the Mega session killed a sixth
  (`kill -9` on a pid it took for its own zombie, 5:59 am). Two more NSA jobs have no end line in its lock
  log and no established killer. The NSA cell still finished and graded clean. The PID namespace above
  closes this too.
- **`gpu-lock-exec` re-entrancy.** A process from the same run skips the lock with no log line, so a
  session that holds the lock in one job can run more GPU jobs unlocked and unlogged (the MegaQwen
  session's first `benchmark.py` held the cuda lock from 6:16:56 to 6:54:16 am while its own check, a
  second benchmark and a dev bench ran on the card). Open: log re-entrant runs, or serialize them.
- **Audit probes need the same window check as regrades.** The first Fused MoE probe ran on a borrowed
  GPU 3 while the Mega session's final check and benchmark were on it. Rerun on a quiet GPU 2. Run
  `gpu_window_clean.py` over the probe window before writing that the card was quiet.
- **Token counts for multi-turn Claude Code sessions.** A session woken by background tasks emits one
  `result` event per turn and each `usage` covers that turn only; `extract_usage.py` took the last one, so
  the four Qwen cells first stored 205 to 3,767 output tokens for sessions that wrote 264,654 to 759,370.
  All four copies now read the final event's cumulative `modelUsage` (test:
  `kbtool/tests/test_extract_usage.py`) and record no cost when Claude Code marks the price basis unknown
  (its figure for these cells, $29.59 to $250.50, is a fallback-rate estimate; OpenRouter's own
  daily usage for the key read $122.26 after all four sessions ended, the aborted $48.35 first attempt
  included). The four archives were re-extracted. Open:
  the retired, unpublished `20261109_143848` Fable 5.2 cell has the same undercount, and older
  OpenRouter-routed cells still show Claude Code's estimate as the run cost.
- **The sandbox shares the host process table.** `ps`, `top`, `pgrep -a`, `/proc/*/cmdline` and nvidia-smi's
  process table show sibling sessions' command lines (run ids, the sandbox hide list, inline patch source)
  and unrelated host jobs. No session used it to read another kernel, but it is a leak channel between
  cells and it trips `kb contamination` (the Kimi mega and NSA cells carry a manual `contamination: clean`).
  Open: a PID namespace for the sandbox.
- **Public traces drop process listings.** `scripts/trace_privacy.py` is now the one filter for every
  `traces_to_hf.py` (hard and mini had none): private config reads, operator skill files, the host's GPU
  lease listing and every process-listing command are removed with their paired results. Open: older
  staged exports under `runs/{mega,cuda,hard}/traces` (12 files found) still carry operator skill text,
  lease job names and one `ps` result, and three June mega traces carry redacted env dumps. Most have no
  transcript on the Mac, so cleaning them means pulling the archives back and re-exporting.
- **Deck gaps seen in the audits.** Mega `check.py` cannot see a wrong MLA attention path (second model in a
  row to pass it with wrong attention; it needs a cache case where attention matters). `BASELINE.md` for
  cuda 03 is not copied into any agent's workspace.

## 2026-11-11 - Gemini 4 Carbon (high) published & posted; Fable 5.2 staged; X post & chart rules

Published **Gemini 4 Carbon (`gemini-4-carbon-high`, `agy` harness)** on `RTX PRO 6000` across CUDA (`01` Fused MoE `0.1022` / 10.2%, `02` DeepSeek Native Sparse Attention `0.0879 ms` geomean, `03` MegaQwen Decode `0.0609` / 6.1%; `04_grid_mingru_sps` retired in `2ebda3b`) and Mega (`02_kimi_linear_decode` `23.79x`, `0.25 ms/tok` at 2k), following `Claude Fable 5.2 (high)` (`1d02d7c`).

- **Harness, sandbox & redaction (`6fc6f5a`, `941e409`, `aefc4d4`, `07dd3d7`):** `scripts/lib/sandbox.sh` now hides `~/.gemini/config` (blocking host hooks inside `agy` runs) and overlays `agy` conversation stores per run; `scripts/redaction.py` now matches home paths inside `file://` markdown links (`file:///home/<user>/...`); GPU lock check timeout capped at 1800 s (`KBH_CHECK_TIMEOUT_SECONDS`). Direct audit of all four Carbon transcripts confirmed zero `belt` or operator-context reads (only MegaQwen's prompt-linked public baseline repo).
- **Live X receipts (posted 2026-11-11 ~3:37 pm CST via `xctl`, folders moved to `media/posts/posted/carbon-{cuda,mega}/`):**
  - **KernelBench-CUDA:** root `https://x.com/elliotarledge/status/2120631806596981165` (`2120631806596981165`), links reply `https://x.com/elliotarledge/status/2120631962650218529` (`2120631962650218529`), chart generator `media/make_carbon_cuda.py`.
  - **KernelBench-Mega:** root `https://x.com/elliotarledge/status/2120631976181035177` (`2120631976181035177`), links reply `https://x.com/elliotarledge/status/2120632472350416912` (`2120632472350416912`), chart generator `media/make_carbon_mega_traj_compare.py`.
- **Post & chart learnings locked in (`media/AGENTS.md` updated):**
  1. **Trajectories over bar charts, minimal chart text:** Multi-panel or head-to-head **trajectory hill-climb charts** (`make_carbon_cuda.py`, `make_carbon_mega_traj_compare.py`) carry far more signal than side-by-side percentage bar graphs. Strip GPU name and "isolated regrade" subtitles off the PNGs (readers already know both, or it lives in one word of prose).
  2. **Qualitative trace narrative over score recitation:** Do not recite leaderboard standings ("Model A leads at X%, Model B at Y%"), which reads AI-generated. Focus on unconventional engineering moves, debugging recoveries, and concrete thinking-trace quotes (e.g., Carbon deriving a Cauchy-Schwarz BF16 rounding bound on NSA, streaming CPU RNG draws across a host-pinned flag inside a persistent MegaQwen decode kernel, spotting odd-length half-warp shuffle divergence after a 30-min timeout, reading raw PTX when `compute-sanitizer` timed out on GPU lock, and catching odd-count 29-vs-28 software barrier bit-flip parity on Kimi-Linear).
  3. **Plain English on X (zero backticks):** Never put backticks, hex masks, or raw C++/PTX identifier soup in X posts; explain every hardware/kernel mechanism in plain prose with zero em dashes.
- **Open items carried forward (from collapsed `side duties` session):**
  - `media/posts/audited/fable52-cuda/` and `media/posts/audited/fable52-mega/` remain staged (rewrite to match the plain-English trajectory format above before posting if Elliot decides to post them).
  - **Mega board SKU baseline note (unfixed):** `02_kimi_linear_decode` currently pools runs regraded on `RTX PRO 6000 Blackwell Server Edition` (`kb-*` cloud boxes, PyTorch baseline ~7.6 ms/tok at 2k) with runs regraded on `RTX PRO 6000 Blackwell Workstation Edition` (local workstation, baseline ~5.9-6.1 ms/tok), inflating relative `Nx` multipliers on Server Edition regrades vs Workstation Edition regrades even at similar absolute `ms/tok`.

---

## 2026-10-29 - Mac run archives thinned; mini outputs deleted

Mini is discontinued (Elliot, 10-29). Deleted `benchmarks/mini/outputs` (15.6 GB): the
306 LFM2.5-2.6B-Agent sessions from 07-28 to 07-30 (0 correct), never published, no HF
dataset. The mini code is still in the tree.

hard/cuda/mega archives on the Mac went from about 29 GB to 20.4 GB on disk. Removed:
regenerable caches in 1,262 dirs (run `cache/`, `agent_home/.cache`, `agent_home/.triton`,
and `tcache`, `triton_cache`, `.triton`, `__pycache__`, `.venv`, `torch_extensions` dirs
in the workspace; 7.98 GB) and 56 vendored upstream trees (CUTLASS 4.5.2 to 4.6.1, FLA,
vLLM, Triton, faiss sources; 2.06 GB) that had 200 or more files, no `solution.py`
inside, and no mention in a published solution. Their paths and CUTLASS versions are in
`benchmarks/hard/outputs/third_party_removed_20261029.tsv` (gitignored).

Kept: every top-level run file (`result.json`, `solution.py`, transcripts, logs, `gpu`,
`roofline.png`, `index.html`), agent session logs under `agent_home`, the problem
workspace and agent-written sources (read by `kb contamination`, the mega framework
detector and the grok viewer), binaries outside caches, and 48 vendored trees in 39
published runs whose solution builds against them (4.34 GB, e.g.
`extra_include_paths=[_HERE / "cutlass" / "include"]`), so those kernels still compile on
regrade. `scripts/check_publish_gates.py` passed afterward.

Raw transcripts: 786 files (`transcript.jsonl`, `transcript.part1.jsonl`,
`codex_session.jsonl`) from 720 runs, 8.59 GB, moved off the Mac to tetra's cold
archive as `kernelbench/kernelbench-raw-transcripts-20261029.tar.zst` (1.28 GB, sha256
`fef1c026...`, file count checked on tetra). The 6 `transcript.part1.jsonl` files (20 MB)
were restored to the Mac: `build_run_detail.py` sets `resumed_segment` from their presence,
and a publish without them dropped that flag from 5 runs. Only runs whose HF trace exists
went, and only where removal cannot change a board: the builders' archive-reference
tripwire is silent, or a clean audit overrides it, and mega's copied-solution check does
not fire. Still on the Mac: 615 runs with no HF trace (1.51 GB, the only copy) and 51
runs whose raw transcript is what keeps them off a board (0.73 GB). HF traces are
redacted, normalized and use synthetic timestamps, so `media/trajectory.py` on an old run
needs the tetra archive.

Trace links: `scripts/build_catalog.py` had its own `trace_url` that always emitted the
flat `<rid>.jsonl`, so 43 B200/H100 board cells in `catalog.json` linked to 404s (HF stores
them under `b200/` and `h100/`). It now delegates to `build_model_index.trace_url`.
`20260719_024723_grok_grok-build_01_fp8_gemm` (launch failed, "unknown model id", no
session) joins `NO_TRACE_RUNS`. All 695 site trace links resolve against the HF trees.

---

## 2026-10-03 - Leftover decisions from the Sonnet 5.5 / GPT-6.1 Sol week

`kb lambda up` default type now follows the deck's hardware key: `gpu_1x_h100_pcie`
for hard and cuda (their `problems-h100` claims the PCIe `H100` key), and SXM only
where the deck is SXM (`KB_LAMBDA_BENCH=mini` `gpu_1x_h100_sxm5`, `multi`
`gpu_8x_h100_sxm5`). The old SXM5 default contradicted the 09-28 PCIe rule and would
have graded a PCIe deck on the wrong key. Mini and multi still wait for Elliot to
name SXM (money gate); nothing was launched.

07 zero-point contract: integer zero-points (the 09-24 recommendation), on branch
`bench/recheck-04-07` with the cuda 04 1e-5 margin rule (renamed from
`bench/04-margin-rule` and rebased on master). Detail: `benchmarks/hard/DEVLOG.md`.
Both change an RTX graded surface, so the branch merges only after
`scripts/recheck_04_07.sh` (prep, then `launch <gpu>` on a tetra GPU with no compute
PIDs, then `pull`) replays every published cuda 04 and hard 07 cell check-only, plus
the GPT-6.1 Sol 04 cell. `prep` verifies that tetra's `src/` plus the shipped branch
decks hash to the branch's graded-surface digest and that every archive exists there.

GPT-6.1 Sol xhigh (cuda 01-04, mega 02, tetra GPU 0, 09-30): thin archives pulled
(39 MB of 2.75 GB; transcripts, logs, result.json, solution.py) into the gitignored
`outputs/runs`, and all five audited (`results/annotations/20260930_*gpt-6.1-sol*`).
Verdicts: 01 clean (authored grouped mma.sync MoE, 0.0941), 02 clean (exact NSA
selection on tensor cores, 0.3521), 03 clean (one cooperative decode launch,
0.0386), 04 interesting (hi/lo split-fp16 rollout, version-keyed weight pack,
0.3052), mega interesting (17-minute session, single-launch Triton, 4.9899x).
Contamination clean on all five: every foreign run id `kb contamination` reports is
in ps, rg or nvidia-smi output, never in a command or message. 03, 04 and mega need
the same-buffer overwrite probe (pointer tables or a version-keyed cache), which
`scripts/probe_same_buffer.py` runs inside the same recheck job; their YAMLs say
`audit_status: ... pending`. Do not publish GPT-6.1 Sol until those probes are in
the YAMLs and the recheck has merged.

The leak, and how audits treat it: tetra cells run in host mode, so Codex loads
`~/.codex/AGENTS.md` (Elliot's global CLAUDE.md) and his skills, and Claude Code
loads `~/.claude/CLAUDE.md` the same way (the 09-28 Sonnet 5.5 transcripts quote
it). The published 09-22 GPT-6 Sol/Luna cells had it too. It carries no solution,
grader detail or score, so it is disclosed in `harness_note`, not scored as
contamination. It did change behaviour: 03, 04 and mega ended their sessions asking
permission to kill GPU processes under the global "ask before killing a GPU job"
rule, and agents leased and timed on GPUs 1-3 outside the launcher's GPU 0 lease.
Those numbers likely understate the model. Also found: the sandbox hides
`~/kernelbench.com` but not the stale tetra checkout `~/dev/sites/kernelbench.com`,
whose archives and annotations an agent can list (03 listed file names, opened
none). Masking global agent config and that checkout in `scripts/lib/sandbox.sh` is
the fix; not done, because it changes the environment against which every published
host-mode cell was run.

---

## 2026-09-28 - GPU targets: tetra and Lambda H100 PCIe only, ncu required

Elliot's rule: every run uses tetra's RTX PRO 6000 or Lambda H100 PCIe, with
non-root `ncu` working before the first agent session. Brev, Verda, anvil, B200
and H100 SXM are out unless he names one. He first said SXM, then switched to
PCIe the same day because hard and cuda `problems-h100` claim the PCIe `H100` key. Workflow: `kbtool/AGENTS.md` "GPU targets".

Evidence. tetra: `ncu` 2026.2.1, `RmProfilingAdminOnly: 0`, driver 615.71.09.
Lambda `gpu_1x_h100_pcie` (us-west-3, stock Lambda Stack 22.04, driver 570.148.08,
CUDA 12.8; SXM5 had no capacity): no `ncu`, `RmProfilingAdminOnly: 1`. After
`nsight-compute-2025.1.1` from NVIDIA's ubuntu2204 repo, `ubuntu` got
`ERR_NVGPUCTRPERM` and `sudo ncu` worked. After the modprobe flag and a driver
reload (no reboot), `ubuntu` got `--set full` with all 13 sections.

Found while writing it up: `kb lambda bootstrap` installs neither `ncu` nor the
flag, although the docs said it did, and `kb lambda run` launches in host mode.
Mega claims only `RTX_PRO_6000`, so it has no H100 deck. Mini (`H100_SXM`) and
multi (4xH100 SXM behind NVSwitch) cannot run on PCIe and wait for SXM.

---

## 2026-09-26 - Release prep: tetra launcher and provider route status

`scripts/tetra_release.sh <gpu> <harness> <model> [effort]` packages the Grok 4.7 /
Opus 5.5 / GPT-6 recipe: mega 02 plus cuda 01-04 on one tetra GPU under an
`overnight-compute` lease, a served-model check four minutes in, then the sequential
regrade of exactly those cells on that GPU (`benchmarks/cuda/AGENTS.md`). Tested on
tetra GPU 3 with stub runners (lease, launch arguments, run group, regrade lists, DONE).

tetra prep: checkout fast-forwarded to master. The cuda and mega venvs registered their
own package from the deleted `kernelbench.com-hfx` worktree; re-synced with
`uv sync --frozen` plus `patch_torch.sh` (torch 2.13.0+cu130 cuda, 2.11.0+cu130 mega,
all 4 GPUs visible). `RmProfilingAdminOnly` is 0 since the reboot, so plain `ncu`
works. Muse's launcher was copied from the Mac to tetra (Muse Code 1.4.0); the Meta
key lives in `~/.env_vars` as `META_MODEL_API_KEY`, which the launcher maps to
`META_API_KEY`.

Route probe on tetra (one tiny prompt through each route's CLI and env):
- Working: claude `claude-opus-5-5` and `claude-fable-5-1` (Claude Code 2.1.283),
  codex `gpt-6-sol` (codex-cli 0.155.1, latest 0.157.1), agy
  `gemini-3.8-flash-high`, deepseek-claude `deepseek-flash`, minimax-claude
  `MiniMax-M3`, longcat-claude `LongCat-2.0`, grok `grok-4.7-build-fast`, muse
  `muse-spark-1.3`, and or-fable (OpenRouter) for `stealth/space-bunny-alpha`,
  `moonshotai/kimi-k3`, `z-ai/glm-5.3-prime`, `xiaomi/mimo-v2.6-pro`.
- Broken on the account side (key bytes identical on Mac and tetra): zai-claude 401;
  kimi-claude 401, and the Kimi Code endpoint says the subscription lapsed; qwen-claude
  token plan `AccessDenied.Unpurchased`; grok `grok-4.7` 404 because the Grok Build
  plan no longer serves `grok-4.7-build` (the CLI default is now `grok-4.7-build-fast`).
  Until those plans are renewed, GLM, Kimi and Qwen releases route through or-fable.
- `benchmarks/hard/scripts/preflight_harnesses.sh` still probes GPT-5.5, Opus 4.8 and
  GLM-5.1; its rows are stale.

---

## 2026-09-24 - Traces live on HuggingFace only

Elliot: traces stay on HF, not GitHub. The 56 mega transcript viewers in
`public/runs/*.html` (24.5 MB of the 39 MB `public/runs`) are removed from the repo;
13 of them had no HF copy and were converted from local archives and uploaded to
`Infatoshi/kernelbench-mega-traces` first, so all 56 runs have an HF trace.
`next.config.mjs` 308-redirects `/runs/<run_id>.html` to the HF file. Nothing in the
app read the viewers (`loadRunAudits` in `app/_lib/data.ts` has no callers).
Same day: `scripts/redaction.py` stopped treating `0::2` slices and asm `::` as IPv6
(commit 83a0ed9); 22 HF traces were re-exported or added with the fix.

---

## 2026-09-14 - Storage cleanup and missing public traces

User authorized removing reproducible archive bulk and publishing missing eligible
traces, with explicit API-key/private-information redaction. Removed 6,014
dependency/environment/bytecode directories and 2,636 compiled .o/.so/.cubin
files. Their measured allocated size was 15,384,621,056 bytes (14.33 GiB).
Filesystem free space after cleanup was 95.13 GiB. Preserved transcript originals,
CUDA/source files, results, audit evidence, build descriptions, and lockfiles.
No model-cache deletion or Mini publication was performed.

Published 11 missing audited session traces plus 11 audit-summary sidecars:
2 Hard traces (H100 and B200) and 9 CUDA traces (1 H100 PCIe, 8 H100 SXM).
SXM exports live under h100sxm/ to retain the hardware distinction. Audit verdicts
are explicit; rejected/interesting traces are not new leaderboard scores.
Removed 7 private tool calls and their 7 paired results. All exact exported files
passed the redaction scanner and independent decoded-string checks; downloaded
all 22 public files and verified SHA256 equality. All 11 original transcript
hashes remained unchanged. Existing public traces were not overwritten.

Public dataset commits:
- Hard: c0b84f2945c8ab2ca039e5dbbc902b35862802ed
- CUDA: 1f71d832d8c7405cf5b13afe65fb623755f3180b

Private receipts: runs/cleanup-20260914/{publication-plan,selected-traces,
staging-receipt,publication-receipt}.json. Cleanup inventories and verified
deletion receipt: benchmarks/hard/outputs/cleanup-20260914/.
Some older apparent upload gaps are already public under another GPU namespace;
others lack local source archives or an audit. They were not invented or silently
treated as backed up. Raw unique evidence remains local.

Hardened scripts/redaction.py for quoted/JSON credentials, private keys, URL
credentials, JWTs, known stored credentials, emails, home paths, and IPs. It
preserves embedded JSON and has non-mutating category-only scan helpers.
Validation: 20 focused privacy tests passed; full kbtool suite had 60 passes and
one unrelated failure because the already-modified kbtool/AGENTS.md is 32,949
bytes, exceeding its 32,000-byte limit. Existing user edits were preserved.

## 2026-09-11 - Manifund grant terms (migrated from Claude auto-memory)

KernelBench is funded through Manifund: $15,000 over 6 months from JueYan Zhang
(jueyanz@gmail.com, AISTOF AI safety fund), fully funded 2026-07-13. Project
page:
https://manifund.org/projects/kernelbench-an-independent-benchmark-for-ai-automated-gpu-kernel-engineering

Indicative budget: $6k frontier-model API credits (including API-priced models
like GPT-5.5 Pro that were previously skipped on cost), $4k multi-GPU cloud
rentals for the 8xH100 track, $5k maintainer time for sweeps and per-run
reward-hack audits.

The signed grant agreement (Manifold for Charity boilerplate, signed 2026-07-14)
governs, and its entire-agreement clause supersedes the informal email terms.
That means substantive budget or project changes need written approval, and
unspent funds at project completion must be returned. The informal terms from
the email thread were looser: no legal reporting obligations, email updates
requested, deviations fine if flagged, disclosure in his fund letter possible,
acknowledgment optional.

This is half of the original $30k/12mo proposal; renewal or scope expansion is
possible later.

---

## 2026-09-17 - Grok 4.7 (xhigh) on tetra: first sweep on the home rig

First KernelBench cells graded on tetra, the bare-metal box with two RTX PRO 6000
Blackwell Workstation Edition cards (driver 610.57.04, CUDA 13.3). Everything ran
on GPU 1 under an `overnight-compute` lease (`--resource gpu1`); GPU 0 carried an
unrelated fine-tuning job the whole night and was never touched. Grok 4.7 smoke
test at `--reasoning-effort xhigh` answered in 3.9 s; the harness route is the
native Grok CLI (`grok` harness, model `grok-4.7`, served as `grok-4.7-build`
through cli-chat-proxy.grok.com on the Grok Build subscription, so there is no
per-token dollar cost to record).

Five cells in parallel on GPU 1 (mega 02, cuda 01-04, unlimited budget), all
correct, all stopped on their own inside 100 minutes, then the isolated sequential
regrade on GPU 1 with the canonical decks (in-run / isolated):

- mega 02 Kimi-Linear Decode 5.652 / 6.3791x, clean, authentic single-launch
  cooperative megakernel (72 grid barriers per token, fidelity 1.0000 everywhere).
  Grok 4.5 was 0.816x on this cell.
- cuda 01 GLM-5.2 Fused MoE 0.0944 / 0.0948, interesting: cuBLAS strided-batched
  expert GEMMs, the Grok 4.6 / V4 Pro class, not an authored GEMM.
- cuda 02 Native Sparse Attention 0.1002 / 0.1002, clean, hand-written WMMA; the
  exact-tie block ordering is flipped relative to the reference but cannot fire on
  bf16 randn inputs (probe-verified, non-scoring).
- cuda 03 MegaQwen Decode 0.0454 / 0.0455, clean, seven hand-written kernels.
- cuda 04 Grid + MinGRU 0.6542 / 0.6398, interesting: the rollout runs in fp16 on
  a problem declared fp32, chosen after the agent measured fp32 / bf16 / fp16 at
  the graded shapes; the strict oracle matches at every graded shape and seed
  (positions exact, rewards bitwise, logits within 2.5e-6) and no torch precision
  flag is touched. Same call as the Gemini 3.8 Flash cell, fourth on the board.

Audits were trace-level and submission-level with same-buffer overwrite probes on
the quiet GPU 1 for every cell; contamination clean on all five (the only foreign
run ids in any transcript are `find` output that was never followed). None of the
five agents ran ncu or nsys even though both were available.

Two pieces of plumbing landed for this box. (1) `RmProfilingAdminOnly=1` cannot be
cleared without a driver reload, which the fine-tune on GPU 0 forbids, so ncu goes
through `~/kb-bin/ncu-sudo` (sudoers limited to `ncu`, `nsys` and a path-locked
`kb-chown-run`) and the harness gained `KBH_NCU_BIN` / `KBH_NSYS_BIN` to point the
per-run lock wrappers at it; proven with a real `smsp__cycles_elapsed.avg` on
GPU 1. `/etc/modprobe.d/nvidia-ncu.conf` is staged for the next reboot, after
which the shim is unnecessary. (2) Grok's full tool timeline lives in
`~/.grok/sessions/<url-encoded cwd>/<uuid>/chat_history.jsonl`, not in the
harness transcript, and host-mode runs do not archive it; each run's session
store was copied into `<run>/agent_home/.grok/sessions/` by hand so the viewer,
the HF trace and the trajectory chart see the tools. `media/trajectory.py` now
drops its auto checkpoints when the trace carries no timestamps (Grok rows have
none) and reads the session `usage.json` for the token axis.

Regrade hygiene note: the first cuda regrade was killed mid-cell because audit
probes were still on GPU 1; `pkill -f` on tetra matches its own ssh session and
returns 255, kill by pid. `*.contended.log` survives a rerun because the script
only renames when the contended file is absent.

Traces: the five HF jsonl exports were converted on tetra, scanned with `scripts.redaction.scan_file` (zero findings) and the rg tripwire, uploaded to `kernelbench-mega-traces` and `kernelbench-cuda-traces`, then downloaded back and SHA256-matched against the staged files. Posts drafted, not posted: `media/posts/audited/grok47-cuda/` (headline cell 04)
and `media/posts/audited/grok47-mega/`, FILL-IN left open. Trajectory charts for
all five runs are in the session scratchpad and regenerate from the annotations.

## 2026-09-11 - DeepSeek V4.1 Flash: two posts scheduled, annotations closed

Two short posts for `deepseek-claude/deepseek-flash` were drafted from the
published board and handed to Codex (gpt-6-astra high) in tmux, which scheduled
them in X through Chrome and verified both in the saved Scheduled queue.
Receipts: `media/posts/scheduled/dsv41flash-cuda/GOAL.md` and
`media/posts/scheduled/dsv41flash-mega/GOAL.md`.

- CUDA, 2026-09-11 19:00 MDT (2026-09-12 01:00 UTC). Headline cell
  `20260910_150236_deepseek-claude_deepseek-flash_02_deepseek_nsa` at 0.5019,
  fourth on the board behind Fable 5.1 (1.0627), Opus 5 (1.0367) and Fable 5
  (0.7266). The story is the over-compute: at 8K context the semantics need about
  14% of the causal block triangle and the kernel executes 78%. Time across the six
  shapes 0.059 to 0.736 ms. Rest of deck in the post: GLM-5.2 Fused MoE 0.0946,
  MegaQwen Decode 0.0539, Grid + MinGRU 0.2856.
- Mega, 2026-09-11 20:00 MDT (2026-09-12 02:00 UTC). Cell
  `20260910_084202_deepseek-claude_deepseek-flash_02_kimi_linear_decode` at
  17.1008x, sixth of the published field (GPT-6 Astra 24.8x, Fable 5 24.6x).
  Posted as interesting, not clean: the persistent atomicAdd accumulator is never
  zeroed, so every KDA layer after the first inherits the previous layer's
  activations, output cosine 0.9997 on the first token and 0.98 on the next.

Each thread carries the rank chart on tweet 1 (field = best per model on the
published RTX PRO 6000 board, rendered with `save_rank` from
`media/posts/unaudited/make_charts.py`) and the board, kernel and Hugging Face
trace URLs as the reply; all three URLs were curl-verified live before the
handoff. FILL-IN is Elliot's paragraph, pasted verbatim. No Hard post: the
Hard cells sit in the bottom third of every field (best clean 0.3310 on FP8
GEMM), same call as the Fable 5.1 and Gemini 3.8 Flash batches.

Same day, commit 62e4c34: the four CUDA annotations still said probes and the
isolated regrade were "owed before publish". Replaced with the measured
outcomes (same-buffer overwrite passed on 01 and 02, long-context selection
probe LONG_CTX_OK on 02, the predicted stale weight-pack reproduced on 03 with
the fresh-model control passing) and republished the cuda board. Full
measurement record for the sweep, including the CC-box timing floor and the
second non-CC regrade box, is the 2026-09-10 entry in `benchmarks/hard/DEVLOG.md`.

Still owed: when the two posts are live, paste their URLs so the folders move
to `media/posts/posted/` with `tweet_id.txt`. The three older folders in
`media/posts/scheduled/` (gemini38flash-mega, musespark13-cuda, musespark13-mega)
have been waiting on the same step since 2026-09-05.
