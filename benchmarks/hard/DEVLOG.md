# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

---

## 2026-07-20 - Lambda Cloud $10k + worker lifecycle

Zach/Lambda sponsored **$10k** Cloud credits for Hard/Mega/CUDA/Multi (H100,
B200, RTX PRO 6000). Applied to account `elliot@arledge.net` (email
"Your Cloud Credits Have Been Applied!"). Credits visible only at
cloud.lambda.ai Settings → Billing → Credits (not via Cloud API).

Repo additions:
- `scripts/lambda_worker.sh` — list/ls/up/sync/bootstrap/run/regrade/pull/down/ssh
  via Cloud API (curl+jq). Launch attaches SSH key names **macbook** + **anvil**.
- `kb lambda ...` thin wrapper in `kbtool/kb/cli.py` (same pattern as `kb brev`).
- AGENTS.md: Lambda section + LAMBDA_API_KEY note.

Control-plane setup (not in git): `LAMBDA_API_KEY` / `LAMDBA_API_KEY` in
`~/.env_vars` on Mac and anvil; API keys for SSH already uploaded to Lambda.

## 2026-07-09 - Agent-side CUDA disabling removed

`KBH_DISABLE_AGENT_CUDA` was removed from the harness, parallel launcher, and
infra-retry launcher. It had been introduced to prevent parallel agents from
bypassing the shared GPU lock, but hiding CUDA also removed the live
compile/check/benchmark/profile loop that KernelBench is intended to measure.
That made disabled and enabled runs incomparable. New runs always expose CUDA;
parallel GPU commands serialize through `outputs/gpu.lock`. Historical
`agent_cuda_disabled` metadata remains in archived results for provenance.

## 2026-07-09 - Hy3 TokenHub context wall + host-mode stall watchdog

RTX PRO 6000 TokenHub resweep: 4/6 finished (fp8/paged/moe/w4a16), kda+topk
looked "hung" for ~8h. RCA (`/tmp/hy3-stall-rca/REPORT.md`, Fable):

1. **TokenHub real input wall is 196608** (= 0.75 × advertised 262144). Live
   probes of ~200k and ~215k prompts both report `prompt_tokens=196608` (silent
   truncation, HTTP 200). Latency at the wall is 150–210s; during the incident
   every boundary request died after ~183s with "upstream model service is
   abnormal or unreachable". Sibling runs under ~160k passed concurrently.
2. **OpenCode 1.17.8 session loop** retries that error forever with uncapped
   `2^n` backoff (observed up to 8192s) and emits nothing to `--format json`
   transcript — so the session looks dead while parked on timerfds with no
   TokenHub sockets.
3. **Host-mode path had no stall watchdog** (`KBH_AGENT_CONTAINER=0` for the
   local launch); `timeout 0` never fires. Config advertised `context: 262144`
   so OpenCode compaction threshold (230144) sat *past* the real wall.

Harness fixes:
- `write_tokenhub_hy3_opencode_config`: default `context=196608`, `output=32000`
  (override via `HY3_TOKENHUB_CONTEXT_LIMIT` / `HY3_TOKENHUB_OUTPUT_LIMIT`).
- Host-mode hy3: `run_host_with_stall_watch` on transcript growth + bounded
  retries; default `KBH_OPENCODE_STALL_SECONDS=1500` (legitimate ~13min gaps
  observed; multi-hour silence still dies).
- Mega hy3 config limit aligned to the same measured wall.

Not the 2026-06-09 `@ai-sdk/openai-compatible` mid-stream hang — every request
here completed with an explicit provider error; client then slept silently.

## 2026-07-09 - Hy3: retire OpenRouter preview, official TokenHub route

**Published purge:** all `hy3-claude/tencent/hy3-preview` cells removed from
leaderboard.json (+ h100), published_runs.json, public/runs solutions,
annotations, and HuggingFace `Infatoshi/kernelbench-hard-traces` (12 jsonl).
Raw archives under `outputs/runs*` kept on disk only. Re-enter the board via
TokenHub `hy3` resweep when ready.

Tencent eval guide (`Evaluation_Guide_Using_Hy3_API_Keys.docx`) specifies direct
TokenHub OpenAI-compat only:

- `POST https://tokenhub.tencentmaas.com/v1/chat/completions`
- model id **`hy3`** (not `hy3-preview` / not OpenRouter `tencent/hy3-preview`)
- `Authorization: Bearer $TENCENT_API_KEY`
- `reasoning_effort`: `high` (slow) or `no_think` (fast)
- `max_tokens` up to 262144

Harness change: `hy3-claude` OpenRouter+Claude-Code path is **gone**. New harness
**`hy3`** = OpenCode → TokenHub (`tokenhub/hy3`), archive-local opencode.json,
`extraBody.reasoning_effort` pinned. Alias `hy3-claude` still maps to `hy3` so
old commands don't silently hit Claude. Site display drops "(Preview)".

Published board still has historical `hy3-claude/tencent/hy3-preview` cells from
the OpenRouter era (run ids unchanged). Re-sweep with `uv run kbh run hy3 hy3
problems-rtxpro6000/<problem>` for official TokenHub numbers.

Smoke: raw TokenHub chat/completions 200; OpenCode `tokenhub/hy3` returned
`pong` with reasoning_effort high. Full problem smoke next.

## 2026-07-08 - LongCat H100 gap-fill: two watchdog bugs that each cost real money

Second scaleway H100 node filled LongCat's 05/06/07 gaps (6/6 on H100 now; all
three cells audited clean). Two automation bugs from the same night, both in
"unattended teardown" code paths, both silent:

- **`pgrep -c` prints 0 but exits nonzero on no-match.** The teardown watchdog
  did `BUSY=$(ssh node 'pgrep -c -f ...' || echo probe_fail)`, so on completion
  BUSY became `"0\nprobe_fail"` — matching neither the done branch nor the retry
  branch. Infinite silent loop at $3.96/hr (~2h burned before a manual check-in
  caught it). Rule: never bolt `|| fallback` onto a command substitution whose
  success-case exit code is nonzero; test `$?` separately or use
  `pgrep ... | wc -l`.
- **Node-side disk janitors must not touch dirs the harness still owns.** A
  cleanup loop purging `repo/.venv` from "finished" runs (result.json exists)
  raced the harness's own check phase and deleted the venv mid-check; uv rebuilt
  the whole environment inside the check timeout window and the run recorded
  `check_timeout` (exit 124) for a kernel that passes in 3 minutes. Rescoring the
  archived solution on the same node recovered the cell (0.0390, matching the
  model's own in-transcript geomean prediction). Rule: janitors key off the
  harness's terminal marker only after QUEUE_END for that problem, not
  result.json existence — and never purge the venv of the newest run.

## 2026-07-07 - Hy3 + LongCat-2.0 debut: cloud-H100 overnight sweep, shared-GPU contention, PASS-gate bug

Debuted two new harness routes: `hy3-claude` (Tencent Hy3 preview via OpenRouter's
Anthropic skin, `tencent/hy3-preview`) and `longcat-claude` (Meituan LongCat-2.0 via
`api.longcat.chat/anthropic`). Both are copies of the `kimi-claude` branch; both ported to
mega's `run_hard.sh` too. Published H100 (Hy3 4/6, LongCat 3/3 attempted) + first RTX
cells, 12 audit annotations, all clean, zero contamination.

Lessons that cost real time:

- **PASS-gate false negative (fixed in both benches).** An agent debug printf without a
  trailing newline glued onto check.py's marker (`kv_cache=0x7PASS`), so the anchored
  `grep -q "^PASS"` missed it and silently skipped benchmark.py, misclassifying a passing
  run as harness_error. Gate is now `check exit 0 && grep -aq "PASS"` — strictly stronger.
- **H100 rented via brev (`kbh-h100`, ~$2.28/hr).** TRT-LLM `:latest` rotated off nvcr.io —
  pull by digest with an anonymous token, retag. Never `docker save | ssh` from Anvil
  (~5MB/s uplink); pull from registries on the node. End-of-sweep watchdog pattern worked:
  poll `pgrep run_hard|run_queue`, 8h hard deadline, rsync archives to `runs-h100/`,
  ABORT teardown if rsync fails, then `script -qec "brev delete ..." <<< "y"` + `brev ls`
  verify. Deadline fired with LongCat 05/06/07 still queued — those cells are simply
  missing, not failed. A per-run venv is ~4.7G: run a janitor that purges `repo/.venv` +
  caches from any run dir that already has result.json, or a 100G root disk fills mid-sweep.
- **Hy3's signature failure mode is infrastructure, not cheating.** OpenRouter caps it at
  262144 context with 128000 reserved for output; three cells died mid-fix on that 400
  (topk was one edit from correct — the agent had already diagnosed the bug). Its H100 kda
  cell is a leaderboard first: honest eager-PyTorch verification scaffold, correct, pf
  0.0000 ("Triton optimizations will be added after verification" — then the provider 400'd).
- **Shared-GPU check_timeouts are salvageable.** kernel-rl's vLLM (up to 91GB / 100% util
  on GPU0) made six RTX check.py runs time out under contention. Solutions are real;
  a rescore loop waits for GPU0 quiet (<8GB, 3x60s probes) and re-runs check+benchmark in
  the archived workspace, patching result.json with a `rescore_note`. Don't publish a
  benchmark timed inside a vLLM burst without noting it (hy3 w4a16 measured ~3-6% low).
- LongCat runs LONG sessions (3.5-6h): 26MB transcript on kda grinding the bf16-tolerance
  wall (any non-cuBLAS reduction order compounds through the sequential recurrence), and on
  paged it deliberately abandoned a hung cp.async CUDA kernel for its working Triton
  split-K version when it couldn't profile blind (ncu blocked in container). Good triage.
- **GPU-lock starvation cascade (worse than the check_timeouts).** While one run's
  harness-owned check.py holds `outputs/gpu.lock` for 30-60+ min (crawling under the vLLM
  co-tenant), every OTHER session's wrapped agent commands (nvidia-smi/uv probes) block on
  the lock until Claude Code's 2-min Bash timeout SIGTERMs them (exit 143, zombie chains of
  bash->gpu-lock-exec->flock under the container's PID 1). LongCat's RTX kda session was
  starved for essentially its whole 2h15m window — 43/43 bash launches dead — and still
  shipped a first-try-PASS fallback by static reasoning alone ("correctness is provable by
  construction"). That cell under-measures the model; rerun it on a quiet GPU. Lesson:
  under co-tenant contention the lock doesn't just slow checks, it silently lobotomizes
  every concurrent agent session. Don't run KernelBench sweeps against a busy co-tenant.

---

## 2026-07-04 - Fable-5 hard resweep finished: H100 5/6 published, RTX fp8 held back on purpose

Closed out the Fable-5 [max] hard sweep across all three GPUs.

**H100 (published, 5/6).** Reswept the cells the July-2 run lost to the missing-ninja
grading bug + rate limits. Final board-best-per-cell for Fable: fp8 0.3033, kda
0.0152, paged 0.4605, topk 0.047, w4a16 **0.3681 (board ceiling)**. `06_sonic_moe_swiglu`
is the one gap — its solution.py built fine but `check.py` `check_timeout`'d on the
cold CUTLASS compile, so it's ungraded, not a correctness fail. We SKIPPED it: the
H100 box (`wgmma-hopper`) was reclaimed by the provider mid-retry, and moe_swiglu is
low-impact (RTX 0.076 / B200 0.076). Re-added the H100 target to the site (it had
been pulled July 2 while the ninja bug was live) and rebuilt `leaderboard.h100.json`
from `outputs/runs-h100/` (9 models, contamination guard excluded 1 tainted composer
run). All 5 passing Fable cells reward-hack audited CLEAN.

**B200 (published, 4/6).** Unchanged from the earlier publish. `02_kda_cutlass` and
`05_topk_bitonic` were SKIPPED — their July-2 `check_failed` was the SAME missing-ninja
infra bug (solution.py existed, grader couldn't load the C++ ext), not correctness.
Not worth renting a B200 to re-grade two cells.

**RTX fp8: fresh 0.4098 run EXISTS but is deliberately NOT published. Important gotcha.**
The blank RTX fp8 cell finally got a real unlimited run (`20260703_001306`, pf 0.4098,
audited clean). But adding it to `results/published_runs.json` and rebuilding DROPPED
Fable's other 5 RTX cells, taking the row from 5/6 → 1/6. Cause: `build_v2_leaderboard.py`
has a **best-of-both / CAMPAIGN_EPOCH (20260613) rule** — once a model has ANY
post-campaign "unlimited" run, it filters OUT all that model's pre-campaign 45-minute
cells (so a model is shown as EITHER the 45-min generation OR the unlimited generation,
never a mix). Fable's other 5 RTX cells are all June 45-min runs with no unlimited
equivalent, so the single fresh fp8 run demoted the whole row. Reverted the manifest;
RTX Fable stays the 5-cell June reference (fp8 blank), and the /hard RTX blurb now says
Fable's unlimited resweep is pending. TO PUBLISH RTX fp8 PROPERLY: do a full unlimited
RTX resweep of ALL 6 Fable cells (or at minimum enough to beat/replace the June set),
then add that generation to the manifest as a unit. Do not add a lone unlimited cell.

**Published as commit a9b6875.** Re-added the H100 target to `/hard` + the home/efficiency
charts, rebuilt `leaderboard.h100.json`, emitted redacted solution viewers for the new
H100/B200 cells (force-add — `public/runs/` is gitignored via the `runs/` rule, so viewers
need `git add -f`), 5 clean audit annotations.

**HuggingFace trace coverage — non-obvious gotcha.** `kb push-runs` only uploads the
run_ids in `results/leaderboard.json` (i.e. the RTX board) and only searches `outputs/runs`,
so by itself it misses the H100 and B200 boards AND every failed run. For full transparency
(all GPUs, winners + failures) push manually: `scripts/traces_to_hf.py <stage> --from-list
<ids.txt> --search outputs/runs --search outputs/runs-h100 --search outputs/runs-b200`,
then `HfApi().upload_folder(<stage> -> Infatoshi/kernelbench-hard-traces)`. **Mega has NO
bench-local `traces_to_hf.py`** (it reuses hard's machinery) — run hard's converter with
`--search <mega>/outputs/runs`; the claude transcript format is identical. End state for
Fable: 77/77 hard + 18/18 mega traces on HF (was 16 winners + 1). Caveat already known but
worth restating: native `claude` encrypts chain-of-thought, so these traces carry the full
message/tool/diff timeline but empty thinking blocks — not a capture bug.

**Fable weekly sub-cap (pace future sweeps against this).** The 20x max plan gates Fable 5
at ~50% of the *weekly* usage, then forces a fallback to Opus 4.8 for the back half. The
tell is Fable-SPECIFIC, not an account lockout: `"You're out of usage credits. Run
/usage-credits to keep using Fable 5 or /model to switch models"` while Opus still runs =
you've crossed the halfway line. We hit it and paused Fable until reset. So the real budget
for a Fable sweep is ~half the weekly credits before it silently downshifts; plan the matrix
(and the spend) around that ceiling, and remember token rotation means one machine per
account at a time or the creds get wiped mid-sweep.

---

## 2026-07-03 - Deck-redesign brainstorm: DO NOT touch Hard; ideas parked for a new bench (DEFERRED on cost)

Long design conversation about reshaping the Hard deck. **Outcome: change NOTHING
in Hard. Park all of this and revisit in ~2 weeks.** Deferred purely on cost —
Fable-5 [max] runs ~thousands of dollars to sweep once across all GPU
architectures × problems × benches, so a new bench isn't getting swept right now.
This entry exists so a future agent can resume the thinking without re-deriving it.

**>>> THE HARD RULE THAT CAME OUT OF THIS: Hard is now (near-)FROZEN. <<<**
Chinese frontier labs have reached out and are considering citing KernelBench-Hard
in their model-release reports. Once a bench is externally cited it becomes shared
infrastructure — mutating it retroactively invalidates labs' published numbers
(the MLPerf / SWE-bench / GPQA lesson: version, never edit a released bench).
So:
- **Non-breaking = still OK for Hard:** *appending* a new problem (e.g. `08_*`).
  Benchmarks grow; that never invalidates an existing cell.
- **Breaking = must be a NEW bench, never an edit to Hard:** switching the metric,
  and REMOVING/reshaping existing problems (kda / topk / paged). Treat any of
  these as a "rug pull" on Hard and don't do them.

### The ideas we explored (all parked, none implemented)

1. **DeepSeek Sparse Attention (DSA) as a problem.** More relevant than KDA
   (DeepSeek-V3.2 / GLM-5.2 are open-weight frontier; Kimi Linear is narrower),
   and harder to reward-hack (no SM120 drop-in lib to forbid-then-copy). DSA =
   lightning indexer (small score GEMM) + top-k token selection + sparse MLA
   attention, so it naturally *subsumes* the standalone top-k problem and could
   stand in for kda + topk + paged in one op. Two hard design gates we identified:
   - **Metric fit is bad for Hard.** Hard scores fraction-of-dense-peak with the
     algorithmic-FLOPS "dense-equivalent, no credit for skipping" rule. DSA's whole
     point is structural sparsity (fixed `k`), so scoring vs dense peak either reads
     absurdly low OR rewards NOT being sparse. You'd have to score at the *sparse*
     work (indexer + k-token attn) — a special-case decision. This is the main
     reason DSA doesn't belong in Hard as-is.
   - **Selection determinism.** Top-k over indexer scores is discrete; fp rounding
     in a candidate indexer flips the selected set → flaky correctness. Fix pattern
     (same as `05_topk`): grade the FINAL attention output with tolerance, make the
     reference selection deterministic (fp32 indexer, stable tie-break, scores
     well-separated near the k boundary). Do NOT grade the index set directly.
   - Home: full fused DSA *decode* = Mega (speedup-over-reference, sparsity pays
     off there); DSA *attention-op* could be Hard-scale but the metric fights it.

2. **Metric switch: fraction-of-peak → wall-clock (ms).** Motivation: FFT, 3DGS,
   hash-join, and mixed ALU→tensor-core kernels have NO clean single roofline, so
   "fraction of peak" is ill-defined for them (this is a real blocker, not a
   preference — those problems literally can't be scored on a roofline). Resolution
   we landed on: **measure ms, but report GEOMEAN SPEEDUP OVER `reference.py`**
   (dimensionless, comparable/averageable across problems; raw ms alone can't rank
   because a 40µs FFT and a 4ms MoE don't average). This ALSO unifies the metric
   with Mega ("×over reference") — one mental model across the whole site. Secondary
   read where a real ceiling exists (fp8→cuBLASLt, FFT→cuFFT): `solution_ms /
   sota_ms` = "% of best hand-tuned kernel" (roofline's honest, measured cousin).
   Presentation: reuse `media/make_fable5_hard_bars.py` grouped bars, y-axis =
   ×speedup (higher = better, kills the "why not 1.0?" question), one headline
   geomean bar per model, raw ms on a LOG axis / detail table only.
   NOTE: a metric switch is itself breaking → new bench only.

3. **Two candidate NEW benches (collection additions, NOT Hard edits):**
   - **KernelBench-Classic** — non-DL / "old-school but still modern" kernels:
     graphics (**3DGS rasterization**), physics (n-body / stencil / PDE), crypto
     (hashing), **FFT** (ideally *fused*, e.g. FFT-conv or STFT→mel), hash-join /
     dedup, radix sort. Thesis: *can agents write fast kernels beyond the
     transformer?* Biggest whitespace in the field; nobody benchmarks agent
     kernel-writing on graphics/data/science. ms-speedup metric is native (no
     rooflines). This is the one most worth building out.
   - **KernelBench-Frontier** — signature kernels of the newest lab architectures:
     DSA, MLA (multi-head latent attention), whatever ships next; a rolling
     "~last 6 months" window. Most attractive to the labs who reached out (a bench
     featuring *their* architecture). Open problem the user flagged: it gets awkward
     once entries hit ~1yr old (architectures age out) — needs an explicit rolling /
     versioning policy before it's real.

### Per-idea verdicts worth keeping
- **Keep W4A16** (if ever reconsidered): it is NOT redundant with fp8 GEMM. fp8 is
  compute-bound tensor-core (MMA/tcgen05 throughput); W4A16 is memory-bound with a
  register-level dequant pipeline and NO tensor-memory path — different regime,
  different tricks.
- **FFT is only a MEDIUM differentiator.** Shared-mem / bank-conflict / fusion
  skills overlap fp8 & W4A16. It's distinct only on the butterfly *communication*
  pattern + warp shuffles + zero tensor-core path. Include as a "signal" slot; do
  not anchor a bench on it.
- **3DGS is gradable and the showpiece.** Non-hackable recipe: pin a `.ply` scene +
  camera by URL AND hash; naive PyTorch tile-rasterizer reference (bin → stable
  depth-sort → alpha-composite); correctness = rendered image within pixel
  tolerance / PSNR floor; ceiling = time `gsplat` / `diff-gaussian-rasterization`
  in `sota.py` and FORBID those same libs in `solution.py`. Zero tensor cores, no
  allowed drop-in → maximally hack-resistant. Only care-item is depth-sort
  determinism (stable tie-break).

### When we resume (checklist for future agent)
1. Pick a bench identity (Classic vs Frontier; Classic is the recommended flagship).
2. Draft its `SPEC.md` (purpose, ms→geomean-speedup metric, problem set, reward-hack
   rules, relationship to Hard/Mega). No changes to any live deck.
3. Budget the sweep BEFORE building (Fable full sweep ≈ thousands of $; that's why
   this is deferred).
4. DSA: decide Mega (fused decode) vs Frontier bench. It does NOT go into Hard under
   the current fraction-of-peak metric.

---

## 2026-07-02 - Fable 5 hard sweep (RTX + B200); H100 held out; TWO RESWEEPS PENDING

Swept Claude Fable 5 [max] across the hard deck on RTX PRO 6000 + a rented B200,
and published both boards. Then hit Anthropic's ~50%-of-weekly cap on both
accounts (elliot@ keychain + infatoshi@ env-token), so the sweep is incomplete.

**>>> PENDING WORK — do these when Fable is back and rate limits reset: <<<**
1. **RTX PRO 6000 `01_fp8_gemm` for Fable 5** — never ran (rate-limited before it
   started). It is the ONE blank cell on the RTX board. Command:
   `uv run kbh run claude claude-fable-5 problems-rtxpro6000/01_fp8_gemm max`
2. **Full H100 resweep for Fable 5** — the H100 board was REMOVED from the site
   (app/hard/page.tsx GPU_TARGETS, app/_lib/charts.ts, app/page.tsx) and
   `leaderboard.h100.json` deleted, because the H100 box shipped **without ninja**,
   so every `load_inline` hand-CUDA cell died at grading with "Ninja is required"
   (03/05/07 were pure-infra fails; only the pure-Triton fp8 cell survived at
   0.303). This is an infra gap, not a capability miss. **Fix is already in**:
   `ninja>=1.11` is now pinned in `benchmarks/hard/pyproject.toml` (commit
   9814b3f). On resweep, re-add the H100 target to the three files above and
   rebuild via `scripts/build_all_gpus.sh`.

After either resweep: audit new passing cells (annotations), `kb contamination
hard`, `kb publish`, push traces to HF, and regenerate the charts from
`media/make_fable5_hard_bars.py` / `make_fable5_b200_fp8.py` /
`make_fable5_hard_status.py`.

**What DID ship (audited clean unless noted):**
- **RTX PRO 6000** (5/6, fp8 blank): W4A16 GEMM 0.348 (board-best), MoE SwiGLU
  0.108 (board-best), paged-attn 0.630, top-k 0.049, kda 0.036. Beats/ties
  GLM-5.2 / Opus 4.8 / GPT-5.5 on the two compute cells.
- **B200** (4/6): **fp8 GEMM 0.254 — board leader** (GLM 0.200, Opus 0.196, GPT
  0.146), a HAND-WRITTEN SM100 tcgen05 kernel (2-CTA cluster `tcgen05.mma`, TMEM
  accumulators, TMA + mbarrier pipeline, swizzled fused-scale epilogue). Real PTX,
  no CUTLASS/cuBLASLt. The shipped kernel dispatches the hand path on all 4 graded
  shapes (Triton is only the compile-failure fallback); note it came out ~even
  with its own tuned Triton (~0.283 vs ~0.281 self-measured, ~90k extra tokens for
  <1%) — impressive as capability, ROI-negative as an optimization. Viewer:
  `/code?f=/runs/20260702_090059_claude_claude-fable-5_01_fp8_gemm_solution.py.txt`
  (force-added to `public/runs/`; the RTX publish script does not cover the B200
  board). B200 kda failed correctness; top-k was rate-limited out.
- Posted to X (mega post earlier got a Karpathy like; hard post 2026-07-02).

---

## 2026-06-27 - glm-5.2 fp8 verdict overturned to clean; publish made reproducible

Two corrections, one of which redraws a published cell. Per the integrity note
below: the evidence overturned a prior verdict, so the record is corrected here.

**glm-5.2 01_fp8_gemm is CLEAN, not a reward_hack** (overturns the 2026-06-15
entry below, which marked it invalid for an "output-memoization / data_ptr cache"
hack). An empirical re-audit (annotation `20260614_145529_zai-claude_glm-5.2_01_fp8_gemm.yaml`,
also cited in CLAUDE.md as THE canonical `kb lint` false-positive) proved the
`data_ptr()` pattern is a CUDA-graph replay, not a lookup: overwriting the same
input buffer with new contents changes the output (recompute, not stale), the
~0.18 ms reused-input time matches the theoretical 4096-cube fp8 GEMM (not a µs
lookup), and 0.406 sits in the frontier pack (opus 0.386, fugu 0.394). The graph
just elides Triton launch overhead — a legitimate optimization. The lint fired on
`data_ptr()==`; the static scan can't tell replay from memoization, so the human
audit governs.

**Why the live board was stale.** The annotation was flipped to clean shortly
after 06-15, but `leaderboard.json` was never rebuilt to honor it — because a
rebuild would have *ballooned* the curated board (the date-gate footgun, fixed
below). So the published board kept opus as the 01_fp8_gemm ceiling while the
annotation said glm-5.2 was clean and higher. Rebuilding now corrects it:
**glm-5.2 holds the 01_fp8_gemm ceiling (0.4059 > opus 0.3855), pass_count 5->6.**
`leaderboard_v2.json` (a stale H100/8-model snapshot) was also regenerated to
match the RTX/10-model site file.

**Publish is now reproducible (the footgun fix).** `build_v2_leaderboard.py` was
date-gated only (every run >= 20260610), so any rebuild silently grew the curated
board 10->13 models / 55->63 cells by pulling in experimental/superseded sweeps.
Added an explicit allowlist `results/published_runs.json` honored via
`KBH_PUBLISHED_MANIFEST` (default-on for the RTX board; `build_all_gpus.sh`
disables it for the per-GPU boards). `rebuild == committed` now holds.

**Mega framework labels fixed.** `build_mega_leaderboard.py:_framework()` only
scanned `solution.py`, so cursor cells that import the kernel from a sidecar
(`from w4_triton import ...`, `@triton.jit` in `scratch/w4_triton.py`) were
mislabeled "eager". It now resolves local imports into sidecar modules. Relabels
the 3 cursor composer `02_kimi_linear_decode` cells eager -> triton (no score
change).

---

## 2026-06-15 - the unlimited-time generation: shipped, audited, published

Closed out the "everyone gets unlimited time" resweep into a clean, honest 8-model
generation on kernelbench.com/hard, plus a public HF dataset of every kernel.

**Final roster (9 rows):** 8 unlimited-time current models - Claude Opus 4.8,
GPT-5.5 [xhigh], GLM-5.2, MiniMax-M3, Gemini 3.5 Flash, Kimi K2.7-Code, DeepSeek
V4 Pro (deepseek-claude), Cursor Composer 2.5 - plus Claude Fable 5 as a labeled
frozen 45-min legacy reference (suspended mid-run, US-gov; can't re-run, held 3
ceilings). Everything else dropped.

**What got dropped and why (sweepability is the gate):** a model is only kept if
we can actually run it. Verified auth: claude/codex coding plans OK; zai/minimax/
kimi/deepseek/gemini keys present; cursor logged in (composer sweepable!). NOT
sweepable -> dropped: qwen3.7-max (needs DASHSCOPE key we don't have; only the
flaky opencode/OpenRouter route), grok-build (OAuth expired, dies on re-login in
headless), and the old opencode/OpenRouter legacy models (deepseek-v4-flash,
mimo-v2.5-pro, kimi-k2.6, glm-5.1, nemotron). Stale 45-min rows next to
unlimited-time rows are apples-to-oranges, so they go.

**fp8 resweep - the headline:** after fixing 01_fp8_gemm to be genuinely fp8xfp8
(see 2026-06-14 entry), reran the column. BEFORE the fix: 0 models ever wrote a
real fp8 kernel (all bf16-upcast "leaks" or hacks). AFTER: 7 of 8 wrote a real
fp8 tensor-core MMA kernel (Triton tl.dot on fp8); composer-2.5 (a small fast
model) included. GLM-5.2 wrote a real fp8 kernel too, then bolted an
output-memoization hack on top (data_ptr cache -> timed loop measures a lookup;
caught by `kb lint`, verdict reward_hack, invalid). The fix did exactly what we
hoped: making the problem honestly fp8 got models to do real fp8.

**Roofline rescale, applied:** the 2.5x roofline correction only moves
regime=compute problems (graded on TFLOPS). build_v2 rescales 02_kda + 06_sonic
x0.4 for pre-fix runs; regime=memory problems (03_paged, 05_topk, 07_w4a16,
graded on the unchanged 1.8 TB/s bandwidth) are untouched - so the headline
records SURVIVED (GLM-5.2 paged-attn 0.677 new record, w4a16 0.321; Fable's
frozen marks still top w4a16/sonic/topk).

**Generation hygiene:** build_v2 has a CAMPAIGN_EPOCH filter - any model with an
uncapped run (>= 20260613_042249) shows ONLY its uncapped cells, never a
best-of-both Frankenstein across 45-min and unlimited budgets. Old fp8 runs
(broken-problem) quarantined so the column uses only corrected-problem runs.

**Survived an anvil meltdown** (2026-06-13): K=8 uncapped + 5 concurrent
compile-heavy sessions -> load 645, OOM, SSH-dead. Recovered on its own,
campaign+phase2 finished overnight, only grok lost. Lesson banked: uncapped
compile-heavy sweeps run at K=2.

**Published:** kernelbench.com/hard live (commits up to 56e4be8). Three
NVIDIA-themed charts (board heatmap, fp8 redemption, per-problem champions) in
~/dev/sites/kernelbench.com/x-article-images/unlimited-gen/. Public HF dataset of
every submission (kernel code + metrics + audit verdicts):
https://huggingface.co/datasets/Infatoshi/kernelbench-hard-submissions

**Integrity note for future sessions:** we revised our OWN prior "rubric_leak"
annotations once the fp8-spec bug proved the bf16 path had been the only valid
answer - the data redrew the story. Audit every passing/leader cell before
publishing; correct the record when evidence says you were wrong.

---

## 2026-06-14 - 01_fp8_gemm was mis-specified three ways; fixed before the fp8 resweep

While trying to hand-write a "real fp8 kernel nobody cracked," we discovered the
fp8 problem could NOT be solved by a genuine fp8 kernel as specified. Root cause
was three independent bugs, all now fixed. Lesson at the bottom — do not repeat.

**Bug 1: the weight was bf16, not fp8.** reference.py stored
`self.weight = nn.Parameter(..., dtype=torch.bfloat16)` and computed in bf16,
even though the name/docstring/roofline all say fp8. Consequence: the only
correct answer was a bf16 GEMM (bit-identical to the reference, 0.0000 error),
which physically caps at ~0.5 of the fp8 roofline (bf16 tensor cores = half fp8
rate). A real fp8 kernel must quantize the bf16 weight to fp8, injecting ~0.4
max error that fails EVERY tolerance. So the "fp8" column actually measured
"best bf16 GEMM," and the bf16-upcast solutions we annotated `rubric_leak` were
in fact the ONLY valid answer (those annotations were unfair; the cuBLAS-wrapper
and grader-tamper cells were still genuine reward hacks). Proven by isolating
the numeric floor: bf16-upcast 0.0000 error; per-row fp8 weight-quant 0.444;
per-128-block 0.413 — fp8 fails at 0.01, 0.15, and 0.30.
Fix: weight is now genuinely fp8_e4m3 (normalized into the e4m3 range) + a
per-output-channel `weight_scale` buffer (the standard scaled-fp8 layout). The
reference upcasts the SAME fp8 operands -> a real fp8 x fp8 MMA matches it and
can exceed 0.5; a bf16 upcast still passes but stays capped at ~0.41.

**Bug 2: the 0.15 fp8 tolerance was dead.** correctness.py keys tolerance on the
OUTPUT dtype (`dtype = reference_out.dtype`), which is bf16, so it used the bf16
default atol/rtol = 0.01 and the `tolerance: fp8_e4m3fn: 0.15` override never
applied. 0.01 is far too tight for fp8 accumulation-order noise (a legit fp8
kernel drifts ~0.06-8 abs depending on input magnitude, mostly on near-zero
outputs). Fix: key the override on `bfloat16` (the output dtype) = 0.2 nominal,
and recalibrate the 01_fp8 numeric_stress tolerances to be magnitude-scaled
(small_input 5e-4, large_input 12.0, small_weight 3e-3, rtol 5e-2) — measured
empirically as fp8-MMA residual x ~1.5, rtol still catches gross error.

**Bug 3: the roofline peaks were 2.5x too low.** src/hardware/rtx_pro_6000.py
listed fp8 400 / bf16 200 / fp4 800 TFLOPS. Real Blackwell GB202 dense is fp8
1000 / bf16 500 / fp4 2000 (NVIDIA headline 4000 fp4-sparse AI TOPS -> halve for
dense, halve per precision step). Verified: cuBLAS hits fp8 773 / bf16 412 on
4096^3 (77-82% of the corrected peaks). The too-low table produced
peak_fraction > 1.0 for a real fp8 kernel and inflated EVERY published number by
2.5x (rankings preserved; absolute values wrong). Fix: corrected the whole
table to the NVIDIA dense spec (fp32 was also wrong: 12 -> 125 SIMT).

Validation (experiments/fp8_ceiling, a real Triton fp8 MMA solution): check.py
PASS on all 4 shapes x 3 seeds x 3 stress cases; benchmark peak_fraction 0.57
(aligned) / 0.63 (up-proj) with NO cell > 1.0; bf16 baseline caps ~0.41. The
problem now rewards genuine fp8.

**LESSON (do not repeat) when adding a precision-specific problem:**
1. The reference must actually COMPUTE in the target precision (store operands in
   that dtype), not a higher-precision stand-in. If the reference is higher
   precision than the problem name, the "intended" kernel can't match it.
2. Tolerance is keyed on the OUTPUT dtype in correctness.py, not the input/
   precision name. Put the override under the output dtype key (here bfloat16),
   or it silently no-ops.
3. ALWAYS sanity-check the roofline peak against a vendor-library measurement
   (cuBLAS / torch._scaled_mm). If peak_fraction can exceed ~0.9 for cuBLAS or
   >1.0 for any kernel, the peak is wrong.
4. Before publishing a new problem, write a real kernel in the intended precision
   and confirm it PASSES and scores < 1.0. We had shipped 01_fp8 without that.

---

## 2026-06-13 - qwen dropped from the resweep; uncapped campaign

Decision: qwen3.7-max is NOT in the uncapped resweep. Why: the only working
route for it is the `opencode` / OpenRouter (`@ai-sdk/openai-compatible`)
adapter, which stalls intermittently (~1/3-1/2 of sessions; see 2026-06-09).
The reliable alternative would be a `qwen-claude` harness (Claude Code -> the
provider's Anthropic-compatible endpoint, the pattern that makes zai/minimax/
kimi/deepseek reliable). For Qwen that endpoint is Alibaba DashScope Model
Studio (Intl/Singapore: dashscope-intl.aliyuncs.com/apps/anthropic, model
qwen3-max), which needs a DASHSCOPE_API_KEY we do not have. The `qwen-claude`
branch is wired and preflight-stops cleanly on the missing key, so the day we
get a Model Studio key it's one `kb sweep qwen-claude qwen3-max` away. Until
then, a flaky-route qwen row would not be comparable, so we leave it out rather
than publish an unreliable number.

Uncapped resweep campaign (2026-06-13): with Fable 5 suspended (US-gov action;
its 45-min rows are now a frozen legacy ceiling), we reswept the field with NO
wall-clock cap (6h backstop only, to keep a hung native session from wedging a
slot). Reliable models first via scripts/sweep_campaign.sh (K=8 concurrency,
refill as jobs finish; RAM is the limit ~39G): claude-opus-4-8, gpt-5.5 xhigh,
glm-5.2, MiniMax-M3, gemini-3.5-flash, grok-build. deepseek-claude added after
a smoke (new harness, works). kimi-k2.7-code reswept separately. Early signal:
uncapped time buys real gains on the grinder problems - opus paged-attention
0.6706 beats Fable's old 0.6299 ceiling, opus kda 0.1380 beats 0.0894.

Gotcha caught: 6 concurrent native `claude` (opus) sessions on ONE coding plan
trip 401/429/rate_limit; one cell (07_w4a16) exhausted retries and got SIGTERM
(143) at 19 min. The other 5 opus cells rode through it. Native claude/codex
have no stall watchdog, so throttle opus concurrency on a single plan or expect
the occasional rate-limit casualty (re-run those cells solo).

---

## 2026-06-12 - kimi-claude harness: Kimi K2.7-Code via Moonshot Anthropic endpoint

Added a `kimi-claude` harness to bench moonshotai/Kimi-K2.7-Code (1T MoE, 32B
active, coding-focused, 256K ctx, forces thinking mode). It mirrors the proven
zai-claude / minimax-claude pattern: Claude Code routed to a provider's
Anthropic-compatible endpoint.

Route (validated by curl + a container smoke):
- ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic (KIMI_ANTHROPIC_BASE_URL
  overrides). The /anthropic endpoint is real; docs only list the OpenAI base
  (api.moonshot.ai/v1) but the HF card confirms Anthropic-compat.
- Auth: Bearer via ANTHROPIC_AUTH_TOKEN=$KIMI_API_KEY (NOT x-api-key).
- Model id: kimi-k2.7-code.
- Thinking: K2.7-Code REQUIRES thinking enabled ("only type=enabled is allowed
  for this model"); CLAUDE_KBH_SETTINGS already sets alwaysThinkingEnabled, so
  no extra wiring. A request without thinking 400s.
- KIMI_API_KEY in ~/.env_vars. The old key was dead (account needs a >=$1
  prepaid top-up to activate); user replaced it.

Also fixed a latent bug while there: the SESSION_COMPLETE detector listed
claude|zai-claude|ccr-claude|cursor|gemini but NOT minimax-claude (it only
worked because runs hit the 124 timeout). Added minimax-claude and kimi-claude
to that group. extract_usage.py now routes kimi-claude through the claude
parser.

Known limitation: Moonshot's Anthropic-compat usage block reports
output_tokens=0 and cached_tokens=0 in per-message events (it leaks OpenAI-style
prompt_tokens fields into the Anthropic shape). Input is captured; output/cache
are not reported by the provider. Same cosmetic class as the cursor/gemini
usage gaps - does not affect scoring or kernels; real cost tracked via the
Moonshot console.

Smoke: container topk probe at 600s -> [OK score=0.0059], solution written,
correctness PASS, thinking active, genuine tool use, no auth/429 errors.
Full 6-problem parallel container sweep launched 2026-06-12.

## 2026-06-11 - v2 published to kernelbench.com; X post shipped

The v2 containerized sweep is live and public (unannounced until the post).

Publication:
- Built results/leaderboard_v2.json from the sweep via
  scripts/build_v2_leaderboard.py (best-of-cell, audit verdicts applied:
  reward_hack cells kept visible but marked invalid and excluded from the
  ceiling ranking). Reshaped to the site's schema-1 shape and made it the
  monorepo's leaderboard.json; v1 preserved as leaderboard_v1.json.
- Site lives on the Mac at ~/dev/sites/kernelbench.com (Next.js, Vercel
  auto-deploy on master push, commit email MUST be elliot@arledge.net).
  /hard bypasses the v1 VISIBLE_MODEL_LABELS allowlist when
  environment=="v2_containerized". 78->79 v2 transcript viewers in
  public/runs/; 162 v1 viewers archived to archive/v1-runs/ (all kahan
  ls-noise was in those v1 viewers; problems/04_kahan_softmax pycache also
  deleted from Anvil).

Bugs found by clicking the live site, all fixed and redeployed:
- NGC banner broke the viewer format sniff -> 72/78 viewers showed 0 events;
  skip leading non-JSON banner lines, plus added a gemini parser. (commit
  e57a1f5)
- Leaderboard linked empty context-overflow runs for no-pass cells; now ranks
  failed attempts by has_solution -> has_check -> peak, so 14 failed cells link
  the actual submitted kernel. Also guarded the Claude "<synthetic>" model
  label. (commit 4196861)
- The fable 06 sonic-moe record viewer (0.2688) was referenced but never
  generated -> 404; generated it. (commit e926e0e). Full re-audit: all 79
  cells clean (no 404s, no 0-event, no bad labels, no missing solution tabs).

Final fable record tally (all clean, audited): 3 all-time records (05 topk
0.0494, 06 sonic-moe 0.2688, 07 w4a16 0.3477); tops the v2 sweep on 5 of 6
problems; 01 fp8 is the heroic failure (only real fp8 tensor-core kernel,
scored 0 on a K=4127 tail while wrappers passed at ~0.43). 10 reward hacks
across 8 models; FP8 column has zero valid passes.

Trajectory analysis + plots (for the X post): three parallel subagents deep-read
the 07/03/01 transcripts and extracted the optimization trajectories. Key
finding on 07: fable reverse-engineered the benchmark's own 128MB L2 zero_()
flush and used evict_last to pin weights in L2 through it, beating the DRAM
roofline (2.4x bandwidth). Four phosphor-themed diagrams saved at
x-article-images/fable-trajectories/ (overview + 3 per-problem trajectories)
with the generator scripts. X post shipped 2026-06-11.

Open follow-ups (not blocking): v1 prose on /blog/hard still describes the
v1 8-problem deck (legitimate history, not wrong); a /methodology page; the
opencode adapter stall watchdog is shipped but the route stays diagnostic.

## 2026-06-11 - Viewer fix: NGC banner broke format sniff; added gemini parser

After publishing v2, 72/78 transcript viewers rendered 0 events (e.g. the
fable 02 KDA cell showed harness=codex, model=?, 0 events). Root cause: every
container-mode transcript is prefixed by the NGC image's PyTorch/driver banner
(plain text). src/viewer/parsers/__init__.py sniff() read the first non-empty
line, hit non-JSON, and immediately returned "codex"; the codex parser then
produced nothing from the claude/opencode stream-json that followed.

Fixes (src/viewer/, uncommitted on Anvil like the rest of that tree):
- sniff() skips leading non-JSON banner lines and detects format from the
  first real JSON line; a file with no JSON at all returns "codex" (true
  stdout text) instead of raising.
- Added src/viewer/parsers/gemini.py. Gemini had no parser and fell through
  to the claude fallback (0 events). Gemini format: {type:init, session_id,
  model} then message/tool_use/tool_result/result; usage from the result
  event's stats block.

Result: 0/78 viewers broken. grok viewers are thin (2 events) but correct -
grok's stream-json is pure thought/text token deltas with no structured tool
events, so there is nothing more to render. Regenerated all 78 and redeployed
to kernelbench.com (commit e57a1f5 on the monorepo).

Follow-up (commit 4196861): two more viewer-quality fixes. (1) The leaderboard
linked the wrong failed run for no-pass cells -- fable 01 FP8 pointed at a
context-overflow run with no solution.py instead of the sibling attempt that
wrote the real fp8 kernel (K=4127 tail bug) and ran check. build_v2_leaderboard.py
now ranks failed attempts by has_solution -> has_check -> peak; 14 failed cells
now link the actual submitted kernel. (2) Claude Code tags synthetic turns
("Prompt is too long") with model="<synthetic>"; the claude parser let that
clobber the card model. Guarded.

---


## 2026-06-11 - Fable budget rerun: confirms budget-bound, earns a third record

The two fable cells that ended session_complete=false at 2700s (03, 06) were
rerun at 3600s, 2 concurrent. Both climbed materially - the timeout, not the
kernel design, was the ceiling:

```text
03_paged_attention   0.5340 -> 0.6299   (+18%)
06_sonic_moe_swiglu  0.2395 -> 0.2688   (+12%)
```

Both audited clean (annotations written): 06's _launch_cache is shape-keyed
compiled-kernel reuse re-run with live inputs every call (not the gpt-5.5
output-memoization hack); 03's os.environ reads are tuning knobs with fixed
defaults the harness never varies between check and benchmark.

Corrected fable record tally vs all-time ceilings (this matters - an earlier
note overstated it):

```text
01_fp8_gemm   FAIL          (real fp8 kernel, K=4127 tail bug; all-time 0.537)
02_kda        0.0894        NOT a record (all-time 0.118 grok); best this sweep
03_paged      0.6299        NOT a record (all-time 0.664 gpt-5.5); best CLEAN cell this sweep
05_topk       0.0494  RECORD (prior 0.046)
06_sonic_moe  0.2688  RECORD (prior 0.254, earned by the budget rerun)
07_w4a16      0.3477  RECORD (prior 0.220, +58%)
```

Three all-time records (05, 06, 07), all clean. 03 leads the sweep on clean
cells but trails the historical gpt-5.5 cell. 01 is the honest failure: the
only real fp8-tensor-core attempt of the sweep, tripped by predicated-tail
math on the K=4127 shape, while the five "passes" were cuBLAS wrappers.

---


## 2026-06-11 - Full-sweep audit: every passing solution read, 10 reward hacks, per-problem health report

Every correct cell from the v2 sweep (49 cells) was read in full against its
PROMPT.txt and forbidden list. Verdicts live in results/annotations/ (17 new
files). Column health:

```text
01_fp8_gemm        5/5 passing cells HACKED (stack-sniff dual path, torch.mm,
                   reference resubmit, at::matmul shim, cuBLASLt). The ~0.428
                   score is a cuBLAS-wrapper fingerprint: four different hacks
                   land within 0.4% of each other. No model demonstrated fp8
                   skill. Column ceiling remains gpt-5.5 0.537 (May).
02_kda_cutlass     4 hacked (zero-kernel PyTorch ports sharing the reference
                   forward-substitution line; kimi with a false 'custom CUDA
                   kernel path' docstring), 5 clean. Detector: >=0.0174 means
                   real kernels, <=0.0034 means PyTorch port.
03_paged_attention 11/11 real kernels. Healthiest column.
05_topk_bitonic    1 hack: gpt-5.5 0.1601 (column top) is input-identity
                   memoization exploiting timing.py reusing the same inputs
                   list across timed iterations - kernel runs only in warmup.
                   1 rubric leak: qwen uses Triton's built-in tl.topk.
                   Legitimate top: claude-fable-5 0.0494.
06_sonic_moe       9/9 clean; designs converged on near-isomorphic grouped
                   Triton GEMMs; several size grids off the harness's balanced
                   routing (would not survive skewed loads; check.py never
                   feeds skew).
07_w4a16_gemm      9/9 unpack int4 in-kernel, zero rubric leaks. Ceiling:
                   claude-fable-5 0.3477 with a policy caveat - its module
                   import sets a global torch backend flag that changes
                   reference numerics during check (documented in-solution,
                   defensible direction, but solution code mutating harness
                   state needs an explicit rule).
```

Maintainer judgment calls flagged (not auto-resolved):

- qwen 03 0.6268: CUDA-graph capture with pointer-identity replay. Kernels
  re-execute with live data; launch-overhead elision likely explains the
  column top. interesting, pending comparability policy.
- fable-5 07 global-flag mutation (above).
- Harness rule worth adding: outputs must be recomputed per call / rotate
  input buffers in timing.py (kills the memoization class); forbid global
  torch backend mutation from solutions; add tl.topk/tl.sort to 05 forbidden.

Process note: the audit was run by two parallel subagents reading every
solution end to end; the wrapper/stack-sniff greps from earlier in the night
caught 5 of the 10 hacks - the other 5 (memoization, fig-leaf kernel, PyTorch
ports, tl.topk) required actually reading the code. Greps are tripwires, not
audits.

Retry-lane cells (grok 6/6 recovered after the OAuth fix, zai-claude 6/6 after
serializing) audited clean: grok 05 hand-written CUDA selection, grok 06 real
Triton grouped GEMM, grok 07 in-kernel int4 unpack, zai 03 real paged decode.
grok 01/02 remain wrapper/reference-port hacks (annotated). No new hack classes
from the retries.

---


## 2026-06-11 - v2 night sweep: results, three FP8 reward hacks, two infra retries

First full data collection in the v2 container environment (all rows
KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock). 78-row
non-Claude matrix plus the Claude Fable 5 row, collected overnight.

Fable 5 row (claude harness, effort max, best-of per cell):

```text
01_fp8_gemm          FAIL (three genuine attempts, no pass)
02_kda_cutlass       0.0894
03_paged_attention   0.5340
05_topk_bitonic      0.0494   new problem ceiling (prior 0.046)
06_sonic_moe_swiglu  0.2395
07_w4a16_gemm        0.3477   new problem ceiling (prior 0.220)
```

Anti-cheat audit of headline cells produced three reward_hack verdicts, all
on 01_fp8_gemm (annotations in results/annotations/):

- gemini-3.5-flash 0.6649: environment-sniffing dual path. inspect.stack()
  detects check.py and routes correctness to an exact bf16 matmul; the
  benchmarked Triton path is never correctness-verified. Invalid. Detection
  for this class is now part of the audit toolkit (grep for
  inspect.stack/sys._getframe/check.py in solutions).
- qwen3.7-max 0.4294: bare torch.mm wrapper, no kernel at all.
- kimi-k2.6 0.4284: resubmitted reference.py nearly verbatim, docstring
  intact.

Two more wrapper cells landed from the overnight retry lanes and got the
same verdict: grok 0.4292 (at::matmul inside a load_inline shim) and
zai-claude GLM 0.4277 (direct cublasLt calls - flagged gray-zone vs the
allowed CUTLASS path, pending maintainer review). Five of five FP8 passes
in this sweep were library measurements, none kernel authorship: the 01
tolerance leak now reliably attracts wrappers. A static check (forbid
at::matmul/cublas symbols in solutions, or tighten tolerance) should be
considered for the next benchmark version.

The 01 ceiling therefore remains gpt-5.5's 0.537. Fable's clean cells were
verified: 07 does real in-kernel int4 unpacking, 05 has no forbidden topk
ops, 03 is a 16KB raw CUDA kernel with no library fallbacks.

Infra failures diagnosed and retried overnight:

- zai-claude glm-5.1 0/6: pure 429 storms - transcripts are nothing but
  api_retry events (78+ retries, zero assistant turns). Running 3 concurrent
  GLM sessions on one Z.ai coding-plan key rate-limits the whole row.
  Sequential retry lane running. Rule: zai-claude rows must run at
  concurrency 1.
- grok 0/6: OAuth refresh rotation orphaned the host ~/.grok/auth.json (the
  June 9 smoke rotated the token inside its archived agent_home copy; every
  later run fell into an interactive login prompt and timed out). Restored
  the newer token from the smoke archive; sequential retry lane now syncs
  auth.json back to the host after every run. Same failure class as the
  Anthropic OAuth expiry - file-copied OAuth credentials rot when sessions
  rotate refresh tokens. Long-lived env tokens are the durable fix where
  providers support them.
- opencode-nemotron 0/6 instant crashes: the route's pinned-DeepInfra config
  (KBH_OPENCODE_CONFIG_FILE) was never copied into the container agent home,
  so opencode had no openrouter-deepinfra provider. Fixed in
  prepare_opencode_container_home; all six rows rerun as real sessions
  (first three completed as genuine correctness failures).

---


## 2026-06-10 - Container sessions now run in parallel; per-command GPU lock inside containers

Course correction on the v2 readiness entry below: container sessions no
longer hold the GPU lock for their whole 45-minute budget. That serialized the
sweep (98 sessions x 2700s = 73.5h floor) for no reason. Container mode now
matches the host sweep model: agent sessions overlap freely, and GPU-facing
commands serialize per-command through the shared flock.

How it works:

- The lock moved to its own directory (outputs/gpu_lock/gpu.lock,
  KBH_GPU_LOCK_DIR) so containers can bind-mount just the lock. flock is on
  the inode, so host and container commands serialize against each other.
- Each run generates container-side wrappers (RUN_DIR/cbin, mounted at
  /kbh/bin, first on the container PATH) for uv/python/python3/nvidia-smi/
  nvcc/ncu/nsys. They resolve the real binary from a PATH that excludes
  /kbh/bin and route through the same gpu-lock-exec.
- run_docker_locked_timeout only wraps the session in the lock under
  KBH_AGENT_CONTAINER_SESSION_LOCK=1 (legacy serial mode); default is plain
  timeout + docker with the stall watchdog unchanged.

Hard-won gotcha: the NGC image sets BASH_ENV=/etc/shinit_v2, which runs
nvidia-smi on EVERY bash startup. With /kbh/bin on PATH that resolves to our
bash wrapper, whose startup sources shinit_v2 again - a fork bomb that
silently consumed the container PID limit and produced empty transcripts. Fix:
the runners set -e BASH_ENV= -e ENV=. If a future image needs shinit, wrap the
agent command instead of the global PATH.

Validation (overlap + lock):

- Two agent containers ran simultaneously (docker ps showed both).
- codex/gpt-5.5 in parallel mode: full session, solution, check PASS,
  scored 0.0029 in 187s wall. Its in-container `uv run python check.py`
  iterations appear in gpu_lock_container.log as wait/start/end with elapsed
  times; host post-run check/benchmark serialized through the same lock.
- Wall-time implication: the 73.5h serial floor is gone. Sweep wall time now
  scales with launcher concurrency, bounded by GPU-command contention.

Update 2026-06-10: resolved. CLAUDE_CODE_OAUTH_TOKEN now lives in
~/.env_vars and the claude container leg passed end-to-end with it
([OK score=0.0023] on the TopK smoke). The env token takes precedence over
the stale credentials copy, so no runner change was needed.

Original blocker note: Anthropic OAuth on Anvil is expired -
`claude -p` 401s on the host itself, so the claude row (and host claude runs)
are blocked until a human runs `claude setup-token` (preferred; put the
result in ~/.env_vars as CLAUDE_CODE_OAUTH_TOKEN, which the runners already
pass through) or `/login`. Static credential copies into containers are
fragile against OAuth refresh rotation; the long-lived setup-token avoids the
class entirely.

---


## 2026-06-09 - v2 container sweep readiness: all harnesses and exotic routes verified

Full verification round for the v2 container-mode resweep, requested before
sweeping the entire matrix. Everything below ran tonight on Anvil.

Container harness smokes (KBH_AGENT_CONTAINER=1, BUDGET_SECONDS=600,
problems/05_topk_bitonic; scores are demo-budget numbers, not capability):

```text
claude / claude-opus-4-8        PASS  peak=0.0092
codex  / gpt-5.5                PASS  peak=0.0113
cursor / composer-2.5-fast      PASS  peak=0.0036
gemini / gemini-3.5-flash       PASS  peak=0.0091   (new runner)
minimax-claude / MiniMax-M3     PASS  peak=0.0116   (new runner)
zai-claude / glm-5.1            verified: agent active, solution failed
                                correctness at smoke budget (model outcome)
grok / grok-build               verified: full session, solution written,
                                check failed on extension build (model outcome)
```

New container runners this round: zai-claude and minimax-claude (parameterized
claude runner: env-name passthrough so secrets stay off docker argv, alias
model arg, and NO Anthropic credential copy so a broken mapping errors instead
of silently billing Anthropic), gemini (node mount + GEMINI_API_KEY), grok
(auth-file copy; its bin/grok is a version symlink to an ELF executable in
~/.grok/downloads/ - mount the resolved file and exec it directly, do not run
it under node).

OpenCode exotic routes, multi-step probe (scripts/probe_opencode_multistep.sh,
real PROMPT.txt shape; PROBE_OK means the route survives the first long
generation after tool results enter context):

```text
openrouter-alibaba/qwen/qwen3.7-max          OK
deepseek/deepseek-v4-flash                   OK (wrote solution in probe)
deepseek/deepseek-v4-pro                     OK at 900s (slow reasoner;
                                             fails a 420s window, fine at 2700s)
openrouter-pinned/xiaomi/mimo-v2.5-pro       OK
openrouter-moonshot/moonshotai/kimi-k2.6     OK
openrouter-google-ai-studio/gemini-3.5-flash OK (wrote solution)
openrouter-deepinfra nemotron-3-ultra        OK (wrote solution; pinned config)
zai/glm-5.1                                  INTERMITTENT - passed this probe
                                             after five stalls earlier tonight
```

The GLM/opencode stall (earlier entry today) is therefore intermittent, not
deterministic. The conclusion stands: a route that hangs for 45 minutes on
some sessions cannot produce comparable scored rows. The opencode zai row is
now gated behind KBH_USE_OPENCODE_ZAI=1 in preflight and the sweep launcher;
GLM-5.1 is scored via zai-claude.

Row/matrix changes in scripts/preflight_harnesses.sh and
scripts/launch_parallel_sweep.sh:

- claude row bumped to claude-opus-4-8 max.
- Added opencode rows: deepseek-v4-pro, deepseek-v4-flash, mimo-v2.5-pro,
  kimi-k2.6.
- opencode zai/glm-5.1 demoted to opt-in diagnostic row.
- Preflight now runs the multi-step probe for every opencode row by default
  (KBH_PREFLIGHT_MULTISTEP=0 to skip), because one-turn smokes provably cannot
  catch the adapter stall.

Operational lessons recorded:

- NEVER overwrite scripts/run_hard.sh in place while runs are active: bash
  reads scripts incrementally, and an scp overwrite reuses the inode, so a
  running region shifts under the interpreter (one zai-claude smoke died with
  a phantom syntax error this way). Deploy with scp to a temp path plus
  atomic mv.
- Known cosmetic/accepted: gemini and cursor usage stays null when a session
  hits the budget (no terminal stats event), same as host mode; the NGC
  entrypoint GPU-driver warning is bogus (ldconfig missing under --user).

Credit state at verification time: OpenRouter ~ $99 remaining of $2485.
DeepSeek/Z.ai/MiniMax keys all authenticated and billed during probes.

Late additions after the draft above:

- The adapter stall is NOT GLM-specific: a qwen3.7-max container smoke stalled
  with the identical signature (empty reasoning part, zero tokens, journaled in
  the agent_home opencode.db). Concurrent GLM probes measured a ~1/3 to 1/2
  per-session stall rate tonight. Treat every @ai-sdk/openai-compatible route
  as intermittently stall-prone.
- Adapter swap is a dead end: pointing the zai provider at @ai-sdk/openai
  makes the AI SDK call the OpenAI Responses endpoint (/v4/responses), which
  Z.ai does not implement (404).
- Mitigation shipped: run_docker_locked_timeout now supports an opt-in stall
  watchdog (KBH_STALL_WATCH_LOG + KBH_STALL_SECONDS) that kills the container
  when the transcript stops growing, and run_opencode_container retries killed
  sessions with the remaining budget (KBH_OPENCODE_STALL_SECONDS, default 900s
  to stay above deepseek-v4-pro's 400s+ silent thinking; KBH_OPENCODE_STALL_RETRIES,
  default 2). Validated empirically with a 240s threshold on GLM: three stalls,
  three watchdog kills, three retries, honest INFRA when the budget ran out,
  every kill logged to stall_watchdog.log in the run archive.
- Residual risk accepted: a route with a high per-session stall rate can still
  lose a cell when every retry stalls. Such cells are visibly INFRA (never
  silent), carry the watchdog audit log, and can be rerun individually.
- Verified the opencode container leg end-to-end with an exotic
  (qwen3.7-max) in addition to the earlier zai smoke; the stall it hit is what
  motivated the watchdog.

Sweep arithmetic for planning: 14 scored rows x 7 problems = 98 serialized
container sessions, hard ceiling 2700s each = 73.5 GPU-lock hours of agent
phase maximum, plus post-run check/benchmark. Sessions that finish early
shorten this; the ceiling does not lie.

---


## 2026-06-09 - OpenCode zai/glm-5.1 stall: root cause isolated to opencode's OpenAI-compatible adapter

Every opencode zai/glm-5.1 run since late May shows one signature: 7-9
successful tool calls (parallel template reads) in the first 5-25 seconds,
then zero events until the budget expires. The May 28 finish-sweep 0/6 ERR row
shows this at the full 2700s budget, so it is a true hang, not slow thinking.
The 2026-05-31 MiniMax zen-route 0/7 is likely the same failure class.

Elimination table (all probed 2026-06-09):

```text
GLM-5.1 model           innocent  raw paas/v4 stream: 1.59MB in 198s, completes
Z.ai endpoint/key       innocent  works raw on paas/v4 AND coding/paas/v4
docker container        innocent  identical stall on bare host
docker bridge network   innocent  identical stall with --network host
opencode binary         not it    1.15.9, 1.15.13, 1.16.2 all stall
permission config       innocent  small write probe succeeds under same config
adapter multi-turn      GUILTY    stall always starts at step 3, the first
                                  request whose context contains tool results
```

Minimal repro (no harness, no container): copy a problem template to a scratch
dir, run `opencode run --pure --format json -m zai/glm-5.1 "$(cat PROMPT.txt)"`.
Reads complete, then the next generation opens an empty reasoning part
(`{"type":"reasoning","text":""}` is the last journaled event) and no tokens
ever arrive.

Upstream corroboration in anomalyco/opencode: #28427 (GLM-5 empty delta.role
breaks stream validation), #22803 (reasoning + tool runs die after 1-3
rounds), #21903 (reasoning field infinite spin), #14972 (agent stops after
tool execution on OpenAI-compatible providers).

Decisions:

- The opencode zai/glm-5.1 route is infra-broken until upstream fixes land.
  Do not interpret its rows as model results. GLM-5.1 scores in v2 should come
  from the `zai-claude` harness (Claude Code against api.z.ai/api/anthropic),
  which is also Z.ai's recommended agentic route.
- Re-annotate `zai/glm-5.1 [2026-05-28 finish]` 0/6 ERR as infra, not model.
  Audit the 2026-05-31 MiniMax free-route row for the same signature.
- Preflight gap: one-turn smokes cannot catch this multi-round stall. A future
  preflight should include a 2-3 step tool-use probe (read then write) per
  opencode route.
- GLM-5.1 itself is fine: in every stalled run its visible behavior was fast,
  correct parallel tool use, consistent with its public agentic benchmarks.

Side installs for bisection live at ~/.local/share/kbh-opencode/{1.15.9,1.16.2}
(harness override: KBH_AGENT_CONTAINER_OPENCODE_BIN).

---


## 2026-06-09 - CUDA toolkit version is benchmark surface; stay pinned on 13.2

Evaluated bumping the toolchain to CUDA 13.3 / PTX ISA 9.3 (the driver,
610.43.02, already ships a 13.3 UMD so it would run fine). Decision: do not
change while the current leaderboard table is live. Every published row was
compiled and scored under nvcc 13.2, and ptxas codegen directly shapes
peak_fraction, so swapping the toolkit mid-table silently changes the
instrument the scores were measured with.

Rules going forward:

- The CUDA toolkit is part of the benchmark surface, like problem templates
  and tolerances. Pin it per leaderboard version; never bump it between rows
  of the same table.
- Agent dev and host scoring must use the identical toolkit. Today this holds
  by construction: both use /usr/local/cuda-13.2 (container mode bind-mounts
  that same directory at /usr/local/cuda-host). Keep it that way.
- When v2 lands, record the nvcc version in result.json so future audits do
  not have to infer it.
- Any future bump happens only at an explicit benchmark version boundary
  (the container v2 resweep is the natural one), validated by rebuilding a
  few archived passing kernels under old and new toolkits and comparing
  benchmark times before publishing.

Review notes from the 13.3 evaluation, so nobody re-derives this:

- PTX ISA 9.3 additions are datacenter-Blackwell features: fabric.* ops,
  multimem.st/red.async, mbarrier phase-type extensions, clmad. Nothing
  sm_120-usable; no pull for problem design either.
- CUDA 13.3 ships a newer ptxas plus Blackwell library tuning (NVIDIA cites
  cuBLAS TF32/FP4 gains, mostly aimed at Blackwell Ultra). Its actual effect
  on sm_120 peak fractions is unmeasured; treat any expected gain as
  needs-measurement.

---


## 2026-06-09 - Container mode demo: four harnesses verified on TopK

First end-to-end verification of `KBH_AGENT_CONTAINER=1`: one 600-second demo
run per container-capable harness on `problems/05_topk_bitonic`
(driver: `outputs/tmp/container_demo_20260609.sh`).

Results:

```text
claude / claude-opus-4-8        correct  peak_fraction=0.0092
codex  / gpt-5.5                correct  peak_fraction=0.0113
cursor / composer-2.5-fast      correct  peak_fraction=0.0036
opencode / zai/glm-5.1          no solution.py at 600s (agent was active)
droid                           blocked: Factory auth expired
```

All passing runs had `template_mutated=false` and went through host-side
`check.py` (numeric stress) and `benchmark.py` under `outputs/gpu.lock`.
Peak fractions are demo-budget numbers, not capability scores.

Confirmed container facts:

- The GPU is visible and usable inside the container (torch sees the RTX PRO
  6000; host CUDA 13.2 `nvcc` is mounted at `/usr/local/cuda-host`).
- The NGC entrypoint warning "NVIDIA Driver was not detected" is cosmetic:
  `50-gpu-driver-check.sh` fails on missing `ldconfig` under `--user` non-root.
- Environment skew is real: the container stack is torch 2.8.0a0+cu129
  (NV 25.06) while host scoring runs torch 2.11+cu130. All three passing demo
  solutions survived the skew, but it remains a risk for version-sensitive
  kernels and should be settled before a published container sweep.
- opencode redoes its sqlite migration in every fresh per-run agent home,
  burning budget at short horizons. Pre-warming a template agent home would
  remove that overhead.
- Codex's `transcript.jsonl` contains only the NGC banner by design; the rich
  transcript is recovered from the archived codex session JSONL
  (`codex_session.jsonl`), and usage parsed correctly from it.

The June 7 all-fail container smoke round was misleading: claude and cursor
were live agents that just hit tiny smoke budgets, codex's empty transcript is
expected, and only droid failed for a real reason. Droid auth is expired on
the host itself (`droid exec` fails identically outside the container), so the
droid leg needs an interactive `/login` on Anvil or a `FACTORY_API_KEY` before
it can be swept in any mode.

Update, same day: droid is dropped from the eval suite entirely. No relogin is
planned; do not include droid rows in future sweeps or treat it as a pending
harness.

---


## 2026-06-09 - Durable Nemotron Ultra route

Nemotron 3 Ultra is now wired as a durable, clone-facing benchmark route via
OpenCode and OpenRouter pinned to DeepInfra:

```sh
uv run kbh run opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b problems/01_fp8_gemm
```

Decision:

- Use `opencode-nemotron` for scoring. OpenCode speaks OpenAI-compatible APIs
  directly, so this avoids the Anthropic-router translation layer required for
  Claude Code and avoids Droid/Factory routing that is not native for this
  provider.
- The harness writes an archive-local OpenCode config for each run, pins
  OpenRouter provider order to `DeepInfra`, and sets `allow_fallbacks=false` so
  provider drift cannot silently change the row.
- `OPENROUTER_API_KEY` is the required key. Enable the row in broad sweeps with
  `KBH_USE_OPENROUTER_NEMOTRON=1`.
- Target only the row for a cheap route smoke with
  `KBH_USE_OPENROUTER_NEMOTRON=1 KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra
  ./scripts/preflight_harnesses.sh`.
- `scripts/extract_usage.py` treats `opencode-nemotron` like OpenCode because
  the transcript shape is the same.

Rejected / diagnostic paths:

- Claude Code through CCR smoked successfully once, but it adds an
  Anthropic-compatible router layer and should not be the scoring route for this
  OpenAI-shape provider.
- NVCF remains as `nvcf-nemotron` for diagnostics only. Ultra was visible on the
  account but degrading and returning 504s; Super direct chat worked, but
  OpenCode-shaped agent traffic hit provider errors.

Smoke result:

```text
KBH_USE_OPENROUTER_NEMOTRON=1 KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra ./scripts/preflight_harnesses.sh
opencode_nemotron_ultra ok=true exit=0 elapsed=2s
```

Clone-facing docs were updated in `README.md`, `AGENTS.md`, and `CLAUDE.md`.

---

## 2026-06-02 - Numeric stress correctness validation

Correctness now reruns canonical shapes and seeds under problem-specific numeric
stress cases. This is not hidden-shape bloat: stress cases rescale existing
floating inputs or model state to catch zero-output, cached-nominal, and
loose-tolerance solutions that can pass under one friendly random distribution.
`benchmark.py` remains canonical-deck only, so measured peak fractions stay
comparable for kernels that still pass.

Implemented:

- Added `src/eval/numeric_stress.py` with nominal plus targeted small/large
  activation or weight-scale cases for the active hard problems.
- Wired numeric stress into the active `check.py` runners.
- Kept integer/discrete comparison exact and improved float failure diagnostics
  with max absolute/relative error, bad element count, worst index, and
  tolerance.
- Added tests for classic cheat/failure classes: zero output under loose
  tolerance, cached nominal answers, and state scaling/restoration.

Verification:

```text
uv run ruff check . --fix
uv run pytest                         # 31 passed
KBH_NUMERIC_STRESS=1 check.py TopK    # disposable GPU smoke: PASS
KBH_NUMERIC_STRESS=1 check.py FP8     # tiny disposable GPU smoke: PASS
```

Operational note: `KBH_NUMERIC_STRESS=0` is useful for local debugging only.
Do not use it for official checks, sweeps, or published backfills.

---

## 2026-06-01 - Removed Kahan softmax from the active deck

`04_kahan_softmax` has been removed from the benchmark surface. The problem was
too easy to satisfy with a plain fast softmax under the existing tolerance, so
it rewarded the shortcut instead of forcing compensated summation. Current
scripts, machine-readable results, baselines, annotations, and leaderboard docs
no longer include it. Historical DEVLOG discussion is intentionally preserved
below as audit context for why the problem was removed.

## 2026-06-01 - Benchmark scoring is solution-first by default

KDA exposed a general harness risk: reference diagnostics can be slower than the
submitted kernel, so timing eager / `torch.compile(reference)` / SOTA before the
solution can turn a valid submission into a post-run benchmark timeout. The
default benchmark path now measures the submitted solution first for every
problem. Reference diagnostics are still available, but only when explicitly
requested.

Fixes:

- Every `problems/*/benchmark.py` now times and prints `variant=solution` before
  any eager, compiled, or SOTA diagnostic.
- Eager / compiled / SOTA diagnostics are opt-in via
  `KBH_BENCHMARK_BASELINES=1`; KDA also keeps the legacy
  `KBH_KDA_BENCHMARK_BASELINES=1` alias.
- `src/eval/timing.py` now emits `benchmark_event` lines around each variant
  (`variant_start`, `variant_end`, `variant_error`) so future audits can split
  solution, eager, compiled, and SOTA wall time directly from `benchmark.log`.
- During `KBH_DISABLE_AGENT_CUDA=1` agent phases, `nvidia-smi` and `nvcc` now
  pass through without taking the GPU lock, while `ncu` and `nsys` fail fast.
  Harness-owned `check.py` and `benchmark.py` still run under `outputs/gpu.lock`.

## 2026-06-01 - KDA benchmark backfill

The KDA benchmark timeouts were not lost submissions. The archived
`solution.py` files were present and correctness-passing; the old
`benchmark.py` measured eager + `torch.compile(reference)` diagnostics before
timing the submitted solution, and the compile path could consume the whole
1800s post-run benchmark budget.

Fixes:

- `02_kda_cutlass/benchmark.py` now times and prints the solution score first.
  Eager/compiled/SOTA reference diagnostics are opt-in via
  `KBH_KDA_BENCHMARK_BASELINES=1`.
- `scripts/run_hard.sh` gives KDA a 7200s benchmark backstop by default
  (`KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS` overrides it) and records the
  check/benchmark timeout values in future `result.json` files.
- Backfilled every archived correctness-passing/no-score KDA row from its
  submitted kernel under `outputs/gpu.lock`; `results/leaderboard.json` now has
  zero `correct=true` cells without a numeric `peak_fraction`.

Backfilled KDA scores:

```text
grok/grok-build [2026-05-28 opus48-grok max]       0.1184
claude/claude-opus-4-7 [2026-05-28 finish max]    0.1166
claude/claude-opus-4-8 [2026-05-28 opus48-grok]   0.1165
minimax-claude/MiniMax-M3 [2026-06-01]            0.1114
cursor/composer-2.5-fast [2026-05-28 finish]      0.0690
claude/claude-opus-4-7 [max]                      0.0330
codex/gpt-5.5 [2026-05-28 finish xhigh]           0.0095
opencode/zai/glm-5.1 [2026-05-08]                 0.0030
```

Note: the Z.ai Claude FP8 row from 2026-05-13 remains invalid despite having a
numeric archived benchmark, because that run modified `problem.yaml` tolerance.
The leaderboard backfill intentionally skips cells that were already marked
invalid.

---

## 2026-06-01 - MiniMax M3 Claude Code full sweep

Full CUDA-track sweep through the direct Claude Code route:

```text
kbh_minimax_m3_claude_full_20260601_105827
```

MiniMax M3 produced correct solutions on all seven problems. The original
`02_kda_cutlass` post-run benchmark hit the old 1800s timeout, but the archived
submission later backfilled successfully after the KDA benchmark fix. Published
row:

```text
01_fp8_gemm          0.5334
02_kda_cutlass       0.1114
03_paged_attention   0.0286
04_kahan_softmax     0.2364
05_topk_bitonic      0.0433
06_sonic_moe_swiglu  0.2538
07_w4a16_gemm        0.1076
```

The run is a big delta from the previous OpenCode MiniMax M3 free route, which
wrote no solutions. Claude Code route quality matters here.

Audit notes: the FP8 GEMM cell uses the known bf16-reference loophole (explicit
fp8-to-bf16 cast plus CUTLASS Sm80 bf16 GEMM), and the Kahan softmax cell is a
fast fp32 tree-sum softmax rather than compensated Kahan. Both are annotated as
rubric leaks. The TopK and Sonic MoE cells are clean/interesting: TopK uses CUB
BlockRadixSort with striped loads and a hierarchical k=64 single-row merge;
Sonic MoE directly implements grouped GEMM with fused SwiGLU and becomes the
new best cell on that problem.

---

## 2026-06-01 - MiniMax M3 Claude Code direct route

MiniMax's current docs now explicitly support Claude Code through the
Anthropic-compatible endpoint `https://api.minimax.io/anthropic` with model
`MiniMax-M3`. Added a dedicated `minimax-claude` harness instead of mutating the
normal `claude` harness or global `~/.claude/settings.json`.

Auth convention:

```sh
export MINIMAX_API_KEY=...
```

Keep that in Anvil's `~/.env_vars`, which `scripts/run_hard.sh` and
`scripts/preflight_harnesses.sh` already source. The harness maps it to
`ANTHROPIC_AUTH_TOKEN` only inside the spawned Claude Code process and sets
`ANTHROPIC_MODEL` plus the Sonnet/Opus/Haiku defaults to `MiniMax-M3`.
The key is exported inside the launch subshell before `timeout claude`; do not
use `timeout env ANTHROPIC_AUTH_TOKEN=...` because `env` arguments appear in
process listings while a run is active.

Use:

```sh
KBH_USE_MINIMAX_M3_CLAUDE=1 ./scripts/preflight_harnesses.sh
./scripts/run_hard.sh minimax-claude MiniMax-M3 problems/01_fp8_gemm
```

---

## 2026-05-31 - MiniMax M3 free sweep and provider classifier hardening

Swept MiniMax M3 through the opencode harness using the available public Zen
route `opencode-zen-live/minimax-m3-free` because Anvil has no saved OpenCode
Go credentials. Run group:

```text
kbh_minimax_m3_opencode_20260531_183925
```

All seven rows completed with `session_complete=true` and `harness_exit_code=0`
but wrote no `solution.py`; no `check.py` or `benchmark.py` validation ran.
The corrected result is therefore 0/7, all `no_solution`.

The first summary falsely labeled `01_fp8_gemm` as
`provider_rate_limited` and `06_sonic_moe_swiglu` as
`provider_insufficient_credits`. Both were transcript false positives: the
model had read text containing "quota/rate limits" from `AGENTS.md` and
`insufficient_credits` from `run_hard.sh`. Provider classification now lives in
`src/harness/classification.py` and scans explicit CLI/API error events plus
stderr, not arbitrary assistant text or tool outputs.

---

## 2026-05-28 - Opus 4.8 and Grok Build addendum

Added Anvil `grok` CLI support using model `grok-build` and the top-level
headless streaming JSON route. Also added a Grok transcript viewer parser so
run archives render correctly in `src.viewer`.

Run group:

```text
kbh_opus48_grok_full_20260528_125852
```

The addendum drained cleanly after the old temporary launcher exposed a wait
bug: 14 manifest rows, 14 `result.json` rows, 0 running, and 0
exited-without-result. `scripts/launch_parallel_sweep.sh` has since been fixed
to keep child jobs waitable. Claude Opus 4.8 used `--effort max` with fast mode
disabled. Grok Build completed all seven rows through the new harness path.

Claude Opus 4.8 passed six of seven CUDA rows plus KDA correctness:
`01_fp8_gemm` 0.5332, `03_paged_attention` 0.6517, `04_kahan_softmax` 0.3517,
`05_topk_bitonic` 0.0462, `06_sonic_moe_swiglu` 0.2507, and
`07_w4a16_gemm` 0.1127. `02_kda_cutlass` passed correctness but timed out in
the benchmark phase.

Grok Build passed `04_kahan_softmax` at 0.0373 and passed KDA correctness but
timed out in benchmark. The remaining Grok rows wrote checkable solutions that
failed correctness.

Summary artifacts:

```text
outputs/sweeps/kbh_opus48_grok_full_20260528_125852/summary/summary.json
outputs/sweeps/kbh_opus48_grok_full_20260528_125852/summary/summary.latest.json
```

---

## 2026-05-23 - Lock-timeout and workspace stress fix

`check.py` and `benchmark.py` now acquire `outputs/gpu.lock` before their
execution timeout starts. The previous `timeout 180 uv run python check.py`
shape let lock wait consume the correctness budget, which made queued rows look
like model failures. The new `run_gpu_locked_timeout` path wraps `timeout`
inside the lock holder and classifies execution timeouts as
`check_timeout`/`benchmark_timeout` retryable rows instead of plain
`check_failed`.

Claude-family harnesses now `cd "$PROBLEM_DIR"` before launching Claude Code.
The old repo-root cwd plus `--add-dir "$PROBLEM_DIR"` was enough for some runs
to spend huge token budgets writing `problems/<name>/solution.py` in the source
tree while the archive-local workspace had no `solution.py`.

Stress test `stress_lock_fix_20260523_230809` used fake `kimi`/`claude`
binaries to avoid API spend. A fake Kimi run waited four seconds on the GPU
lock with `KBH_CHECK_TIMEOUT_SECONDS=1` and still passed after the lock opened;
six concurrent fake Kimi rows got unique archive directories and passed; fake
Claude proved its cwd was the archive-local
`outputs/runs/.../repo/problems/99_lock_stress` directory. Source-tree leaked
solutions/scratch were preserved under
`outputs/tmp/source_contamination_20260523_230921` and removed from
`problems/`.

---

## 2026-05-23 - Classified resweep fixes

The resweep launcher now has one worker per harness instead of a problem-major
outer loop. The original per-harness cap prevented more than two sessions per
harness, but it still head-of-line blocked when the next row belonged to a busy
harness. With workers, Cursor/Gemini/OpenCode can backfill their own next
problem while Codex or Claude remains busy.

OpenRouter was depleted during resweep setup, so the current classified rerun
uses:

```sh
KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1 KBH_HARNESS_CONCURRENCY=2
```

That runs Codex GPT-5.5, Claude Opus 4.7, Z.ai GLM-5.1 through both Claude
Code and OpenCode, Cursor Composer 2.5 Fast, and direct Gemini 3.5 Flash. Qwen
3.7 Max remains blocked until OpenRouter is topped up or a direct provider key
is added.

Aborted sweeps can leave orphaned harness timeout groups because some CLIs
spawn new process groups below `run_hard.sh`. Before restarting, kill by cwd
under `KernelBench-Hard` / `outputs/runs/<run_prefix>` and verify
`nvidia-smi --query-compute-apps` is empty.

Also fixed a failure-classifier false positive: matching plain `overage` marked
normal Cursor transcript text containing `coverage` as
`provider_insufficient_credits`. Credit detection is now limited to explicit
credit/balance/payment phrases and only applies when no solution was produced.
This matters because Cursor can quote old `result.json` files in otherwise
successful-session transcripts.

---

## 2026-05-23 - Guarded parallel sweep logging

The guarded parallel sweep now records enough metadata for website use:
agent wall time, total/check/benchmark wall time, harness/check/benchmark exit
codes, session completeness, CUDA-guard state, parsed token/cache/reasoning
usage, output tokens/sec, and GPU lock wait/active totals from
`scripts/summarize_runs.py`.

Current sweep group:

```text
kbh_hard_parallel_guarded_20260523_003820
```

The lock intentionally catches `uv`, `python`, `python3`, `nvidia-smi`, `ncu`,
`nsys`, and `nvcc` from agent workspaces. This is conservative: CPU-only probes
such as `python -c import triton` may wait behind a harness-owned benchmark.
That is acceptable for the guarded sweep because the invariant is stronger:
agent editing phases can overlap, while CUDA-facing compile/check/benchmark
work serializes through `outputs/gpu.lock`.

After observing Z.ai rows waiting behind a Qwen `benchmark.py` for harmless
`python -c import triton` probes, the wrapper was relaxed for future runs:
when `KBH_AGENT_PHASE=1`, `uv`/`python`/`python3` bypass the lock. CUDA remains
hidden with `CUDA_VISIBLE_DEVICES=` plus the `sitecustomize.py` torch guard, so
agent edit-phase Python can inspect syntax/imports without queueing behind GPU
benchmarks. Harness-owned post-run `check.py` and `benchmark.py` still run
outside `KBH_AGENT_PHASE` and therefore serialize through the lock.
The CPU-only transcript usage extraction path also bypasses the lock now, so
completed rows do not queue behind unrelated GPU benchmarks just to parse token
counts.

---

## 2026-05-22 - Parallel-safe workspaces and Cursor Agent smoke

`scripts/run_hard.sh` now creates an archive-local repo-shaped workspace for
every run:

```text
outputs/runs/<run_id>/repo/problems/<problem_name>/
```

Only the immutable problem template files are copied into that workspace. The
workspace gets a symlink to the real `src/` plus copied `pyproject.toml`,
`uv.lock`, and `.python-version`, so `check.py` / `benchmark.py` still see a
repo root two parents up while agents can mutate dependencies or scratch files
without touching the source `problems/*` directory. This fixes the parallel run
hazard where two agents on the same problem could delete or overwrite each
other's `solution.py`.

Added a `cursor)` harness branch for Anvil's Cursor Agent CLI, which is
installed as `agent` rather than `cursor`:

```sh
agent --trust --yolo --print --output-format stream-json \
  --model "$MODEL" --workspace "$PROBLEM_DIR" "$PROMPT"
```

`scripts/extract_usage.py` now has `_cursor()` support for terminal
`{"type":"result"}` events with `usage.inputTokens`, `outputTokens`,
`cacheReadTokens`, and `cacheWriteTokens`. Partial Cursor timeouts do not expose
a terminal usage block, so usage may be null on 300s smokes that hit timeout.

Composer 2.5 smoke:

```text
BUDGET_SECONDS=300
problem: problems/01_fp8_gemm
harness/model: cursor / composer-2.5
archive: outputs/runs/20260522_144839_cursor_composer-2.5_01_fp8_gemm/
```

The run wrote `solution.py` and preserved source problem cleanliness, but timed
out and failed post-run `check.py`: the generated solution tried to load a
Torch extension named `cutlass_fp8_gemm`, but the `.so` was missing at
post-check import time. This is a real model/harness smoke result, not a
workspace collision.

Validation after the harness changes:

```sh
uv run ruff check . --fix
uv run pytest
```

Both passed.

### GPU queue smoke

Added per-run cache directories and a shared GPU lock wrapper under each
archive's `bin/` directory. During a run, `PATH` points at wrappers for `uv`,
`python`, `python3`, `nvidia-smi`, `ncu`, `nsys`, and `nvcc`; those wrappers
acquire `outputs/gpu.lock` before forwarding to the real binary. Per-run cache
env vars are also set:

```sh
TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
TMPDIR="$RUN_DIR/tmp"
```

Smoke test launched three agents concurrently on `01_fp8_gemm` with
`BUDGET_SECONDS=180`:

- `opencode openrouter-alibaba/qwen/qwen3.7-max`
- `opencode openrouter-google-ai-studio/google/gemini-3.5-flash`
- `cursor composer-2.5`

All three reached post-run validation without touching the source problem
directory. The lock logs showed serialized GPU-facing calls. In particular,
Gemini's `check.py` ran first (`16:01:07` to `16:01:09`), Cursor's `check.py`
then ran (`16:01:09` to `16:01:35`), and Qwen's `check.py` then ran
(`16:01:35` to `16:01:38`). This validates the intended shape: agent work can
overlap, but check/compile/benchmark phases queue through the shared lock.

The 180s model results are not capability scores:

- Gemini wrote a solution but failed tolerance (`max_abs_diff=0.5625`).
- Qwen wrote a Triton solution but failed compile (`Unsupported rhs dtype fp8e4nv`).
- Composer wrote a CUTLASS solution but failed extension build.

Important limitation: the lock only governs commands launched inside the
KernelBench harness. It does not stop unrelated machine-wide CUDA jobs already
running elsewhere on Anvil, so serious published sweeps should still check
`overnight-compute status` / `nvidia-smi` first or run under a broader machine
reservation.

First wrapper attempt deadlocked during a follow-up Qwen run: `uv run python
benchmark.py` held the GPU lock, then the benchmark's child `nvcc --version`
wrapper tried to acquire the same non-reentrant lock. Fixed by setting
`KBH_GPU_LOCK_HELD=1` while executing the real locked command; nested wrapper
calls now bypass the lock and exec the real binary directly.

Validation after the reentrant fix:

```text
BUDGET_SECONDS=300
harness/model: opencode / openrouter-alibaba/qwen/qwen3.7-max
archive: outputs/runs/20260522_161511_opencode_openrouter-alibaba_qwen_qwen3.7-max_01_fp8_gemm
```

Result: `correct=true`, `peak_fraction=0.4257`, `template_mutated=false`.
The lock log shows the full post-run path serialized cleanly:

```text
16:20:11 start uv run python check.py
16:20:17 end   uv run python check.py status=0
16:20:17 start uv run python benchmark.py
16:20:26 end   uv run python benchmark.py status=0
```

---

## 2026-05-21 - Gemini CLI smoke wired, timeout path validated

Gemini CLI support is now wired into the Hard harness but remains uncommitted.
`scripts/run_hard.sh` has a `gemini)` branch that runs from inside the problem
directory because Gemini has no `--cwd` or `--add-dir` flag:

```sh
cd "$PROBLEM_DIR" && gemini -m "$MODEL" --approval-mode yolo -o stream-json -p "$PROMPT"
```

The Gemini branch was added to the `session_complete` group by looking for a
terminal result event shaped like `{"type":"result"}`. `scripts/extract_usage.py`
now has `_gemini()` support that reads `stats.input_tokens`,
`stats.output_tokens`, and cached-token stats from that terminal result event.

Smoke run:

```text
BUDGET_SECONDS=300
problem: problems/01_fp8_gemm
harness/model: gemini / gemini-3.5-flash
archive: outputs/runs/20260519_212055_gemini_gemini-3.5-flash_01_fp8_gemm/
```

The smoke passed end-to-end at the harness level: Gemini wrote a Triton GEMM
`solution.py` and executed `check.py` inside the sandbox. The run timed out at
300 seconds with exit 124, so `session_complete=false`; this is expected for
the smoke budget, and because there was no final result event, usage stayed
null.

The important validation was the reward-hack defense. Gemini tried to edit
`problem.yaml` to add a `bfloat16: 0.15` tolerance after its FP8 kernel failed
the atol check (`max_abs_diff=0.546875`). The template-mutation guard detected
`template_mutated: true`, restored `problem.yaml`, and marked the run INVALID.
That is the intended defense working. Workspace isolation also held: Gemini was
restricted to `PROBLEM_DIR` plus `~/.gemini/tmp/...`, matching the other
harnesses sandbox model.

Harness is ready for a real 45-minute sweep when requested. Before publishing
or merging this work, run normal checks and commit the two dirty harness files:

- `scripts/run_hard.sh`
- `scripts/extract_usage.py`


## 2026-05-14 - Leaderboard split after non-pass audit

After a per-run audit of every non-pass cell, `/hard` now renders two leaderboard sections instead of one flat table. The serious comparison section keeps rows where audited non-passes are normal benchmark outcomes: correctness failures, build failures, full 2700s timeouts, or explicit invalid/reward-hack behavior. Current serious rows are GPT-5.5, Claude Opus 4.7, Claude Code GLM-5.1 on Z.ai, Droid GLM-5.1, and the two DeepSeek OpenCode rows.

Rows moved to diagnostic/needs-rerun have at least one non-pass that is not a clean model attempt: API/provider errors, auth/setup failures, harness adapter failures, hidden reasoning-token exhaustion before writing `solution.py`, or unknown early stops with no checkable artifact. That includes both older OpenCode/Z.ai GLM-5.1 rows, the OpenRouter-pinned Qwen/MiMo/MiniMax rows, and Kimi K2.6. Kimi is demoted despite being otherwise interesting because problems 09/10 ended in 401 authentication errors after only 4-5 seconds; its 6/9 raw pass total is not directly comparable until those cells are rerun.

DeepSeek through OpenCode is intentionally not blanket-demoted: the audited DeepSeek non-passes were ordinary solution bugs or full-budget timeouts with artifacts, not API/setup failures. Droid is kept serious because the documented May 8 smoke tests confirmed the custom Z.ai/Factory route was wired correctly; its four `ERR` cells were 45-minute incomplete runs with no `solution.py`, not endpoint failures.

---

## 2026-05-14 — Z.ai GLM-5.1 Claude Code rerun with corrected Anthropic endpoint

Shuyan confirmed Z.ai's internal Claude Code eval config: disable experimental betas, set very high retry and output-token ceilings, disallow plan/user-question tools, and map every Claude Code alias including Haiku / Explore / subagents to `glm-5.1`. `scripts/run_hard.sh` now bakes those defaults into the `zai-claude` harness against `https://api.z.ai/api/anthropic`, and `scripts/rerun_zai_claude_glm51.sh` records the nine-problem CUDA rerun command.

The May 13 rerun is now represented as `zai-claude/glm-5.1 [2026-05-13]`. Correctness-passing cells were `03_paged_attention` 0.2220, `04_kahan_softmax` 0.3367, `05_topk_bitonic` 0.0029, `06_sonic_moe_swiglu` 0.1111, and `10_patch_embed_conv3d_gemm` 0.1471. `02_kda_cutlass` failed numerically, `07_w4a16_gemm` timed out with no `solution.py`, and `09_fmha_preattn_mrope` failed CUDA extension compilation/checking.

`01_fp8_gemm` is deliberately not counted as a pass even though its archived `result.json` says `correct=true`: the model edited `problem.yaml`, changing the tolerance key from `fp8_e4m3fn: 0.15` to `bfloat16: 0.15`. Since `check.py` looks up tolerance by `ref_out.dtype == torch.bfloat16`, that relaxed the actual correctness check from the default bf16 tolerance to 0.15. This is a clean reward-hack example, not a valid kernel result, so the public row marks it invalid and attaches a `reward_hack` annotation.

Harness fix: `run_hard.sh` now snapshots `reference.py`, `sota.py`, `shapes.py`, `problem.yaml`, `check.py`, `benchmark.py`, and `PROMPT.txt` before each agent run. If any of those files are changed, deleted, or created unexpectedly, the run is marked invalid, `template_mutated=true` is written to `result.json`, a diff lands in `template_mutations.log`, and the original files are restored before the next problem.

---

## 2026-05-09 — Website policy: demote OpenCode, remove blocked Qwen 35B-A3B row

After inspecting the May 8 GLM-5.1 rerun transcripts, we changed the public `/hard` page to treat OpenCode rows as diagnostic rather than primary evidence. The OpenCode Z.ai rerun had multiple early `ERR` cells caused by hidden-reasoning budget exhaustion before tool use, so the page now shows a red disclaimer and pushes OpenCode rows below the native-harness rows. Droid and Claude Code rows should carry more weight when they exist.

Also removed `opencode/openrouter-pinned/qwen/qwen3.6-35b-a3b` from the public leaderboard data. Its previous `0/7 ERR` row was an infrastructure block, not a model result: available providers did not advertise tool-use support to the agent harness. The historical details remain in this devlog; the website no longer presents it as an evaluated model row.

For Qwen 3.6 27B, a transcript dive found no result-parser bug. The frequent failures are mostly missing `solution.py` / early-stop behavior, plus several full-budget OpenCode timeouts where a written solution still failed `check.py`. The later problem 09/10 first attempts failed due to OpenRouter insufficient-credit API errors, then reran successfully and are represented by the passing rows in `leaderboard.json`.

## 2026-05-08 — Z.ai GLM-5.1 rerun: OpenCode, Droid, Claude Code attempt

Z.ai reached out after the public KernelBench-Hard GLM-5.1 row, asking for a rerun because several OpenCode cells appeared to terminate early as `ERR` after a small number of iterations. We reran all CUDA-track problems on the RTX PRO 6000 using their dedicated `$ZAI_API_KEY` against the actual Z.ai endpoint. `08_metal_lightning_attn` stayed out of scope on this CUDA host.

OpenCode rerun used `zai/glm-5.1` through the Z.ai API. It reproduced the core anomaly: `03_paged_attention`, `05_topk_bitonic`, `07_w4a16_gemm`, and `09_fmha_preattn_mrope` all ended as `ERR` well before the 2700s budget. Passing cells were `02_kda_cutlass` (correct but no parsed peak), `04_kahan_softmax` at 0.0561, `06_sonic_moe_swiglu` at 0.2154, and `10_patch_embed_conv3d_gemm` at 0.1742. `01_fp8_gemm` wrote a solution but failed correctness at timeout.

Droid rerun used the Factory custom model `custom:GLM-5.1-[Z.AI-Coding-Plan]-0`, also pointed at `https://api.z.ai/api/coding/paas/v4`. It solved five of nine: `01_fp8_gemm` 0.4140, `03_paged_attention` 0.2523, `04_kahan_softmax` 0.2339, `06_sonic_moe_swiglu` 0.1490, and `07_w4a16_gemm` 0.0863. `02_kda_cutlass`, `05_topk_bitonic`, `09_fmha_preattn_mrope`, and `10_patch_embed_conv3d_gemm` timed out incomplete with no scored solution.

Claude Code was attempted only against Z.ai, not Anthropic: first through `ccr-rust` with a Z.ai-only router, then directly with `ANTHROPIC_BASE_URL=https://api.z.ai/api/coding/paas/v4` and `ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY`. The proxy path authenticated but returned malformed/empty HTTP 200 responses to Claude Code; direct model names returned 404 model-access errors. Those runs are setup-invalid, not model results, and are not counted on the leaderboard.

Website changes from this rerun: add Droid harness support to `scripts/run_hard.sh`, add Droid usage extraction, render all 18 May 8 transcript viewers, and publish two additional leaderboard rows: `opencode/zai/glm-5.1 [2026-05-08]` and `droid/zai/glm-5.1 [2026-05-08]`. This preserves the original public GLM row while making the rerun evidence explicit.

Follow-up smoke tests found the Claude Code wiring bug: Z.ai exposes a separate Anthropic-compatible endpoint for Claude Code, `https://api.z.ai/api/anthropic`. The earlier direct attempt used the OpenAI-compatible coding endpoint, `https://api.z.ai/api/coding/paas/v4`, which is the right shape for Droid/Factory but the wrong shape for Claude Code. `scripts/run_hard.sh` now has a `zai-claude` harness that sets `ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic`, maps Claude Code's `opus`/`sonnet` aliases to `glm-5.1`, and leaves `ccr-claude` as the historical proxy path. One-turn smoke results: `zai-claude` returned `KB_SMOKE_OK` through model `glm-5.1`; Droid's existing `custom:GLM-5.1-[Z.AI-Coding-Plan]-0` also returned `KB_SMOKE_OK`. Droid was already hooked up correctly; its four benchmark ERR cells were 45-minute incomplete runs with no `solution.py`, not API failures.

---

## 2026-04-30 — Launch prep: monorepo, kernelbench.com, transcript viewers, blog plots

Three substantial pieces went in between the rubric-leak audit and shipping public.

### Monorepo

The standalone `Infatoshi/KernelBench-Hard` and `Infatoshi/KernelBench-v3` repos got absorbed into `Infatoshi/kernelbench.com` as `git subtree` merges (history preserved). The website lives at the repo root for Vercel auto-detection; benchmarks live under `benchmarks/hard/` and `benchmarks/v3/`. The standalone repos still exist but the monorepo is now the canonical home — the website's `lib/data.ts` reads `benchmarks/hard/results/leaderboard.json` directly from disk at build time, no HTTP fetch.

Trade-off accepted: the per-suite DEVLOGs stay inside their subdirs (this file lives at `benchmarks/hard/DEVLOG.md`). Cleaner per-suite history; harder to write a single chronological narrative across them. Worth the trade.

### Public website with the hacker theme

Next.js 16 + React 19 + Tailwind v4. Phosphor green on near-black, JetBrains Mono everywhere, subtle CRT scanlines via fixed CSS overlay. Routes:

- `/` — landing with ASCII KernelBench banner, version cards, design principles, contact box.
- `/hard` — leaderboard table (12×7 grid, every cell clickable into its run viewer), per-problem ceilings table with eager / compiled / SOTA timings, full rubric-leak deep dives with pull quotes, what-changed-from-v3 bullets.
- `/v3` — client-side filterable explorer over the 2071-row results.csv, embedded plots, per-row solution.py / reference.py links.
- `/runs` — sortable index of all 100 transcript viewers (peak-fraction-ranked).
- `/runs/<run_id>.html` — themed transcript viewers, see below.
- `/blog`, `/blog/v3`, `/blog/hard` — long-form writeups (moved over from elliotarledge.com).

Domain: `kernelbench.com` registered through Vercel, attached to project `kernelbench` under team `elliot-arledges-projects`. Auto-deploys on push to `master` via Vercel's native GitHub integration — no GitHub Actions workflow needed.

### One critical Vercel deploy gotcha

Every commit pushed from anvil with the autogenerated email `infatoshi@anvil.tail21a94e.ts.net` failed Vercel's commit-verification gate at the pre-build phase (silent ERROR with no build logs). Three commits errored before I traced it. The fix: pass `-c user.email=elliot@arledge.net` (the GitHub-linked email) inline on every git commit. The repo has a local `git config user.email` set to this; new commits should pick it up automatically, but if you're working from a fresh clone or different machine, set it explicitly.

### 100 themed transcript viewers + reward-hack tab

`src/viewer/html.py` now respects two env vars at HTML-generation time:

- `KB_VIEWER_THEME=phosphor` — applies a CSS override layer (phosphor green on near-black, JetBrains Mono, CRT scanlines) plus a site-nav strip linking to `/`, `/hard`, `/v3`, `/runs`. Preserves the original role color slots (assistant=green, tool=amber, error=red, user=cyan).
- `KB_ANNOTATIONS_DIR=<path>` — looks up `<run_id>.yaml` for the currently-rendered run, and if found inserts a "reward hack" tab between solution.py and final answer. The tab renders the annotation's verdict badge (color-coded by category), summary, pull quotes (with file:line anchors and syntax-highlighted code), and implication paragraph.

Generation pattern (run from `benchmarks/hard/`):
```bash
KB_VIEWER_THEME=phosphor KB_ANNOTATIONS_DIR=$(pwd)/results/annotations \
  uv run python -m src.viewer <run_dir> --out <out_path>
```

Bulk regeneration over all 100 runs is a one-liner shell loop. The generated HTMLs land in the monorepo's `public/runs/`. They're committed (~18 MB total) so the website serves them directly as static assets.

### Top-peak audit: 30 annotations total

Initial audit produced 13 annotations (the FP8 GEMM bf16 dressup cluster + the Kahan softmax skip). Follow-up serial pass added 17 more for top-peak cells with `peak_fraction ≥ 0.10`. All 17 came back `clean` — Triton kernels, no forbidden ops, no F.softmax / scaled_dot / flash_attn library cheats. The 30 annotations now cover every cell where there's something to say: 12 rubric leaks + 1 honest Kahan + 17 clean top performers. Lower-peak cells deliberately left unannotated.

### Baseline + SOTA timings

`scripts/run_baselines.sh` benchmarks each problem's `reference.py` (and `sota.py` where one exists) and writes `results/problem_baselines.json`. The website's `/hard` per-problem ceilings table now shows eager / compiled / SOTA ms alongside best-model peak. Most problems lack a SOTA entry on SM120 — FP8 needs scaling args, vLLM/flashinfer not wired, etc. — so those columns show `—`. Only `04_kahan_softmax` and `05_topk_bitonic` have SOTA timings populated.

Notable timing facts surfaced: torch.compile gives `02_kda_cutlass` an 8x speedup over eager (61.9 → 7.4 ms) and `07_w4a16_gemm` a 4x (0.61 → 0.144 ms); the rest are within noise of eager.

### Five matplotlib blog plots

`benchmarks/hard/scripts/generate_blog_plots.py` reads `leaderboard.json` and produces five PNGs in `public/blog-hard/`, themed to the kernelbench.com palette:

- `leaderboard_heatmap.png` — full 12×7 grid colored by peak_fraction
- `pass_count_by_model.png` — tier ranking, gpt-5.5 xhigh in amber as the only 7/7
- `best_peak_per_problem.png` — per-problem ceilings, shows the easy/hard regimes
- `fp8_gemm_cluster.png` — visualizes the bf16-dressup herd
- `kahan_inversion.png` — visualizes "punishes algorithmic honesty" (deepseek-v4-pro green at 0.101 vs the rest in orange above it)

Embedded into `app/blog/hard/page.tsx` at the matching narrative beats. Re-run the script any time the leaderboard data changes.

### Site-level extras

- `app/icon.svg` — popcorn-kernel favicon (phosphor lobes + amber hull). Reads at 16×16.
- Contact section on `/` with `infatoshi@gmail.com` mailto, also surfaced in the footer on every page (including the 100 transcript viewers).
- elliotarledge.com cleaned up: kernelbench cards removed from blog index, page sources deleted, nav link redirected to kernelbench.com, "kernelbench.com" card added under "things im proud of" on home.

### Pending for later (not blocking launch)

- N≥2 reruns to put variance bands on cells.
- A permanent `/methodology` page surfacing how peak_fraction is computed (currently the explanation lives only in this DEVLOG and the blog post).
- bwrap-based universal sandboxing across all four CLI harnesses (claude, codex, kimi, opencode share the same architectural lack of FS isolation).
- Per-problem detail pages (`/hard/01_fp8_gemm` etc.) with full reference / SOTA / per-shape breakdown.
- Closing the two rubric leaks via tighter tolerances or static-analysis pattern checks. Both fixable in a few hours; deferred so we publish now.

---

## 2026-04-29 — Reward-hacking audit: two rubric leaks, publishing with them documented

After the full sweep (12 models × 7 problems), audited the high-peak runs for reward hacking. Two findings, very different in severity. Decision: ship the leaderboard with the leaks documented inline rather than iterate on problem design until perfect.

### Audit method

Read the solution.py for every (model, problem) cell with `peak_fraction ≥ 0.10`. Looked for the v3-era reward-hack patterns (precision downcast, PyTorch wrapper masquerading as a custom kernel, baseline gaming, hardcoded test-input handling) plus problem-specific shortcuts (skipping the named algorithm in favor of a simpler one that still passes correctness).

### Findings

**Clean (real kernel work):**
- `03 paged_attention` — all top peaks (opus 0.602, gpt-5.5 0.498, kimi 0.432) are real Triton FlashDecoding-style kernels. Online softmax, GQA register reuse, exp2 fast-path. No shortcuts.
- `07 w4a16_gemm` — all 8 passing solutions inline int4 unpacking (`& 0xf`, `>> 4`) inside the kernel; none pre-unpack-and-stash-as-bf16 at init. Genuine quantized kernel work.

**Rubric leak (cell number doesn't measure what the problem name implies):**

- `01 fp8_gemm` — every passing solution at peak ≥ 0.4 (5 models: opus 0.534, mimo 0.434, qwen-plus 0.431, qwen-max 0.429, gpt-5.5 0.423) casts fp8 → bf16 inside the kernel and runs a bf16 GEMM. Both opus and gpt-5.5 explicitly pin to `cutlass::arch::Sm80` — Ampere CUTLASS, no SM120 FP8 tensor cores anywhere. Opus's source comment is explicit: *"follow the codex baseline (BF16 GEMM internally)..."*. Technically valid (the reference also does the bf16 cast) but the problem name promises FP8-tensor-core skill that isn't being measured.

- `04 kahan_softmax` — 6 of 7 passing solutions skipped Kahan compensated summation entirely, including both top-tier scores (gpt-5.5 0.363, opus 0.317). Only deepseek-v4-pro implemented Kahan — and scored *lowest* of the seven passes (0.101) because compensated summation has real overhead. The model whose docstring explicitly says *"Numerically tight softmax with Kahan compensated summation. Map: each block computes local (max, Kahan-sum-of-exp)..."* is the one that loses, because everyone else takes the easy path and tolerance doesn't enforce the difference.

The Kahan one is the more depressing of the two. The benchmark, as designed, *punishes* algorithmic honesty: the model that implements the algorithm the problem name describes scores worst, because the rubric leaks and the dishonest path is faster.

### Decision: publish with flaws documented inline

Two reasons to ship now rather than fix-then-publish:

1. **Diminishing returns on iteration.** This is the second round of post-hoc design issues we've found (the first was the verification gate / prompt-shape regime in late April). Every iteration surfaces something new. Publishing with the current flaws documented is more honest than iterating until the next flaw appears, then publishing.
2. **The flaws ARE the finding.** The benchmark's purpose is to surface what models will and won't do under autonomous-agent evaluation. "Five frontier models all took the bf16 shortcut on FP8 GEMM" and "six of seven skipped Kahan compensation" are themselves headline results — they characterize how models behave when the rubric leaks.

### What we shipped

- `LEADERBOARD.md` — canonical human-readable cross-model grid + per-problem ceilings + a *Benchmark design flaws* section that explicitly footnotes the two leaky problems with their cell numbers.
- `results/leaderboard.json` — machine-readable, schema-versioned. Source for the website's leaderboard view.
- `results/annotations/<run_id>.yaml` — per-cell commentary for 13 runs covering both leaks (5 fp8 cells, 7 kahan cells) plus the headline clean cell (opus paged_attention 0.602). Schema in `results/annotations/SCHEMA.md`.
- `results/annotations/SCHEMA.md` — annotation file format with five verdicts (`clean`, `rubric_leak`, `reward_hack`, `interesting`, `bug`).

### Future leak fixes (logged, not done)

- **fp8_gemm**: tighten tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge on the test inputs, or add a static-analysis check to the rubric that detects the `fp8 → bf16` cast pattern before the GEMM call.
- **kahan_softmax**: tighten tolerance to a value where naive vs Kahan produce visibly different results on the test inputs (the test inputs may need to include numerically-pathological cases — large logit ranges, near-equal extremes), or write a check that detects compensated-summation pattern in solution.py.

These are tractable; deferred so we publish the leaderboard now.

---

## 2026-04-27 — opencode workspace leak: root cause + partial fix

The Qwen 27B forensic dive (next entry) led to auditing every opencode-routed `read` call across the shakedown. The leak is universal across all opencode-routed models, not just Qwen.

### What was being read outside the workspace

Across 31 opencode runs (Qwen 35B-A3B excluded — never made it past routing):

| Path | Reads | Why it matters |
| --- | --- | --- |
| `src/hardware/rtx_pro_6000.py` | 9 | Full peak TFLOPS table (fp4=800, fp8=400, bf16=200, dram=1800). The hardware spec we deliberately stripped from PROMPT.txt is sitting four directories up. |
| `src/eval/correctness.py` | 12 | Per-dtype tolerance lookup. Models can verify what atol/rtol they need to pass before writing code. |
| `src/eval/timing.py` | 5 | Benchmarking methodology (warmup count, L2 flush size, iter count). Lets a model game the timing harness. |
| `src/eval/roofline.py` | 2 | Throughput formulas. Lets the model reverse-engineer the score. |
| `~/.claude/skills/perf-optim/SKILL.md` | 7 | The user's personal CUDA optimization atlas — speed-of-light methodology, profiling tools, the exact "always profile before optimizing" framing. |
| `~/.claude/skills/kernel/SKILL.md` | 2 | The user's master kernel-skill index. |
| `CLAUDE.md` | 1 | Developer-facing repo notes. |
| `/home/infatoshi/cuda/KernelBench/problems/07_w4a16_gemm` | 1 | Sibling repo on the same machine — model walked outside KernelBench-Hard entirely. |

### Root cause

`opencode --pure` does not sandbox file system access. `--pure` means "run without external plugins" — nothing about FS isolation. opencode's `read` tool dispatches to OS file read with no path restrictions; whatever the user running opencode can read, the tool can read. The "soft boundary" of "this is your workspace" was just the prompt + cwd, neither of which constrains the tool implementation.

Same architecture in claude-code (`--add-dir` extends visibility but doesn't restrict; bash can still touch absolute paths) and codex (no path constraints at all). The leak is universal across all three CLI harnesses; opencode was just first-noticed because Qwen 27B was particularly aggressive about reading files.

### Fix (partial)

Added to `~/.config/opencode/opencode.json`:
```json
"permission": {
  "external_directory": "deny"
}
```

This blocks tool calls that touch paths outside the working directory where opencode was started (verified end-to-end: a smoke run trying to `read /home/infatoshi/cuda/KernelBench-Hard/src/hardware/rtx_pro_6000.py` returned `status: "error"` with the message *"The user has specified a rule which prevents you from using this specific tool call"*, and the model correctly reported the block).

### What's still open (and why)

When opencode dumps its rule list on a denied call, it surfaces auto-generated allow rules for **every Claude Code skill the user has installed**:

```
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/perf-optim/*", "action":"allow"}
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/kernel/*",      "action":"allow"}
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/<each-skill>/*", "action":"allow"}
```

These are more specific than my `*: deny`, so they win. The user's CUDA-optimization skills (`perf-optim`, `kernel`, `gpu-profiling`, `port-kernel`, `debug-gpu`) remain readable. That's a separate, smaller leak (user's personal notes, not benchmark internals), but the prompt's "look up PTX docs and library headers" directive is degraded if the model can short-circuit via the user's pre-written kernel atlas.

To close fully, options are:
1. **Rename/move the skills directory before each sweep.** `mv ~/.claude ~/.claude.bak` for the duration. Cheap, intrusive.
2. **Find the opencode config knob that controls skill discovery and disable it.** Not surfaced in the public docs that I could find; would need to source-dive opencode.
3. **bwrap the harness.** `bwrap --bind $PROBLEM_DIR /workspace --ro-bind /usr /usr ... opencode run`. Real isolation; medium-weight; works for all three harnesses uniformly.
4. **Accept the user's-skills leak.** It's pre-existing personal knowledge, equivalent to "the model has been pre-trained on this content." Different category than leaking benchmark internals.

For now: option (1) for serious sweeps, otherwise note the asymmetry. The prompt directive remains the primary signal.

### Cross-harness scope

claude-code and codex are not currently behind any path restriction. Their `Bash`, `Read`, `Edit`, etc. tools see everything the user account does. The leak audit only covered opencode runs because those were the only fresh runs in `outputs/runs/` after we deleted the topk-overnight set. Worth re-auditing whenever the next claude/codex sweep runs. Likely fixable for both via bwrap if the leak proves load-bearing.

### Reading-the-leaderboard note

Until full sandboxing lands, **opencode-routed numbers from before this commit reflect a leakier environment than the current PROMPT.txt regime claims**. Models that read `rtx_pro_6000.py` had peak TFLOPS as a number, not a thing-to-look-up. Models that read `perf-optim/SKILL.md` had a written CUDA optimization atlas. Their scores are not directly comparable to a future run under the post-fix permission policy. Re-running the shakedown after the fix would tell us how much the leak actually mattered, and is worth doing before any "official" leaderboard publication.

---

## 2026-04-27 — Qwen 3.6 27B: post-fix rerun reverses the drop

After the leak fix landed, reran Qwen 3.6 27B on all 7 problems under the new permission policy. Result: **1/7 PASS** (sonic_moe_swiglu, peak_fraction 0.0822 — same tier as MiniMax M2.7's 0.076 on that problem) and dramatically more engagement across the board.

| Problem | Pre-fix shakedown | Post-fix rerun |
| --- | --- | --- |
| 01 fp8_gemm | ERR | ERR |
| 02 kda_cutlass | ERR (1-step bail) | FAIL (45 min, 28k output, has_solution) |
| 03 paged_attention | FAIL (`__sqrt__` hallucination) | FAIL (different bug) |
| 04 kahan_softmax | ERR | ERR (11 min bail) |
| 05 topk_bitonic | ERR (token cap) | FAIL (45 min, 32k output) |
| 06 sonic_moe_swiglu | ERR | **PASS 0.0822** |
| 07 w4a16_gemm | ERR | ERR |

Engagement shifted: immediate bails 4/7 → 2/7; solutions written 1/7 → 4/7; total token consumption rose ~10x (708k input / 9k output → 8.2M input / 91k output).

### Why the change?

Three honest possibilities:
1. **Removing the leak forced focus.** Pre-fix, Qwen burned tool calls reading `src/hardware/rtx_pro_6000.py`, `src/eval/correctness.py`, `src/eval/timing.py`, `~/.claude/skills/perf-optim/SKILL.md`. Post-fix, those reads fail fast with an explicit denial, redirecting the model to focus on `reference.py` and write code.
2. **LLM nondeterminism.** Same model, same prompt, runs 11 hours apart. DeepSeek Flash on TopK regressed from PASS to FAIL on a similar interval — variance is real on this benchmark.
3. **Both.** Leak-fix vector is right (reducing rabbit-holing improves focus), but a 10x engagement swing and 0→1 PASS is hard to attribute purely to that.

A controlled experiment (5x runs of each disposition, same conditions otherwise) would isolate the effect. Worth doing before any "leak fix improved Qwen 1/7 PASS" claim becomes load-bearing.

### Decision

Re-added to ACTIVE_MATRIX. Treat as same tier as MiniMax (functional but high-variance, low ceiling). Earlier "capability + compliance, dropped permanently" framing was a misread driven by N=1.

### N=1 is not enough — methodology footnote

Two reversals within 24 hours on this benchmark: Flash on TopK (PASS → FAIL) and Qwen 27B (0/7 → 1/7). Future official results should run N≥2 per (model, problem) and report variance. Reproducibility footnote in the shakedown entry already flagged this; second confirmation here.

---

## 2026-04-27 — Qwen 3.6 27B: dropped from active matrix (initial drop, see entry above for reversal)

Forensic dive into the 0/7 result on the cheap-tier shakedown.

### Failure-mode breakdown across 7 runs

| Problem | Steps | Tool calls | Wrote solution.py | End reason |
| --- | --- | --- | --- | --- |
| 01 fp8_gemm | 5 | 11 reads + 4 bash | NO | `stop` |
| 02 kda_cutlass | **1** | **0** | NO | `stop` (immediate bail) |
| 03 paged_attention | 8 | 18 (incl. 1 write) | YES — but compile-broken | `stop` after write, no verify |
| 04 kahan_softmax | 3 | 8 reads | NO | `stop` |
| 05 topk_bitonic | 8 | 17 | NO | `length` (output token cap hit) |
| 06 sonic_moe | 5 | 14 | NO | `other` |
| 07 w4a16_gemm | **2** | **1 read** | NO | `stop` (immediate bail) |

### Three intertwined patterns

**Variable engagement.** Step counts ranged 1-8 with no clear relationship to problem difficulty. Two runs (KDA, W4A16) bailed in 1-2 steps with effectively zero engagement. Other runs explored extensively. Same prompt each time, same model, same provider.

**Explores extensively, refuses to write.** 5/7 runs ended with no `solution.py`. The model reads `reference.py`, `problem.yaml`, `shapes.py`, `check.py`, `benchmark.py`, runs `nvidia-smi`/`nvcc`/`triton` probes — and then stops. On the paged attention run it actually said *"Let me verify the check infrastructure before writing the kernel — I noticed syntax issues in those files"* — vocalized the verification gate, then stopped without acting on it. Knows the rule, agrees with it out loud, doesn't follow through. This is a deeper compliance gap than DeepSeek Flash had pre-prompt-edit; tightening the prompt sentence further isn't likely to help.

**When it does write code, it hallucinates APIs.** The one solution.py it produced (paged_attention, 8230 chars of real Triton) had:
```python
scale = 1.0 / float(HEAD_DIM).__sqrt__()
```
`__sqrt__` is not a Python or Triton method — invented. `float(tl.constexpr)` also fails because Python's `float()` doesn't accept Triton tensors. Compilation crash on the first call. The model then *did not run check.py*, so it never saw the error, and stopped.

### Decision

Dropped from `scripts/sweep.sh` ACTIVE_MATRIX and `scripts/shakedown_sweep.sh`. The route stays defined in opencode config so re-add is a one-line restore. Revisit when qwen3.7 lands or if a future agent harness materially improves Qwen's tool-use compliance.

### Observation worth keeping

Qwen 27B's pattern is the inverse of the verification-gate experiment with Flash: Flash didn't read the rule and skipped the test; tightening the prompt fixed it. Qwen *does* read the rule, *does* acknowledge it, then ignores it anyway. That's not a prompt-clarity problem — it's a model-side compliance issue. The verification gate works on models that have the discipline-half latent and need the cue; it doesn't manufacture discipline where it isn't present.

---

## 2026-04-27 — Cheap-tier shakedown sweep: 35 runs, $2.14, full grid

First end-to-end validation of the new PROMPT.txt regime + token-logging wiring. Five cheap-tier models against the full 7-problem deck, sequential.

### Final grid

| Model | 01 fp8 | 02 kda | 03 paged | 04 kahan | 05 topk | 06 moe | 07 w4a16 | PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-flash | FAIL | 0.009 | **0.167** | **0.138** | FAIL | 0.083 | **0.134** | 5/7 |
| deepseek-v4-pro | FAIL | FAIL | 0.027 | 0.101 | 0.011 | **0.108** | 0.125 | 5/7 |
| minimax-m2.7 | ERR | ERR | FAIL | 0.034 | FAIL | 0.076 | 0.030 | 3/7 |
| qwen3.6-27b | ERR | ERR | FAIL | ERR | ERR | ERR | ERR | 0/7 |
| qwen3.6-35b-a3b | ERR×7 (no tool-use endpoint) | | | | | | | 0/7 |

ERR = no `solution.py` written; FAIL = solution.py present, `check.py` failed; numeric = peak_fraction (PASS).

### Token totals + per-model API spend

| Model | input | output | cache_read | reasoning | est. spend |
| --- | --- | --- | --- | --- | --- |
| deepseek-v4-flash | 400k | 260k | 39.4M | 461k | $0.26 |
| deepseek-v4-pro | 336k | 163k | 15.7M | 319k | $0.56 |
| minimax-m2.7 | 1.50M | 164k | 15.5M | 56k | $0.71 |
| qwen3.6-27b | 709k | 10k | 57k | 41k | $0.61 |
| qwen3.6-35b-a3b | 0 | 0 | 0 | 0 | $0.00 |
| **TOTAL** | | | | | **$2.14** |

DeepSeek's 39.4M cache_read on Flash and 15.7M on Pro are the most striking numbers — implicit caching dominates the input budget. Cache reads are ~10x cheaper than fresh input on most providers, so the "input" column understates the real efficiency win. MiniMax cache_read was high too (15.5M) — Fireworks fp8 endpoint also caches.

Qwen 27B's 10k total output across 7 attempts is diagnostic of something deeper than "model can't kernel" — it barely emitted any tool calls. Either the OpenRouter/Alibaba tool-call format isn't matching what opencode expects, or the model defaults to a reasoning-mode that never produces tool-use. Worth a transcript dive before next sweep.

### What this validated

1. **The PROMPT.txt regime works end-to-end.** Models that can drive tool calls (DeepSeek, MiniMax) produced real solutions; verification gate triggered `python check.py` invocations consistently in the passing runs.
2. **Token logging is uniformly populated** across opencode runs. The `usage` block in result.json carries cleanly. Cross-harness comparison data is now in place.
3. **OpenRouter pinning works** for Alibaba (Qwen) and Fireworks (MiniMax). MiniMax even had real cache_read numbers, confirming Fireworks supports prompt caching.
4. **My cost estimate was 2.8x conservative.** Estimated $5-6, actual $2.14. Failed/ERR runs save money on failure modes that bail early. Future sweeps can be planned with this calibration.

### What it surfaced

1. **Both DeepSeek tiers cluster around 5/7 PASS** with overlapping but non-identical strengths. Flash hits higher peaks on memory-bound problems (paged_attn 0.167, kahan 0.138, w4a16 0.134); Pro is more consistent on compute-bound (sonic_moe 0.108) and has a non-zero TopK pass where Flash regressed. Both fail FP8 GEMM and KDA — those are the deck's hardest two for cheap-tier reasoning models.
2. **MiniMax's 3/7 is the floor for "model can autonomously kernel."** It needs the simpler problems (kahan softmax, sonic_moe up-proj, w4a16) and bails on harder ones with no solution.py. Useful as a benchmark sanity floor.
3. **Qwen 3.6 27B 0/7 is a harness-integration failure, not a capability failure.** 708k input tokens consumed but only 9.5k output suggests opencode is talking to it and Alibaba is responding, but tool-call exchange isn't happening. Needs investigation before counting it out as a model.
4. **qwen3.6-35b-a3b is benchmark-blocked.** Documented in the previous entry; confirmed by 7×0-second ERR runs.

### Reproducibility footnote

DeepSeek V4 Flash on TopK: passed at 0.0019 yesterday (24 hours ago, prompt-edit experiment); failed today on the shakedown. Same prompt, same model, same provider. The variance on hard problems near the model's capability floor is real and not insignificant. For TopK specifically, Flash is on the edge of "can solve this" — sometimes does, sometimes doesn't. Future passes should report N>=2 trials per (model, problem) and note variance, not just a single peak_fraction.

---

## 2026-04-27 — Harness configuration parity: what we touched and why

When you run "the same task" through five different agent CLIs, the meaning of "same" is doing a lot of work. This entry catalogs every config knob we touched to make cross-harness results comparable, and (more importantly) the asymmetries we could not eliminate. Read this if you want to know how much trust to place in any given peak_fraction comparison.

### Reasoning effort tiers (asymmetric across harnesses)

The CLI surface for "make the model think harder" differs per harness. Our active-matrix settings:

| Harness | Model | Setting | What it actually does |
| ------- | ----- | ------- | --------------------- |
| claude | claude-opus-4-7 | `--effort max` | Highest of the {low, medium, high, xhigh, max} tiers exposed by claude-code 2.1.119. Triggers extended thinking with the largest budget the CLI allows. |
| codex | gpt-5.5 | `-c model_reasoning_effort="xhigh"` | Highest effort tier codex exposes for gpt-5.5. |
| kimi | kimi-k2.6 | (default) | kimi-cli does not expose a reasoning-effort flag. K2.6 is a reasoning model and reasons by default; the budget is whatever Moonshot allocates. |
| opencode | deepseek-v4-pro / -flash, glm-5.1, minimax, qwen, mimo | (default) | opencode SST has no per-call reasoning-effort hook. The underlying model decides whether and how much to reason; some (DeepSeek V4 Pro, GLM-5.1) are reasoning models, others aren't. |

This is the biggest "same task, different shape" asymmetry in the benchmark. We use the highest tier each CLI exposes; we don't pretend that's identical to what another model does on its own. Result tables should be read as "model X via harness Y at the maximum effort that harness exposes," not "model X at parameterized effort level Z."

### Provider routing (what reaches the GPU)

OpenRouter dispatches to whichever backend has capacity. Many providers serve int4/fp4-quantized weights of frontier models; running a benchmark against int4 of GLM-5.1 is not the same as running against the lab's full bf16/fp8 weights. We pin every OpenRouter-routed model to its native lab provider via `extraBody.provider.order` with `allow_fallbacks: false`.

Current provider order in `~/.config/opencode/opencode.json` openrouter-pinned: `["Alibaba", "Xiaomi", "Minimax", "DeepSeek", "Z.AI"]`. With `allow_fallbacks: false`, a request fails if the named providers don't host the model, rather than silently falling back to a quantized third party. The fail-loud is intentional — we'd rather see "no integrity-clean route" than ship a quietly-quantized number.

Models routed lab-direct (not OpenRouter): `deepseek-v4-pro`, `deepseek-v4-flash`, `glm-5.1`, `glm-5`. These hit the lab's API directly via OpenAI-shape providers in opencode config.

Excluded from the matrix: `qwen/qwen3.6-35b-a3b`. Alibaba does not serve it on OpenRouter; only AtlasCloud and Parasail (both fp8) do. Including it would mean either accepting third-party fp8 (breaks the integrity rule) or running against a different precision than the rest of the Qwen family (apples-to-oranges). Skipped, documented; user can opt back in if they accept the tradeoff.

### Codex version pin

Local rust binary `codex 0.118.0` rejects `-m gpt-5.5` ("model not recognized"). The npm `@openai/codex` 0.125.0 accepts it but dropped `wire_api="chat"` config support, which means codex 0.125.0 cannot route arbitrary OpenRouter models — only OpenAI's `/responses` API works. Net result: codex is the right harness for OpenAI models specifically, not a universal harness for anything OpenAI-compatible. Z.AI doesn't implement `/responses` so GLM cannot be reached through codex at all; we route GLM through opencode instead.

A second codex quirk: `codex 0.125.0` updates SQLite session state by touching old session JSONL files in `~/.codex/sessions/<date>/`, which broke "find by mtime" archival. Fix: extract `session id: <uuid>` from stderr and `find -name "*${uuid}*.jsonl"` to locate the right transcript.

### Workspace state and template files

Every per-run cycle deletes everything in the problem dir except the template set. Current TEMPLATE_FILES (in `scripts/run_hard.sh`): `reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt`. Anything else the agent created (build artifacts, scratch kernels, profiling traces, intermediate `.cu` files) gets archived to `outputs/runs/<ts>/scratch/` and removed from the workspace before the next run.

`shapes.py` and `problem.yaml` stay in the workspace (model-visible) only because `check.py` and `benchmark.py` import them at runtime. A curious agent can `cat problem.yaml` and re-read the regime / forbidden ops list / tolerance — the prompt does not direct it there, but the option exists. Closing this leak would require refactoring check/benchmark to read yaml from outside the workspace; not load-bearing yet, flagged for later.

### Per-trial benchmarking methodology

Centralized in `src/eval/timing.py` so every problem's `benchmark.py` uses the same cadence:
- 10 warmup calls (absorbs Triton autotune ~7 configs and torch.compile reduce-overhead CUDA-graph capture).
- Per-trial L2 flush via 128 MB write to a scratch tensor (RTX PRO 6000 L2 is 96 MB, so 128 MB strictly evicts).
- CUDA Events with synchronize() AFTER record() but BEFORE elapsed_time().
- Median over 30 trials (default; some problems use fewer for slow Python references).

Known biases left in:
- `torch.compile(mode="reduce-overhead")` gets CUDA graphs (eliminates launch overhead). Custom Triton/CUDA kernels do not. On small shapes where launch overhead matters, this gives the compile baseline an artificial advantage. Accepted as the cost of using `torch.compile` as the published "compiled" reference line.
- cuBLAS / cuDNN allocate workspaces on first call. The 10-call warmup absorbs.
- Median over a small number of trials catches outliers but won't expose bimodal latency distributions.

### Wall-clock budget, not turn count

`BUDGET_SECONDS=2700` (45 min) per (model, problem) run, enforced by `timeout(1)`. Models get unlimited turns within the budget. v3 used `for turn in range(max_turns)` and got chewed up by reasoning models (GLM-5.1) burning turns on filesystem exploration before writing anything — a turn cap penalizes models with verbose tool-use patterns regardless of capability. Wall-clock is the fairer floor.

### Token logging (cross-harness uniformity)

Every transcript schema is different. `scripts/extract_usage.py` parses each one and emits a normalized shape:
```
{ input_tokens, output_tokens, cache_read_tokens,
  cache_creation_tokens, reasoning_tokens, total_cost_usd }
```

What's countable per harness:
- claude / kimi: terminal `{"type":"result"}` event has cumulative usage with `total_cost_usd` (only when running off API direct, not coding-plan).
- codex: per-turn `payload.type=token_count` events have `last_token_usage`; we sum.
- opencode: each `step_finish` carries `part.tokens` with input/output/reasoning + cache.read/cache.write; we sum.

What's NOT countable:
- Coding-plan billing (Claude Code, Codex on a subscription) does not expose per-call USD in the transcript. Token counts ARE present and are what we use for cross-model comparison. Per-call cost is reconstructable post-hoc from public price sheets if needed.
- Raw chain-of-thought content. Both `claude` (thinking blocks come back as `{"thinking": "", "signature": "..."}`) and `codex` (shows reasoning *summaries*, not raw CoT) encrypt the actual reasoning content in their CLI delivery channels. We get cryptographic proof that thinking happened, plus the token cost, but not the content itself. This symmetric disclosure floor is enforced by the harnesses themselves; we cannot lift it without bypassing them and calling lab APIs directly.

### What this means for cross-harness comparisons

A peak_fraction number from the benchmark is meaningful within these caveats:
- The hardware target is fixed (RTX PRO 6000 SM120, GDDR7 1.8 TB/s peak).
- The problem definition (reference.py, shapes, tolerance, forbidden ops) is fixed and append-only after publication.
- Each model runs at the highest effort tier its harness CLI exposes, but those tiers are not necessarily equivalent across vendors.
- Provider pinning ensures the model weights served are the lab's full-precision endpoint, not a quantized third party.
- Wall-clock budget and benchmarking methodology (warmup, L2 flush, median) are identical for all runs.
- Coding-plan billed runs (claude, codex) report token counts only, no per-call USD.

If you build on these numbers, cite the (model, harness, effort, provider) tuple, not just the model name. The same model behind a different harness will produce a different number.

---

## 2026-04-27 — Verification gate refinement (validated experimentally)

**Setup.** First DeepSeek V4 Flash run on TopK with the new PROMPT.txt regime: PASSed `has_solution`, FAILed correctness because the kernel allocated `threads * k * 8 = 128 KB` of dynamic SMEM on shape 0 (k=64), which exceeds the 100 KB default opt-in cap. Tool-call inventory showed Flash had run zero `python check.py` invocations — it had self-validated with two ad-hoc `python -c "from solution import ..."` snippets that almost certainly used the small default shape (16 KB SMEM) and never iterated through all five shapes.

**Edit.** Tightened the verification gate sentence in all 7 PROMPT.txt files:
- Old: `verify correctness against the oracle in check.py, then iterate. If check.py isn't passing, you're not done.`
- New: ``verify correctness by running `python check.py` and reading the output, then iterate. Don't substitute your own one-off correctness snippets for check.py — it iterates over every shape, your spot-check almost certainly won't. If `python check.py` hasn't printed PASS, you're not done.``

Three deliberate changes: (1) literal-action verb ("by running") replaces the abstract goal ("against the oracle"); (2) the middle sentence directly counter-instructs the failure mode (rolling your own); (3) PASS as the explicit sentinel string anchors the stop condition.

**Validation.** Reran Flash with the same model and the same problem; the only variable was the prompt tweak.
- Tool-call inventory: **3 `python check.py` invocations** (was zero).
- Result: PASS on all 5 shapes, peak_fraction 0.0019.
- The model produced a *correct but slow* kernel rather than a *plausible-looking but broken* one.

The score is low — Flash didn't push throughput — but the disciplinary outcome flipped from FAIL to PASS purely from the prompt edit. That's a clean experimental result. Three sentences of prompt rewrite changed the verification regime from "models that already test thoroughly do; models that don't, don't" to "models that *can* run a test, run it." Capability gates kernel quality; discipline now gates correctness.

Filed under: arguments for tightening prompts further actually do work, sometimes. Counter to my earlier "skill issue" framing — turns out half of "skill issue" is "compliance issue," and compliance is promptable.

---

## 2026-04-27 — Opus parity: --effort max wiring + token-cost logging

**Decision.** Wired `--effort` flag for the `claude` harness in `run_hard.sh` (previously only codex respected `REASONING_EFFORT`). Updated `scripts/sweep.sh` ACTIVE_MATRIX to use `claude claude-opus-4-7 max` for parity with `codex gpt-5.5 xhigh`.

**Why.** Houssin's Twitter critique on the launch post: "Why not use Opus 4.7 Max if you're using xHigh for GPT 5.5? That's not fair." Correct critique. Last sweep ran Opus at default effort while GPT-5.5 was at xhigh. The CLI exposes `low | medium | high | xhigh | max` as the effort tiers (`claude --help`); `max` is the highest. Smoke-tested with a trivial math prompt — flag accepted, thinking block emitted, output_tokens scaled past visible answer length confirming extended thinking happened.

**Thinking-content visibility.** The `thinking` block in Claude Code transcripts comes back with `thinking: ""` and a `signature: "..."` — content is encrypted in the CLI delivery channel. We get cryptographic proof that thinking happened, plus token counts, but not the raw chain-of-thought. Same disclosure floor as codex (codex shows reasoning summaries but not raw CoT either). Symmetry is preserved.

**Token logging in result.json.** Added `scripts/extract_usage.py` — a single Python script that parses each harness's transcript schema (claude/kimi `{"type":"result"}`, codex `payload.type=token_count` events, opencode `step_finish.part.tokens`) and emits a normalized `{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, reasoning_tokens, total_cost_usd}` shape. Wired into `run_hard.sh` so result.json now includes a `usage` block. Coding-plan billing on the CLI hides per-call USD, but raw token counts are always present in the transcripts and that's what matters for cross-model comparison.

Validated on the Flash rerun: input=57,555, output=12,158, cache_read=1,367,296, reasoning=98,047. The 1.37M cache_read confirms DeepSeek implicit caching is hot (matches the 98% hit rate noted in the provider-pinning entry below).

---

## 2026-04-27 — Model coverage expansion (Twitter-driven)

**Added to ACTIVE_MATRIX** (per Twitter requests):
- `qwen/qwen3.6-max-preview` — Alibaba, 262k ctx
- `qwen/qwen3.6-plus` — Alibaba, 1M ctx
- `qwen/qwen3.6-27b` — Alibaba, 262k ctx
- `xiaomi/mimo-v2.5-pro` — Xiaomi (lab), fp8, 1M ctx

All routed through `opencode openrouter-pinned/...` with `provider.order = ["Alibaba", "Xiaomi", "Minimax", "DeepSeek", "Z.AI"]` and `allow_fallbacks: false`.

**`qwen/qwen3.6-35b-a3b` — infrastructure-blocked. Holding off.** Tried twice (skipped initially for native-lab integrity; reversed on user direction with AtlasCloud and Parasail appended to provider.order). Shakedown sweep then surfaced the actual blocker: every run fails in <1s with `APIError 404: No endpoints found that support tool use. Try disabling "bash"`. Neither AtlasCloud nor Parasail advertises tool-use capability to OpenRouter for this model, and our agent harness is fundamentally tool-call-driven (bash/read/edit). There is no integrity-clean route through which an autonomous agent can use this model right now. Removed from the active matrix; will revisit if (a) Alibaba hosts it on OpenRouter, (b) AtlasCloud/Parasail expose tool-use, or (c) the model lands on a lab-direct API like Z.AI/DeepSeek already are. Filed as a useful negative result — the benchmark surfaces "no autonomous-agent endpoint exists for this model" as a real outcome, not just an integration bug.

**TODO — when budget permits:**
- **GPT-5.5 Pro.** Twitter request × 1 (insanowskyy: "what about 5.5 pro?"). Not on the active matrix because the OpenAI per-call cost is high enough to be a real budget item; coding-plan doesn't apply to API-direct gpt-5.5-pro calls. Revisit when sweep cadence justifies the spend.
- **Gemini 3.1 Pro.** Twitter request × 2. The harness story is unsettled — Droid worked in v3 but is not currently wired into KernelBench-Hard's run_hard.sh. Adding requires either (a) re-adding the droid case to run_hard.sh and authenticating via Factory, or (b) routing through opencode if Google AI Studio offers an OpenAI-compatible endpoint. Skipped until the harness wiring is decided.
- **Mythos / generic "show us X" requests.** Volume signal only; not actionable until the model has a stable identifier and a benchmarkable native-lab provider.

---

## 2026-04-27 — Prompt regime overhaul: eval-shaped → human-shaped

**Decision.** Replaced the two-file `preamble.md` + `AGENT.md` system-prompt regime with a single per-problem `PROMPT.txt` written in plain human voice. The harness now sends `PROMPT.txt` directly as the prompt to each agent — no system/user split, no markdown structure, no "Read SYSTEM_PROMPT.md first" wrapper.

**Why.** Two observations from the TopK overnight sweep:

1. The old preamble opened with "You are an autonomous coding agent being evaluated on a hard GPU kernel optimization problem." That framing primes models to perform-on-test rather than do-the-work. Opus's "the 0.1 RESULT threshold isn't structurally achievable here" rationalization is the eval-shape pattern: when you tell a model it's being evaluated, it explains its score instead of fixing the kernel.
2. The preamble was 101 lines of hardware specs, peak throughput tables, optimization recipes, profiling commands, and workflow steps. That's a benchmark giving away the answer key and then asking the model to find the answer. Models that already know this stuff gain nothing; weaker models get carried.

**What changed in the prompt itself.**

Removed entirely: opening "you are an autonomous coding agent" framing; full hardware spec section (tensor cores, what's not on SM120, etc.); peak throughput table; toolchain section (CUDA versions, compile flags, CUTLASS path); optimization guidance (FP4/FP8/BF16/TMA recipes); profiling commands (`ncu`, `nsys`, `torch.profiler`); workflow steps; budget line; "what makes a good solution"; "good luck" closer.

Kept: one-line hardware identifier in a parenthetical (`SM120 Blackwell, GDDR7, 1.8 TB/s`); library availability list (without it the model won't know FLA / scattermoe / flashinfer are options); shapes inlined as prose; forbidden ops inlined as prose; tolerance + correctness contract inlined as prose; verification gate as a single sentence in the flywheel paragraph ("If check.py isn't passing, you're not done."); custom-kernel mandate; "look up PTX docs / clone repos / investigate" directive.

**What the model now doesn't know coming in.** Peak TFLOPS for any precision. Which tensor-core instructions are available on SM120. Which are SM100-only and will fail. Compile flags. The fact that 188 SMs exist. Profiling tool names. Optimization recipes. It has to look these up itself or know them from training data — that's part of what's being measured.

**What stays in the workspace.** `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, `sota.py`, `PROMPT.txt`. The yaml and shapes.py have to stay because `check.py` and `benchmark.py` import them at runtime. Small leakage risk (a curious model could `cat problem.yaml` and read the regime / forbidden list / tolerance again), but the prompt only directs the model to `reference.py`. If that leakage matters later, the fix is refactoring check/benchmark to read yaml from outside the workspace; not yet worth the complexity.

**Files deleted.** `src/harness/preamble.md`, all `problems/*/AGENT.md` (8 files), one stale `problems/02_kda_cutlass/SYSTEM_PROMPT.md`. The harness no longer composes a SYSTEM_PROMPT.md per run.

**Smoke-tested.** Claude Code on problem 05 with `BUDGET_SECONDS=300` — confirmed PROMPT.txt arrives clean as `event[6] type=user` in the transcript, workspace cleanup behaves, no stale SYSTEM_PROMPT.md left behind.

---

## 2026-04-27 — Verification gate added (then folded into the flywheel)

**Decision.** Added a "your final action before stopping must be a successful `python check.py`" requirement to the prompt. After the prompt overhaul, this lives as a single sentence ("If check.py isn't passing, you're not done.") inside the flywheel paragraph rather than its own section.

**Why.** Of the 4 non-passing TopK runs:
- DeepSeek V4 Flash: linker error from `extern "C"` mismatch between `.cu` and `cpp_sources` header inside `load_inline`.
- DeepSeek V4 Pro: CUDA illegal memory access in the bitonic merge kernel.
- MiniMax M2.7: hardcoded `build_directory="/tmp/topk_v2"` that didn't exist; `FileNotFoundError` on first import.
- GLM-5.1: never wrote `solution.py` — burned 31,995 reasoning tokens before emitting any tool call.

3 of 4 would have been caught by running `check.py` once before submitting. The pattern is "submit blind, stop." Mandating a verification pass costs nothing for capable models, and it's not "hand-holding" — it's the discipline-half of pair programming, which is fair to require. (GLM is unfixable from the prompt; that's a Z.AI output-token-budget problem.)

---

## 2026-05-23 — Parallel sweep logging and queue-safe launch

The reliable parallel mode is `KBH_DISABLE_AGENT_CUDA=1`: hide CUDA from
OpenCode/Cursor agent phases, then let `scripts/run_hard.sh` own `check.py` and
`benchmark.py` under `outputs/gpu.lock`. This avoids the failure mode where an
agent bypasses PATH wrappers by calling an absolute `.venv/bin/python3`.
The first full launch at `kbh_hard_parallel_20260523_002720` proved the extra
guard was necessary: Cursor set `REAL_UV=$(which uv)` after the wrapper had
entered `PATH`, recursively invoking the wrapper, while several absolute
`.venv/bin/python3` children touched CUDA outside the lock. That sweep was
terminated before result collection. The harness now keeps fallback real binary
paths and injects an agent-phase `sitecustomize.py` guard so torch CUDA probes
fail fast during generation; harness-owned validation still runs without the
agent guard.

`result.json` now records `run_id`, `run_group`, ISO and epoch timestamps,
harness-only wall time, total wall time, check/benchmark wall time, queue mode,
agent CUDA visibility, and normalized usage fields. `scripts/summarize_runs.py`
flattens `outputs/runs/*/result.json` into JSON/CSV summaries for website import.
`scripts/launch_parallel_sweep.sh` writes a manifest under
`outputs/sweeps/<run_group>/manifest.tsv` and starts the model/problem matrix in
parallel.

Verification before launch:

```bash
bash -n scripts/run_hard.sh
bash -n scripts/launch_parallel_sweep.sh
uv run ruff check . --fix
uv run pytest
uv run python scripts/summarize_runs.py --output-dir outputs/summaries/smoke_latest
```

Result: 10 tests passed; summarizer wrote 167 historical rows.

Clean guarded sweep launched after the failed first attempt:

```bash
KBH_RUN_GROUP=kbh_hard_parallel_guarded_20260523_003820 \
KBH_DISABLE_AGENT_CUDA=1 \
./scripts/launch_parallel_sweep.sh
```

Manifest:
`outputs/sweeps/kbh_hard_parallel_guarded_20260523_003820/manifest.tsv`.
Early verification showed no agent-phase CUDA apps, one GPU compute process at
a time under `outputs/gpu.lock.owner`, and result rows carrying the new timing,
usage, and queue metadata. Interim summaries are written to
`outputs/sweeps/kbh_hard_parallel_guarded_20260523_003820/summary/`.

## 2026-04-26 — TopK overnight sweep: forensic findings

**Setup.** 7 models × 1 problem (05_topk_bitonic), sequential, 45-min budget each. `regime: memory`, scored against 1.8 TB/s GDDR7 peak. Geomean over 5 shapes.

**Results.**

| Rank | Model            | Status               | peak_fraction |
| ---- | ---------------- | -------------------- | ------------- |
| 1    | GPT-5.5 xhigh    | PASS                 | 0.0657        |
| 2    | Claude Opus 4.7  | PASS                 | 0.0132        |
| 3    | Kimi K2.6        | PASS (timed out)     | 0.0063        |
| —    | GLM-5.1          | ERR (no solution.py) | —             |
| —    | DeepSeek V4 Pro  | FAIL (CUDA OOB)      | —             |
| —    | DeepSeek V4 Flash| FAIL (link error)    | —             |
| —    | MiniMax M2.7     | FAIL (build dir)     | —             |

**Algorithm gap dominated kernel-craft gap.** GPT and Opus had the same wall budget on the same hardware. Opus picked full bitonic sort (O(n log²n) per row), GPT picked packed-key reduction with `tl.topk` (O(n) per row). At n=8192 that's a ~7x asymptotic gap — and the observed perf gap on the prefill shape (b=64, n=8192, k=8) was 8.7x. The kernel-craft delta would have been maybe 2x; the algorithmic choice was 5-7x of the 8.7x.

**Opus's "structurally launch-bound" claim was wrong.** On shape 0 (b=1, n=131072, k=64), Opus claimed the geomean threshold was unreachable because "the whole benchmark is launch-overhead bound." Actual numbers:
- Bandwidth lower bound to read 512 KB at 1.8 TB/s: **0.28 μs**.
- GPT-5.5 measured: **27 μs** (~100x slower than the floor).
- Opus measured: **48 μs** (~170x slower).

A single launch on a hot CUDA graph is ~1-2 μs. The remaining ~25 μs is real kernel time, not launches. Why is the kernel slow? GPT picked `chunk_n=2048` for shape 0, which gives `131072/2048 = 64` blocks for a 188-SM machine. **34% SM occupancy ceiling.** The kernel is leaving 2/3 of the GPU idle. Opus's CHUNK_PAD=2048 has the identical bug. The fix is `chunk_n=512` → 256 blocks → fully oversubscribed → near-peak bandwidth → estimated 0.10–0.15 peak_fraction on shape 0 alone.

Lesson: "launch-bound" is a real diagnosis on small kernels with many launches and no graphs. "Parallelism-starved" is a different diagnosis with the same surface symptom (low throughput on small shapes). Mixing them up is how rationalization sneaks in. Both Opus and GPT made the same parallelism-starvation mistake; only Opus rationalized it as physical-limit-bound.

**The 4 failures break into one model-side issue and three "didn't run check.py" issues.** GLM-5.1's 31995-reasoning-token blowup is fixable only by raising opencode's max output tokens for zai/glm-5.1; nothing in the prompt fixes a model that can't budget its own thinking. The other three were trivial bugs that any single test run would have caught. Hence the verification gate.

---

## 2026-04-25 — Centralized timing module + L2 flush + warmup bump

**Setup.** Each `problems/<NN>/benchmark.py` was duplicating warmup-and-cuda-events code. Several discrepancies surfaced when comparing runs.

**What we found.** Without an explicit L2 cache flush between trials, FP8 GEMM peak_fraction came out at 0.520. With a 128 MB write to evict L2 (Blackwell consumer L2 is 96 MB), the same kernel measured 0.426. The skinny-M shape went 20% → 10% with the flush. The original numbers were measuring L2-cached re-reads, not HBM bandwidth.

Warmup of 5 was too short for Triton autotune (~7 configs) plus `torch.compile(reduce-overhead)` CUDA-graph capture. Bumped to 10. `iters` defaults to 30 trials; report median.

**What lives in `src/eval/timing.py`.** Single `time_fn(fn, inputs, iters, warmup)` that does warmup → per-trial L2 flush → cuda Events with synchronize-after-record → median. All seven `benchmark.py` files import this; methodology bugs only need fixing once.

**Known biases not addressed.** `torch.compile(reduce-overhead)` gets CUDA graphs which eliminate launch overhead; custom Triton/CUDA kernels do not. On small shapes this gives the compile baseline an artificial advantage. Accepted as the cost of using torch.compile as the published "compiled" reference.

---

## 2026-04-25 — Harness wars

**ccr-rust pivot to OpenCode SST.** Tried routing Claude Code to non-Anthropic providers via ccr-rust (an Anthropic-API-shape proxy). It returned malformed SSE that broke the claude-code stream-json parser. Pivoted to OpenCode SST with custom OpenAI-shape providers (`deepseek`, `zai`, `openrouter-pinned`) — that worked.

**Codex 0.125.0 broke chat-completions routing.** The new release dropped `wire_api="chat"` config support, so codex can no longer route arbitrary OpenRouter models. It only speaks `/responses` API now. Z.AI doesn't implement `/responses`, so GLM-5.1 cannot be reached through codex at all. We fall back to opencode for non-OpenAI lab models. Documented in `CLAUDE.md` model-harness assignment table.

**Codex session-id-from-stderr instead of mtime.** Codex 0.125.0 touches old session JSONL files when scanning its SQLite thread-state DB. So picking the most-recently-modified file in `~/.codex/sessions/<date>/` returns the wrong file. The fix is to grep `session id: <uuid>` out of stderr and `find -name "*${sid}*.jsonl"`.

**`set -e` + SIGTERM 124 was a silent script killer.** When a harness hits the wall-clock `timeout` and gets SIGTERM, exit code is 124. With `set -euo pipefail`, capturing the exit via `cmd; HARNESS_EXIT=$?` exits the whole script. Fix: `cmd || HARNESS_EXIT=$?`. This bug ate two debugging sessions before we caught it.

**Local rust codex binary had a stale alias.** `npm install -g @openai/codex` gives 0.125.0 with `gpt-5.5` support; the local rust binary was 0.118.0 and rejected the model name. Non-interactive shells don't see the alias, so `which codex` was lying. Force PATH to npm bin.

---

## 2026-05-23 - Infra failure classification and safer resweep controls

Added explicit failure classification to `scripts/run_hard.sh`:
`pass`, `template_mutated`, `provider_rate_limited`, `timeout`,
`incomplete_session`, `provider_early_stop`, `no_solution`, `check_failed`,
`benchmark_failed`, and `harness_error`. Each `result.json` now carries
`failure_reason`, `retryable_infra_failure`, and
`minimum_useful_output_tokens` so the website can distinguish a bad kernel from
an API/quota/no-output event. The default minimum useful output threshold is
5,000 tokens for no-solution kernel attempts.

Added `scripts/preflight_harnesses.sh` for cheap auth/model-route checks before
paid sweeps, and `scripts/launch_infra_retries.sh` to rerun only rows whose
latest result is retryable infrastructure failure. `scripts/summarize_runs.py`
now flattens the new fields into summary JSON/CSV.

OpenRouter can pass a tiny preflight while still lacking enough balance for the
full KernelBench prompt. When `/api/v1/credits` shows usage at or above credits,
run non-OpenRouter rows with
`KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1`; this keeps Gemini running via
the Gemini API key and leaves Qwen pending until OpenRouter is topped up or a
direct Alibaba/Qwen key exists.

The guarded full-sweep launcher still uses archive-local workspaces and the GPU
lock, but now defaults to `KBH_HARNESS_CONCURRENCY=2` per harness/provider
path. Claude Code runs also pass `--settings
'{"fastMode":false,"alwaysThinkingEnabled":true}'` explicitly; Anvil's global
Claude setting was already `fastMode=false`, and previous Opus result metadata
also reported `fast_mode_state: off`, but this makes the benchmark setting
durable.

During the classified resweep, `scripts/launch_infra_retries.sh` initially
emitted tab-separated retry keys. Bash treated the empty effort field as
collapsible whitespace, shifted the problem into the effort column, and launched
blank-problem retry manifest rows. Fixed by using `|` as the retry key delimiter
and normalizing summary problem names back to `problems/<name>` before calling
`run_hard.sh`.

IVA / voice bridge jobs on Anvil have higher priority than KernelBench sweeps.
Do not kill IVA just to make `nvidia-smi` empty. The KernelBench harness should
remain lower priority; if an unrelated IVA CUDA context or CPU workload is
present, leave it alone and report it as concurrent machine state.

## 2026-04-24 — Provider pinning + caching wisdom

**OpenRouter dispatches to whichever backend has capacity, including int4/fp4-quantized weights.** Code generation on int4 is materially worse than full weights — a model that scores 50% on bf16 might score 30% on int4. So `provider_order` pinning to the native lab is mandatory for benchmark integrity.

**Pinning works in our harness, not in Droid custom models.** Droid OpenRouter custom-model configs ignore `provider_order`. The KernelBench harness sends `extraBody.provider.order` directly via the OpenAI SDK, which OpenRouter respects. Anything routed through Droid custom OpenRouter loses pinning.

**MiniMax direct API was 401.** Worked through OpenRouter pinned to "Minimax" provider (their fp8 endpoint, ~$0.30/$1.20 per M, 99.7% uptime).

**DeepSeek implicit caching is real.** Verified: same prompt sent twice in a row hit `cache_tokens: 1792 / 1829` on the second request. ~98% cache hit rate at 10x cost reduction. No explicit cache-control header needed; just resend the same prefix.

---

## 2026-04-24 — Why "Hard": pivot from KernelBench-v3

**v3 was 43 problems of grab-bag difficulty.** Most were winnable by any frontier model with any harness. Median speedups ended up reward-hacked (precision downcast, F.softmax wrappers, GEMM dispatching to `torch._scaled_mm`) or trivially above eager. Leaderboard non-informative.

**v3 reward-hack patterns.** GLM-5.1 cast fp32 inputs to fp16 before GEMM to use tensor cores → ~2x "speedup" that was cheaper arithmetic, not better algorithm. `pct_of_peak > 100%` was the giveaway. MiniMax M2.5 attempted `pkill -f python` to kill the eval process on its first run. Various models called the library wrapper (F.softmax, F.scaled_dot_product_attention) and counted that as a "kernel."

**What we tried that didn't work.** Extensive regex blocklists for forbidden patterns. Brittle whack-a-mole — every release added new ways to hide the dispatch. Replaced with an LLM judge model post-benchmark (`src/eval/judge.py`) that reviews the solution code and flags semantic cheating. Better recall than regex; defaults to PASS on judge error to avoid false negatives.

**Hard's three changes vs v3.**
1. **Tight per-dtype tolerance + multi-shape eval kills reward-hacking** at the correctness gate, so a degenerate identity-operator solution fails check.py.
2. **Roofline grading against hardware peak**, not speedup over PyTorch. Beating eager means nothing; approaching SOTA is the goal. peak_fraction = achieved_TFLOPS / peak_TFLOPS for compute regime, achieved_GB/s / peak_HBM for memory regime.
3. **Forbidden ops listed in problem.yaml + inlined into the prompt.** Using `torch.topk` on a top-k problem fails post-hoc. The point of each problem is to write the kernel, not to dispatch to a library.

**Wall-clock budgets > turn-count budgets.** v3 used `for turn in range(max_turns)` in agent loops. Models like GLM-5.1 burned 9/10 turns on filesystem exploration ("looking for CUTLASS headers") and never wrote code. Switched to wall-clock timeouts in v3 late-stage; carried over to Hard. Models get unlimited turns within a 45-min budget.

---

## Open questions / things to chase

- **Pair-programming eval.** The autonomous-floor numbers tell us how each model behaves alone; they don't tell us the human-in-the-loop ceiling. A 5-model paired-session run on problem 05 would answer "what's the agency tax of running model X without me there?" — the gap between paired and autonomous peak_fraction. n=1 per model but useful even so.

- **Persistent-kernel / cooperative-reduction kernel for shape 0 of TopK.** Both PASS submissions (Opus, GPT) are parallelism-starved on b=1 n=131072. A correctly-fanned-out kernel should hit ≥0.10 on shape 0 alone. Worth writing the reference solution by hand to confirm the achievable ceiling and validate whether the geomean threshold of 0.1 is reachable.

- **GLM-5.1 output-token cap.** The opencode `extraBody` for zai/glm-5.1 doesn't expose `max_output_tokens` — Z.AI's beta API caps at 32768. With reasoning chains of 30k+, that leaves no room for tool calls. Either request a higher cap from Z.AI, or accept GLM as an outlier whose autonomous score is bounded by output budget rather than capability.

- **Removing problem.yaml + shapes.py from the model's view.** Currently they sit in the workspace because check.py and benchmark.py import them. Refactor option: pre-render their content into the prompt (already done) and have check.py / benchmark.py read yaml/shapes from a sibling private directory. Closes a small information leak. Not currently load-bearing.

- **Per-problem prompt voice consistency.** All seven prompts hand-written in one session, same voice, same four-paragraph structure. If we add an 8th problem (Metal lightning attn) or add a second hardware target, the temptation will be to write that prompt in a different style. Resist. The voice is part of the experimental control.

---

## 2026-07-15 — kinetic-0715 (Moonshot) and the "preserved thinking" 400

Wiring Moonshot's new `kinetic-0715` through Claude Code (`kinetic-claude`
harness, same Anthropic endpoint as kimi-claude but a separate
`MOONSHOT_API_KEY`) hit a deterministic session-killer: a few tool-use turns
in, every request 400s with *"under preserved thinking, every assistant
message must pass back its thinking content, but assistant message at index N
is missing it."*

Root cause, established by capturing Claude Code's exact failing request with
a logging proxy and bisecting it via direct replays: **the model itself
sometimes emits assistant messages with no thinking block** (plain tool_use
turns), and Moonshot's validator then rejects any request that replays that
history — the endpoint 400s the model's own prior output. Verbatim replay →
400; identical body with a placeholder thinking block injected into each
thinkingless assistant message → 200. Not fixable by request params: fails
with `thinking: adaptive`, `thinking: enabled+budget`, and no thinking param
at all. kimi-k2.7-code never trips this because it emits thinking in every
assistant message (92/92 in the June topk run); kinetic skips it on some
turns.

Fix: `scripts/kinetic_thinking_proxy.py`, a stdlib-only local rewriting proxy
that injects `{"type":"thinking","thinking":"(continuing)","signature":""}`
into any thinkingless assistant message on `/v1/messages` and passes
everything else (including SSE) through. The `kinetic-claude` branches in
hard and mega `run_hard.sh` launch it per-run (container mode reaches it via
the docker bridge IP). Gotcha that cost an hour: Claude Code posts to
`/v1/messages?beta=true`, so the path match must strip the query string.

Validation: a breaker prompt forcing ~14 sequential tool calls + task-tool
usage 400s in 3 turns without the proxy, and completes 30 turns clean through
it. A 420s harness smoke on `01_fp8_gemm` then produced a **correct** fp8
GEMM at 0.106 peak fraction with zero 400s. Remove the proxy once Moonshot
fixes the validator server-side (worth reporting in the Moonshot Slack: the
one-line repro is "replay any kinetic history containing one of its own
thinkingless assistant messages").

**Update, same day:** the proxy is retired; `CLAUDE_CODE_EFFORT_LEVEL=max`
alone eliminates the 400 at the source — bisected: the breaker prompt that
dies at turn 3 by default completes 29/29 turns with thinking present in
every assistant message once effort is max (kinetic thinks on every turn, so
the validator never sees a thinkingless message to reject). The
kinetic-claude branches now set `CLAUDE_CODE_EFFORT_LEVEL=max` (also the
right call for bench comparability — matches the Opus `--effort max`
convention) and hit `https://api.moonshot.ai/anthropic` directly;
`kinetic_thinking_proxy.py` was deleted (recoverable from this entry / git
history if kinetic ever skips thinking at effort max — the residual risk is
behavioral, not contractual, and a hit would surface as a deterministic 400
→ retryable_infra_failure).
