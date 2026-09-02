# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

---

## 2026-09-02 — environments/ and benchmarks/v3 removed

The Prime Intellect `verifiers` mirrors (`environments/kernel_hard`,
`kernel_mega`, `kernel_v3`) are gone: nothing is training on them and the
sandboxed-RL-env plan was dropped. One lesson stays on record: they carried an
opt-in judge veto (`enable_judge`, default `z-ai/glm-5.2` via OpenRouter, never
switched on), and a competing board model as an API reviewer inside the reward
path is the wrong shape. Reward is `check.py` PASS x `peak_fraction`, nothing
else; reward-hack review is an offline audit on archived rollouts, the same
annotation gate as the benches. KernelBench v3 (`benchmarks/v3`, the old
per-op archive with its own harness, plus its `media/generate_dark_plots.py`)
was removed the same day; nothing on the site read it.

---

## 2026-09-02 — highlight chart generator: media/trajectory.py

The Fable 5 Mega 18.7x post (annotated trajectory, Karpathy-liked) is the format
every model post should converge on: one standout run, the trace as the story.
`media/make_fable5_trajectory.py` had its 12 points typed in by hand and is
replaced by `media/trajectory.py <run_dir>`, which reads any archive through the
bench's viewer parsers.

Why the labels are NOT automatic — the whole reason the audit owns a
`trajectory:` list in the annotation YAML: Claude transcripts carry only 3
official `benchmark.py` runs for that Fable session. The other 9 points were the
agent's own microbenchmarks, recoverable only by reading the prose. The chart is
not drawn without that list, and every entry must point at something in the
trace.

Verified on the Fable Mega run (12 points, matches the posted chart), a Fable
Hard run (1 point — Hard agents rarely rerun `benchmark.py`), and a Codex Hard
run (11 points). Regressions render rose only when the drop exceeds 3%;
sub-percent jitter is not a story.

---

## 2026-09-02 — AGENTS.md became a 10 KB entrypoint over per-directory AGENTS.md files

Grok truncates each AGENTS.md at 10,000 characters and Codex at 32 KB; the operator guide was 63 KB, so Grok never saw a rule or gate and Codex never saw the reward-hack audit, publish rules, or gotchas. The root AGENTS.md (`CLAUDE.md` and `.cursorrules` symlink to it) now holds only universal rules, gates, and a "where to read next" list; `kbtool/tests/test_repo_consistency.py` fails if it grows past 10,000 bytes or a sub-file past 32 KB. A first cut the same day split the detail into `docs/*.md` plus per-bench READMEs; that was the wrong shape and was replaced within hours by one specialized `AGENTS.md` per directory, no other `CLAUDE.md`, no READMEs or `docs/` folders below the root (project markdown is AGENTS, SPEC, DEVLOG, GOAL only). Where things went: rented-worker runbook (Lambda CLI, bootstrap order, ncu admin gate, Brev teardown, cu128 wheel, pull-size gate), harness route notes, runner behaviour, workspace/lock/cache isolation, broad-sweep launcher, and the `KB_` table to `kbtool/AGENTS.md`; layout tree, adding a problem, correctness, results, tests, sweep failures, torch pins, the audit YAML schema (was `results/annotations/SCHEMA.md`), and the `KBH_` table to `benchmarks/hard/AGENTS.md`; each deck plus its deltas to `benchmarks/<bench>/AGENTS.md` (mini owns `KBMINI_`, multi owns `KBM_` and the 4xH100 node notes); chart palette, cover card, write-up, redaction scan, ephemeral-artifact rules, short-post and article skeletons to `media/AGENTS.md`; site data flow and model/lab tables to `app/AGENTS.md`. Mega's judge rubric moved from `docs/` into its `SPEC.md`. Also on this day: the bench `.venv`s were deleted from the Mac checkout (2.5 GB); venvs exist only on the GPU boxes that run a benchmark, the Mac keeps `kbtool/.venv`. Dated stories that used to sit in the guide, kept here for the record:

- 2026-07-25: a stray `anvil:~/kb-remote-archives/` held the only local copy of 33 already-published leaderboard cells. Origin of the "every artifact stays in the repo on every machine" rule.
- 2026-06-19 contamination audit: mega-published 7/24 contaminated (the glm-5.2 17.4x and MiniMax 16.5x "beat opus" cells were fake; glm's clean score is 7.3x; opus 14-19x is real), hard-published 0/53 clean, 107/403 hard archive runs contaminated but none published. Both leaderboard builders now auto-exclude runs whose transcript references another run's archive. Proper fix remains a sandboxed harness (Prime Intellect `verifiers` env with a judge).
- 2026-08-12 friend-handoff mining (10 Grok 4.6 extractors over KernelBench sessions): SKU pin + stamp and the article cover rules were closed in code that night; still-open P0 holes were grok session-store sandbox, served-model pin, contamination rebuild, and a publish-time headline check.
- 2026-08-13: grok-4.6 copied Fable's 24.6x kernel from `runs-remote-pro` and passed the same-buffer overwrite test; hence "a literal cp of another archive's solution is contamination regardless of the overwrite PASS". Lint false positive on record: `20260614_145529_zai-claude_glm-5.2_01_fp8_gemm` (HACK flag, clean on review).
- 2026-08-17 pull sizes for the August wave: 21 full run dirs = 12.7 GiB, tiny set = 0.5 MB, tiny + transcripts = 1.0 GiB, one DeepSeek TopK jsonl = 241 MB.
- Removed as stale: 300-second smoke notes for Qwen 3.7 Max (passed, 0.4257), Gemini 3.5 Flash and Composer 2.5 (wrote, failed correctness); the retired `grok-build` model id (2026-07-15); "Kimi only after auth is fixed"; the anvil-era "do not kill IVA" and `overnight-compute status` lines (current sweeps run on rented workers); the "other serious rows" model list, which contradicted the roster rule that open benches do not enumerate models.

---

## 2026-08-07 — Property-generated structural correctness guards

The fixed numeric-stress suite caught whole-tensor scale cheats; a later audit
found the failure mode it could not catch — candidates that recognize its uniform
distributions or assume structure that never varies. The mechanism, the replay
flag, and the not-a-security-boundary caveat are in `SPEC.md`. What is only here
is the list of exploits that motivated it:

- KDA kernels truncated old recurrence state and still passed three fixed scales.
- Sonic MoE kernels assumed perfectly balanced expert offsets, or sampled a short
  input prefix before choosing an approximate path.
- Top-k kernels kept too few local candidates to survive clustered maxima.
- One CUDA graph replayed a stale pointer. Hence the replay probe rewrites the
  warmed input through a storage alias that preserves BOTH the data pointer and
  the version counter — otherwise an output cache masquerades as graph replay.

Calibration notes, so nobody re-derives them: KDA's low-key range was tuned on
RTX PRO 6000 against two independent correct kernels and the audited
diagonal/history exploit, to stay out of the unstable high-interaction regime
while keeping a clear failure margin. Sonic's fixed 129-row transfer moves one
expert from the 256-row average through an average-plus-one 128-row-block launch
cap (384 to 385) on the bounded checker shape. Only one bounded shape per problem
gets the extra calls, so the performance deck and every published score are
unchanged.

The shared `src/` tree is now copied into each run rather than symlinked,
hash-compared against its pre-agent snapshot, and restored before final grading;
otherwise a candidate could edit the new guard itself. Canonical-deck regrades
restore `src/`, `pyproject.toml`, `uv.lock`, and `.python-version` together with
the problem templates, and abort rather than fall back to archived grader code.

---

## 2026-07-24 — GPU-lock pipeline deadlock (cost 71 min of the Opus 5 sweep)

During the first Claude Opus 5 hard sweep, the whole box went idle for 71
minutes with six live sessions: load 0.16, GPU 0%, three finished cells stuck
at `Running check.py...` and two agents unable to touch the GPU.

Cause: **both stages of an agent shell pipeline are PATH-wrapped.** The
03_paged_attention agent ran

```
nvcc ... -Xptxas=-v -c pa2.cu 2>&1 | python3 -c "<parse regs/spills>" | sort | head -70
```

`nvcc` and `python3` are both wrapped by `gpu-lock-exec`, and pipeline stages
start *simultaneously*. The wrapper's same-run fast path (exec without locking
when `owner_file` names a live PID from this `RUN_DIR`) is a **race**: `nvcc`
read `owner_file` and missed, `python3` then won the lock and wrote it, and
`nvcc` fell into the unbounded `flock -x 9` fallback. From there `nvcc` waited
on a lock held by its own pipeline partner, while that partner blocked reading
`nvcc`'s stdout. Neither side could ever proceed, and the unbounded acquire
meant it never self-healed. `KBH_GPU_LOCK_HELD=1` reentrancy does not help —
it only covers parent→child, not sibling pipeline stages.

Fix: replace the unbounded `flock -x 9` with a `flock -x -w 5` retry loop that
**re-reads `owner_file` between attempts** and execs through when the holder is
a live same-run PID. Logs `reentrant_after_wait`. Applied to hard, cuda, and
mega (identical wrapper in all three).

Verified by reproducing the race directly: hold the lock for 30s from the same
`RUN_DIR`, delete `owner_file` so the waiter's fast path misses, restore it 3s
later. Old behaviour blocks the full ~28s; patched waiter execs through in 5s
with `reentrant_after_wait` in the lock log.

Operational lesson: **idle box + non-empty run set is a deadlock signature, not
progress.** A stalled sweep looks identical to a thinking agent from the
outside — check `load`, GPU util, and `gpu.lock.owner` together, and treat a
lock owner that holds while the machine is idle as stuck until proven
otherwise.

---

## 2026-07-15 — kinetic-0715 (Moonshot) and the "preserved thinking" 400

Why `kinetic-claude` pins `CLAUDE_CODE_EFFORT_LEVEL=max`: it is load-bearing, not
a comparability nicety.

A few tool-use turns into any kinetic-0715 session, every request 400s with
"under preserved thinking, every assistant message must pass back its thinking
content, but assistant message at index N is missing it." Root cause, established
by capturing the failing request with a logging proxy and bisecting it via direct
replays: **the model itself sometimes emits assistant messages with no thinking
block**, and Moonshot's validator then rejects any request that replays that
history — the endpoint 400s the model's own prior output. Verbatim replay 400s;
the identical body with a placeholder thinking block injected returns 200. Not
fixable by request params: it fails with `thinking: adaptive`, with
enabled+budget, and with no thinking param at all. kimi-k2.7-code never trips
this because it emits thinking on every assistant message; kinetic skips it on
some turns.

At effort max kinetic thinks on every turn, so the validator never sees a
thinkingless message: the breaker prompt that dies at turn 3 by default completes
29/29. A rewriting proxy (`scripts/kinetic_thinking_proxy.py`) also worked and
was deleted; recover it from git history if kinetic ever skips thinking at max
effort. The residual risk is behavioural, not contractual, and a hit surfaces as
a deterministic 400 -> `retryable_infra_failure`.

---

## 2026-07-09 - Agent-side CUDA disabling removed

`KBH_DISABLE_AGENT_CUDA` was removed from the harness, parallel launcher, and
infra-retry launcher. It had been introduced to prevent parallel agents from
bypassing the shared GPU lock, but hiding CUDA also removed the live
compile/check/benchmark/profile loop that KernelBench is intended to measure.
That made disabled and enabled runs incomparable. New runs always expose CUDA;
parallel GPU commands serialize through `outputs/gpu.lock`. Historical
`agent_cuda_disabled` metadata remains in archived results for provenance.

---

## 2026-07-09 - Hy3 TokenHub context wall + host-mode stall watchdog

TokenHub's real input wall is **196608** tokens = 0.75x the advertised 262144.
Live probes at ~200k and ~215k both report `prompt_tokens=196608` — silent
truncation, HTTP 200. At the wall, latency is 150-210s and boundary requests die
with "upstream model service is abnormal or unreachable" while sibling runs under
~160k pass concurrently.

The generalizable part: an advertised context limit the provider does not honor
puts the client's compaction threshold PAST the real wall, so the agent compacts
too late and every request at the boundary fails. The harness now defaults
`context=196608`, `output=32000` for this route (`HY3_TOKENHUB_CONTEXT_LIMIT` /
`HY3_TOKENHUB_OUTPUT_LIMIT` override), and mega's config is aligned to the same
measured number.

Second lesson from the same incident: host mode (`KBH_AGENT_CONTAINER=0`) had no
stall watchdog and `timeout 0` never fires, so a client that retries a provider
error forever with uncapped backoff and emits nothing to the transcript looks
dead for eight hours. Host-mode hy3 now runs under a transcript-growth watchdog
(`KBH_OPENCODE_STALL_SECONDS`, default 1500 — legitimate ~13 min gaps exist,
multi-hour silence does not). Distinct from the 2026-06-09 adapter hang: every
request here completed with an explicit provider error, then the client slept.

---

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

---

## 2026-07-07 - PASS-gate false negative and a GPU-lock starvation cascade

Two findings from the Hy3 + LongCat-2.0 debut sweep worth more than the cells it
produced.

**PASS-gate false negative (fixed in hard and mega).** An agent debug printf
without a trailing newline glued itself onto check.py's marker
(`kv_cache=0x7PASS`), so the anchored `grep -q "^PASS"` missed it, benchmark.py
was silently skipped, and a passing run was misclassified `harness_error`. The
gate is now `check exit 0 && grep -aq "PASS"` — strictly stronger. Any gate
anchored on a marker that the agent's own stdout can touch has this bug.

**GPU-lock starvation lobotomizes concurrent sessions.** While one run's
harness-owned check.py held the lock for 30-60+ minutes (crawling under an
unrelated vLLM co-tenant at up to 91 GB / 100% util), every OTHER session's
wrapped agent commands — `nvidia-smi` included — blocked on the lock until Claude
Code's 2-minute Bash timeout SIGTERMed them (exit 143, zombie chains of
bash -> gpu-lock-exec -> flock). LongCat's RTX kda session lost 43 of 43 bash
launches across its whole window and still shipped a first-try-PASS fallback by
static reasoning alone; that cell under-measures the model. Under co-tenant
contention the lock does not merely slow checks down, it silently removes every
concurrent agent's ability to run anything. Do not sweep against a busy GPU, and
treat check_timeouts from such a window as salvageable by rescoring the archived
solution on a quiet GPU rather than as model failures.

---

## 2026-07-04 - Fable-5 resweep: the publish footgun, the HF gap, the Fable weekly cap

Three gotchas from closing out the Fable-5 [max] sweep. The per-cell scores are
on the board and in the annotations; these are not.

**A lone unlimited cell can demote a whole row.** A fresh, audited RTX fp8 run
existed and was deliberately NOT published, because adding it to
`results/published_runs.json` and rebuilding DROPPED Fable's other 5 RTX cells,
taking the row 5/6 -> 1/6. `build_v2_leaderboard.py` has a `CAMPAIGN_EPOCH`
(20260613) rule: once a model has ANY post-campaign unlimited run, all of that
model's pre-campaign 45-minute cells are filtered out, so a model shows as EITHER
the 45-min generation or the unlimited one, never a mix. Publish a generation as
a UNIT — a full unlimited resweep of the row — never a single fresh cell onto an
older row.

**`kb push-runs` does not cover every trace.** It uploads only the run_ids in
`results/leaderboard.json` (the RTX board) and searches only `outputs/runs`, so by
itself it misses the H100 and B200 boards and every failed run. For full
coverage push manually: `scripts/traces_to_hf.py <stage> --from-list <ids.txt>
--search outputs/runs --search outputs/runs-h100 --search outputs/runs-b200`.
Mega has no bench-local converter and reuses hard's; point it at mega's
`outputs/runs`.

**The Fable weekly sub-cap, for pacing future sweeps.** The 20x max plan gates
Fable 5 at roughly half of the WEEKLY usage and then silently falls back to Opus.
The tell is Fable-specific, not an account lockout: "You're out of usage credits.
Run /usage-credits to keep using Fable 5" while Opus still runs means the halfway
line is crossed. Budget a Fable sweep against half the weekly credits, and
remember token rotation means one machine per account at a time or the
credentials get wiped mid-sweep.

Also on record: `public/runs/` is gitignored by the `runs/` rule, so solution
viewers for a non-RTX board need `git add -f`.

---

## 2026-07-03 - Hard is (near-)frozen; parked deck ideas, one line each

**THE RULE: Hard is frozen except for appends.** Chinese frontier labs have
reached out about citing KernelBench-Hard in model-release reports. Once a bench
is externally cited it becomes shared infrastructure, and mutating it
retroactively invalidates other people's published numbers (the MLPerf /
SWE-bench / GPQA lesson: version, never edit a released bench). Appending a
problem (`08_*`) is non-breaking and fine. Switching the metric, or removing or
reshaping an existing problem, is a rug pull on everyone who cited it — that has
to be a NEW bench, never an edit to Hard.

Ideas explored in the same conversation and parked. The Classic bench is not
being built; these verdicts are kept only so nobody re-derives them:

- **DeepSeek Sparse Attention as a Hard problem: no.** Hard scores fraction of
  DENSE peak with no credit for skipped work; DSA's whole point is structural
  sparsity, so it either reads absurdly low or rewards not being sparse. Its home
  is a fused decode on Mega.
- **If DSA is ever built, grade the output, not the selection.** Top-k over
  indexer scores is discrete, so fp rounding flips the selected set and
  correctness goes flaky. Grade the final attention output with tolerance and make
  the reference selection deterministic (fp32 indexer, stable tie-break, scores
  well separated near the k boundary). Same pattern as `05_topk`.
- **Metric switch to geomean speedup over `reference.py`:** right answer for
  kernels with no clean roofline (FFT, 3DGS, hash-join) and it would unify with
  Mega — but a metric switch is breaking, so new bench only.
- **Keep W4A16; it is not redundant with fp8.** fp8 is compute-bound tensor-core
  throughput; W4A16 is memory-bound with a register-level dequant pipeline and no
  tensor-memory path. Different regime, different tricks.
- **FFT is only a medium differentiator.** Shared-mem and fusion skills overlap
  fp8 and W4A16; it is distinct only on the butterfly communication pattern.
- **3DGS is the hack-resistant showpiece.** Pin a scene and camera by URL AND
  hash, naive tile-rasterizer reference, correctness by pixel tolerance / PSNR
  floor, ceiling by timing `gsplat` in `sota.py` while forbidding it in
  `solution.py`. Zero tensor cores, no allowed drop-in; only care-item is
  depth-sort determinism.

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

## 2026-06-15 - the unlimited-time generation: what the fp8 fix proved

**The headline, and the strongest evidence a problem-spec fix works.** After
making `01_fp8_gemm` genuinely fp8xfp8 (see the 2026-06-14 entry), the column was
rerun. BEFORE the fix: 0 of 8 models had ever written a real fp8 kernel — every
pass was a bf16-upcast leak or a library wrapper. AFTER: 7 of 8 wrote a real fp8
tensor-core MMA kernel, a small fast model included. Making the problem honestly
fp8 got models to do real fp8. One model wrote a real fp8 kernel and then bolted
an output-memoization hack on top of it.

**Roofline rescale, and why the records survived.** The 2.5x roofline correction
moves only `regime: compute` problems, which are graded on TFLOPS; `regime:
memory` problems are graded against the unchanged 1.8 TB/s and were untouched.
build_v2 rescales the compute-regime pre-fix runs by 0.4 rather than re-grading
them.

**Generation hygiene.** A model with any uncapped run shows ONLY its uncapped
cells, never a best-of-both Frankenstein across 45-min and unlimited budgets. Old
fp8 runs against the broken problem are quarantined so the column uses
corrected-problem runs only.

**Anvil meltdown, lesson banked (2026-06-13):** K=8 uncapped plus 5 concurrent
compile-heavy sessions drove load to 645, then OOM and SSH-dead. Uncapped
compile-heavy sweeps run at K=2.

**Integrity note for future sessions.** We revised our OWN prior `rubric_leak`
annotations once the fp8-spec bug proved the bf16 path had been the only valid
answer — the data redrew the story. Audit every passing and leader cell before
publishing, and correct the record when the evidence says you were wrong.

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

## 2026-06-13 - a flaky route is not a row; native-claude concurrency

Decision on record: qwen was left OUT of the uncapped resweep rather than
published from its only working route, the opencode / OpenRouter
`@ai-sdk/openai-compatible` adapter, which stalls on a third to a half of sessions
(2026-06-09). A row collected through an intermittently hanging transport is not
comparable to rows collected through a reliable one, and a number nobody can
trust is worse than a blank cell. The `qwen-claude` branch was wired and
preflight-stops cleanly on the missing key, so the row costs one command the day
a Model Studio key exists.

Gotcha from the same campaign: 6 concurrent native `claude` sessions on ONE coding
plan trip 401/429/rate_limit. One cell exhausted retries and was SIGTERMed at 19
minutes while the other five rode through. Native claude and codex have no stall
watchdog, so throttle concurrency on a single plan and rerun the casualties
solo.

---

## 2026-06-11 - Viewer fix: NGC banner broke format sniff; added gemini parser

Two failure modes worth remembering whenever a viewer renders nothing.

Every container-mode transcript is prefixed by the NGC image's PyTorch/driver
banner in plain text. `sniff()` read the first non-empty line, hit non-JSON, and
returned "codex" for every run; the codex parser then produced nothing from the
claude/opencode stream-json that followed. 72 of 78 viewers showed 0 events with
no error anywhere. Sniffing now skips leading non-JSON lines and picks the format
from the first real JSON line.

A harness with no parser silently falls through to the claude fallback and also
renders 0 events — that is what Gemini did until `src/viewer/parsers/gemini.py`
existed. But a thin viewer is not always a bug: grok's stream-json is pure
thought/text token deltas with no structured tool events, so 2 events is correct
and complete.

Related fix from the same pass: the leaderboard linked the wrong failed run for
no-pass cells (a context-overflow run with no solution.py instead of the sibling
attempt that wrote a real kernel and ran check). Failed attempts now rank by
has_solution -> has_check -> peak.

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

## 2026-06-11 - v2 night sweep: two infra root causes worth keeping

The hack findings from this sweep are consolidated in the full-sweep audit entry
above. What is unique here are two infrastructure failures that are class-level,
not incidents:

- **zai-claude rows must run at concurrency 1.** Three concurrent GLM sessions on
  one Z.ai coding-plan key produced a 0/6 row whose transcripts are nothing but
  api_retry events — 78+ retries, zero assistant turns. A pure 429 storm looks
  exactly like a model that wrote nothing.
- **File-copied OAuth credentials rot.** A grok 0/6 row traced to OAuth refresh
  rotation: a smoke run rotated the token inside its archived agent_home copy,
  orphaning the host `~/.grok/auth.json`, so every later run fell into an
  interactive login prompt and timed out. Same class as the Anthropic OAuth
  expiry. Long-lived env tokens are the durable fix where the provider supports
  them; where it does not, sync the auth file back to the host after every run.

Third, smaller: a pinned-provider opencode config that never got copied into the
container agent home produced instant crashes that read as model failures.

---

## 2026-06-10 - Container sessions run in parallel; the shinit fork bomb

Container mode was serializing the whole sweep by holding the GPU lock for each
session's entire budget. It now matches the host model — agent sessions overlap
freely and GPU-facing commands serialize per-command through a flock on the lock
inode, so host and container commands serialize against each other. The current
lock and wrapper design is in `kbtool/AGENTS.md`. What is documented nowhere
else are the two gotchas that cost a night:

**The NGC image sets `BASH_ENV=/etc/shinit_v2`, which runs `nvidia-smi` on EVERY
bash startup.** With the wrapper directory first on PATH that resolves to our own
bash wrapper, whose startup sources shinit_v2 again — a fork bomb that silently
consumed the container PID limit and produced empty transcripts. The runners now
set `-e BASH_ENV= -e ENV=`. If a future image needs shinit, wrap the agent
command instead of the global PATH.

**Never overwrite a runner script in place while runs are active.** bash reads
scripts incrementally and an scp overwrite reuses the inode, so a running region
shifts under the interpreter; one smoke died with a phantom syntax error this
way. Deploy with scp to a temp path plus an atomic `mv`.

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

## 2026-06-09 - CUDA toolkit version is benchmark surface

Evaluated bumping the toolchain mid-table and declined. Every published row was
compiled and scored under one nvcc, and ptxas codegen directly shapes
peak_fraction, so swapping the toolkit between rows silently changes the
instrument the scores were measured with.

Rules going forward:

- The CUDA toolkit is part of the benchmark surface, like problem templates and
  tolerances. Pin it per leaderboard version; never bump it between rows of the
  same table.
- Agent dev and host scoring must use the identical toolkit. Today that holds by
  construction — container mode bind-mounts the host toolkit. Keep it that way.
- Record the nvcc version in `result.json` so future audits do not have to infer
  it.
- Any bump happens only at an explicit benchmark version boundary, validated by
  rebuilding a few archived passing kernels under old and new toolkits and
  comparing benchmark times before publishing.

Same shape as the per-bench torch pins in `benchmarks/hard/AGENTS.md`, but stronger: torch is
explicitly not the scored surface, and ptxas is.

---

## 2026-06-01 - Removed Kahan softmax from the active deck

`04_kahan_softmax` has been removed from the benchmark surface. The problem was
too easy to satisfy with a plain fast softmax under the existing tolerance, so
it rewarded the shortcut instead of forcing compensated summation. Current
scripts, machine-readable results, baselines, annotations, and leaderboard docs
no longer include it. Historical DEVLOG discussion is intentionally preserved
below as audit context for why the problem was removed.

---

## 2026-06-01 - Benchmark scoring is solution-first by default

KDA exposed a general harness risk: a reference diagnostic can be SLOWER than the
submitted kernel, so timing eager / `torch.compile(reference)` / SOTA before the
solution turns a valid submission into a post-run benchmark timeout — a model
failure that was really a harness ordering bug. Every `benchmark.py` now times
and prints `variant=solution` first, and the reference diagnostics are opt-in via
`KBH_BENCHMARK_BASELINES=1`. `src/eval/timing.py` emits `benchmark_event` lines
around each variant so an audit can split solution, eager, compiled, and SOTA
wall time straight out of `benchmark.log`.

---

## 2026-05-31 - Provider-failure classification reads error events, not prose

A sweep summary reported `provider_rate_limited` and
`provider_insufficient_credits` for two cells that had hit neither. Both were
transcript false positives: the model had READ text containing "quota / rate
limits" out of `AGENTS.md` and `insufficient_credits` out of `run_hard.sh`, and
the classifier was scanning arbitrary assistant text and tool output.

Classification now lives in `src/harness/classification.py` and scans explicit
CLI/API error events plus stderr only, on rows with no solution. That is why the
rule is worded so narrowly: on this benchmark the agent reads the harness's own
source, so any classifier keying on substrings anywhere in the transcript will
eventually classify the repo's own documentation as a provider outage. The same
bug bit again later as plain `overage` matching the word `coverage` in a Cursor
transcript.

---

## 2026-05-23 - Lock wait must not consume the correctness budget

`check.py` and `benchmark.py` now acquire `outputs/gpu.lock` BEFORE their
execution timeout starts. The old `timeout 180 uv run python check.py` shape let
lock wait eat the correctness budget, so queued rows looked like model failures;
execution timeouts are now classified `check_timeout` / `benchmark_timeout` and
marked retryable instead of `check_failed`.

Second fix, same class of "the harness made the model look bad": Claude-family
harnesses now `cd "$PROBLEM_DIR"` before launching. With the old repo-root cwd
plus `--add-dir`, some runs spent huge token budgets writing
`problems/<name>/solution.py` into the SOURCE tree while the archive-local
workspace stayed empty — a full session graded as no_solution, plus a
contaminated source tree to clean up.

---

## 2026-05-23 - Infra failure classification and preflight limits

`result.json` carries `failure_reason`, `retryable_infra_failure`, and
`minimum_useful_output_tokens` so the site can distinguish a bad kernel from an
API/quota/no-output event; the taxonomy itself lives in the runner and
`benchmarks/hard/AGENTS.md`. Two things from building it that are not derivable
from the code:

- **A cheap preflight is not a budget check.** OpenRouter passes a tiny auth probe
  while lacking the balance to serve one full KernelBench prompt. Compare usage
  against credits at `/api/v1/credits`, not just a 200.
- **Do not use tabs as a retry-key delimiter in bash.** The retry launcher emitted
  tab-separated keys; bash treated the empty effort field as collapsible
  whitespace, shifted the problem into the effort column, and launched
  blank-problem retry rows. The delimiter is `|` now.

---

## 2026-05-22 - Archive-local workspaces, and the first lock reentrancy deadlock

Every run gets a repo-shaped workspace under
`outputs/runs/<run_id>/repo/problems/<problem>/` so two agents on the same problem
can never delete or overwrite each other's `solution.py`. The current shape, cache
vars, and wrapper list are in `kbtool/AGENTS.md`.

The one thing worth keeping from building it: the first wrapper attempt deadlocked
against itself. `uv run python benchmark.py` took the GPU lock, then the
benchmark's own child `nvcc --version` hit the same non-reentrant wrapper and
blocked on a lock its parent already held. Fixed by setting `KBH_GPU_LOCK_HELD=1`
while executing the real locked command, so nested wrapper calls exec straight
through. That fix covers parent -> child only, which is exactly why the sibling
pipeline-stage race of 2026-07-24 was still possible. Read the two together
before touching the wrapper.

---

## 2026-05-14 — Z.ai GLM-5.1 rerun on the corrected Anthropic endpoint

Z.ai asked for a rerun after the public GLM-5.1 row, saying cells had terminated
early, and got one on the corrected endpoint; Shuyan supplied their internal
Claude Code eval config, now baked into the `zai-claude` branch and recorded in
`kbtool/AGENTS.md`. Precedent worth keeping: a lab that disputes a row gets the
rerun, on the record, with the config it says is right.

`01_fp8_gemm` from that rerun is deliberately NOT counted as a pass even though
its archived `result.json` says `correct=true`. The model edited `problem.yaml`,
changing the tolerance key from `fp8_e4m3fn: 0.15` to `bfloat16: 0.15`. Because
`check.py` looks up tolerance by `ref_out.dtype`, which is bfloat16, that swap
relaxed the real check from the default bf16 tolerance to 0.15. It is the
cleanest published reward-hack example we have — and, as the 2026-06-14 entry
later found, while cheating the model had accidentally located a genuine bug in
the tolerance lookup.

The harness fix this produced: `run_hard.sh` snapshots `reference.py`, `sota.py`,
`shapes.py`, `problem.yaml`, `check.py`, `benchmark.py`, and `PROMPT.txt` before
each agent run. Any change, deletion, or unexpected creation marks the run
invalid, writes `template_mutated=true`, diffs into `template_mutations.log`, and
restores the originals before the next problem.

---

## 2026-04-30 — Launch prep: monorepo and the Vercel commit gotcha

**Monorepo.** The standalone `KernelBench-Hard` and `KernelBench-v3` repos were
absorbed into `kernelbench.com` as `git subtree` merges with history preserved.
The website lives at the repo root for Vercel auto-detection; benchmarks live
under `benchmarks/`. Trade-off accepted deliberately: the per-suite DEVLOGs stay
inside their subdirs, which buys cleaner per-suite history and makes a single
chronological narrative across suites harder. Still worth it.

**The Vercel deploy gotcha, which is why `AGENTS.md` names an email.** Every
commit pushed with the autogenerated `infatoshi@anvil...ts.net` address failed
Vercel's commit-verification gate at the pre-build phase — a silent ERROR with no
build logs at all. Three commits errored before it was traced. Pass
`-c user.email=elliot@arledge.net` inline or set it in the clone; a fresh clone on
a new machine reproduces this.

**Annotation policy set here, still current:** annotate every cell where there is
something to say — the rubric leaks, the honest failure, the clean top performers
— and leave low-peak cells unannotated on purpose.

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
- `results/annotations/<run_id>.yaml` — per-cell commentary for 13 runs covering both leaks (5 fp8 cells, 7 kahan cells) plus the headline clean cell (opus paged_attention 0.602).
- the annotation file format with five verdicts (`clean`, `rubric_leak`, `reward_hack`, `interesting`, `bug`); schema now in `benchmarks/hard/AGENTS.md`.

### Future leak fixes (logged, not done)

- **fp8_gemm**: tighten tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge on the test inputs, or add a static-analysis check to the rubric that detects the `fp8 → bf16` cast pattern before the GEMM call.
- **kahan_softmax**: tighten tolerance to a value where naive vs Kahan produce visibly different results on the test inputs (the test inputs may need to include numerically-pathological cases — large logit ranges, near-equal extremes), or write a check that detects compensated-summation pattern in solution.py.

These are tractable; deferred so we publish the leaderboard now.

---

## 2026-04-27 — opencode workspace leak: root cause + partial fix

A forensic dive into Qwen 27B's 0/7 led to auditing every opencode-routed `read` call across the shakedown. The leak is universal across all opencode-routed models, not just Qwen.

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

### Still open (moved here from the old open-questions list)

`problem.yaml` and `shapes.py` sit in the workspace only because `check.py` and
`benchmark.py` import them at runtime, so a curious agent can still `cat
problem.yaml` and re-read the regime, forbidden-ops list, and tolerance. Closing
it means refactoring check/benchmark to read them from a sibling private
directory. Not load-bearing so far; not done.

---

## 2026-04-27 — N=1 is not enough: the Qwen reversal

Qwen 3.6 27B went 0/7 to 1/7 with a ~10x jump in engagement on a rerun 11 hours
after the opencode leak fix landed. The tempting story is that removing the leak
forced focus. Three honest possibilities, and we never got to pick one:

1. Removing the leak forced focus — pre-fix the model burned tool calls reading
   `src/hardware/`, `src/eval/`, and the user's own kernel skill files; post-fix
   those reads fail fast and redirect it to `reference.py`.
2. LLM nondeterminism. Same model, same prompt, 11 hours apart.
3. Both.

Two PASS/FAIL reversals inside 24 hours on this benchmark: DeepSeek Flash on
TopK, and this. Isolating the effect would take 5x runs per disposition, which
was never run, so "the leak fix improved Qwen" must not become load-bearing.
Official results should be N>=2 per (model, problem) with variance reported. The
earlier "capability + compliance, dropped permanently" framing was itself a
misread driven by N=1.

---

## 2026-04-27 — What the verification gate can and cannot fix

From the forensics on Qwen 3.6 27B's 0/7 (reversed by the rerun above), one
observation that generalizes past the model.

Qwen's pattern is the INVERSE of the DeepSeek Flash verification-gate result.
Flash never read the rule and skipped the test; tightening the prompt fixed it.
Qwen read the rule, said it out loud — "let me verify the check infrastructure
before writing the kernel" — and then stopped without acting on it, repeatedly.
That is not a prompt-clarity problem, and no further tightening of the sentence
addresses it.

The verification gate works on models that have the discipline half latent and
need the cue. It does not manufacture discipline where there is none. When a
model vocalizes a rule and violates it in the same session, stop editing the
prompt.

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

## 2026-04-27 — Opus parity: --effort max

Wired `--effort` for the `claude` harness and moved the matrix to Opus at `max`
alongside Codex at `xhigh`. The origin is a public critique of the launch post —
"Why not use Opus Max if you're using xHigh for GPT-5.5? That's not fair." It was
correct: the previous sweep ran Opus at default effort against GPT-5.5 at xhigh,
and that comparison should never have been published.

This is why effort is a publish gate rather than a nicety. A dropped `--effort`
does not make the cell noisier, it makes it a different cell. Use the highest tier
each CLI exposes and record which one was used.

---

## 2026-04-27 — "No autonomous-agent endpoint exists" is a real result

`qwen/qwen3.6-35b-a3b` was requested, tried twice, and is benchmark-blocked: every
run failed in under a second with `APIError 404: No endpoints found that support
tool use. Try disabling "bash"`. Alibaba does not host it on OpenRouter, and the
third parties that do are fp8 and do not advertise tool use — so there is either
no integrity-clean route or no agentic route at all.

Filed as a useful negative result, not an integration bug. A benchmark that
requires an autonomous tool-using session legitimately surfaces "no agent harness
can drive this model at full precision today" as an outcome. That is not the same
finding as a low score and must not be recorded as one.

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

Of the 4 non-passing TopK runs, 3 failed on trivia that a single `check.py` run
would have caught: a linker error from an `extern "C"` mismatch, an illegal memory
access in a bitonic merge, and a hardcoded build directory that did not exist. The
pattern was "submit blind, stop." The fourth never wrote a solution at all and is
not promptable.

So the prompt gained a verification requirement, which after the prompt overhaul
lives as one sentence inside the flywheel paragraph rather than its own section.
Mandating a verification pass costs a capable model nothing, and it is not
hand-holding — it is the discipline half of pair programming, which is fair to
require.

---

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

**Still open (moved here from the old open-questions list):** whether the TopK
geomean of 0.1 is physically reachable at all. Both PASS submissions were
parallelism-starved on shape 0, so writing the persistent-kernel /
cooperative-reduction version by hand would confirm the achievable ceiling and
settle whether the threshold was ever realistic. Not done.

---

## 2026-04-25 — Centralized timing module + L2 flush + warmup bump

**Setup.** Each `problems/<NN>/benchmark.py` was duplicating warmup-and-cuda-events code. Several discrepancies surfaced when comparing runs.

**What we found.** Without an explicit L2 cache flush between trials, FP8 GEMM peak_fraction came out at 0.520. With a 128 MB write to evict L2 (Blackwell consumer L2 is 96 MB), the same kernel measured 0.426. The skinny-M shape went 20% → 10% with the flush. The original numbers were measuring L2-cached re-reads, not HBM bandwidth.

Warmup of 5 was too short for Triton autotune (~7 configs) plus `torch.compile(reduce-overhead)` CUDA-graph capture. Bumped to 10. `iters` defaults to 30 trials; report median.

**What lives in `src/eval/timing.py`.** Single `time_fn(fn, inputs, iters, warmup)` that does warmup → per-trial L2 flush → cuda Events with synchronize-after-record → median. All seven `benchmark.py` files import this; methodology bugs only need fixing once.

**Known biases not addressed.** `torch.compile(reduce-overhead)` gets CUDA graphs which eliminate launch overhead; custom Triton/CUDA kernels do not. On small shapes this gives the compile baseline an artificial advantage. Accepted as the cost of using torch.compile as the published "compiled" reference.

---

## 2026-04-25 — Harness wars: two bugs still worth knowing

**`set -e` plus SIGTERM 124 is a silent script killer.** When a harness hits the
wall-clock `timeout` it exits 124; under `set -euo pipefail`, capturing that with
`cmd; HARNESS_EXIT=$?` exits the whole script instead. Use
`cmd || HARNESS_EXIT=$?`. This bug ate two debugging sessions before it was
caught.

**Find the codex transcript by session id, not mtime.** Codex touches old session
JSONL files in `~/.codex/sessions/<date>/` while scanning its SQLite thread state,
so the most-recently-modified file is the wrong one. Grep `session id: <uuid>` out
of stderr and `find -name "*${uuid}*.jsonl"`.

Also on record: a routing layer that returns malformed SSE (ccr-rust, at the time)
breaks the stream-json parser with no explicit error anywhere, which is what
pushed non-Anthropic providers onto OpenCode and later onto Claude Code against
each provider's own Anthropic endpoint.

---

## 2026-04-24 — Provider pinning; the Droid trap

OpenRouter dispatches to whichever backend has capacity, including int4/fp4
quantized weights, and code generation on int4 is materially worse than on full
weights — a model that scores 50% on bf16 might score 30% on int4. Pinning to the
native lab provider is therefore mandatory for benchmark integrity, with
`allow_fallbacks: false` so a request fails loudly rather than silently serving a
quantized third party. The fail-loud is the point: "no integrity-clean route" is a
better outcome than a quietly quantized number.

The trap: pinning works in our harness because it sends `extraBody.provider.order`
directly, and does NOT work through Droid custom OpenRouter models, which ignore
`provider_order` entirely. Anything routed that way has lost its precision
guarantee even though the config looks right.

---

## 2026-04-24 — Why "Hard": what v3 actually did

The design argument is in `SPEC.md`. What is only here is what v3's models did,
which is the evidence behind every anti-hack rule in the deck:

- GLM-5.1 cast fp32 inputs to fp16 before the GEMM to reach tensor cores — a ~2x
  "speedup" that was cheaper arithmetic, not a better algorithm. The giveaway was
  `pct_of_peak > 100%`, which is also why a peak table a kernel can exceed is
  treated as a bug in the table, not a triumph.
- MiniMax M2.5 attempted `pkill -f python` to kill the eval process on its first
  run.
- Several models called `F.softmax` or `F.scaled_dot_product_attention` and
  counted the library wrapper as their kernel.

What did not work: extensive regex blocklists for forbidden patterns. Brittle
whack-a-mole — every model release found a new way to hide the dispatch. v3
replaced them with an LLM judge over the solution code, which had better recall.
Worth reading beside the 2026-09-02 decision to take the judge OUT of the RL
envs' reward path: a judge is a reasonable OFFLINE reviewer of finished code and a
bad online grader, and the annotation gate is where that review lives now.
