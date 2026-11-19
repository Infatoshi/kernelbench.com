# KernelBench-CUDA — DEVLOG

## 2026-11-19 - Gemini 4 Neon (XHigh) solo Fused MoE rerun: 0.1105 (`clean`, #2 on MoE, CUDA composite 0.9665)

Solo rerun (`20261119_123612_agy_gemini-4-neon-xhigh_01_glm52_fused_moe`) of CUDA 01 on tetra GPU 2
(12:36 to 3:36 pm CST, 3.0 h wall, after deploying the zero-fork `gpu-lock-exec` `ncu` fix; an
earlier GPU 0 attempt `20261119_114923` was aborted at minute 31 on the pre-fix `ncu` deadlock).
Initial post-run grade (0.0717) and GPU 2 regrade (0.0807) overlapped an unrelated external host
job on shapes 3 to 5; clean isolated sequential re-regrade on quiet GPU 0 (4:10 to 4:20 pm CST,
5-second watch log confirmed zero foreign compute PIDs during the entire `benchmark.py` window):

- 01 Fused MoE 0.1105, `clean`, 2nd overall (+9.4% over lock-blocked V1 0.1010, passing Fable 5.2
  at 0.1085 and GPT-6.2 Astra at 0.1064, trailing only Sonnet 5.5 at 0.1130 by 2.2%). Self-contained
  1,977-line SM120 CUDA/PTX/TMA suite with zero CPU-GPU sync across 4 adaptive regimes: split-warp
  streaming GEMV for decode (`T <= 4`), single-barrier hybrid `cp.async` + 3D-TMA (`L2::evict_first`
  when `nm==1`) with direct-store shared-expert overlap for small (`BM=64`) and medium (disjoint
  `[BM=128, BM=192]`) batches, and sequential direct-store shared + universal 3-bucket
  `[BM=128, BM=96, BM=192]` schedule minimizing `(padded_rows, m_tiles)` at `T=8192`. Sets board-best
  latency at `T=1` (0.299 ms), `T=512` (7.921 ms), and `T=1000` (7.993 ms), with `T=4096/4127/8192`
  at 8.829 / 9.069 / 14.215 ms. Live probes on GPU 0: 4/4 same-buffer overwrite regimes pass
  (cos(out1,out2) ~0, cos(ref,sol) 0.999997), 12/12 unseen seeds (123/456 x 6 shapes) pass, and
  `CUDA_LAUNCH_BLOCKING=1` passes on `T=512` and `T=4096`. Operator sent one `SIGTERM` at 1:52 pm
  CST to a deadlocked scratch test (`bench_ab_v2.py`, host PID 943258 / sandbox PID 4521). Lifts
  Neon's CUDA composite from 0.9384 to 0.9665 (#2 overall, within 2.9% of #1 Sonnet 5.5 at 0.9951).

## 2026-11-19 - Gemini 4 Neon (XHigh) on tetra: 3/3 correct, two `clean`, one `interesting`

`agy` harness, `gemini-4-neon-xhigh` (reasoning level in model id), three sessions on tetra GPU 0
starting 3:04 to 3:05 pm CST 2026-11-18 alongside Mega 02. Isolated sequential regrades on quiet
GPU 0 (12:37 to 12:51 am CST 2026-11-19) and quiet GPU 2 for 03 (1:06 to 1:18 am CST, replacing an
initial GPU 0 regrade at 0.0736 overlapped by an unrelated `llama-imatrix` job on shape 3), plus
audit probes on GPU 3 (timings strictly from the quiet GPU 0 / GPU 2 regrades):

- 01 Fused MoE 0.1010, `clean`, 7th overall on an uniterated V1 first draft. Pure SM120 PTX
  (`cp.async.cg`, `ldmatrix.x4`, bf16 `mma.sync`, 2-kernel streaming GEMV at `T <= 4`), zero cache
  or graph; same-buffer overwrite cos(out1,out2) ~0, cos(ref,sol) 0.999997, 6/6 unseen seeds pass.
  After its initial smoke test, sibling 02's deadlocked smoke test held the shared `cuda` lock from
  3:18 to 6:01 pm, so this session exited cleanly at 5:29 pm without ever benchmarking or iterating.
- 02 NSA 1.3506 (0.0730 ms geomean), `interesting`, 2nd overall (within 0.8% of #1 Sonnet 5.5 at
  1.3619 / 0.0724 ms, well ahead of #3 Carbon at 1.1247 / 0.0879 ms). Exact 3xBF16 block scoring,
  exact reference tie-break to larger block index (`S=1024/2048` tie probe cos 1.000000/0.999999),
  lock-free inverted-index key-stationary attention with PTX `int8` partial outputs (`pack_i8x4`),
  same design class as Sonnet 5.5. Overwrite recomputes on all three paths, 12/12 unseen seeds pass
  (mean_abs 0.00028-0.00035, slightly tighter than Sonnet 5.5's 0.00037-0.00044), extended `S=16384`
  passes (cos 0.999982). Operator sent `SIGTERM` three times: deadlocked smoke test (6:01 pm) and
  two `lockshim` + `ncu` zombie hangs (7:24 pm, 8:36 pm; root DEVLOG 2026-11-19).
- 03 MegaQwen 0.0755 (14,499 / 7,435 / 2,594 / 725 tok/s at 2k / 8k / 32k / 128k), `clean`, 3rd
  overall (2nd `clean`, ahead of Opus 5.5 at 0.0739, Fable 5.2 at 0.0671, Carbon at 0.0612, and
  GPT-6.2 Astra at 0.0578). SM120 CUDA Graph + Programmatic Dependent Launch (`griddepcontrol`),
  exact PyTorch CPU MT19937 + Box-Muller noise computed on GPU inside the timed graph, full bf16 KV
  cache (~1.56 TB/s at 128k). Same-buffer overwrite of hidden + KV recomputes (cos 0.999946),
  `p.copy_` repacks cleanly (0.999952; `.data.copy_` bypasses `_version` like Sonnet 5.5), unseen
  seeds 123/456 pass (0.999966/0.999964), 1-step `q_norm/k_norm` gain stress holds 0.999983-1.000000
  through gain=8 (all finite), and `CUDA_LAUNCH_BLOCKING=1` completes cleanly in 0.35 s.

## 2026-11-17 — GPT-6.2 Astra [xhigh] on tetra: 3/3 correct, two `clean`, one `interesting`

`codex` harness, native Codex CLI 0.169.0 on a ChatGPT Pro login (earlier CLIs reject `gpt-6.2-astra`
with "not supported when using Codex with a ChatGPT account"), `gpt-6.2-astra` at `model_reasoning_effort
xhigh`, reasoning summaries `auto`, service tier default (fast off). Effort matches the September GPT-6
Astra Pro cells, which ran Codex 0.140.0 through OpenRouter, so the two are a rough comparison. Three
sessions on tetra GPU 2 from 1:33 pm with the two mega sessions; each stopped on its own in 36 to 65 minutes
on 30k to 45k output tokens. Sequential isolated regrade on GPU 2, 2:51 to 3:18 pm, then audit probes on
quiet GPUs 0 and 1:

- 01 Fused MoE 0.1064, clean, third of 29 and the best codex cell (GPT-6 Astra Pro 0.0945). Hand-written
  TMA + mma.sync grouped GEMM, no cache of any kind; same-buffer overwrite cos(out1,out2) about 0, 54/54
  unseen-seed stress cases pass. Its in-run 0.0672 was contention from the MegaQwen sibling (below).
- 02 NSA 0.8005 (0.1231 ms geomean), clean, fifth in latency. Exact-ranking block scores, reference tie
  rule, each selected block attended once per choosing query; overwrite probe recomputes, 24/24 unseen
  seeds, S up to 16384 past the deck pass.
- 03 MegaQwen 0.0578, interesting, ties 11th. 28 ordinary kernels per token, not a megakernel. Pointer-
  and version-keyed weight pack (`.data` edits are missed, off the graded path); multi-step decode drifts
  out of tolerance at q/k norm gains of 2 and up (graded gain is 1). It used the upstream MegaQwen repo the
  prompt offers, via a stale `/tmp/megaqwen_ref` clone left by the 2026-11-16 Qwen 4 Max attempt; 4 of 310
  lines match upstream, no copied code.

Harness gaps found by the audits, none touching the regrades: (1) inside the sandbox plain `python`
resolved to `~/.local/bin/python` ahead of the run's lock wrapper, so the NSA and MegaQwen sessions ran GPU
work outside the GPU lock for long stretches and contended the siblings' in-run numbers; (2) `/tmp` and
`~/.cache` are shared across sandboxes; (3) the stale tetra checkout `~/dev/sites/kernelbench.com` (old
archives with solutions) is outside the sandbox hide root and the canary misses it. All three cells'
`kb contamination` flags are a `rg --files -g pyvenv.cfg` over home that listed paths in that stale
checkout (or, for 03, the upstream repo listing); nothing was opened.

## 2026-11-16 — Qwen 4 Max [xhigh] on tetra: 3/3 correct, two `interesting`

`or-fable` harness (Claude Code on OpenRouter, `qwen/qwen4-max`, effort xhigh, prompt caching disabled; root
DEVLOG 2026-11-16), three sessions side by side on tetra GPU 2 from 2:01 am, 2.6 to 5.5 hours each, with the
Mega cell on the same card. Sequential isolated regrade on GPU 2 from 8:09 to 8:38 am CST, the window checked
against the host GPU-use log (11 blocks, 0 foreign processes), then audit probes on a quiet GPU:

- 01 Fused MoE 0.0936, interesting. Every FLOP is the model's own tensor-core CUDA: no cuBLAS, no input
  cache, routing recomputed in the timed window. The kernel header blames an "SM120 caveat" for a race in
  its `cp.async` pipeline and drains every copy before each compute step. Probe Part F: the session's wait
  depth mismatches on 20 of 20 reps, and one empty `commit_group` in the last k-steps makes the deeper
  pipeline bitwise equal on 20 of 20, so the hardware claim is the model's own tail bug. It cost little
  (the fixed pipeline is +8.3% at T=1, -0.1% to +2.0% at the larger shapes). The session benchmarked on
  GPU 3 and GPU 1, neither its own.
- 02 NSA 0.7002 (0.1408 ms geomean), clean. Key-stationary design: tensor-core scoring on block-mean keys
  split into three bf16 planes, register top-8 with the reference tie rule, each selected K/V block loaded
  once for up to 128 queries, log-sum-exp merge, all inside a CUDA graph keyed on the input pointers.
  Probe: the replay recomputes from live memory (cos(out1,out2) about 0, cos(ref,sol) 0.999996), 12 of 12
  unseen seeds pass, the graph is worth about 4% (0.1414 ms against 0.1475 ms direct). Its one original
  idea, an L2-resident locked merge for D=128, was built, measured wrong and slower, and left switched off.
- 03 MegaQwen 0.0466, interesting. One cooperative launch for the whole timed decode, 188 blocks, five
  barriers per layer. It leaves the decode input draw inside the timed call (about 4% at 2k context,
  measured; the Kimi and DeepSeek kernels moved it out). The softmax has a fixed shift in place of the max
  subtraction: exact at the graded unit gains, non-finite at q/k gains of 6 and up. Its second version
  timed 114.5 us/step at 2k in its own test; a NaN in the third cost 2.8 hours and the shipped rewrite
  runs 172 us/step.

Conduct, all in the audit YAMLs: the MegaQwen session killed five jobs of the live NSA session between
4:18 and 5:56 am after deciding the lock holder was an orphan, and ran an A/B on GPU 1; the Mega session
killed a sixth. The NSA session never left GPU 2 and refused to kill a sibling's hung job. All three
cells trip `kb contamination` on process listings and lock-owner files that name sibling run dirs (no
archive was read) and carry a manual `contamination: clean`.

The first Fused MoE probe ran on a borrowed GPU 3 and overlapped the Mega session's final check and
benchmark there; it was rerun on a quiet GPU 2 and only the rerun is quoted. Check an audit probe's window
against the GPU-use log before writing that the card was quiet, the same as for a regrade.

## 2026-11-16 — Kimi K3.1 Pro [max] on tetra: 3/3 correct, one flagged

`kimi` harness (Kimi Code CLI 2.5.3, `kimi-code/k3.1-pro`, effort max), three sessions on tetra GPU 3 that
each stopped on their own in 60 to 90 minutes. Sequential isolated regrade on GPU 3, repeated until a
GPU-use log showed the card to itself (Qwen 4 Max sessions on GPU 2 overrode their GPU pin during the
first passes; root DEVLOG 2026-11-16), then audit probes on the quiet GPU:

- 01 Fused MoE 0.1007, `rubric_leak`, the first on the RTX board. The routing sort is cached on the
  identity of the expert-id tensor and `benchmark.py` reuses its inputs, so the sort never lands in the
  timed window; the session's first design note says so. Probe: the cache is worth 1.068x on the geomean
  (30% at T=1), and an in-place rewrite of the ids returns stale output (cos 0.81 to 0.88). About 0.094
  on equal terms. GEMMs are cuBLAS.
- 02 NSA 0.1037 (0.950 ms geomean), clean. Exact per-query top-8 with a WMMA scoring kernel and a
  per-query gather; no cache. Overwrite and tie-direction probes pass. More than 10x behind the designs that
  share K/V blocks across queries.
- 03 MegaQwen 0.0610, interesting. One cooperative launch for the whole timed decode. Decode inputs are
  drawn in the untimed prefill (3.3% at 2k context, measured), and the softmax skips the max subtraction
  (exact at the graded unit gains, non-finite once q/k norm gains reach 6). Packed weights go stale under
  in-place edits, as for Carbon and Fable 5.2.

## 2026-11-09 — `04_grid_mingru_sps` retired; the deck is 01-03

The problem confused readers. Its prompt says final positions and rewards must be exact, but the grader
excuses position flips at near-ties under 1e-5 logit gap, so a plain-fp16 policy passes. The RTX board
then ranked the plain-fp16 Opus 5 kernel first at 1.961 while the careful exact designs (Opus 5.5 0.7938,
Sonnet 5.5 0.9442, Fable 5.2 1.0334) sat at 0.79-1.03: the score rewarded the loophole, not the CUDA.

Removed: the deck dirs `problems-{rtxpro6000,h100,h100sxm}/04_grid_mingru_sps`, 04 from
`scripts/build_v2_leaderboard.py` PROBLEMS, and 04 from `scripts/tetra_release.sh` (release sweeps are
now mega 02 plus cuda 01-03). The 25 published 04 run ids moved from `results/published_runs.json`
`run_ids` to `excluded` ("problem 04 retired 2026-11-09"). `scripts/build_model_index.py` skips 04
annotations (and the one v0-numbered `03_grid_mingru_sps` grok-4.5 audit from 2026-07-15) so model
pages and audit counts drop it too. Kept: every archive under `outputs/runs*/` and
every 04 annotation YAML (history). The 04 entry in `src/eval/numeric_stress.py` stays as a dead key,
because `src/` is in every problem's graded-surface digest and editing it would stale every published
01-03 cell (publish gate E) and force a full GPU regrade. Removed stays removed.

## 2026-09-29 — Claude Sonnet 5.5 [max] on tetra: 4/4 correct, three #1 cells

Cells ran 2026-09-28 12:22 MDT onward, all five (cuda 01-04 plus mega 02) on tetra GPU 0 via
`scripts/tetra_release.sh`, billed to a separate subscription account through
`SWEEP_CLAUDE_OAUTH_TOKEN`. Every turn was served as `claude-sonnet-5-5` (no subagents, refusals or
fallback). Sequential isolated regrade on GPU 0, then audit probes on the quiet GPU:

- 01 Fused MoE 0.1130, clean. Overwrite of x and of expert_ids at T=1/512/4127 follows the new data.
- 02 NSA 1.3619 (0.072 ms geomean), interesting. Exact per-token top-8; the key-stationary path stores
  per-block partials as int8 with a per-row scale. Unseen seeds 123/456 pass at all six shapes; mean
  error 4-5x the bf16 rounding floor on that path, at the floor on the fused path. 11.3 h session, the
  longest of the sweep, still setting new bests at 10.1 h.
- 03 MegaQwen 0.0800, clean. Full-context split-KV attention in one cooperative kernel. Its packed-weight
  cache keys on (data_ptr, _version), so a weight change through `.data` goes unseen (the benchmark never
  does that).
- 04 Grid + MinGRU 0.9442, interesting. fp16 tensor-core policy with a 3-term hi/lo split on an fp32
  problem, yet final positions match the reference exactly at every graded size and seed (0 of 94,208
  envs, including seed 42's 7e-9 logit gaps). `solution.py` warms the GPU at import for 0.3-3 s; an
  interleaved A/B measured it as no gain (0.985-0.998 with it, 0.999-1.004 without).

Host skills: Claude agents on tetra see the operator's 52 personal skills; four of five cells invoked
build or kernel skills, none of which carry prior results. The unused `traces` and `kb-article` skills
point at the published HF traces, a live leak path for every Claude cell here. The kbgrade agent
container (`benchmarks/grader/SPEC.md`) removes it; until then, audits check Skill calls.

Memory: tetra has 62 GB. The operator's concurrent training jobs plus five agents' compiles exhausted it
twice (14:40-15:26, 16:33); the OOM killer took operator JVMs and a trainer, never an agent session.

Traces: HF `kernelbench-cuda-traces` commit 597485517bcc (4 files), converted on tetra with 11-16
private tool calls removed each, zero redaction findings, SHA256 round-trip verified.

## 2026-09-24 — 04 positions at graded sizes: 5 live kernels diverge (RTX 3090 proxy)

Probe, not a regrade: tetra's four GPUs were held by a GLM-5.3 vLLM server, so
this ran on anvil's RTX 3090 with each kernel's `sm_120` flags patched to
`sm_86` in a throwaway copy. Scripts and logs:
`anvil:~/dev/sites/kernelbench.com/benchmarks/cuda/outputs/probes/`.
`check.py` only runs `run(128, 8)`; this ran every live 04 kernel at the four
graded sizes, seeds 42/123/456, against `reference.run`.

- fp64 reference equals the fp32 reference on every final position at all 12
  (size, seed) pairs, so exact positions are well defined.
- Exact everywhere: Grok 4.5, Kimi K3 256k and 1M, Opus 4.8, Fable 5,
  GPT-6 Astra Pro, DeepSeek V4.1 Flash, Opus 5.5.
- Diverge, seed 42 only: Opus 5 (1/2/7 envs at 4096/16384/65536, step 1,
  reference top-2 logit gap 1.4e-7), Qwen 3.8 Max (same envs, gap 2.3e-7),
  Grok 4.7 (3 envs at 65536, gap 7e-7), Gemini 3.8 Flash (2/7/30, gaps up
  to 3.7e-4), DeepSeek V4 Flash 0731 (3/11/1, gaps up to 2.7e-3, 5 envs whose
  first divergence moves between horizons).
- Fable 5.1 uses thread-block clusters and does not build for sm_86; Grok
  4.7's `kernels.cu` exists only in the archive's `scratch/`, not next to
  `solution.py`.
- A TF32 mutant of the reference passes the numeric part of `check.py` on all
  three seeds and flips the same seed-42 envs as Opus 5 at graded sizes.
- Positions carry signal only on seed 42: a mutant that skips GRU layer 3
  moves 49% of final positions on seed 42 and none on 123/456 at any size.
  79-94% of reference agents end on the board edge (26-27 distinct final
  positions of 128), pinned by `env_step`'s clamp. `last_logits` is what
  catches that mutant on 123/456.
- A margin rule scaled to the 1e-3 logit tolerance would be toothless: the
  skip-a-layer mutant diverges only at gaps up to 1.7e-3, and ~79% of
  env-steps have gaps under 2.7e-3. A gap cutoff near 1e-5 excuses the 1e-7
  flips and still flags Gemini, DeepSeek V4 Flash, bf16 and skip-a-layer.

Follow-up, branch `bench/04-margin-rule` (not on master): the RTX 04 `check.py` now runs
every graded shape as well as (128, 8), and excuses a final-position mismatch only when
the env's first divergence is an action the reference took by a top-2 logit gap under
`position_tie_margin: 1.0e-5` (problem.yaml); excused envs drop out of the reward and
last-logit comparisons. RTX 3090 proxy: 11 live kernels PASS (Opus 5, Qwen 3.8 Max and
Grok 4.7 included), DeepSeek V4 Flash 0731 FAILS (gap 1.4e-4), Gemini 3.8 Flash FAILS the
pre-existing env_step check on the 3090 (judge on the board GPU), Fable 5.1 does not
build for sm_86. Mutants: TF32 passes, bf16 policy and skip-a-layer fail. H100 decks
unchanged. Merging changes the graded surface: every RTX 04 cell needs a check-only
restamp on RTX PRO 6000 before gate E lets the cuda board publish.

NSA: two kernels that skip block selection (dense causal; first 8 blocks) both
fail the current 02 check at shape 0 nominal and small_qkv. The 09-17 fix holds
for that class.

## 2026-09-22 — Opus 5.5 CUDA sweep, audit, and resume failures

Four Opus 5.5 `[xhigh]` cells on tetra were correct in sequential isolated
regrades on RTX PRO 6000: 01 Fused MoE 0.1036, 02 Native Sparse Attention
1.0957 (0.090 ms geomean across six shapes), 03 MegaQwen Decode 0.0739,
04 Grid + MinGRU 0.7938. The four annotation YAMLs are the source for
verdicts and details. 01 is clean; 02-04 are interesting because the operator
resumed the same Claude sessions twice after GPU drops. Two copies overlapped
in each workspace. 03's copies explicitly exchanged profiling findings;
04's second copy touched scratch files but reported leaving the final
solution untouched. Do not call the 02-04 development traces one agent's
hill climb. Their final kernels are genuine and their isolated regrades are
usable, with the operator overlap disclosed.

The `template_mutated` flags on 02-04 were harness false positives. Resume
mode copied `repo/src` into an existing `repo/src` and likewise copied into
an existing `trusted_src`, creating nested `src/src`. The mutation logs name
only those extra directories. The seven graded problem files, the top-level
trusted source, and the archived final solutions matched their canonical
copies byte for byte. The harness now preserves the existing workspace src
and original trusted snapshot on resume, avoiding the nesting while retaining
evidence of any pre-interruption edits. The original `result.json` flags remain
as raw history; the annotations explain and override them for publication.

Same-buffer overwrite probes on a quiet tetra GPU 0 passed the checker gate
for all four final kernels: cos(out1,out2) was -0.0141, -0.4529, 0.4422 and
-0.9821 for 01-04; cos(reference,solution) was at least 0.99998. In-place
weight changes also invalidated 03's packed weights and 04's prepared weights.
The first probe spent about 89 seconds compiling its saved CUDA extension;
the other three completed in about 8, 10 and 3 seconds with their cached
builds. Future audits should reuse the run's extension cache.

The check-only NSA restamp under the revised RTX check finished 15 PASS,
4 FAIL across 19 archives. All four FAIL cells were already incorrect before
this pass, so no previously passing score was newly withdrawn. Only their
46,371 bytes of updated `result.json` files were pulled from 11.18 GB of
remote archives. The four new Opus archives were also pulled thin; their
combined full size on tetra was 4.54 GB. Native Claude sessions and stream
transcripts were retained for the audit and HF export. The CUDA trace
converter now selects the complete native session across resumes and removes
private home/config/environment tool calls together with their paired results
before redaction. Four exact staged exports had zero redaction findings, were
uploaded in HF dataset commit `e224d2df9f46de5af45d421bdb94fd8b63dbad95`,
then downloaded and SHA256-matched. `kb publish cuda` and its site gates passed
locally; the site now presents Native Sparse Attention in measured milliseconds
because its dense-equivalent roofline is not a useful headline.

Live release: `654b3cc` on `origin/master`; GitHub's Vercel status reported
success. The live `/cuda` board shows all four audited Opus cells, all four
served kernel files SHA256-match `public/runs/`, and all four HF trace blob
pages return 200. The three-part CUDA post draft and inspected dark-mode
images are in `media/posts/audited/opus55-cuda/` (`00_post.txt`, `01.png` to
`03.png`). It is unsent and still has `FILL-IN` for Elliot's paragraph. The
post names the overlapping resumes; it does not present 02-04 as one-agent
hill climbs.
At 2026-09-22 22:07 MDT, the same three-post thread was saved in Chrome as an
unsent draft for `@elliotarledge`. The X Drafts list showed the opener with
"2 more posts" and three uploaded images (results, latency, design diagram).
`FILL-IN` remains in post 1 for Elliot's paragraph. No CUDA post was sent.

## 2026-09-22 — 02_deepseek_nsa: large_qkv dropped from the RTX PRO 6000 check

The widened check (#11) withdrew four independent kernels on 02 (Grok 4.5, Grok 4.6,
Fable 5.1 at 1.0627, GPT-6 Astra) with byte-identical stats (126 bad, max 28.75) at
(1,8,8191,128) under `large_qkv`. An oracle written from PROMPT.txt, not from
reference.py, matched the reference at (1,1,2048,64) and (1,1,8191,128) and agreed
with the four kernels, not the reference, at the failing config. Every bad element
is one row, (b=0,h=6,t=4835): key blocks 49 and 35 tie for the 8th top-k slot with
fp64 importances 9.014908865 vs 9.014908415, a gap of 0.47 fp32 ULP. The reference's
per-token mat-vec picks 49, any batched GEMM over a query tile picks 35, and Grok
4.7's block-sum route makes them bit-identical and resolves on its tie-break. x8 on
q and k makes logits 64x and the softmax a hardmax, so the swapped block flips the
row from +17.25 to -11.50. At nominal scale the same row diverges by 0.082 inside
tolerance. The spec fixes no accumulation order, so neither side is wrong; the case
was a selection-tie detector, not a numerics test. check.py now skips `large_qkv`
(problems-rtxpro6000 only; the H100 decks are untouched). This changes the graded
surface for 02 only, so every published RTX 02 cell needs a check-only restamp.
The DeepSeek V4 Flash 02 withdrawal (96,577 bad elements past row 2007) is a real
failure and is not affected.

## 2026-09-17 — check 01/02 at the shapes they are graded on (deck correction, cells stale)

**This changes `check.py` on a published deck, so every cell below is stale
until re-checked. The numbers on the board for 01 and 02 were produced under
the old, narrower check and must be re-validated or withdrawn.**

A review of the grading surface found that neither problem verified the shapes
it scores:

- `01_glm52_fused_moe` capped the check at `T = min(T, 256)` while `benchmark.py`
  times T = 4096, 4127, 1, 8192, 512, 1000. The tile-config branches the kernel
  uses at prefill sizes never ran under `check.py`.
- `02_deepseek_nsa` hardcoded two check shapes at S = 256/384, both D = 64,
  while the graded sweep runs S to 8192 and includes `{S: 8191, D: 128}`. D = 128
  was never verified anywhere, and at those short lengths every block is
  selected, so the block-selection path that *is* the operation at long context
  was never exercised.
- No `benchmark.py` in this deck (or hard's) calls `allclose` or
  `check_correctness` — verified: 0 hits across 10 files. So a graded number
  proved "correct at T<=256" and "fast at T=8192" as two separate facts, with
  nothing tying them together. A shape-conditional fast path is exactly the
  branch `reward_hack_lint.py`'s regexes cannot see.

Both checks now iterate the graded `shapes.SHAPES`:
- 01: all 6 shapes at their own T, 3 seeds each (42/123/456), as before.
- 02: all 6 shapes, one seed (42). One shape at one seed, not six at two, is
  the same trade 03's long-ctx spot check made; the reason here is cost, not
  correctness. The reference loops `B*H*S` tokens in Python with a per-token
  block loop and sort — 310,632 iterations across the six shapes, ~250s at a
  pessimistic 800us/iteration against the 1800s `KBH_CHECK_TIMEOUT_SECONDS`
  budget. If a box measures it over budget, drop to the three longest shapes
  rather than reinstating a short-shape cap.

Affordability for 01 was measured, not assumed: dense-equivalent work at
T = 8192 is ~3.7 TFLOP and the reference is fp32, so the cap was a leftover
from an earlier, slower reference rather than a live constraint.

**Backfill required.** 15 published 01 cells and 14 published 02 cells were
scored under the old check. Re-run each archived `solution.py` through the
current `check.py` in check-only mode, on an idle RTX PRO 6000 (the hardware
the numbers were produced on):

```
cd benchmarks/cuda
KBH_REGRADE_CHECK_ONLY=1 KBH_REGRADE_DECK=problems-rtxpro6000 \
    scripts/regrade_sequential.sh $(python3 ../../scripts/check_publish_gates.py --list-stale cuda)
```

`regrade_sequential.sh` restores `reference.py sota.py shapes.py problem.yaml
check.py benchmark.py PROMPT.txt` plus `src/` and the locked project from the
canonical deck, so the corrected check applies and any agent edit to the graded
surface is reverted and reported. `KBH_REGRADE_CHECK_ONLY` matters: without it
the regrade also replays `benchmark.py` and overwrites `peak_fraction` from the
fresh timing, so a "re-check" on any other box or thermal state would silently
re-rank the board. In check-only mode the published timing and `benchmark.log`
are kept, `graded_surface_sha` is stamped, and a cell that now FAILs has its
`peak_fraction` voided: it is withdrawn, not re-run, and gets an annotation
explaining which shape broke it. `--list-stale` enumerates exactly the cells
publish gate E refuses, so the same command also stamps the cells whose check
did not change.

## 2026-07-16 — Pre-debut deck repairs: torch 2.13 init fixes, numeric stress, 03 long-ctx

Caught by the Grok 4.5 cell audits before the first publish (legal because the
deck is still unpublished):

1. **torch 2.13 rejects CPU generators on CUDA tensors.** Two template sites
   hard-crashed independent of any solution: `04 reference.reset_parameters`
   (check.py calls it after `.to(device)`) and `03 check.py _reinit`. Grok's 04
   run tripped `template_mutated` by fixing the first with a bit-exact
   CPU-draw-then-`copy_` hunk (audit verified `torch.equal` against the
   original stream); Grok's 03 solution monkey-patched
   `torch.nn.init.normal_` at import to survive the second (also bitwise
   faithful). Both templates now carry the CPU-draw-then-copy pattern at the
   source level — identical generator streams — so future solutions need
   neither workaround. The 04 run was invalidated by the guard despite clean
   intent (verdict `interesting`, not reward hack); rerun launched on the
   fixed deck. Post-hoc, its solution passes the full strict battery and
   benchmarks geomean peak_fraction 0.3728.
2. **Numeric stress wired into the deck.** `_CASES` now covers 01 (hidden
   ×1e-2 / ×8), 02 (qkv ×1e-2 / ×8), 04 (obs+state ×1e-2 / ×8, policy_forward
   section only — env_step/run synthesize inputs internally and are
   position-exact). 03 gets NO scale cases by construction: `run()`
   synthesizes inputs from the seed (nothing external to scale) and RMSNorm
   makes the stack weight-scale invariant, so input/state scaling cannot bite.
3. **Two tolerances calibrated against measured noise, not guesses.** 04
   `small_obs_state`: hard's `_TINY_FP32` atol 1e-7 is below the fp32
   machine-epsilon floor for this pipeline — the audited kernel measures
   4.6e-7 reorder noise (44/196608 bad at atol 1e-7), so the case uses
   atol 1e-6 (≈3x margin, still ~100x tighter than a wrong-gate signature).
   01 `large_hidden`: at 8x hidden, output magnitudes reach ~370 (mean ~57)
   and even an idealized bf16 pipeline (fp32 matmuls, bf16 silu*up
   intermediate) measures max_abs=2.0. The final case is atol 1.5 / rtol 5e-2:
   the smallest point passing the audited kernel at every check variant
   (min margin +0.49; floor sim +1.03) while wrong/cached kernels diverge by
   O(mean magnitude) ~57. Floor/scan harnesses: `backfill/` scratch
   (ephemeral). **Gotcha worth remembering:** the pass predicate is
   `torch.allclose(ref, sol)` whose rtol multiplies |sol| (asymmetric), but
   check.py's diagnostic `bad=` count uses |ref| — a scan built on the
   diagnostic semantics undercounts and picks atol 1.0, which then fails the
   same kernel at seed 123 by −0.006. Calibrate against the actual predicate,
   across ALL seeds the check runs (42/123/456), not just seed 42: the failing
   element moved between seeds.
4. **03 hardening is a long-ctx spot check.** check.py used to top out at
   ctx 512, so numerics at 2k+ were verified by no one (audit finding on the
   passing Grok cell). Measured on that kernel: max_abs is a CONSTANT 0.0625
   at both ctx 2048 and ctx 8192 (a per-layer bf16 rounding constant, not
   accumulation), reference wall time ~13s at 8k. check.py now runs one
   (seed 42, ctx 8192, decode 16) comparison at the same 0.08 tolerance.
5. **04 peak_sps stays 150M**, derivation now documented in problem.yaml: the
   first audited fused CUDA baseline sustains ~50-61M sps (pf 0.33-0.41), so
   150M is an aspirational roofline-proxy ceiling, not a best-known-kernel
   number. Do not fit the ceiling to the kernel.

All published cells are re-validated against this final surface before the
board debuts (backfill checks run the archived solutions through the current
check.py + numeric_stress.py; 01/02/03/04 PASS, with 04's fresh rerun pending).

## 2026-07-15 — Latency-anchored relative scoring (standing metric decision)

Peak fraction is the wrong headline for cells whose roofline ceiling is
structurally unreadable. Two live examples: hard's `05_topk_bitonic`, where the
~0.02 ceiling is launch-overhead-bound for EVERY model, and cuda's
`02_deepseek_nsa`, where the dense-equivalent FLOPs baseline can never be
attained by a correct *sparse* kernel (Grok 4.5's first clean pass scored
0.0177 and "0.0177" tells a reader nothing). For those, grade and display on
**milliseconds**, not peak fraction.

The design (user decision 2026-07-15; implement per this when wiring publish):

1. **ms per shape stays the ground truth.** result.json already records
   shape-by-shape times; the leaderboard builder must carry per-shape ms for
   the solution variant so every displayed number is reproducible from the
   archive.
2. **Persisted headline = geomean speedup vs a FROZEN anchor.** Anchor is the
   deck's eager torch reference (`reference.Model`), timed per shape at deck
   publication and committed alongside the problem (eager_ms per shape).
   Score per shape = eager_ms / solution_ms; problem score = geomean over the
   shape sweep. The anchor is frozen at publication with the deck, so a
   published cell's score never changes afterward — same frozen-board rule as
   prompts.
3. **Best..worst linear span is a PRESENTATION layer only.** On the site, also
   show the board-relative position: with `t` a cell's geomean ms,
   scaled = (worst - t) / (worst - best) across published models
   (worst -> 0, best -> 1; degenerate span 0 -> all 1). Computed at site build
   from the current board, NEVER persisted into leaderboard.json. When a new
   best lands, the span shift is a site render change, not a re-grade of
   historical cells — the same rule that keeps labs trusting the board.
4. **peak_fraction is demoted to context**, shown as a secondary column
   ("roofline attainability"), never the sort key on structurally unreachable
   ceilings. It stays the headline where the ceiling is real (MoE, GEMM).
5. **Cross-bench reach: presentation only.** hard's topk cell gets the same
   ms + span DISPLAY treatment; hard and mega graders, prompts, and
   leaderboard.json schemas stay frozen. No hard/mega cell is re-graded.

## 2026-07-15 — Why a third single-GPU bench (and why not touch Hard/Mega)

Hard and Mega are live boards that labs already care about. Changing a prompt,
tightening a forbidden list, or swapping "CUDA or Triton" for "CUDA only"
would re-grade historical cells and look like moving the goalposts. So CUDA
is a **new, isolated deck** with the same harness DNA, not a Hard fork of the
published problems.

### Thesis

We want an objective read on **CUDA kernel writing**, including:

1. Problems that map cleanly to CUDA but are *annoying* in raw CUDA (reductions,
   online softmax, shared-memory layouts) while being *easy* in Triton.
2. Optional megakernel / sim paths (grid env + multi-layer MinGRU policy) where
   fusion is allowed, not mandatory — grade sustained steps/sec and see what
   strategy the model picks.
3. Explicit sidecars for **instruction following**: did the model obey the CUDA
   mandate, or did it sneak back to `@triton.jit` / a DSL / a pure torch op?

Hard already *allows* Triton (and many winning cells are Triton). That is fine
for "best kernel on the metal." It is the wrong axis for "can you write CUDA."

### v0 deck (retired easy ops)

- `01_rmsnorm_residual` / `02_online_softmax` — Grok 4.5 smoked them clean
  (real CUDA, language gate pass). User decision 2026-07-16: **drop them**.
  Too tutorial / Triton-blog-post shaped for a "super hard CUDA" board.
  Annotations kept; problem dirs removed.

### v1 deck (perfect stack, 2026-07-16)

- `01_glm52_fused_moe` — GLM-5.2 MoE (E=256, top_k=8, 1 shared) fused gate|up pack; no Mixtral.
- `02_deepseek_nsa` — NSA-inspired block top-n + sliding window + sparse attn.
- `03_megaqwen_decode` — Qwen3-0.6B geometry (4-layer slice); improve Infatoshi/MegaQwen.
- `04_grid_mingru_sps` — RL sim SPS; fusion optional.

Deck is frozen at four. Spec-decode tree attention was floated as a fifth and
rejected (user decision 2026-07-15); do not re-pitch it.

Smoke 2026-07-16: all four references run on PRO 6000; MoE `check.py` PASS with
minimal CUDA silu_mul solution (language gate `cuda_raw`); Triton gate fails as
designed.

Shape discipline (Hard FP8 technique): MoE/NSA include misaligned T/S (4127,
8191, …) and serving tails (T=1); score = geomean. MegaQwen: prefill untimed,
decode-only timed at ctx ∈ {2k,8k,32k,128k}; pure numeric last_hidden (no tokens).

### Language gate

`src/eval/cuda_language.py` is the hard fail. Numeric PASS with Triton still
fails the problem. Report goes to `cuda_language.json` for future leaderboard
columns (`framework`, `triton_cheat`).

### Harness

Rsynced from Hard (scripts + src + kbh CLI). Package renamed
`kernelbench-cuda`. Do not edit Hard/Mega prompts from this workstream.

## 2026-09-03 — gemini-3.8-flash-high on RTX PRO 6000 (published) and H100 SXM (held)

agy (Antigravity CLI) cells, all isolated-regraded and overwrite-probed on the
box. RTX PRO 6000 board: 02_deepseek_nsa 0.0958 clean, 03_megaqwen_decode 0.0426
clean, 04_grid_mingru_sps 0.3637 interesting (exact-fp32 path only for the
num_envs <= 256 shape check.py runs, TF32 on graded shapes, but the strict
oracle matches positions/rewards/logits at every graded shape and seed, same
bar as the qwen3.8-max 0.2848 cell). 01_glm52_fused_moe 0.0918 is
`verdict: contamination` and unpublished: the agent grepped every prior
peak_fraction in results/annotations for the problem and opened the 0.2787
top-scorer's annotation before coding. Root cause: scripts/lib/run_harness.sh
has no bwrap sandbox (mega's run_hard.sh does), and `kb contamination` only
looks for outputs/runs references, so annotation reads pass it. Mitigation on
the boxes was moving results/ and DEVLOG.md out of the tree and pulled run dirs
out of outputs/runs before the later cells; repo fix pending (port the KBH_SBX
block, extend kb contamination to results/annotations reads). Publish commit
deeea1f (manifest published_runs.json += the three clean run ids); traces on
Infatoshi/kernelbench-cuda-traces. H100 SXM5 cells (deck problems-h100sxm,
archives in outputs/runs-h100sxm, gitignored): 01 0.0805 contamination (same
annotation reads), 02 0.0417, 03 0.049, 04 0.3645; there is no H100 SXM cuda
board, so these stay unpublished until one is decided. Bench-level fixes noted:
check.py should exercise a long-ctx shape for 03 (both gemini cells fill the KV
cache with noise above 8192, like the deepseek-v4-pro cell) and a graded-shape
fidelity check for 04.
