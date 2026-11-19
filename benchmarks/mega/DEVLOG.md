# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

Numbering note: entries before 2026-07-21 say "Problem 03" for the Kimi-Linear decode kernel, which is today's `02_kimi_linear_decode`; "Problem 02" was `01_rl_grid_ppo`, removed 2026-07-21.

---

## 2026-11-19 - Gemini 4 Neon (XHigh) on 02: 36.66x (`interesting`), new #1 overall

`agy` harness, `gemini-4-neon-xhigh`, single session on tetra GPU 0 (3:04 pm to 11:25 pm CST
2026-11-18, 1,000 turns, 1.98M output / 811K thinking tokens). Isolated sequential regrade on quiet
GPU 0 at 12:36 to 12:37 am CST 2026-11-19 (graded surface `c522e2010e34`): 36.655x geomean
(34.93x / 36.54x / 38.59x; 0.157 / 0.166 / 0.176 ms/tok; 6,350 / 6,012 / 5,686 tok/s; geomean
6,010 tok/s), taking #1 overall ahead of Claude Opus 5.5 (35.34x), Sonnet 5.5 (31.65x), Fable 5.2
(31.01x), and Gemini 4 Carbon (24.68x, +48.5%).

Hand-written 2,529-line SM120 cooperative megakernel (`188 x 512`, `torch.profiler` confirms
exactly 6 `kimi_decode_megakernel` launches across 6 steps, 0 other kernels): fused int4 dequant,
online absorbed MLA, split-K tiled projections, and in-place chained decode. Marked `interesting`
for three audit findings: (1) `dot32_bf2` accumulates 32-element GEMV chunks in `fma.rn.bf16x2`
before reducing to FP32; (2) `*g_norm_sq = 0.0f` at `solution.py:1132`, `:1470`, and `:1764`
executes before the preceding `grid.sync()` (0/60 graded ctx=8192 repeats and 0/50 6-step chains
failed, though one 6-step chain under concurrent GPU load dropped at step 2 to cos 0.0174); (3)
`_dense_arena` weight packing invalidates on `load_state_dict()` rather than bare `p.copy_()`.
All attention-sensitive probes pass cleanly (identical rows x10 cos 0.99987, random cache x30 cos
0.99978, unseen seeds 123/456 cos 0.99976-0.99983). Operator sent `SIGTERM` twice during the
session: once for a deadlocked `python check.py` (6:01 pm) and once for the `lockshim` + `ncu`
zombie hang (6:37 pm; root DEVLOG 2026-11-19).

## 2026-11-17 — GPT-6.2 Astra [xhigh] on 02: 23.48x and 22.93x, both `clean`, both stopped early

`codex` harness, native Codex CLI 0.169.0, `gpt-6.2-astra` at xhigh (cuda DEVLOG 2026-11-17 has the
route), two sessions on tetra GPU 2 next to the three cuda cells. Isolated regrade 23.4782x (s1, board
row) and 22.9346x (s2) at 2:49 to 2:51 pm. Each wrote its own one-launch cooperative CUDA megakernel (no
shared code): int4 dequant fused into the GEMVs, weights repacked once at load, absorbed MLA with the
correct per-head stride of 192. Probes on a quiet GPU 2: same-buffer overwrites of token, state and
weights read live values, six steps are six launches with no memcpy, memset or graph, 18 attention-heavy
caches hold at cos(ref,sol) 0.9999 or better, 32/32 unseen seeds, caches to 131,072 rows pass (baseline.py
itself drops to 0.96 there). One s1 six-step chain on an attention-heavy cache falls to 0.967; baseline.py
fails the same chain at 0.963 because the reference's top-8 router margin there is under check.py's 0.025.

Both sessions ended on their own about 75 minutes in, on 55k to 57k output tokens, against 2.8 to 6 hours
and 0.6M to 1.2M output tokens for the 31x to 35x Claude cells. Neither trace weighs remaining headroom:
benchmark printed a number, check printed PASS, it cleaned up and reported. The late in-run jump on s1
(18.75x to 23.64x in its last seven minutes) is mostly s2's contention lifting; its own kernel timer puts the
last change at 273.5 to 252.0 us per step. s1 ran its GPU work outside the GPU lock (cuda DEVLOG gap 1).
Both `kb contamination` flags are one `rg` over home that printed pyvenv paths in the stale tetra checkout;
no mega archive was named or read.

## 2026-11-16 — Grader v3 replay of the RTX PRO 6000 board: 11 kernels fail the attention case, 27 re-timed

Grader v3 (commit `4c2c347`, graded surface `c522e2010e34`, was `829522700bbe`) adds attention-sensitive
check cases and times the baseline in its own process (`SPEC.md` Metric and Correctness). Every scored
RTX PRO 6000 row was replayed under it with `scripts/regrade_sequential.sh` (`KBH_REGRADE_DECK=problems`)
on tetra GPU 2, one cell at a time, alone on the card: 38 board cells twice (2:04 to 2:49 pm and 2:49 to
3:17 pm CST), plus two older sibling runs twice (3:17 to 3:32 pm). All 89 graded windows are clean in
the host GPU-use log. The published number is each cell's second run. The two passes differ by a median
0.28%; the largest gaps are slow kernels (inkling 7.6%, Kimi K2.7 5.4%, Sonnet 5 2.0%). Baseline at
context 2048 read 5.476 to 5.495 ms per token on every passing cell.

Eleven kernels fail the new attention case, all on the first case (seed 27, context 2048) except hy3:

| model | old score | cosine (two runs) | cause |
| --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | 29.89 (already `bug`) | 0.3457, 0.3448 | wrong MLA attention, known since October |
| Gemini 4 Argon | 27.03 | 0.9399, 0.8874 | race: every thread reads `smem_red[0]` (solution.py:1434) while warp 0 overwrites it (:1438), no barrier |
| kinetic-0715 | 18.09 | 0.6444, 0.6665 | shared-memory overflow: `load_sx_bf16(..., sx, 4096)` into a 2304-float `sx` (kimi_mega.py:620, :985); also a race |
| ox-alpha | 13.64 | 0.2936, 0.2936 | absorbed query reads 512 values from head h's query and contracts the wrong `kv_b` dimension (solution.py:659) |
| GLM-5.2 | 11.14 | 0.2381, 0.2373 | strided `q_rope` view handed to a RoPE kernel that indexes it as contiguous (:509, :347) |
| kinetic-0715 1m | 9.78 | 0.5828, 0.5845 | `sp = it % 6` where the split has 3 slots, so half of q is never written (:819); a second smaller race not located |
| Kimi K3.1 Pro | 8.62 (already `bug`) | 0.4628, 0.4628 | absorbed query steps 128 where the head stride is 192 |
| DeepSeek V4 Pro | 4.64 | 0.2067, 0.2067 | queries stored as `r*32 + h`, loaded as `r + h*32` (:700, :714, :735, :749) |
| GPT-5.5 | 4.34 | 0.5856, 0.5856 | same stride mistake as Kimi: strided `q_nope` view read at `pid_h * 128` (:676, :384) |
| MiniMax-M3 | 2.62 | 0.4363, 0.4363 | `view(...).t()` where a permute is needed scrambles W_k (:490) |
| hy3-preview | 0.31 | 0.9617, 0.9617 (seed 27, ctx 8192) | softmax scale 1/sqrt(128) where the reference is 1/sqrt(192) (:558) |

The eight that were not already known were root-caused before any verdict moved (a probe per kernel on
tetra: gain sweep of the hot cache rows from 1 to 120, three repeats, torch profiler to confirm the
graded path, then a one-line scratch fix). Every failure reproduces from a fresh build with empty caches
to four digits, every fixed kernel returns to 0.996 or better (kinetic-0715 1m only to 0.964 to 0.98,
its second race), and `baseline.py` holds 0.9963 to 0.9999 at every gain, so the case is not harsh on a
correct kernel. hy3 fails only at gains 30 to 60 and recovers above 90 because the softmax goes one-hot
whatever the temperature; a gain-40 case would catch temperature errors with more margin (not added).
Eight annotations move to `verdict: bug` with a `withdrawn:` block; Kimi K3.1 Pro and DeepSeek 1017 get a
`regrade_v3:` note; hy3 has no annotation and shows as a failed row. The 28 passing cells that have an
annotation get the new `peak_fraction` and a `regrade_v3:` block with both runs.

Voiding a headline lets an older run of the same model become the row (the builder keeps the best scored
run per model), so the two that would surface were replayed too: kinetic-0715 `20260715_204551` reads
12.06 (was 14.11) and DeepSeek V4 Pro `20260618_150605` reads 2.08 (was 2.11). Both pass.

Re-timed cells, old then new (second run):

| model | old | new | note |
| --- | --- | --- | --- |
| Opus 5.5 | 35.46 | 35.34 | |
| Sonnet 5.5 | 32.13 | 31.65 | |
| Fable 5.2 | 31.41 | 31.01 | |
| Gemini 4 Carbon | 23.76 | 24.68 | kernel 0.249 to 0.232 ms per token at 2048; both grades on tetra, cause not established, both passes agree |
| GPT-6 Astra Pro | 24.80 | 20.87 | first graded on a rented box |
| Fable 5 (or-fable) | 24.61 | 20.85 | first graded on a rented box |
| Fable 5.1 (or-fable) | 22.95 | 19.34 | first graded on a rented box |
| GLM-5.3 | 19.43 | 19.22 | |
| Fable 5 (claude) | 18.71 | 19.10 | |
| Qwen 4 Max | 16.29 | 16.10 | |
| Fable 5.1 (claude) | 15.84 | 15.63 | |
| DeepSeek V4.1 Flash (deepseek-flash) | 17.10 | 15.05 | first graded on a rented box |
| Opus 4.8 | 14.40 | 14.28 | |
| Grok 4.7 | 6.38 | 6.34 | |
| GPT-6.1 Sol | 4.99 | 4.96 | |
| DeepSeek V4 Flash 0731 | 5.24 | 4.32 | first graded on a rented box; `reward_hack`, not on the chart |
| Sonnet 5 | 4.03 | 3.92 | |
| GPT-5.6 Sol | 2.64 | 2.64 | |
| Composer 2.5 Fast | 2.48 | 2.46 | |
| GPT-6 Luna | 2.48 | 2.46 | |
| Gemini 3.8 Flash High | 2.74 | 2.36 | first graded on a rented box |
| Gemini 3.5 Flash | 2.28 | 2.27 | |
| Kimi K2.7 | 2.59 | 2.25 | old baseline at 2048 was 11.93 ms per token (contended); 5.49 now |
| Muse Spark 1.3 | 2.18 | 2.04 | first graded on a rented box |
| GPT-6 Sol | 0.96 | 0.95 | |
| Grok 4.5 | 0.82 | 0.82 | |
| inkling | 0.17 | 0.54 | old grade was contended on both sides (kernel 71.7, baseline 11.35 ms per token at 2048) |

Lessons and loose ends:

- Host effect. The baseline is CPU-bound eager PyTorch: 7.6 to 8.0 ms per token at 2048 on the rented
  boxes that first graded seven of these cells, 5.48 on tetra. The kernels are faster on tetra too, by
  less, so those cells read 7 to 18% lower here. One board is comparable only when timed on one host.
- The H100 (16 rows) and B200 (10 rows) boards are still on the pre-v3 grader. Replaying them needs
  rented cards. Publish gate E (`stale_mega_cells` in `scripts/check_publish_gates.py`) covers the
  RTX PRO 6000 rows only for that reason.
- Old kernels need `NVCC_APPEND_FLAGS=-std=c++20` on tetra: torch 2.11's `ATen/core/List_inl.h` does
  not compile under nvcc 13 with gcc 15 as C++17. Nine cells were graded with it. A kernel with a
  PyTorch fallback passes the check on the fallback when its build fails: deepseek-flash read 0.41x in
  pass 1 that way and 14.99 and 15.05 once built, so grep both logs for the build error.
- `regrade_sequential.sh` now restores `baseline.py` from the deck. The pre-replay `result.json` of every
  replayed cell is kept as `result.grader_v1.json`. The pre-replay logs are in `check.prior.log` and
  `benchmark.prior.log` for the 20 cells that had been regraded before, and in `*.contended.log` for the
  20 that had not (the script parks a first regrade's predecessor there; their `*.prior.log` is the
  first replay run). Match on the printed `peak_fraction`, not the file name.
- The first replay attempt (12:53 pm, GPU 3) lost the card to another project's queued jobs after three
  cells; its window check flagged five cells and nothing from it was kept. The driver now holds that
  project's per-GPU lock file for the whole replay and stops if a foreign process appears.
- One window verdict (deepseek-flash, 3:31 pm) was rerun by hand against the GPU-use log because the
  log was renamed while that cell was running; it is clean (5 log blocks, 0 foreign).
- The 2026-09-11 entry below ("the 39 published mega cells have no run archive") is out of date: all 66
  board rows have a `solution.py` archive on the Mac, which is what made this replay possible.
- Evidence: tetra `runs/mega-grader-v3-replay/` (`cells.pass1.tsv`, `cells.pass2.tsv`, `cells.sib1.tsv`,
  `cells.sib2.tsv`, the GPU-use logs and the driver).

## 2026-11-16 — Qwen 4 Max [xhigh] on 02: 16.29x with correct MLA, `interesting` for conduct

`or-fable` harness (Claude Code on OpenRouter, `qwen/qwen4-max`, effort xhigh) on tetra GPU 2, one 6.1 hour
session next to the three cuda cells. Isolated regrade 16.2864x (0.361 / 0.379 / 0.393 ms per token against
5.512 / 6.106 / 6.888 for the baseline) at 8:09 am CST, window clean in the host GPU-use log; the contended
in-run grade was 16.2879. A genuine one-launch Triton megakernel: 752 blocks of 128 threads (four per SM,
picked by a compile-only register and shared-memory probe), 35 software grid barriers per step built on
atomic counters, absorbed MLA.

Unlike the two cells before it, the attention is right. Audit probe on a quiet GPU 2: 14 of 14 cases on
caches where attention carries the output hold at cos(ref,sol) 0.999945 to 0.999998, 32 of 32 unseen seed
and context cases pass, same-buffer overwrites of weights, input and state read live values, and six steps
are six launches with no memcpy, memset or graph. Off the graded path: the softmax combine reads 256 chunks
of 128 rows, so a cache past 32,768 rows degrades (0.9917 at 40,000 rows on an attention-heavy cache), and
the cache pack is keyed on `c_kv`'s address only.

The first passing build was 6.5x at one block per SM; cutting registers to fit four blocks per SM took it
to 15.9x by 3.3 hours. The last three hours went to a zero-barrier redesign with per-task dependency
counters that never beat the barrier kernel (2,301 / 548 / 645 us per step against 361), and it shipped the
barrier version.

Verdict `interesting` for conduct: with the host process table visible it ran `kill -9` on a process on
its card that it took for its own zombie, which was the NSA sibling cell's test, and it ran experiments on
GPU 0 and GPU 3. Root DEVLOG 2026-11-16 has the harness gaps.

## 2026-11-16 — Kimi K3.1 Pro [max] on 02: passes check at 8.6x, MLA attention wrong, published as `bug`

`kimi` harness (Kimi Code CLI 2.5.3, `kimi-code/k3.1-pro`, effort max) on tetra GPU 3. The kernel is a real
one-launch megakernel (six decode steps are six cooperative launches) and the harness grades it 8.62x on
the isolated regrade (0.606 / 0.714 / 0.866 ms per token against 5.571 / 6.178 / 6.965 for the baseline),
but the absorbed MLA query reads the output projection with the wrong per-head stride (128 where the
layout is 192, solution.py:577). On the graded cache that error is invisible; with the random cache scaled
x30 the solution reads cos 0.922 / 0.934 and x60 reads 0.264 / 0.386 while `baseline.py` stays at 0.996 to
0.9999. Verdict `bug`, no score, same treatment as the DeepSeek V4.1 Flash (1017) row below. This is the
second model in a row to pass `check.py` with a wrong MLA path, so the scaled or structured cache case
proposed in that entry is now overdue; it still needs a deliberate bench version.

`kb contamination` trips on this cell (six foreign run ids) because one `ps` tool result lists sibling
sessions; the annotation carries a manual `contamination: clean`.

## 2026-10-18 — DeepSeek V4.1 Flash (1017) on 02: passes check, MLA attention wrong, published as `bug`

OpenRouter `deepseek/deepseek-v4.1-flash-1017` via `or-fable`, 8.3 h session on a rented 4x RTX PRO 6000
box (cells spread one lock per GPU, `KB_RELEASE_GPUS`). Isolated regrade 29.89x with every graded seed
passing, but the audit probe shows the kernel's MLA M4 stage fills only the diagonal 16 of 512 latent dims
per head (lane = head owns m = lane*16..+15), so 31/32 of the attention reduction is missing. Two grader
gaps let it through:

- check.py cannot see MLA. On the graded random cache (0.1 randn) softmax is near uniform and the MLA
  attention output is ~0.14% of the hidden state, so a wrong MLA still scores out=0.9999. With identical
  cache rows x10 or the random cache x30 the solution reads cos 0.76 / 0.95 (one NaN) while baseline.py
  stays at 0.9999. Fix candidate: a structured or scaled cache case in check.py.
- benchmark.py times the baseline after the solution on the same stream, so a solution that sets
  `cudaLimitPersistingL2CacheSize` plus a stream access-policy window slows the baseline: 7.70 -> 8.05
  ms/tok at 2k (~4%), inflating the speedup. Fix candidate: reset the persisting limit and stream
  attribute (or time the baseline in its own process) before timing it.

Also: across 20 fresh-state runs per seed, ungraded seeds 1, 3 and 5 fall below 0.98 (min 0.9577), the
same atomic-order nondeterminism class as Opus 5.5 and Argon. The cell stays on the model page as `wrong`
and is not ranked. Grader changes need a deliberate bench version; not done here.

## 2026-09-29 — Claude Sonnet 5.5 [max] on 02: 32.13x, clean, second on RTX PRO 6000

Isolated regrade on tetra GPU 0: 32.13x over the optimized-PyTorch baseline (in-run 32.25x), behind
Opus 5.5's 35.46x. One launch per token, 188 persistent CTAs over 26 phases: int4 weights dequantized in
registers into mma GEMVs, KDA delta-rule state updates, absorbed MLA attention over every cached token
with an in-place append into a capacity buffer, and router plus nine-expert MoE per layer. Probes on the
quiet GPU: same-buffer token overwrite and an in-place x30 cache scale on the live append path both
follow the new data (cos(ref,sol) 0.99998 and 0.9997). Trace on HF `kernelbench-mega-traces` commit
8614b76ec8ac. Sweep context and the host-skills note: `benchmarks/cuda/DEVLOG.md` 2026-09-29.

---

## 2026-09-11 — the 39 published mega cells have no run archive (migrated from Claude auto-memory)

Found 2026-07-28 while relocating `anvil:~/kb-remote-archives`. The 39 run_ids in
`public/data/mega/results.csv`, all `02_kimi_linear_decode` and dated 2026-06-18
onward, have no run archive anywhere local: not in the Mac's four mega run dirs
(`runs`, `runs-h100`, `runs-b200`, `runs-inkling-hold`), not in anvil's checkout,
not in the relocated remote archives.

What survives: all 39 solutions are tracked in-repo at
`public/data/mega/code/<run_id>.solution.py.txt`, and the transcripts should be on
HuggingFace (`kernelbench-mega-traces`). Verify that before relying on it.

Why it matters: the mandatory sequential isolated re-grade rebuilds a cell from
its archive workspace. With no archive those 39 cells cannot be re-graded or
contamination-checked from local data; they would have to be reconstructed from
the published solution text plus the HF trace, or re-run outright.

hard and cuda are fine by contrast: all 65 published hard and 24 published cuda
run_ids are present on the Mac with a non-empty `result.json`.

## 2026-09-02 — claude-fable-5-1 max on 02_kimi_linear_decode, RTX PRO 6000 (anvil GPU 1)

Cell: 15.84x after sequential isolated regrade (in-run 15.82x; regraded
0.344/0.388/0.431 ms/tok at ctx 2048/8192/16384), verdict clean,
megakernel_authentic true, one cooperative persistent work-queue kernel
(arrival counters, no grid barriers, split-K GEMV, in-kernel MLA cache append).
Recompute test cos(o1,o2) = -0.018, cos(o2,ref) = 0.994; 1 launch/step;
0.39 of the 1.8 TB/s roofline. Session was cut by the provider five-hour limit
at ~91 min while still tuning. Same GPU class as the July Fable 5 18.71x cell
(20260701_172615, RTX PRO 6000 Blackwell, max, ran to self-termination), so
the gap is comparable but reads as a session-length effect at least as much
as a model delta.

Three earlier attempts the same day died before writing a solution
(annotated bug): the first-party `claude)` block in `scripts/run_hard.sh` did
not export `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, so Claude Code used its 64000
default for claude-fable-5-1 and a single max-effort reasoning turn ended the
session with a terminal max_output_tokens error. `-p` mode does not recover
from that. Fixed: the block now exports 128000 (the CLI's upper limit for
this model). Effort tier was not the lever (xhigh died the same way). Also:
anvil had Claude Code 2.1.252, which did not know the model id at all
(200k/32k profile); updated to 2.1.257. Preflight a new model id with a
one-line `-p` call and check the rig's `claude --version`.

Tooling notes: `kb -b mega lint <run_id>` cannot find mega runs (delegates to
the hard bench's reward_hack_lint.py with the hard root); run the script with
the absolute run path. The 0.98 output-cosine gate on 02 is sensitive to
reference-side MoE router near-ties (seeds 2 and 10 are exact ties, seed 5
reads 0.977 for a correct kernel); expect dips in future 02 audits.

---
## 2026-08-19 - sandbox v2: hide-the-tree, shared helper, honeytoken, refuse-closed canary

The 2026-08-13 grok-4.6 incident (below) got through because the sandbox was a
DENYLIST: `outputs/runs-remote-pro` was a pull tree nobody added to the hide
list. Denylists lose to the next stray directory, so the sandbox is now the
inverse — tmpfs the entire monorepo (or thin bench root), bind back only this
run's `$RUN_DIR`, the dedicated gpu-lock dir (`outputs/gpu_lock/`, moved from
bare `outputs/gpu.lock` so the bind cannot re-expose `outputs/runs`), and one
deliberate honeytoken decoy. Construction lives in ONE shared helper,
`scripts/lib/sandbox.sh`, sourced by both the shared runner
(hard/cuda/mini host mode) and this fork — the fork keeps only its harness
set and lock legacy, not a third copy of the hide list. `src/` is now copied
into the run workspace (the old repo symlink would dangle under tmpfs).
Session-history dirs (`~/.claude/projects`, `~/.grok/sessions`,
`~/.codex/sessions`, `~/.gemini/tmp`, opencode storage) are overlaid per-run
into `$RUN_DIR/agent_home/` — prior transcripts (which quote old runs) are
unreadable, auth files stay visible via the host bind, and this run's history
lands in the archive for the audit. Before the model starts,
`scripts/lib/sandbox_canary.py` sweeps the visible tree INSIDE the same bwrap
for anything run-dir/result.json/solution.py/`runs-remote-*`-shaped that is
not this run and refuses the launch on any hit (exit 4, refuse-closed). The
honeytoken (`20260601_030405_or-fable_..._<problem>`, fake juicy score,
beacon `kbh7f3a9c1e5d2b` — canonical in `kbtool/kb/contamination.py`) turns a
copy into an automatic `verdict: contamination`. `kb contamination` also grew
the beacon check, `runs-lambda-*`/multi-segment `runs-remote-*` path
matching, and an own-artifact URL tripwire (kernelbench.com / HF traces /
kernelbench raw.githubusercontent paths) since the sandbox deliberately does
NOT `--unshare-net` (provider APIs and the localhost proxies must work).

Honest scope: bwrap still runs `--dev-bind / /` so ncu/nsys/CUPTI/driver and
CLI auth keep working. That full-root bind means this is HYGIENE — a live
agent cannot SEE a foreign archive — not containment: docker.sock, ssh keys,
and the network remain reachable, and a hostile agent is not confined. The
property defended is "no foreign archive visible"; authorship verdicts stay
with the refuse-closed audit. Linux smoke (no GPU):
`scripts/smoke_sandbox_linux.sh`. Unit tests (no bwrap, run on the Mac):
`kbtool/tests/test_sandbox_canary.py`, `test_contamination.py`.

## 2026-08-17 - retracted grok-4.6 21x (copied Fable)

`20260813_152200_grok_grok-4.6_02_kimi_linear_decode` listed `runs-remote-pro`,
read Fable `20260719_121747` (`peak_fraction` 24.6091), then `cp` that
`solution.py` (tool description: "Copy proven megakernel into workspace
solution.py"). The annotation was wrongly `clean` after a same-buffer
overwrite pass. Flip to `contamination`. The Mega board drops the row.
`build_mega_leaderboard.py` no longer lets `clean` override a literal `cp`
of another archive's `solution.py`. `outputs/runs-remote-*` is now a
tripwire path. Isolated regrade and overwrite cosines are not authorship.

---

## 2026-07-31 - run_hard.sh stays a deliberate fork of the shared runner

hard/cuda/mini's run_hard.sh were unified into thin wrappers over
`scripts/lib/run_harness.sh` (monorepo root). Mega deliberately keeps its own
fork: it carries the bwrap anti-contamination sandbox (KBH_SANDBOX — hides run
archives, results/, DEVLOG, public/ solutions, and ~/.claude memory from the
agent) and the legacy anvil machine-wide gpu-lock-exec machinery, neither of
which the shared runner has yet. If the sandbox is ever ported into the shared
runner, fold mega in and delete this fork; until then, harness fixes land in
the lib and must be ported here by hand when relevant.

---

## 2026-07-21 - 01_rl_grid_ppo removed from the deck

Mega is now a single-problem bench: `02_kimi_linear_decode`. The RL-env
megakernel skill that `01_rl_grid_ppo` graded is covered by the CUDA bench's
craftax problem, so keeping both double-counted the same surface. It had
already been soft-hidden from the site (MEGA_HIDDEN_PROBLEMS) since the CUDA
deck landed; this makes the removal real: the deck dir is deleted,
`build_mega_leaderboard.py` filters the problem out of results.csv
(REMOVED_PROBLEMS), and the site constants are cleared. Archived runs, traces,
and annotations for old PPO cells are untouched. Do not re-add — same standing
rule as hard's `04_kahan_softmax`.

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

## 2026-07-01 — SPS credibility cap recalibrated 20x → 100x

The fabrication guard on `01_rl_grid_ppo` set `max_credible_sps_multiple: 20`
(anything over 20x `peak_sps` scores 0.0) when the best known honest kernel was
the ~2.1M SPS reference on the 3090. One day later a clean, audited fable-5
megakernel landed 357M SPS — leaving only ~1.4x honest headroom before the cap
would have zeroed a legitimate kernel. Recalibrated to 100x, still orders of
magnitude below the ~20,000x a no-work fabricator posts.

The problem and the flag are both gone now; the rule is what to keep. Calibrate
a fabrication cap against the physics, orders of magnitude above the best honest
result you currently know of — not just above it. The honest frontier moves, and
a cap tuned to yesterday's record silently starts failing real work.

---

## 2026-07-01 — fabrication red-team: an unfalsifiable metric is not a metric

Red-teaming `01_rl_grid_ppo` before handing the deck out found a critical hole:
the return curve was unfalsifiable. `train()` just returns floats, and nothing
proved an env step ever happened. Two cheats sailed through the gates with
absurd scores:

- **Fabricated curve** (analytic exponential ramp, no GPU work at all): check.py
  PASS (lands in the return band, "improves" from its own early window),
  benchmark **22,006x** peak_fraction.
- **Memoized replay** (`{(steps, seed): curve}` dict; the benchmark reran the
  same BENCH_SEED every trial): **26,849x**.

The fix that generalizes beyond that problem: **benchmark.py draws a fresh
random seed per timed trial** (SystemRandom). Throughput does not depend on the
seed, but a memoized lookup misses every trial, so the cached cheat pays full
cost. Any bench whose timed region can be keyed and cached needs this.

Residual, and the reason the authenticity judge exists at all: a fabricator that
also *sleeps* to fake a plausible elapsed time defeats every mechanical timing
check. A "trainer" with no environment and no policy update fails only on sight,
in an audit. Mechanical guards plus a judge are the enforcement pair; neither
half is sufficient.

---

## 2026-07-01 — megakernel authenticity: judge gate + advisory tripwires (not a substring ban)

Audit finding that started this: **zero of the archived mega submissions are
true single-launch megakernels**, despite the bench being branded one. Every
high-scoring decode cell wins via CUDA graphs (Opus 19.35x = 9 Triton kernels
replayed under a captured graph — cuts launch overhead without fusing anything);
the honest single-custom-kernel attempts are "1 fused GEMV + eager everything
else." The Sonnet 5 (4.03x) vs Opus (19.35x) gap is *entirely* this axis: same
algebra, but Opus captured the whole step as a graph (~1 replay/token) while
Sonnet drove ~9 kernels/token from Python. Neither actually fused. The prompts
and graders never *required* a megakernel, so models rationally reached for the
cheapest launch-overhead fix. Distribution over the 36 decode runs with source: 9 multi-kernel + CUDA graph (every top cell), 20 multi-kernel eager, 4 single custom kernel + eager rest (all <= 2.74x), 1 torch.compile, 2 with no custom kernel at all.

**First attempt (v2) was wrong: string-forbid `torch.compile`/`CUDAGraph` in
check.py.** Before shipping it I red-teamed the gate with a 7-case adversarial
battery (`tests/test_megakernel_evidence.py`). A raw substring scan turned out to
be the worst of both worlds:
- **Leaky** — `getattr(torch.cuda, "CUDAGra"+"ph")` (A5) and `importlib`-based
  runtime codegen that writes+imports a kernel module (A6) carry no literal
  banned token, so they sail through.
- **Brittle** — an honest solution whose *comment* says "no torch.compile, no
  CUDA graphs" (A7) gets false-failed on its own disclaimer.

A substring gate therefore punishes honesty and rewards obfuscation. Killed it.

**Shipped (v2.1): judge gate fed by deterministic advisory evidence.**
- The one bright line that stays a hard fail in check.py is **importing a banned
  library**, matched by **AST import statements** (not substring), recursively
  over solution.py + every local module it imports (incl. `scratch/` sidecars,
  where archived claude/cursor runs stash the real kernel). Naming a lib in a
  comment no longer fails; `marlin` no longer matches `marlinx`.
- `src/eval/megakernel.py` (CLI `scripts/megakernel_evidence.py`) extracts
  objective signals: recursive source, kernel count, and graph/compile/codegen/
  obfuscation tripwires. graph/compile are matched on **comment+string-stripped
  code** (so a disclaimer can't trip them); obfuscation (getattr string-concat,
  banned-token folding) is caught at the AST level so it survives stripping.
- The mandatory pre-publish audit renders the judge prompt from that evidence and
  records `megakernel_authentic: true|false` in `results/annotations/<run_id>.yaml`.
  The judge reasons from code, treating tripwires as hints and docstrings as
  untrusted. Red-team result: judge PASSed A1, FAILed A2–A6 (incl. the obfuscated/
  codegen evasions the substring scan missed) and FAILed A7-as-eager.
- `build_mega_leaderboard.py` **excludes** runs annotated `megakernel_authentic:
  false` (alongside the contamination exclusion), and now emits a **megakernel
  column**: the custom-kernel count in the timed path (launches-per-step proxy;
  for RL, coarse fusion of many steps into one launch is expected, budget ≤8) +
  a green/red marker. Green = genuine fused megakernel within the launch budget;
  red = hides launches / unfused / eager. The marker uses the judge verdict when
  present, else a provisional evidence-based read (hollow dot + trailing `?`).
  `/mega` renders it. Rubric + integration: `SPEC.md`, Megakernel authenticity section.

**Prompts (both problems) updated** to state the timed path must be one fused
kernel and that a *post-run authenticity judge* (not check.py) rejects graph/
compile/per-op-loop escapes — and that obfuscating them is itself a red flag.
Decode mandates one launch/step; RL allows coarse fusion (many env-steps per
launch) but forbids launch counts that scale with steps/horizon/minibatches.

---

## 2026-06-18 — contamination prevented at the source: bwrap sandbox on run_hard.sh

The near-term, easy fix for cross-run contamination (agents reading prior winning
solutions from the shared outputs/runs archive via absolute paths). The proper
sandboxed harness was going to be a Prime Intellect verifiers env (dropped
2026-09-02 along with `environments/`); this is the cheap, in-place prevention that ships now.

Each agent launch in run_hard.sh is now wrapped in:
  bwrap --dev-bind / / --tmpfs $REPO_ROOT/outputs/runs --bind $RUN_DIR $RUN_DIR --chdir $PROBLEM_DIR
`--dev-bind / /` keeps EVERYTHING working (toolchain, src symlink, GPU, codex/
node/claude auth, outputs/gpu.lock — which lives at outputs/gpu.lock, NOT under
outputs/runs, so it stays visible). The `--tmpfs` over outputs/runs HIDES every
other run from the agent; `--bind $RUN_DIR` re-exposes just this run's own dir
(writable, persists). Net: the agent physically cannot read other solutions, and
nothing else changes. Validated: GPU visible under bwrap, codex runs (auth+node
intact), other runs invisible (ls -> 0), find solution.py -> 0, own writes
persist. Only the agent `timeout "$BUDGET_SECONDS"` launches are wrapped (13 of
them); the harness-owned post-run check.py/benchmark.py scoring runs OUTSIDE the
sandbox (different timeout var) and writes results normally.

Toggle: `KBH_SANDBOX=0` disables; auto-off if bwrap is absent (e.g. a box without
bubblewrap). The publish-time tripwire (build_mega_leaderboard.py +
the contamination scan, now `kb contamination mega`) stays as defense-in-depth. See
[[cross-run-contamination]].

Origin: hard DEVLOG 2026-04-27 opencode workspace leak — that entry has the
audit of what agents actually read outside the workspace, and why no CLI
harness isolates the filesystem on its own.

---

## 2026-06-18 — the June rows ran under a 3-hour ceiling

All June runs on the decode problem used `BUDGET_SECONDS=10800` — a 3-hour
wall-clock ceiling, not "unlimited". Mega runs unlimited wall-clock now, so the
June H100 and B200 rows still on /mega are **not budget-comparable** to anything
published since; read them as a 3h-capped cohort. Within June they are
internally comparable, and nothing hit the cap at the time: every run
self-terminated, and the deepest worker (opus on B200) peaked at ~2.5h.

---

## 2026-06-18 — three-GPU leaderboard (Blackwell / H100 / B200)

The decode problem swept across three GPU generations, codex + opus on each:

      GPU                     codex/gpt-5.5    opus-4-8
      RTX PRO 6000 Blackwell      4.34x         14.40x
      H100                        5.62x         15.50x
      B200                        9.37x         19.35x   <- highest

Reads: opus dominates codex on every GPU (deeper kernel engineering under
unlimited time). Both models scale *up* the speedup ratio from Blackwell ->
H100 -> B200, because the baseline (naive int4 materialize) gets relatively
worse on the bigger datacenter cards while the fused dequant-GEMV keeps pace --
so the fusion win compounds with bandwidth.

Why this ports across generations at all: the score is a same-GPU
speedup-over-baseline ratio, so it needs no recalibration per card, and
int4/bf16-acc needs no special tensor-core format (Ampere/Hopper/Blackwell all
do bf16), so the same problem ran on all three with stock cu128 torch and zero
code changes. `build_mega_leaderboard.py` requires a per-run `gpu` marker for
exactly this reason — the requirement is what excludes the legacy bf16 runs that
share the problem name.

The ladder the problem was designed to expose showed up directly: my own hand
solution stalled at 1.1x (launch-bound, 126 tiny kernels), while codex reached
5.62x on the H100 by batching and fusing — 11 specialized Triton kernels, a
4-way fused q/k/v/g GEMV, in-kernel MLA absorption, a hand-written KDA conv +
recurrence. Fusion, not kernel micro-craft, is the whole spread.

---

## 2026-06-17 — Problem 03 went W4A16 (int4 weights), because fp8 loses at decode

Problem 03 (Kimi-Linear hybrid decode) started bf16, then the plan was to make
the baseline fp8 to force quantization into play. Benchmarking killed that:
`torch._scaled_mm` fp8 is a tensor-core compute path with ~7us fixed overhead
and M-padding, so at batch-1 decode (memory-bound) it is *slower* than bf16.

  fp8/bf16 overall speedup across the decode projection set:
    M=1 (decode): 0.84x   M=8: 0.97x   M=32: 1.05x   M=128: 1.16x   M=256: 1.45x

fp8 only pays at large batch. This also explains the dispersion runs where none
of codex/opus/gemini took fp8 — at decode it is not a free win; the real win
needs a hand-written fused dequant-GEMV, and stock fp8 makes it worse.

The fix (Elliot's call): **W4A16** — int4 weights, group-128 asymmetric, bf16
accumulation (the AWQ/GPTQ format OSS actually ships). At batch-1 it is a
*memory-bound* dequant-GEMV, not a compute path, so the 4x weight-traffic
reduction is realizable exactly where decode lives. Empirically grounded by
Hard's own `07_w4a16_gemm` (`regime: memory`, M=1 bandwidth-bound) where models
already wrote fused W4A16 GEMVs hitting 0.15-0.35 of peak DRAM bandwidth on
SM120. Bonus: int4/bf16-acc needs no special tensor-core format, so it runs on
any bf16 GPU (Ampere 3090, Hopper, Blackwell) — the benchmark can travel across
GPU generations, and the speedup-over-baseline metric is already a same-GPU
ratio so it ports with zero recalibration.

Shape of the problem now:
- Weights stored W4A16 (reuses Hard 07's exact pack/dequant; format in
  reference.py). MoE experts quantized too (one int4 set per expert).
- reference.py dequantizes naively in fp32 (oracle). baseline.py is bf16 +
  batched-MoE but still *materializes* each bf16 weight (int4 read + bf16 write
  + bf16 read = ~9x the traffic of fusion) — the floor, deliberately leaving the
  fused dequant-GEMV on the table. Naive int4 is *slower* than plain bf16; int4
  only pays with fusion, which is the whole test.
- Correctness is cosine >= 0.98 of the next-token hidden + decode state vs the
  oracle. The int4 quant noise is in both sides (same weights), so they match at
  ~0.9999 — no tolerance loosening needed, unlike Hard 07 (which compares int4
  against an un-quantized bf16 reference and had to loosen).
- Forbidden now also bars prebuilt int4 kernels (bitsandbytes, torchao, marlin,
  gptq/awq, exllama) so the model writes its own fused dequant-GEMV.
- int4 buffers: 1068 MB vs 4273 MB bf16. Baseline floor ~5.6 ms/tok at ctx 2048.

The "epic" target is now concrete: fused int4 dequant-GEMV + MLA absorption +
KDA/MoE fusion, stacked. None of the three models stacked all of it in the bf16
version; W4A16 makes quantization the central, realizable lever.

---

## 2026-06-16 — RL training megakernel (throughput-graded), v0

The problem was removed from the deck on 2026-07-21. Kept for the design
argument and the hole it exposed, both of which apply to any future
throughput-graded problem.

**Why RL and not a standalone LLM training megakernel.** A fused fwd+bwd
transformer block is low-signal: at frontier scale the training step is already
a chain of big compute-bound GEMMs that cuBLAS/FlashAttention saturate, so there
is nothing to fuse; the only people who care about fused training kernels are
small-model fine-tuners, and Liger-Kernel already owns that. The training that
*is* overhead-bound — tiny nets, millions of tiny steps, env↔learn ping-pong —
is RL. So the "training megakernel" that matters lives inside the RL loop.

**Why throughput, not roofline.** An RL step is control-flow / launch-overhead
bound and has no clean FLOPS ceiling, so peak_fraction-vs-FLOPS is meaningless.
Score `achieved_sps / peak_sps` instead, with algorithm, hyperparameters, and
total step budget fixed and seed-determined — "fastest time to train" then
collapses to "fastest to run N steps".

**Correctness is the learned return level, not allclose.** A different kernel
will never reproduce the reference trajectory bit-for-bit (RNG stream and float
reduction order differ), so check.py trains both from scratch on several seeds
and requires the solution's final-window mean return to land in a band around
the reference's. Validated by codex's honest run: its independent RNG
reproduced the return level closely but not bit-exact (3.985 vs 3.998), which
allclose would have failed.

**DEMONSTRATED HOLE — throughput was credited against the nominal budget, not
actual work.** A wrapper that ran *half* the iterations and padded the returned
curve to full length with the converged value passed check.py (the reference
saturates by ~iter 20, so the final-window mean stays in band and the curve
still climbs from its early baseline) and scored 2x the honest reference floor
for half the work. `benchmark.py` divided `TOTAL_ENV_STEPS / wall_time`,
trusting the solution to have run the budget. If a throughput problem ever comes
back, the harness must own the work accounting — count actual steps inside the
timed region — or grade time-to-threshold instead of steps-per-second.

**Related, untested:** the band pins the outcome, not the algorithm. Nothing
checked that PPO (clip/GAE/stochastic sampling) was used; on a learnable enough
env a greedy controller could clear the band faster. A throughput metric
structurally rewards doing less.

## 2026-09-02 — agy (Antigravity CLI) harness, gemini-3.8-flash-high on Verda RTX PRO 6000

New harness `agy` in run_hard.sh + scripts/lib/run_harness.sh and a stream-json
parser (src/viewer/parsers/agy.py, events init/step_update/result; completion
marker `"event":"result"`). Flag-order trap: a bare `--print` swallows the next
flag as the prompt, so `-p "$PROMPT"` goes last. Auth is OS-keyring OAuth; on a
headless box run the sign-in inside tmux, capture the URL with a 1200-col pane,
paste the user's code. The GEMINI_API_KEY provider path is dead (project quota
0). Box: Verda kb-agy-rtx (default profile, 1x RTX PRO 6000 Blackwell Server
Edition, driver 580, cu130 torch 2.11). Run 20260902_201226_agy_gemini-3.8-flash-high_02_kimi_linear_decode
launched 20:12Z, unlimited budget, bwrap sandbox; ~/cuda_queue.sh then runs the
four cuda problems in sequence on the same GPU. Results pending.

## 2026-09-03 — gemini-3.8-flash-high cells published (RTX PRO 6000 + H100), muse-spark-1.3 queued

Both agy mega cells are on the board after isolated regrade and a same-buffer
overwrite probe on the box: RTX PRO 6000 Blackwell 2.7406 (in-run 2.752) and
H100 SXM5 2.0676 (in-run 2.1075), both `verdict: clean`,
`megakernel_authentic: true` (one cooperative launch per step, 31 and 14
grid.sync sites). The H100 run carries gpu marker `H100` so it lands on the
existing H100 tab (the board self-normalises against baseline.py on the same
box; the annotation says SXM5). Sandbox gap seen in the H100 trace: bwrap
hides results/, outputs/runs and DEVLOG but not src/ or scripts/, and the agent
read src/eval/megakernel.py and ran scripts/megakernel_evidence.py on itself.
Not gaming (PROMPT.txt announces the judge), but those paths should be
tmpfs-hidden too. The problem prompt is worded for the RTX PRO 6000 even on the
H100 box, as for every H100 mega cell. Publish commit deeea1f; traces pushed to
Infatoshi/kernelbench-mega-traces. New harness `muse` (Meta Muse Code CLI,
`muse exec --json --yolo ... --model muse-spark-1.3 --reasoning-effort ultra`,
parser src/viewer/parsers/muse.py) runs 02_kimi_linear_decode on both boxes
after the gemini queues; results pending.
