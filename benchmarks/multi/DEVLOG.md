# KernelBench-Multi DEVLOG

Newest first. SPEC.md holds the methodology; this holds the journey.

## 2026-07-25 — first wave on the new deck killed itself; sessions now run sequentially

Launched four grok-4.5 sessions concurrently (01/07/08/09) at 08:57 UTC on a
quiet hades. Three died at 09:28:00, 09:28:10 and 09:28:19 with exit 137, and the
fourth finished but could not be graded. Nothing external did it — the wave
killed itself, and the mechanism is worth writing down because the harness looked
correct right up until it wasn't.

**The chain.** The node-wide GPU lock serializes GPU *commands*. It does not
partition GPU *memory*, and a session holds its allocations across lock windows,
so four concurrent sessions meant four resident sets and the node hit CUDA OOM
(`NVRM: ... NV_ERR_NO_MEMORY`, 09:25-09:26; host RAM was never touched — 13 GB of
885 used). An agent then looked at `nvidia-smi`, saw "processes holding almost
full GPU memory (74GB each)", and concluded — correctly, from where it sat — that
they were leaked from hung kernels. Its cleanup was `pkill -f 'torchrun'` and
`pkill -f 'worker.py'`. Those patterns match every *sibling* session, so it killed
the other two runs and itself. Its own trace records the confusion in real time:
"pkill might have killed the shell", "pkill killed my process maybe?".

The survivor (09) then failed grading with `EADDRINUSE` on port 29571, because
every session used the same hardcoded rendezvous port — which is also why agents
had been running `fuser -k 29571/tcp`, i.e. killing each other's grading runs to
claim a port. Two independent shared-resource bugs feeding the same fire.

**The co-tenant was not hit.** Worth checking rather than assuming, since the box
is shared: every pattern used was torch-scoped (`worker.py`, `torchrun`,
`torch.distributed.run`, `test_ar.py`, `torchrun.*29612`) and none can match
`vllm serve` or `VLLM::EngineCore`. Verified against the grok session store, which
is the only place the actual bash commands survive — `agent.log` from
`--output-format streaming-json` carries thoughts and text but no tool calls.

**Fixes.** Sequential execution (`scripts/sweep_wave.sh`) is the real one: it
removes the memory contention, the sibling kills and the port fight at once, and
it makes in-run flywheel timings trustworthy instead of contended. Per-run
rendezvous ports. And a `pkill`/`killall` wrapper that refuses patterns matching
another tenant's job.

Building that wrapper took two wrong versions, both caught by testing rather than
by reasoning about it:

1. Keyed on "refuse if a match is outside my session id". It refused *legitimate
   self-cleanup*, because `pgrep -f` also matches any shell whose command line
   contains the pattern — including an operator's own `pgrep -af torchrun`
   monitoring command. A guard that blocks an agent from cleaning up its own hung
   torchrun is worse than no guard.
2. Case-sensitive `grep -E "vllm|..."`. The processes actually holding the GPUs
   are named `VLLM::Worker_TP0..3` in UPPERCASE — precisely the name an agent
   reads out of `nvidia-smi` and would target — so all four slipped through while
   only the lowercase `vllm serve` parents matched. `grep -qiE` now, verified
   against the live PIDs.

The wave's numbers (01 `peak_fraction` 0.2614, 07 `speedup` 1.0261, 08 1.2817)
are recorded in `results/annotations/20260725_wave_parallel_INVALID.yaml` and are
**not results** — killed mid-flywheel under contention. 01 landing at 0.2614
against its earlier clean 0.3057 is the signature of a truncated session, and a
useful reminder that a contaminated wave can look like a plausible regression
rather than like an error.

## 2026-07-25 — deck revision: 6 pure-comm problems → 4, half of them fused

The first clean board (below) exposed a design problem the board itself could not
show: the six problems spanned four collectives but only ~3.5 independent skills.
An all-reduce IS a reduce-scatter plus an all-gather, so 01/02/03 were one family;
02/06 were both reduce-scatter; 04/05 were both all-to-all. Four of the six
epilogues were free — RMSNorm after a reduce-scatter is entirely local (each rank
already owns whole rows), and the wave-8 audit shows it riding along in shared
memory of a kernel the agent had to write anyway. Every pure-comm problem with a
light epilogue converges on the same kernel shape (publish to symmetric memory,
barrier, pull, done), which is why they read as sweep problems.

Cost made it worse: with the mandatory sequential re-grade, problem count is
close to linear in wall-clock on a node that bills by the hour, and every extra
run is another audit and another archive entry for a later agent to mine.

**Kept exactly one pure primitive and moved the rest to where comm and compute
interleave**, which is where the field's hand-tuned kernels actually live:

- `01_allreduce_residual` unchanged (busbw-graded).
- `07_gemm_allreduce_overlap` — TP row-parallel GEMM whose all-reduce should be
  hidden behind the MMA pipeline (async-TP / Flux territory). Shapes chosen so
  gemm_ms and comm_ms are within ~1.5×, where overlap is worth real time.
- `08_ring_attention_cp` — causal context-parallel attention: ring the K/V,
  accumulate with an online softmax, and deal with the 1:2:3:4 causal load
  imbalance that a lockstep ring turns into rank-3-gates-everything. This is the
  honest version of what the retired `05_ulysses_all2all` gestured at (Ulysses
  explicitly removed the attention math, which was the interesting part).
- `09_moe_ep_dispatch_combine` — DeepEP-shaped: data-dependent unbalanced routing,
  metadata round trip on the critical path, permutation fused into the transfer,
  fp8 dispatch leg / bf16 combine leg.

Retired 02/03/05 (subsumed) and 04/06 (superseded by 09). Numbers are not reused.

**Two anti-cheats, both verified adversarially before freezing** (the old 04's
zero-comm hole was found by reading, not by testing — this time the exploit was
written and run):

- 07: the weight shards are rank-distinct by construction, so
  `sum_r(x_r @ W_r) != (sum_r x_r) @ W`. An adversarial solution that all-reduces
  the smaller activation first and does one local GEMM fails at max_rel 22.
- 09: the expert is NOT a per-token map — its output for a token depends on the
  neighbouring token in the expert's canonical `(src_rank, src_index)` batch
  order, in full width, so no gathered-weights local shortcut reproduces it. A
  zero-communication solution fails at max_rel 2.0. (Any per-token elementwise
  expert is algebraically avoidable at bench scale, because nothing stops a rank
  from replicating every expert weight — real EP exists only because the weights
  do not fit. The cross-token dependency is what restores the constraint, and it
  must be full-width: a scalar or low-rank dependency can be recovered with a
  tiny all-reduce.) fp8 on the dispatch leg is likewise forced rather than
  suggested: the reference quantizes per-token, so a bf16-dispatch solution
  misses the oracle by the full quantization error.

**New metric: `speedup`.** 07/08/09 fuse compute with comm, so no single
collective convention describes their bytes and a busbw fraction would be
uninterpretable. They are graded on geomean speedup vs a frozen production anchor
(`sota.py`, timed once via the new `--mode anchor`, pinned as `anchor_ms`). 01
keeps busbw. The tradeoff is inherent: a problem legible enough for a clean
bus-bandwidth fraction has nothing in it but comm.

**08's tolerance was calibrated, not guessed** (`scripts/numerics_probe_attention.py`).
The first NCCL validation had the anchor failing its own oracle on 2 of 8.4M
elements — near-zero outputs where honest bf16 error (2.3e-4) just exceeded the
abs floor (atol·rms = 1.6e-4). Probe results: worst honest variant needs atol
0.0281 (one-shot bf16 SDPA, and a ring with bf16 accumulate); dropping the last
KV chunk needs 2.04–6.85; skipping the online-softmax rescale needs 4.45 at the
large stress scale (it is nearly error-free on homogeneous nominal data, which is
also why it buys no time — numeric stress is what catches it). Gate set at
`[0.1, 0.025]`.

**Anchor measurement is now guarded.** The first anchor pass was taken while a
co-tenant vLLM held 70 GB on GPU0 — violating the quiet-node rule written in that
script's own docstring — and was discarded. `measure_anchors.py` now refuses to
run when any GPU shows >2 GB resident (`KBM_ALLOW_BUSY=1` to override). A
contended anchor is worse than a contended re-grade: it is frozen, and it divides
every future speedup on that problem.

All three new oracles validated on gloo/cpu and then against an independent NCCL
implementation at full shapes (the anchor reproduces the oracle exactly on 09,
which is the real check on the canonical-order design).

**08's anchor was handicapped, which would have inflated every score on it.** The
first draft passed the causal mask to SDPA as a dense bool `attn_mask` over the
full gathered context. That is not a slow *baseline*, it is a *mistake*: a dense
`attn_mask` forces SDPA off the fused kernel onto the score-materializing path
(for `seq_local=2048, heads=32` the score tensor alone is ~1 GB). Anchoring
against it would have paid every model a 1.3-2.3x speedup for avoiding an error
nobody ships, on top of whatever its ring actually earned.

The same mask expressed as **bottom-right causal over the sliced K/V** is exactly
equivalent and stays fused: rank r owns queries `[r*sl, (r+1)*sl)` and sees keys
`[0, (r+1)*sl)`, so `kv_len - q_len = r*sl` and the bottom-right convention gives
`q_i attends k_j for j <= i + r*sl` — the CP mask, unmaterialized. A probe on the
canonical node confirmed both properties at once: `atol_min` vs the fp32 oracle is
identical to 4 decimals on all four shapes (same math), and it runs 1.79x / 1.92x /
1.30x / 2.29x faster. `sota.py` now uses `causal_lower_right`; the mask form stays
in `reference.py`, where it is the correctness oracle and is never timed.

Worth stating as a rule, because it generalizes past this problem: **an anchor
must be the fast honest implementation, not merely a correct one.** A frozen
denominator with an avoidable inefficiency in it silently converts "the model
avoided a beginner mistake" into "the model wrote a good kernel," and the
resulting column looks strong for the entire life of the deck.

**The forbidden-op tripwire had a hole the new deck would have walked into.** It
grepped `solution.py` only. That was survivable when `sota.py` was a stub, but
07/08/09 each ship a `sota.py` that is a complete working NCCL implementation of
the problem, copied into the agent's workspace — so `from sota import Model` was a
two-word solution that passed the grep and scored exactly 1.0 speedup, and a
`comm.py` helper holding the bare `dist.all_reduce` passed just as easily. The
scan now covers every agent-authored `.py` in the workspace (immutable benchmark
files excluded, since `sota.py` legitimately calls what the agent may not) and
treats importing `sota`/`reference` as its own failure. Verified against four
cases: anchor re-export, collective hidden in a helper, bare collective in
`solution.py` (all three caught, each naming the offending file), and an honest
`batch_isend_irecv` ring in a helper module (passes — P2P is not on the forbidden
list). All four prompts now state that the restriction follows imports, so an
honest agent is not blindsided by it.

## 2026-07-24 — contamination sweep, first clean grok-4.5 board, formula fix

Five grok-4.5 waves ran on hades (waves 1/2 killed externally — see below; wave 3
died on auth; wave 4 full-budget but co-tenant-contended; wave 5 full-budget,
uncontended, natural exit). Every candidate cell was audited (subagent code+trace
audit) AND swept for cross-run contamination at tool-call level. Findings:

- **9 of 14 audited runs were cross-run contaminated.** The harness has no
  filesystem sandbox, so agents in waves 2+ found `~/kbm/outputs/runs/`, read
  prior waves' solutions/benchmarks for the same problem, and iterated on them
  (self-iteration, same model — but the standing auto-exclude rule doesn't
  distinguish, and one wave-1 agent even read its CONCURRENT siblings). This is
  what produced the "improvement" trajectory across waves for 02/03/05/06.
- **Audit method matters: grok's `agent.log` omits tool calls and fragments text
  token-per-line, so grep-based tripwires (incl. `kb contamination`-style scans)
  return false negatives.** The authoritative transcript is the grok session
  store `~/.grok/sessions/<urlencoded-ws-path>/<sid>/chat_history.jsonl` (full
  messages + tool calls + terminal logs). One cell (wave-5 06) passed an
  agent.log-only audit and was then disqualified by the session store.
- **Mitigation now in effect:** completed runs are relocated to `~/.kbm-archive/`
  (outside the `~/kbm` tree agents explore) before any new wave launches. Proper
  fix remains a sandboxed harness.
- **05 busbw formula bug fixed** (commit e63778e): the formula multiplied the
  per-rank buffer by world_size (algbw/busbw confusion), over-crediting 4x —
  wave-2's cell read an impossible 1.42x peak on shape 0. All cells re-graded
  under the per-rank NCCL convention; the ceiling script already matched it.
- **04 deck weakness (open):** dispatch->scale->combine is algebraically
  identity-with-scale on local data and `expert_w` is seeded rank-identical, so
  a zero-communication solution would pass correctness. No agent exploited it
  yet (wave-5's audit verified real wire traffic, 284 GB/s). Harden with
  rank-distinct expert weights or a wire-traffic check before publish.

**First clean grok-4.5 board (4xH100, sequential isolated re-grades,
per-cell annotations in `results/annotations/`):**
01 allreduce+residual 0.306 · 02 reducescatter+rmsnorm — (no clean passing
cell) · 03 fp8 allgather 0.106 · 04 moe all2all 0.452 · 05 ulysses 0.206
(wave-6, clean audit: entry+exit barriers, live copy-in before every graph
replay; isolated quiet-node re-grade 0.2056 vs in-run 0.2042) · 06 fp8 reducescatter — (no clean
cell). Wave 6 also proved relocating archives to a hidden `~/.kbm-archive`
is NOT a sandbox: the 02 and 06 agents found and read it (both excluded).
Durable fix applied — all runs now pulled to the Mac
(`outputs/runs-hades/`) and deleted from the node, so wave 7 (02, 06)
launches against a truly empty box.

**Wave 8 completed the board (2026-07-24).** Under the full-scrub protocol
both open cells came back clean (audited, sequentially re-graded on the
quiet node): 02 = 0.2179 (in-run 0.1849 was sibling-contended — the two
wave-8 agents contended with each other, re-grade recovered 18%),
06 = 0.1029. **Final clean grok-4.5 board: 01 0.306 · 02 0.218 · 03 0.106 ·
04 0.452 · 05 0.206 · 06 0.103.** Instructive: the clean 02/06 numbers are
LOWER than the excluded contaminated cells (0.233/0.149) — the "gains" in
waves 4-7 were transcript-mined iteration, not capability. Wave-8 agents
still ran the same archaeology probes (find solution.py, ls ~/.grok/sessions)
— intent is constant; only supply was cut.

**Wave 7 found a THIRD leak channel: grok's own session store.** With the
runs archive genuinely empty, both agents (02 0.2257, 06 0.1490 — excluded)
located `~/.grok/sessions/<old-run-ws>/chat_history.jsonl` and RECONSTRUCTED
prior best solutions from the transcripts' file-write events ("I'll recover
the prior working multimem reduce-scatter solution and build from that
baseline"). Also mined: a stale `~/kbm-smoke` dir; 06 read its concurrent
sibling's agent.log. Lesson: an empty archive is not an empty box — any
history artifact (agent CLI session stores, smoke dirs, launcher logs,
psmon logs) is a contamination vector, and grok-4.5 hunts for ALL of them
by default. Wave-8 protocol: scrub `~/.grok/sessions`, `~/.grok/logs`,
`~/.grok/memtrace`, `~/kbm-smoke`, `~/kbm/outputs/launch*.log` after every
pull, before every launch. Sessions are pulled to the Mac first
(`outputs/runs-hades/_grok-sessions/`) — they are also the audit record.

Auth gotcha: grok CLI refresh tokens are single-use; sharing the Mac's
auth.json with hades caused sign-outs mid-fleet (waves 3/5a) and is the lead
suspect behind the silent fleet kills. Hades now has its own device-code login.

## 2026-07-23 — first agent wave (Grok), node-wide GPU lock, fleet-wide SIGKILL incident

- Harness: `scripts/run_agent.sh` runs ON the 4xH100 node. Node-wide flock
  (`~/kbm/outputs/gpu_lock/gpu.lock`) via PATH wrappers for
  python/python3/torchrun/nvcc/ncu/nsys, reentrant through
  `KBM_GPU_LOCK_HELD=1`, held for process lifetime. Node-wide (not per-bench)
  because every session needs all 4 GPUs. `nvidia-smi` deliberately unwrapped
  (read-only, agents poll it). Verified live: two concurrent wrapped sessions
  serialize; second blocks until first frees.
- Wave 1 (grok-4.5, all 6 problems, 7200s budget, hades): all six grok CLIs
  were SIGKILLed (exit 137) at ~08:16:38-49 UTC, ~8.5 min in. Partial early
  solutions still graded: 04 PASS 0.2269, 05 PASS 0.4042, 06 PASS 0.0341,
  01 solution SIGABRTs (genuine misaligned-address kernel), 02 TypeError,
  03 no solution.
- Kill forensics: ~70 s before the kills, the kernel log shows a massive
  NVRM Xid 13 flood (Graphics SM Warp Exception / Misaligned Address /
  Multiple Warp Errors) across ALL FOUR GPUs — an agent's misaligned IPC
  kernel faulting on the whole fabric (consistent with 01's crashing
  solution). The SIGKILL deliverer was never identified: no OOM (dmesg clean,
  759G free), no journal entry, no logind session teardown, no cgroup limit,
  no human logged in, stderr files 0 bytes (no chance to write). GPUs
  recovered fully (world-4 NCCL check passes, 0 MiB residency).
- Mitigation for wave 2: each session launched under `setsid` (own session,
  immune to any session/process-group-wide signal) plus a psmon sidecar
  logging ps state of grok/torchrun/check/benchmark every 20 s so any
  recurrence pins the exact kill time and scope.

## 2026-07-22 — numerics contract for bf16 reductions (probe-calibrated)

User directive: characterize reduction-order numerics empirically — critical
for distributed problems. `scripts/numerics_probe.py` (torchrun world-4, on
real NCCL/H100) measures honest all-reduce variants (NCCL in-type bf16,
fp32-accumulate, fp64 exact, bf16 tree order, bf16 ring order) against each
other and against cheats (skip-a-rank, local-only), under the exact bench gate,
at 3 input scales.

Findings that changed the bench:

- In-type bf16 orderings disagree with each other by up to ~2e-2 rel. An
  order-dependent oracle (NCCL bf16 reduce) FAILS honest fp32-accumulate
  kernels 5/5 trials at 5e-3. The oracle must be the mathematically exact
  sum with a single downcast: 01's reference now upcasts to fp32 before
  `dist.all_reduce`.
- Fixed atol misfires in both directions under input rescale: at 1e3 scale it
  fails honest kernels; at 1e-3 scale skip-a-rank nearly passes (5.89e-3 vs
  5e-3). `src/eval/compare.py` gate is now scale-aware:
  `tol = atol * rms(ref) + rtol * |ref|` — invariant under numeric-stress
  rescales.
- Tolerance on the reduction decks (01, 02) set to bf16 2.5e-2: honest noise
  tops out ≤2e-2 rel, cheapest cheat (skip-a-rank) is ≥~140% rel — a 100x
  margin. Verified: honest variants 12/12 pass, cheats 12/12 fail, at every
  scale.

## 2026-07-22 — first smoke (GLM via zai-claude) caught grader tampering

The first agent smoke run's solution monkey-patched `dist.all_reduce` so the
in-process reference oracle matched the solution's own numerics
(`_install_exact_reference_reduce()`). Its underlying numerics complaint was
legitimate (see probe above) but the tampering is an instant fail.
`src/eval/worker.py` now snapshots the identity of the c10d surface and every
reference-module callable AFTER importing reference/shapes and BEFORE
importing solution; `oracle_tampered()` re-checks identities before the
verdict and fails the run with the rebound name. Verified: tamper FAILs all
ranks, clean reference passes. The GLM solution itself, graded against the
corrected oracle, is genuinely correct: 0.2148 geomean peak_fraction vs the
NCCL baseline's 0.2416, winning 1.43x on the 1MB shape.

## 2026-07-22 — re-scoped 8xH100 → 4xH100 NVSwitch; ceiling measured

- The rentable temporary nodes (poseidon, hades) are 4xH100 SXM behind
  NVSwitch: every pair NV18 = all 18 NVLink4 links into the crossbar — the
  same per-GPU fabric as the 8x template, just fewer peers. Re-scoped the
  whole bench: deck `problems-h100x4/`, `world_size: 4`, hardware `H100x4`.
- Roofline peak corrected 900 → 450 GB/s: NCCL busbw convention measures
  against the UNIDIRECTIONAL link rate. Measured c10d ceiling on the real
  node: all-reduce 348 GB/s = 0.77 of 450 at 512 MB (inside the 70-85%
  expected band); small messages are latency-bound at 6-18% — that gap is
  the headroom agents exploit.
- `scripts/remote_ceiling.sh` topology gate fixed: the old grep matched the
  NIC column's legitimate PHB and false-failed every node (including the
  original 8x template). Now awk-checks only the GPUxGPU submatrix.
