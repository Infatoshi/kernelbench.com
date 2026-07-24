# KernelBench-Multi DEVLOG

Newest first. SPEC.md holds the methodology; this holds the journey.

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
cell) · 03 fp8 allgather 0.106 · 04 moe all2all 0.452 · 05 ulysses 0.102
(wave-1 8.5-min artifact) · 06 fp8 reducescatter — (no clean cell).
The waves-1/2 external SIGKILLs (below) plus contamination exclusions mean
02/05/06 deserve a fresh wave under the relocated-archive protocol.

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
