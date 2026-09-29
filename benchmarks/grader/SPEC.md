# kbgrade: out-of-reach grading for every KernelBench bench

Status: architecture, 2026-09-28. Nothing here is built yet. Build order is at the end.

## Goal

The agent never shares a filesystem, process, or interpreter with anything that decides its score.
It works in its own container with the GPU, the spec, and its code. When it wants feedback it calls
a grading API. Every grade, dev or official, runs in a fresh sandbox that the agent cannot reach,
from a harness process that never imports the agent's code. The official number comes from the same
service on a quiet, clock-locked GPU, with hidden cases, and is signed so the site can refuse
anything else.

Design rule (Kerckhoffs): assume the agent can read this file and the grader source on GitHub.
Security comes from isolation, secret per-release seeds, and verification, not from hiding code.

## Why now: what the current path cannot stop

Today `check.py` and `benchmark.py` import `solution.py` into their own interpreter, inside the
agent's workspace, and read the same input tensors on every timed call. The bwrap sandbox is
hygiene, not containment (its own header says so): `/` stays bound, and docker.sock, ssh keys and
the network remain reachable. The isolated regrade on the canonical deck fixes grader-file edits
and contended timing, but not in-process attacks. Those are exactly the classes other benchmarks
measured:

| Class | Seen here | Seen elsewhere | Current defense | kbgrade defense |
| --- | --- | --- | --- | --- |
| Read other runs' archives | 30 contamination-family verdicts (grok 4.5/4.6, qwen, gemini) | | bwrap hide + honeytoken + canary | Agent container has no repo, no host paths |
| Edit grader files | template_mutated history | Terminal-Bench, METR esolang | canonical-deck regrade | Grader not in agent container at all |
| Sniff check vs graded path | 04 TF32 only on the unchecked path; 04 positions only checked at 128x8 | SOL-ExecBench one-time correctness | graded-shape checks added piecemeal | Hidden seeds, full graded sizes, randomized validated iterations |
| Cache on input identity or data_ptr | overwrite probes in every audit | KernelBench #171, KernelGuard, CUDA-L1 | manual audit probe | Fresh values and fresh addresses every iteration; outputs validated after timing |
| Work off the timed window (streams, threads, jit.fork) | | CUDA-L1 32.8% of RL kernels; SOL-ExecBench 2.5% | device sync in time_fn | Device-wide sync + thread/stream census + CUPTI cross-check |
| Monkeypatch timers or grader | | METR o3 (synchronize, perf_counter, stack-walk for the reference tensor); SOL-ExecBench 3.3% | trusted_entrypoint blocks SystemExit(0) only | Timer and verdict live in a process the solution never enters |
| Lazy or fake tensors | | SOL-ExecBench FakeTensor | none | `type(t) is torch.Tensor`, materialized in harness memory |
| Import-time or setup-time work | 04 import-time GPU warmup (today) | | none | Harness owns warmup and clock state; CUPTI flags GPU work during import |
| Precision downgrade | fp16/TF32 04 cells | SOL-ExecBench 6.4% (largest class) | tolerance + human audit | Spec-level precision probes plus the audit |
| Grader bugs / weak teeth | 07 rounding mutant passes; 02 zeros pass nominal | | teeth probes on anvil | Mutant suite is a CI gate on the grader itself |
| Leftover processes | | Terminal-Bench sleeper agents | none | cgroup kill of the grade sandbox |

## Trust zones

```
 host (trusted)                                    agent container (untrusted, one per cell)
 ┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
 │ orchestrator (launcher)              │ starts  │ agent CLI (claude/codex/grok/...)    │
 │ kbgrade daemon ── GPU queue per dev  │───────▶ │ /work: PROMPT, reference.py, shapes, │
 │   signing key (never leaves host)    │         │   public spec, solution.py, scratch  │
 │   submission log + result store      │◀────────│ check.py / benchmark.py = API shims  │
 │ official socket (not mounted)        │ dev.sock│ GPU visible for dev + ncu/nsys       │
 └───────────────┬──────────────────────┘ (1/cell)│ no repo, no host home, no docker.sock│
                 │ spawns per request             └──────────────────────────────────────┘
                 ▼
 grade sandbox (fresh, ephemeral, no network, read-only root)
 ┌──────────────────────────────────────────────────────────────┐
 │ harness (uid grader): spec, hidden cases, reference, timer,  │
 │   verdict. Never imports submission code.                    │
 │        ▲ pipe + CUDA IPC tensors                              │
 │        ▼                                                      │
 │ worker (uid sub, no_new_privs, seccomp, Landlock): imports    │
 │   the submission snapshot (ro), runs forward calls only       │
 └──────────────────────────────────────────────────────────────┘
```

Three zones, three rules:

1. Host: orchestrator, `kbgrade` daemon, Docker, the HMAC key, the submission log. Nothing untrusted
   runs here.
2. Agent container (`kb-agent` image): the agent's whole world. GPU visible (never hide CUDA; the
   agent must still profile and self-time), `CAP_PERFMON` for ncu, `no-new-privileges`, non-root user,
   workspace `/work` rw, session store `/home/agent` archived. Its only path to grading is a per-cell
   unix socket mounted at `/run/kbgrade/dev.sock`. The socket identifies the cell; it cannot grade
   another cell or reach official routes.
3. Grade sandbox (`kb-grade` image): one fresh container per request. The submission snapshot is
   mounted read-only, the grader and hidden cases are baked in, network is off, rootfs is read-only,
   `/tmp` is a small tmpfs, and the cgroup is killed at the end. Inside it, the harness process and
   the worker process are different uids. The worker cannot ptrace, cannot write the harness's
   memory, and cannot see the harness's files.

## The harness/worker split

The verdict and the timer never share an interpreter with submission code. Timing still needs the
kernel launch and the timer on the same device context, so it splits like this:

- Inputs: the harness generates every input on its own CUDA context and shares it with the worker
  as CUDA IPC tensors (`torch.multiprocessing`, spawn). Each iteration gets new values and a new
  address (rotating offset in a harness-owned pool), so identity and `data_ptr` caches never hit.
- Calls: the worker runs `Model.forward` (or `run`, `prefill`/`decode_steps`) and hands outputs back
  as IPC tensors. The harness checks `type(t) is torch.Tensor`, dtype, shape, contiguity, then
  validates values on its own context. The worker never sees a reference output.
- Correctness: public cases (same seeds as the dev shim) plus hidden cases from a per-release secret
  salt: every graded shape at full size, three or more seeds, numeric stress and property stress,
  and the probes we now run by hand (same-buffer overwrite of every input, in-place weight change,
  permuted routing, special values). Correctness runs in a different worker process from timing,
  so state from a checked call cannot leak into a timed one.
- Timing: the worker runs the timed loop because microsecond accuracy needs it, but under a timing
  shim compiled into the worker before the submission is imported. The shim records CUDA events
  around a device-wide synchronize. Its function addresses are snapshotted and verified after the
  run (SOL-ExecBench), and its pages are sealed with `mseal` where the kernel supports it
  (pygpubench). The harness does not trust those numbers alone. Four cross-checks:
  1. Harness wall clock around the whole timed block, from pipe signals, bounds the reported mean.
  2. A random subset of timed iterations, unknown to the worker, is validated after the fact.
     Timed and validated calls are indistinguishable from the worker's side.
  3. Thread census and stream census: thread count before and after, and in the official profile
     a CUPTI activity trace (nsys, launched by the harness, not the worker) of every kernel on
     every stream. Reported time must agree with GPU busy time, and no kernel may run outside the
     window.
  4. Physics floor: a reported time below the bytes/bandwidth and flops/peak floor for the shape
     is rejected before any human sees it (KernelGuard layer 2).
- Clock and cache: the official profile locks graphics clocks (`nvidia-smi -lgc`), flushes L2
  between iterations for every problem (today only some do), and runs its own warmup at the graded
  shape until timings settle. Import-time warmups like today's 04 cell stop mattering.
- Build vs run: phase 1 builds extensions (`load_inline`, `.cu` sidecars, Triton warm compile) with
  exec allowed but no network, into a content-addressed cache. Phase 2 runs with Landlock limiting
  execute to the Python and toolchain paths, and a seccomp filter denying `ptrace`,
  `process_vm_writev`, `socket` (except the harness pipe), `bpf`, and `perf_event_open`. Import-time
  GPU activity is recorded (CUPTI) and reported, not banned: compiling is legitimate, warming clocks
  is not.
- Audit hooks: the worker runs `sys.addaudithook` logging `open`, `subprocess`, `socket`, `ctypes`
  and `exec`/`compile` events to the harness. `exec()` on runtime-built strings is a lint finding
  (KernelGuard's obfuscation class).

## API

HTTP+JSON over unix sockets. Two sockets, two audiences.

Dev socket (mounted into the agent container, one per cell):

- `POST /v1/submit {mode: "check"|"benchmark"}` with a tarball of the submission. The submission is
  `solution.py` plus every regular file in `/work` except `scratch/` and the provided spec files,
  capped at 16 MB. Returns `{id}`.
- `GET /v1/submit/{id}/events` streams the same lines today's scripts print (`PASS`,
  `FAIL: shape 2 seed 123 case large_qkv: ...`, `shape=0 variant=solution ... peak_fraction=...`), so
  prompts and agent habits do not change.
- `python check.py` and `python benchmark.py` in `/work` become shims for exactly those two calls.
  The agent can read the shims; they contain no grading logic.
- Dev grades run public cases only, on the cell's GPU, queued behind other dev work. Numbers stay
  "in-run, not a result" and say so in the output.

Official socket (host only, never mounted into any container):

- `POST /v1/grade {run_id, profile: "official", gpu}` takes the archived final submission. The
  daemon waits for zero compute PIDs on that GPU, locks clocks, runs build, hidden correctness,
  probes, timing, and CUPTI, then writes `grade.json`.
- `grade.json` carries submission sha256, grader image digest, problem spec sha (replacing
  `graded_surface_sha`), GPU UUID, driver, locked clock, every case and probe result, timing
  distribution, CUPTI summary, lint and audit-hook findings, and an HMAC over all of it. The key
  lives only on the host.

Every dev and official submission is logged (hash, time, verdict, timings). The hill climb becomes
data: trajectory charts stop depending on transcript parsing, and an audit can diff consecutive
submissions.

## Container runtimes, in order

1. Docker + NVIDIA Container Toolkit. tetra has both (Docker 29.1.3, `nvidia-ctk`); Lambda and Brev
   images do too.
2. Podman rootless with CDI GPU specs, same images, for hosts without a docker group.
3. Apptainer `--nv` for HPC hosts, built from the same Dockerfiles.
4. bwrap containment fallback: today's sandbox v2 grown into real containment. No `--dev-bind / /`;
   ro-bind only the toolchain, Python, and driver paths; `--unshare-net --unshare-pid --unshare-ipc
   --new-session --die-with-parent --cap-drop ALL`; the dev socket and a model-API egress proxy
   socket bound in. Same harness/worker split inside.
5. Cloud sandboxes (Modal, Daytona) later, only if we outgrow owned GPUs. Harbor's task format is an
   export target, not a runtime dependency.

The agent container never gets `docker.sock`. tetra's `docker` group is root-equivalent, which is
why the daemon, not the agent, owns every `docker run`.

## Images

- `kb-agent:<ver>`: CUDA 13.3 devel, Python and uv, each bench's locked venv prebuilt,
  `patch_torch.sh` applied, ncu and nsys, and every agent CLI at a pinned version (Claude Code,
  Codex, Grok, agy, Muse, opencode, Droid, Cursor agent). Pinning the CLIs in the image also records
  the harness version per cell. Credentials are injected per run: one env token or one read-only
  config file per harness, never `~/.env_vars` wholesale.
- `kb-grade:<ver>`: the same CUDA, torch, and Triton stack as `kb-agent`, so code that builds in dev
  builds in grade, plus the harness, specs, references, and the hidden-case generator. Its digest is
  part of every grade.
- Network: grade sandbox none. Agent container egress through a logging proxy, with the model API
  endpoints and an allowlist (GitHub, PyPI, docs) the prompt already promises (MegaQwen,
  flashinfer, marlin).

## Problem spec

Decks stay as they are. The harness reads today's `reference.py`, `shapes.py`, and `problem.yaml`
and adds one declarative file per problem, `grade.yaml`:

- `api`: `forward(inputs)`, `run(num_envs, horizon, seed)`, `prefill+decode_steps`, or `step(state)`.
- `inputs`: generator reference, which inputs are weights (probe with in-place change) and which are
  activations (probe with same-buffer overwrite).
- `outputs`: dtype, shape, tolerance policy (atol/rtol, cosine for mega, exact for 04
  positions/rewards with the tie margin rule once decided).
- `precision`: declared compute precision, for the precision probe.
- `cases`: public seeds, hidden seed count, stress families, graded sizes.
- `score`: roofline fraction, speedup vs frozen eager anchor, or ms geomean, plus the physics floor.

`check.py` and `benchmark.py` in the deck become thin adapters over the same harness, so a human can
still run them locally on a trusted box, and the four copies of `regrade_sequential.sh` retire into
`kbgrade grade --official`.

## Publish gate

New gate: every cell on a board needs a `grade.json` whose HMAC verifies, whose grader digest and
spec sha match the current release, and whose GPU maps to the deck. Hand-edited results fail
closed. The audit YAML stays mandatory. `grade.json` pre-fills its evidence fields (overwrite
cosines, graded-size correctness, CUPTI agreement, import-time GPU activity, served model id from the
orchestrator), and the human or LLM auditor still reads the code and the trace and signs the verdict.

## What this does not solve

- A kernel that is genuinely correct on every tested input but exploits an ambiguity in the spec
  (the 07 zero-point question, the 04 tie margin). That is a spec decision, and hidden cases must stay
  inside the spec or they become gotchas.
- A determined in-worker attack on the timing shim below the Python level. CUPTI and the physics
  floor bound the damage.
- Contaminated knowledge from the public web (old traces on HF). The contamination scan stays.

## Build order

1. Harness and worker without containers. `benchmarks/grader/` uv package: spec loader over
   today's decks, harness/worker split, IPC inputs, fresh addresses, validated timed iterations,
   census, clock lock, L2 flush, audit hooks. Gate: a mutant suite where every row of the table above
   is a failing submission (data_ptr cache, stale CUDA graph replay, side stream, side thread,
   jit.fork, FakeTensor, patched `perf_counter`/`synchronize`/`Event.elapsed_time`, stack-walk for
   the reference, `SystemExit(0)`, import-time warmup, one-time-correct, shape sniffing, fp16 on an
   fp32 spec), and each must be caught. Second gate: every published RTX PRO 6000 cell replayed
   through it keeps its correctness verdict. Timing shifts get measured and reported, not tuned away.
2. Daemon, API, shims, submission log, HMAC, `grade.json`, official profile with CUPTI.
3. `kb-agent` and `kb-grade` images, the Docker launcher backend and bwrap containment fallback,
   credentials, egress proxy. One real cell per harness family end to end on tetra.
4. Cut over: grader v1 regrades every published cell once (check plus timing) under the new timing
   rules, the boards flip together, the publish gate enforces `grade.json`, and host-mode runs stop
   being publishable.

## Sources

- METR, [Recent frontier models are reward hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking/)
  (o3 patched `torch.cuda.synchronize` and timers, stack-walked for the reference tensor; 43x more
  hacking when the scorer was visible) and [task-protected-scoring](https://github.com/METR/task-protected-scoring).
- Harbor [separate verifier mode](https://github.com/meridianlabs-ai/inspect_harbor/issues/187) and
  [Terminal-Bench mitigations](https://github.com/harbor-framework/terminal-bench/issues/2086).
- NVIDIA [SOL-ExecBench](https://arxiv.org/html/2603.19173v1): subprocess isolation, clock lock,
  L2 clear, input cloning, thread census, function-address checks; 14.5% of agent submissions
  flagged.
- GPU MODE [KernelGuard / pygpubench](https://lacuna.tiptreesystems.com/work/we-let-agents-compete-and-they-tried-to-cheat-kernelguard-defending-gpu/wrk_8addb4ee547c113ab8c7edb2c64b9408):
  seccomp + mseal runner, physics floor, and the admitted in-process limit.
- [CUDA-L1](https://arxiv.org/pdf/2507.14111) (side-stream timing hack in 32.8% of RL kernels) and
  [KernelBench #171](https://github.com/ScalingIntelligence/KernelBench/issues/171) (identity cache).
