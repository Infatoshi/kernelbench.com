# KernelBench-Multi

4xH100 NVLink multi-GPU bench (unpublished). Agents turn a PyTorch + NCCL reference into a fine-grained NVLink kernel; graded on busbw (01) or geomean speedup vs a frozen production anchor (07/08/09), not single-GPU TFLOPS. **Frontier-only** roster gate at launch (`roster.yaml`), enforced before any GPU time because one session holds four H100s; off-roster needs `KBM_ALLOW_OFF_ROSTER=1` and is archived but never publishable. Deck: `problems-h100x4/`. Methodology: `SPEC.md`. History (the contamination waves, the wave that killed itself, the grade-stack split): `DEVLOG.md`. Publish gates: root `AGENTS.md`.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_allreduce_residual` | pure all-reduce + residual (busbw) |
| 07 | `07_gemm_allreduce_overlap` | TP GEMM overlapped with all-reduce |
| 08 | `08_ring_attention_cp` | context-parallel ring attention |
| 09 | `09_moe_ep_dispatch_combine` | MoE EP dispatch/combine (fp8) |

```bash
cd benchmarks/multi && ./scripts/run_agent.sh grok grok-4.5 01_allreduce_residual high
./scripts/sweep_wave.sh grok grok-4.5 high          # full sequential wave (never parallel sessions: memory, sibling kills, port fights)
```

## Multi deltas

- Own driver `scripts/run_agent.sh` (a fork; harness rows in `kbtool/AGENTS.md`), node-wide GPU lock (`KBM_GPU_LOCK_DIR`), per-run rendezvous ports, and a `pkill`/`killall` wrapper that refuses patterns matching another tenant (`KBM_PROTECTED_PROCS`). Provider keys come from the worker's `~/.kbm_env`.
- Grades run on the bench venv python (`GRADE_PY`), never system `python3`; the in-run grade and the regrade must share the stack.
- The forbidden-op scan covers every agent-authored `.py` in the workspace and treats importing `sota` / `reference` as a failure; the workspace an agent leaves is the workspace that gets graded.
- Anchors (`measure_anchors.py`) refuse a busy node; a contended anchor is frozen and divides every future score on that problem.
- Regrade with `scripts/regrade.py`; annotations in `results/annotations/` (Hard schema plus multi's `speedup_clean` / `fail_honest` / `fail_canonical_stack` fields as used there).

## The 4xH100 node

The graded SKU is 4xH100 SXM behind NVSwitch: every GPU pair shows `NV18` in `nvidia-smi topo -m`. A PCIe or switchless node produces meaningless busbw numbers; `scripts/remote_ceiling.sh` has a topology gate that enforces this. Lambda `gpu_8x_h100_sxm5` (pin `CUDA_VISIBLE_DEVICES=0,1,2,3`) or Brev Nebius 8xH100 match it (`kb lambda ... KB_LAMBDA_BENCH=multi`, Brev via `scripts/brev_worker.sh`; bring-up, pullback, and teardown rules in `kbtool/AGENTS.md`). Bench venv is torch 2.13+cu130; on a stock Lambda image retarget `/usr/lib/x86_64-linux-gnu/libcublas.so` and `libcublasLt.so` to the cuda-13.0 versions and delete any `build_*/` extension dir built before the retarget (the bad `NEEDED libcublas.so.12` is baked in at link time; `ldd` the artifact). Validate correctness for free on a single GPU first via gloo+cpu (`KBM_BACKEND=gloo KBM_DEVICE=cpu KBM_WORLD_SIZE=4 python check.py`); the rented node should never see a correctness bug for the first time. Contamination scan reads `~/.grok/sessions/*/chat_history.jsonl`; scrub session stores, smoke dirs, and launcher logs between waves, since an empty archive is not an empty box.

## `KBM_` environment variables

`kbtool/tests` fails on a variable read by code that no AGENTS.md documents. Danger list in `kbtool/AGENTS.md`.

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KBM_ALLOW_BUSY` | `benchmarks/multi/scripts/{regrade,measure_anchors}.py` | `0` | Skips the quiet-node precondition for regrades or frozen anchor measurements. | Can contaminate published timings; value must be `1`. |
| `KBM_ALLOW_DEVICE_MISMATCH` | `benchmarks/multi/src/eval/worker.py` | `0` | Allows heterogeneous ranks or a GPU name that does not match the problem hardware key. | Deliberate off-SKU experiments only. |
| `KBM_ALLOW_OFF_ROSTER` | `benchmarks/multi/scripts/sweep_wave.sh` | `0` | Bypasses the frontier roster launch gate. | Result is exploratory and not publishable until rostered. |
| `KBM_ANCHOR_REPEATS` | `benchmarks/multi/scripts/measure_anchors.py` | `5` | Sets the number of frozen-anchor measurements. | Changes anchor statistics used by speedup scores. |
| `KBM_BACKEND` | `benchmarks/multi/src/eval/{launcher,worker}.py`; `benchmarks/multi/scripts/numerics_probe.py` | `nccl` | Selects the distributed backend. | `gloo` is for local CPU validation, not official GPU scores. |
| `KBM_CU` | `benchmarks/multi/scripts/remote_ceiling.sh` | `cu128` | Selects the CUDA-tagged PyTorch wheel for a remote ceiling run. | Runtime/toolchain control. |
| `KBM_DEVICE` | `benchmarks/multi/src/eval/worker.py` | CUDA local rank | Forces CPU when set to `cpu`. | Local correctness smoke only with `gloo`. |
| `KBM_GPU_LOCK_DIR` | `benchmarks/multi/scripts/run_agent.sh` | `<multi>/outputs/gpu_lock` | Selects the node-wide lock domain for all GPU-facing agent commands. | A wrong domain permits 4-GPU fabric contention. |
| `KBM_GPU_LOCK_HELD` | generated wrappers in `benchmarks/multi/scripts/run_agent.sh` | `0` | Makes GPU wrappers reentrant when a parent already owns the lock. | Set internally; manual `1` bypasses serialization. |
| `KBM_ITERS` | `benchmarks/multi/src/eval/worker.py`; `benchmarks/multi/scripts/{nccl_ceiling.py,remote_ceiling.sh}` | problem `num_perf_trials` or `100`; ceiling `50`/remote `50` | Sets measured performance iterations. | Changes timing statistics. |
| `KBM_MASTER_PORT` | `benchmarks/multi/src/eval/launcher.py`; set by `scripts/run_agent.sh` | `29571`; runner assigns a per-run port in `29600..29999` | Selects the local torchrun rendezvous port. | Runner-generated to prevent sibling collisions. |
| `KBM_NUMERIC_STRESS` | `benchmarks/multi/src/eval/stress.py` | `1` | Disables scaled-input stress cases when `0`. | Never disable for official runs. |
| `KBM_OR_CONTEXT` | `benchmarks/multi/scripts/run_agent.sh` | `262144` | Sets the advertised OpenRouter model context limit in the generated OpenCode config. | `opencode-or` only. |
| `KBM_OR_PROVIDER` | same | `Moonshot AI` | Pins the OpenRouter serving provider with fallbacks disabled. | `opencode-or` only; changes serving stack. |
| `KBM_PROTECTED_PROCS` | same | `vllm\|sglang\|trtllm\|nanbeige\|laguna\|dspark\|demon/harness` | Replaces the regex used by wrapped `pkill`/`killall` to protect other tenants. | Safety boundary on shared nodes. |
| `KBM_QUIET_MB` | `benchmarks/multi/scripts/wait_quiet.sh` | `2048` MiB per GPU | Sets the memory threshold below which a GPU counts as quiet. | Used before long waves. |
| `KBM_QUIET_MINUTES` | same | `10` | Sets the required sustained quiet interval. | Samples once per minute. |
| `KBM_QUIET_TIMEOUT_MINUTES` | same | `0` (wait forever) | Sets how long the quiet watcher waits before failing. | Operational only. |
| `KBM_SKIP_FORBIDDEN` | `benchmarks/multi/src/eval/launcher.py` | `0` | Skips the forbidden-import/source tripwire when `1`. | Debug only; changes correctness policy. |
| `KBM_SKIP_GRADE` | `benchmarks/multi/scripts/run_agent.sh` | `0` | Runs the agent but omits post-session check and benchmark when `1`. | Launch-only mode; no publishable score until separately graded. |
| `KBM_TRIALS` | `benchmarks/multi/src/eval/worker.py` | problem `num_correct_trials` or `5` | Sets the number of correctness trials. | Changes validation strength. |
| `KBM_WARMUP` | `benchmarks/multi/src/eval/worker.py`; `benchmarks/multi/scripts/{nccl_ceiling.py,remote_ceiling.sh}` | problem `num_warmup` or `500`; ceiling `200`/remote `200` | Sets warm-up iterations before timing. | Changes timing methodology. |
| `KBM_WORLD_SIZE` | `benchmarks/multi/src/eval/launcher.py` | problem `world_size` or `4` | Overrides torchrun process count. | Official problems expect their declared fabric size. |
