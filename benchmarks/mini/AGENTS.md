# KernelBench-Mini

Small-model (< 200B open-weight) kernel bench on a fresh 4-problem deck. WIP and **unpublished**: do not post the deck or results publicly until debut. Site surface at debut is a homepage scroll category on `/`, not a `/mini` route.

| | |
| --- | --- |
| Deck | `problems-h100/` |
| GPU | H100 SXM (`H100_SXM` peaks), typically Lambda `gpu_1x_h100_sxm5` |
| Session cap | 30 min (`BUDGET_SECONDS=1800`); Mini measures best kernel in 30 minutes and is never compared against unlimited-time Hard |
| Cell | (model, harness, problem) x **5** repeats -> pass rate k/5 + best-of-5 |
| Docs | **`SPEC.md`** = methodology + operator runbook (start here) · `DEVLOG.md` = history |

| NN | problem | language | note |
| --- | --- | --- | --- |
| 01 | `01_dequant_gemv` | Triton ok | vibe-check prompt; int4 gated GEMV, group size 96 |
| 02 | `02_segmented_decay_scan` | Triton ok | decay scan with per-token resets |
| 03 | `03_topp_mask` | CUDA-only | sort-free nucleus mask; ms-anchored |
| 04 | `04_flash_attention` | CUDA-only | causal flash forward; S up to 16384 |

## Commands (cwd = `benchmarks/mini`)

```bash
./scripts/sweep_mini.sh opencode <model>            # one column: 4 problems x 5 repeats
./scripts/sweep_mini.sh <provider>-claude <model>
./scripts/run_hard.sh opencode <model> problems-h100/01_dequant_gemv   # one session
kb -b mini run opencode <model> 01_dequant_gemv      # or from anywhere
./scripts/launch_matrix.sh lfm25-agent-bf16          # local-vLLM subject (all five LFM harnesses), server already up on :8765
./scripts/regrade_sequential.sh outputs/runs/<run_id> ...   # after the wave: stop vLLM/ccr, then clean grades only
```

Use `sweep_mini.sh` / `launch_matrix.sh`. Ignore hard-shaped `sweep.sh` / `launch_parallel_sweep.sh` unless debugging those copies.

Architecture: API models -> provider APIs + eval H100. API-less models -> vLLM co-hosted on the same eval H100 (`localhost:8765`, ~35% mem); publish-grade numbers still require a sequential re-grade with the server stopped. Full runbook, calibration debts, and the missing codex-anchor status: `SPEC.md`.

Status: first subject (LFM2.5-2.6B-Agent, bf16 + NVFP4, 200 sessions) scored 0 correct (see DEVLOG); campaign complete, node torn down. The frontier solvability anchor must be re-run before debut (archive lost with the lease nodes).

Layout, correctness, results, torch policy, audit YAML schema, `KBH_` variables: `benchmarks/hard/AGENTS.md`. Harness routes (`lfm-*`, `hermes`, `pi`), the shared runner, rented workers: `kbtool/AGENTS.md`. Publish gates: root `AGENTS.md`.

## `KBMINI_` environment variables

`kbtool/tests` fails on a variable read by code that no AGENTS.md documents.

| Var | Read by (paths) | Default | What it changes | Notes |
| --- | --- | --- | --- | --- |
| `KBMINI_API_KEY` | `benchmarks/mini/scripts/run_hard.sh` | `local` | Supplies the placeholder/auth key to local LFM OpenAI-compatible routes. | Used by `lfm-opencode`, `lfm-claude`, `hermes`, `pi`, and `lfm-grok`. |
| `KBMINI_BASE_URL` | same | `http://127.0.0.1:8765/v1` | Selects the local OpenAI-compatible inference server. | vLLM is co-hosted on the eval node. |
| `KBMINI_GPUS` | `benchmarks/mini/scripts/launch_matrix.sh` | `0` | Supplies the space-separated GPU IDs used to place matrix workers round-robin. | Each GPU receives its own lock directory. |
| `KBMINI_HERMES_MAX_TURNS` | `benchmarks/mini/scripts/run_hard.sh` | `1000` | Caps Hermes agent turns. | `hermes` only. |
| `KBMINI_HERMES_MODEL` | same | runner model argument | Overrides the model passed to Hermes. | `hermes` only. |
| `KBMINI_HERMES_PROVIDER` | same | `lfm` | Overrides the Hermes provider name. | `hermes` only; endpoint still comes from `KBMINI_BASE_URL`. |
| `KBMINI_PI_API_KEY` | inline Python in `benchmarks/mini/scripts/run_hard.sh` | forced from `KBMINI_API_KEY` | Transfers the local API key into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PI_BASE_URL` | same | forced from `KBMINI_BASE_URL` | Transfers the local endpoint into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PI_MODEL` | same | forced from the runner model argument | Transfers the model ID into pi's generated provider JSON. | Internal one-command environment bridge; a caller-supplied value is overwritten. |
| `KBMINI_PROBLEMS` | `benchmarks/mini/scripts/{sweep_mini,launch_matrix}.sh` | four `problems-h100/*` Mini problems | Replaces the Mini problem list. | Changes deck coverage. |
| `KBMINI_REPEATS` | `benchmarks/mini/scripts/sweep_mini.sh` | `5` | Sets repeats per model/harness/problem cell. | Changes pass-rate and best-of-N methodology. |
| `KBMINI_SPLIT_BY_PROBLEM` | `benchmarks/mini/scripts/launch_matrix.sh` | `0` | Launches one worker per `(harness, problem)` instead of one per harness when `1`. | Changes concurrency, not the intended result set. |

`KBMINI_PI_API_KEY`, `KBMINI_PI_BASE_URL`, and `KBMINI_PI_MODEL` are private pass-through names populated by `run_hard.sh`; callers configure `KBMINI_API_KEY`, `KBMINI_BASE_URL`, and the runner model argument instead.
