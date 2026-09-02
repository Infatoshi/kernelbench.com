# KernelBench-Mini

Small-model (< 200B open-weight) kernel bench on a **fresh 4-problem deck**.
WIP / **unpublished** — do not post the deck or results publicly until debut.

| | |
| --- | --- |
| Deck | `problems-h100/` |
| GPU | H100 SXM (`H100_SXM` peaks), typically Lambda `gpu_1x_h100_sxm5` |
| Session cap | 30 min (`BUDGET_SECONDS=1800`) |
| Cell | (model, harness, problem) x **5** repeats → pass rate k/5 + best-of-5 |
| Docs | **`SPEC.md`** = methodology + operator runbook (start here) · `DEVLOG.md` = history · root `AGENTS.md` · `docs/HARNESSES.md` · `docs/ENV.md` |

| NN | problem | language | note |
| --- | --- | --- | --- |
| 01 | `01_dequant_gemv` | Triton ok | vibe-check prompt; int4 gated GEMV, group size 96 |
| 02 | `02_segmented_decay_scan` | Triton ok | decay scan with per-token resets |
| 03 | `03_topp_mask` | CUDA-only | sort-free nucleus mask; ms-anchored |
| 04 | `04_flash_attention` | CUDA-only | causal flash forward; S up to 16384 |

## Commands (cwd = `benchmarks/mini`)

```bash
# one column: 4 problems x 5 repeats
./scripts/sweep_mini.sh opencode <model>
./scripts/sweep_mini.sh <provider>-claude <model>

# one session
./scripts/run_hard.sh opencode <model> problems-h100/01_dequant_gemv
# or from anywhere:
kb -b mini run opencode <model> 01_dequant_gemv

# local-vLLM subject (all five LFM harnesses) — server already up on :8765
./scripts/launch_matrix.sh lfm25-agent-bf16

# after the wave: stop vLLM/ccr, then clean grades only
./scripts/regrade_sequential.sh outputs/runs/<run_id> ...
```

**Use `sweep_mini.sh` / `launch_matrix.sh`.** Ignore hard-shaped
`sweep.sh` / `launch_parallel_sweep.sh` unless debugging those copies.

## Architecture in one line

API models → provider APIs + eval H100. API-less models → **vLLM co-hosted on
the same eval H100** (`localhost:8765`, ~35% mem); publish-grade numbers still
require a **sequential re-grade with the server stopped**. Full runbook,
calibration debts, and missing codex-anchor status: **`SPEC.md`**.

## Status snapshot

- First subject (LFM2.5-2.6B-Agent, bf16 + NVFP4, 200 sessions): **0 correct** —
  see DEVLOG. Campaign complete; node torn down.
- Frontier solvability anchor: **must re-run before debut** (archive lost with
  lease nodes).
- No homepage Mini deck section and no published leaderboard until debut (site surface is a scroll category on `/`, not a `/mini` route).
