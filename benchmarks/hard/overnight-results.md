# Overnight Results: glm51-claude-droid-sweep

Date: 2026-05-10
Agent: `glm51-claude-droid-sweep`
Repo: `/home/infatoshi/cuda/KernelBench-Hard`

## Lease

- Checked GPU with `nvidia-smi` before work. RTX PRO 6000 was essentially idle.
- Scheduled and acquired `overnight-compute` lease for `glm51-claude-droid-sweep`.
- Heartbeated during the run.

## Harness Fixes

Committed and pushed:

- `2f10fa9 Add direct Z.ai Claude Code harness`

The new `zai-claude` harness routes Claude Code directly to Z.ai's Anthropic-compatible endpoint:

- `ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic`
- `ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY`
- `ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.1`
- `ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.1`
- `ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.5-air`

Important attribution detail: Claude Code used top-level `glm-5.1`, but its internal Explore/tool subagent calls used `glm-4.5-air` through the configured Haiku alias. The transcript records both model names.

## Smoke Tests

Both harnesses completed prompt-response API smoke tests:

- `zai-claude glm-5.1`: `KB_SMOKE_OK`, `session_complete: true`
- `droid custom:GLM-5.1-[Z.AI-Coding-Plan]-0`: `KB_SMOKE_OK`, `session_complete: true`

This confirms the earlier Claude Code issue was endpoint wiring, not a general Z.ai API outage. Droid's existing OpenAI-compatible BYOK route remained valid.

## Real Benchmark Run

Command:

```bash
BUDGET_SECONDS=1800 ./scripts/run_hard.sh zai-claude glm-5.1 problems/04_kahan_softmax
```

Archive:

- `outputs/runs/20260510_000707_zai-claude_glm-5.1_04_kahan_softmax`

Final `result.json`:

- `has_solution: true`
- `correct: false`
- `elapsed_seconds: 1800`
- `harness_exit_code: 124`
- `session_complete: false`
- `peak_fraction: null`

The run timed out at the harness budget while the model was still iterating. It was not an API error or early provider abort.

## Measured Iterations Inside Transcript

The model produced and measured multiple implementations before the timeout:

1. Triton online/Kahan softmax
   - `check.py`: PASS
   - `benchmark.py`: `peak_fraction: 0.0299`, `RESULT: LOW`
   - Diagnosis: one CTA per row underutilized the 188-SM GPU, especially for batch 4/8 large-vocab shapes.

2. Segmented CUDA extension
   - `check.py`: PASS
   - `benchmark.py`: `peak_fraction: 0.0992`, `RESULT: LOW`
   - Shape-level solution bandwidth:
     - shape 0: 72.336 GB/s, fraction 0.0402
     - shape 1: 231.372 GB/s, fraction 0.1285
     - shape 2: 243.064 GB/s, fraction 0.1350
     - shape 3: 171.729 GB/s, fraction 0.0954
     - shape 4: 260.451 GB/s, fraction 0.1447
   - This was just under the `0.1` OK threshold.

3. Fused-only CUDA variant
   - The model changed to always use the fused kernel.
   - The combined `check.py && benchmark.py` command ended with exit code `137` at the end of the harness budget.
   - The harness then marked the run incomplete and failed post-run check.

## Breakpoints

- Z.ai Claude Code endpoint is now functional with `https://api.z.ai/api/anthropic`.
- Claude Code can run GLM-5.1 long enough to implement, test, benchmark, and optimize a real KernelBench problem.
- The failure mode on this bounded run was wall-clock timeout during optimization, not request failure.
- The best measured candidate before timeout was correct and narrowly missed the threshold: `peak_fraction: 0.0992`.
- The final archived solution is the later fused-only experiment, not the best segmented candidate. It should not be scored as the best observed model output without considering the transcript.
- Fresh post-run validation of the archived final solution timed out locally as well, consistent with `load_inline` compile/runtime instability after the final edit.

## Next Steps

- For scoring GLM-5.1 through Claude Code, use the corrected `zai-claude` harness and allow the normal 45-minute budget rather than the 30-minute overnight smoke cap.
- Consider forcing Claude Code's Haiku alias to `glm-5.1` if strict single-model attribution is required, but expect higher cost/latency.
- For this specific problem, preserve the segmented CUDA candidate from the transcript as the useful intermediate; it passed correctness and reached `0.0992`.
- Treat archived final artifacts from incomplete sessions carefully. If `session_complete: false`, prefer transcript-level diagnostics over final `solution.py` scoring.
