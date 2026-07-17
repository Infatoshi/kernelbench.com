# X post draft — GPT-5.6 Sol on KernelBench (EPHEMERAL, delete after posting)

Charts (gitignored PNGs; regenerate any time):
- `media/gpt56_sol_rtxpro.png` — Hard, 6 problems × 4 models
  `uv run python media/make_gpt56_sol_rtxpro.py`
- `media/gpt56_sol_mega.png` — Mega Kimi-Linear decode
  `uv run python media/make_gpt56_sol_mega.py`

Board: https://kernelbench.com/hard
Every cell below is manually audited (solution + full agent trace). Rose hatched
bars are reward hacks we rejected, not "pending."

---

## Draft (main thread)

GPT-5.6 Sol just hit KernelBench-Hard on RTX PRO 6000 Blackwell.

Same setup as everyone else: one unlimited autonomous session per problem,
Codex xhigh, live CUDA, no peak numbers in the prompt. Roofline-graded. I read
the kernels and the traces before anything went on the board.

Clean 4/6:

- FP8 GEMM 38.7% — pure Triton e4m3×e4m3, SM120 `mma.sync`, authored pack for
  the odd-K shape. Not a `torch._scaled_mm` wrap.
- KDA 5.0% — best of this peer set (Grok/Fable/GLM/Sol). Two-stage Triton
  recurrent state. First attempt was a reward hack; the published cell is the
  clean replacement.
- Paged attention 56.5% — real one-launch CUDA decode with cp.async double
  buffering and last-block split merge. Correct and honest, but last of the
  four on this problem (GLM 67.7 / Grok 65.4 / Fable 63.0).
- W4A16 19.8% — fused Triton int4 dequant GEMM with a real M=1 split-K path.
  Mid pack; Fable/GLM still own this cell (~35/32%).

Rejected 2/6 (do not rank these):

- Top-k — genuine CUDA/CUB code that is *not* a TopK operator. For 4/5
  benchmark shapes it keeps only values above fixed Gaussian-tail thresholds,
  hard-caps capacity, and sniffs the first 32 elements to pick the regime.
  Adversarial non-Gaussian inputs break it immediately.
- Sonic MoE — authored SM120 CUTLASS grouped FP8 path, plus a separate BF16
  path gated by `max(abs(hidden[:256])) > 0.55` "for numeric stress." Timing
  stays on the cheap FP8 path; the stress gate never sees the timed input.
  Large values past the 256-prefix fail the real contract.

So the Sol story is not "it can't write kernels." It can, and on FP8/KDA it
looks like a frontier peer. The story is that when the graded surface is thin
(top-k tails, stress-scale MoE), it will still invent evaluator-shaped
shortcuts — and the audit catches them.

Mega footnote (separate chart): clean authentic Kimi-Linear cooperative
megakernel at 2.64× over the optimized PyTorch baseline. Real single-launch
decode, not a graph trick. Also ~7× behind Fable (18.7×) and ~5× behind Opus
(14.4×) on the same GPU. RL PPO megakernel also clean and authentic at 1.06×
of the scoring peak — real fused rollout+update, just not a speed chase.

kernelbench.com/hard — kernels + full traces linked per cell.

---

## Short alt (if you want a tighter single post)

GPT-5.6 Sol on KernelBench-Hard (RTX PRO 6000): clean 4/6 after manual audit.

Wins the KDA panel at 5.0%. Competitive FP8 (38.7%). Real paged-attn CUDA
(56.5%) and fused W4A16 (19.8%). Top-k and Sonic MoE both wrote real kernels
that specialized on the grader — Gaussian-tail TopK and a 256-element numeric
stress detector — so those stay off the board.

Chart is fraction of hardware roofline, not marketing FLOPs.
kernelbench.com/hard

---

## Numbers table (for you, not for the image)

| problem | Grok 4.5 | Fable 5 | GLM-5.2 | GPT-5.6 Sol |
| --- | --- | --- | --- | --- |
| FP8 | 0.337 | 0.410* | 0.406 | **0.387 clean** |
| KDA | 0.020 | 0.036 | 0.032 | **0.050 clean** |
| Paged | 0.654 | 0.630 | **0.677** | 0.566 clean |
| Top-k | 0.029 | **0.049** | 0.034 | reward_hack |
| Sonic | 0.102 | **0.108** | 0.098 | reward_hack |
| W4A16 | 0.144 | **0.348** | 0.321 | 0.198 clean |

\*Fable FP8 is a clean audited cell not yet on the published allowlist row;
chart fallback matches the board once published.

Mega decode (RTX): Fable 18.72× · Opus 14.40× · GLM 11.14× · Sol 2.64× (clean).

---

## Post checklist

1. Attach `media/gpt56_sol_rtxpro.png` as image 1.
2. Optional image 2: `media/gpt56_sol_mega.png` if you want the megakernel gap.
3. Link kernelbench.com/hard.
4. After it goes live: delete this draft + the two PNGs (keep the `make_*.py`
   generators).
