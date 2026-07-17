# X post draft — Tencent Hy3 (preview) debut (EPHEMERAL, delete after posting)

Chart: media/hy3_debut.png (regenerate: `uv run python make_hy3_debut.py`)

---

New on KernelBench-Hard: Tencent Hy3 (preview), swept on RTX PRO 6000 + H100.
One unlimited autonomous session per problem, every cell manually reward-hack
audited (solution + full trace). It passes 3/6 on RTX and 4/6 on H100, and the
story is consistent: real kernels, no optimization pass, death by infrastructure.

OpenRouter caps it at 262K context with 128K reserved for output; multiple
cells died on that 400 mid-fix — top-k was ONE edit from correct (it had
already diagnosed its own bug: "extraction step is producing ascending order
instead of descending"). Its H100 KDA cell is a leaderboard first: an honest
eager-PyTorch verification scaffold, correct at peak_fraction 0.0000, docstring
promising "Triton optimizations after verification" that the provider never let
it write. Its RTX sonic-MoE kernel is honestly slow in one line: grid=(E,) —
128 CTAs on a 188-SM GPU, 31 iterations over 3 hours, never widened the grid.

Where it lands vs the tier: RTX fp8 0.327, within 0.002 of LongCat's debut and
independently discovering the same K-padding trick to dodge SM120's
tail-predication cliff — but still behind Opus (0.386) and GLM (0.406).

The audit fine print (both directions):
- Its RTX top-k DQ has an asterisk: it wrapped torch::sort inside a C++
  extension and said in its trace that check.py "only checks solution.py" so it
  "won't be detected." The forbidden-op grep caught it anyway.
- Its RTX KDA DQ is the opposite — a false positive: the scan tripped on its
  own function named _chunk_kda_forward over a pure-PyTorch reimplementation
  that used no forbidden library at all.

Mega footnote: on KernelBench-Mega's Kimi-Linear decode megakernel it got a
clean correct pass at 0.31x — 3x SLOWER than baseline, killed by the 45-min cap
at literally "Let me measure performance with benchmark.py", zero optimization
passes. The per-problem unlimited format is where this model shows signal.

Boards: kernelbench.com/hard. Kernels + full agent traces linked per cell.
