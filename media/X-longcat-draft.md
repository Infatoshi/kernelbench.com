# X post draft — Meituan LongCat 2.0 debut (EPHEMERAL, delete after posting)

Chart: media/longcat_debut.png (regenerate: `uv run python make_longcat_debut.py`)

---

New on KernelBench-Hard: Meituan LongCat-2.0, swept on RTX PRO 6000 + H100.
One unlimited autonomous session per problem, every passing cell manually
reward-hack audited (solution + full trace). Zero reward hacks. It debuts at
5/6 passing on RTX and 6/6 on H100 — the only model outside the Claude frontier
tier to complete the full H100 deck.

Its best cell is paged attention (0.319 RTX / 0.242 H100): on H100 it wrote a
split-K FlashDecoding kernel, but on SM120 it explicitly reasoned "seq_len is
modest, no split-K needed" and spent the parallelism budget on GQA KV reuse
instead — one CTA per (batch, kv_head), each KV tile loaded once and dotted
against all query heads in the group. The Llama-3-8B decode shape hits 1.35
TB/s, 75% of GDDR7 theoretical. It also worked out on its own that check.py
warms the Triton autotune cache before benchmark.py, and shipped the
warm-cache-verified build. Grader-environment savvy without a single grader edit.

The wildest trace: its sonic-MoE session ran with a completely dead shell — an
infra cascade on our side killed every subprocess it launched, even /bin/true.
It shipped a grouped-GEMM + fused-SwiGLU Triton kernel BLIND in five file
writes, wrapped in a first-forward self-check that falls back to a reference
loop on mismatch. We verified empirically: the Triton path is what scores, the
fallback never fires, and the blind kernel landed within 12% of cuBLAS.

The H100 sweep added more character notes. On top-k it measured the deck is
launch-overhead-bound, reached for CUDA graphs, discovered graph capture in the
container returned zeroed buffers — the exact stale-output pattern our audits
hunt for — recognized it as wrong, and walked away on its own. On sonic-MoE it
dumped compiled PTX and counted wgmma instructions to prove tensor-core
lowering, then closed with an unusually honest final report: "FAILED to hit the
0.1 geomean target ... RESULT: LOW." And w4a16 shows the flip side of its
7-hour grinds: it nailed an even/odd nibble-split dequant layout in its third
message, hit 0.122 (beating GLM-5.2's 0.083), declared a self-invented ">= 0.10
target" met, and quit at 1.5h. A satisficer when things go well, a climber when
they don't.

vs the tier: H100 fp8 0.218 beats GLM-5.2 (0.078), just under Opus (0.245).
RTX fp8 0.329 within 0.06 of Opus. Paged attention stays the frontier
separator: incumbents 0.63-0.68 on RTX, LongCat 0.319.

Boards: kernelbench.com/hard. Kernels + full agent traces linked per cell.

Caveat: its RTX KDA cell (0.0007) under-measures the model — that was the
dead-shell session — and gets a rerun.

Mega footnote: on KernelBench-Mega's Kimi-Linear decode megakernel (45-min
cap) its session hit the cap mid-write — no Model class in the file yet. It
needed the unlimited-time format to show anything.
