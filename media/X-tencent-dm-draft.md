# X DM draft — Tencent contact (EPHEMERAL, delete after sending)

Attach: media/hy3_debut.png (regenerate: `cd media && uv run --with matplotlib --with numpy python make_hy3_debut.py`)

---

Hey — results are in. We put Hy3 (preview) on KernelBench-Hard: frontier
coding agents get one unlimited-time autonomous session per problem to write
CUDA/Triton kernels that beat SOTA libraries, roofline-graded on real
hardware. Hy3 ran the full deck on RTX PRO 6000 Blackwell (SM120) and H100,
head-to-head with Claude Fable 5 / Opus 4.8 / GPT-5.5 and GLM-5.2 / Kimi K2.7 /
DeepSeek V4 Pro. Chart attached; every passing cell was manually audited
(solution + full agent trace), and Hy3's cells are all real kernels — zero
reward hacks.

Best cell: fp8 GEMM on SM120 at 0.327 of roofline — a real Triton
tensor-core kernel (fp8e4nv tl.dot, fp32 accum, per-channel scales), and it
independently worked out the same K-padding trick to dodge SM120's
tail-predication cliff that other frontier models found. That's within 0.06
of Opus 4.8 on the same problem. Full trace:
https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces/blob/main/20260706_182954_hy3-claude_tencent_hy3-preview_01_fp8_gemm.jsonl

The honest read on the rest: the model's kernel instincts are real, but most
failed cells died on serving infrastructure, not capability. We could only
reach Hy3 through OpenRouter, which caps it at 262K context with 128K
reserved for output — several sessions got killed by that 400 mid-fix. The
clearest one: top-k on H100 was literally one edit from correct (the agent
had already diagnosed its own bug — "extraction step is producing ascending
order instead of descending") when the context cap ended the session:
https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces/blob/main/20260707_104519_hy3-claude_tencent_hy3-preview_05_topk_bitonic.jsonl
With a first-party Anthropic-compatible endpoint (like Kimi/GLM/MiniMax run)
and the full context window, I'd expect the board to look meaningfully
better — happy to rerun the failed cells the moment there's an endpoint I
can point Claude Code at.

Everything is public: kernelbench.com/hard has every cell with the kernel,
the audit verdict, and the full agent trace linked. Would love to talk about
what a sponsorship collab could look like — rerunning Hy3 on a first-party
endpoint at full context would be the natural first piece.

---

Notes for Elliot (not part of the DM):
- The two trace links are live (verified via HF resolve).
- Hy3 traces were missing from HF until today (parser bug, fixed + pushed).
- If they offer a direct endpoint, we add a hy3-claude variant pointing at it
  in run_hard.sh — same pattern as kimi-claude, ~10 min of work.
- CONCRETE ASK for the DM/call: a Tencent Cloud TokenHub API key. TokenHub has
  an official Anthropic-compatible Claude Code route (ANTHROPIC_BASE_URL=
  https://tokenhub.tencentmaas.com, ANTHROPIC_AUTH_TOKEN=<key>, model
  hy3-preview / hy3) with first-party serving. Our OpenRouter runs were 13/14
  served by GMICloud (declares bf16, so quantization probably isn't the issue)
  but GMICloud produced the 400s and empty-200 responses that killed sessions.
  TokenHub also unlocks `hy3` (non-preview) which OpenRouter doesn't list.
