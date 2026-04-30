"""Shape sweep for Kahan-corrected softmax.

The point of this problem is numerical accuracy on long reductions. Shapes
mix typical LLM vocab sizes with deliberately adversarial regimes:

  - small vocab (sanity check; naive fp32 should pass)
  - Llama3 vocab 128K (real-world, where fp16 accumulation starts to drift)
  - 256K (DeepSeek-V3 / Gemma-3 class vocab; naive fp16 sum DOES drift past
    the 1e-5 tolerance — this row is what proves Kahan was needed)
  - extreme-logit edge case (large positive logits stress max-subtract +
    summation; if the implementation accidentally exps before subtracting
    max, this row overflows)

The 'extreme' flag is read by check.py to switch input generation to a
distribution that produces a few very large logits per row.
"""

SHAPES = [
    {"batch": 32, "vocab": 4096, "extreme": False},      # sanity
    {"batch": 16, "vocab": 32768, "extreme": False},     # GPT-2 class
    {"batch": 8,  "vocab": 131072, "extreme": False},    # Llama3 vocab
    {"batch": 4,  "vocab": 262144, "extreme": False},    # 256K — Kahan needed
    {"batch": 8,  "vocab": 131072, "extreme": True},     # extreme logits edge
]
