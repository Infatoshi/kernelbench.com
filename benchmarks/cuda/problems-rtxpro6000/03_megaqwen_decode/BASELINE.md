# MegaQwen baseline (improve-this skill)

Public repo: https://github.com/Infatoshi/MegaQwen

- Geometry: Qwen3-0.6B (`H=1024`, `I=3072`, GQA 16/8, `D=128`)
- Published ~530 tok/s full-model decode on RTX 3090 (cooperative megakernel,
  `__ldg`, many `grid.sync` — sync-bound, not BW-bound)
- This problem uses **4 stacked blocks** of that geometry (not 28 layers + embed
  + lm_head) so check/benchmark stay practical without HuggingFace downloads

## Protocol

1. **Prefill** to `ctx_len` (untimed) — real KV fill, sequential block steps.
2. **Decode** `decode_steps` tokens at that context (timed only).
3. Context grid: **2k / 8k / 32k / 128k** (`shapes.py`). Score = geomean tok/s.
4. **No tokens.** Numeric `last_hidden` match ⇒ greedy tokens would match.

## Your job

1. Match `reference.py` block semantics for each layer in the stack.
2. Write real CUDA (language gate). Prefer `prefill` + `decode_steps` split.
3. Maximize **decode-only** tok/s at long contexts (not prefill).
4. **Read MegaQwen**, retarget past it on this GPU (cut `grid.sync`, residency,
   SM120). From-scratch allowed; improve-known-baseline is the point.

