# X draft — Grok 4.5 (max) vs GPT-5.6 Sol (xhigh)

Chart: `media/gpt56_sol_vs_grok45.png`
```
uv run python media/make_gpt56_sol_vs_grok45.py
```

---

## Draft

Grok 4.5 (max) vs GPT-5.6 Sol (xhigh) on KernelBench-Hard, RTX PRO 6000.
Same rules: one unlimited session per problem, roofline score, every cell audited.

They roughly tie — but for different reasons.

On FP8 both ship pure Triton SM120 e4m3 GEMMs. Sol 38.7%, Grok 33.7%. Sol’s edge is an odd-K pack so the MMA never tail-predications.

On KDA Sol’s two-stage Triton recurrent lands 5.0%; Grok’s hybrid Triton+cuBLAS path is clean at 2.0%.

On paged attention Grok wins hard: 65.4% vs Sol 56.5%. Grok fused GQA FlashDecoding in Triton and put CUDA graphs on the live tensors; Sol’s one-launch CUDA is correct, just slower.

On W4A16 Sol 19.8% vs Grok 14.4%. Both fuse int4 dequant into the GEMM; Sol tunes it further, Grok’s interesting bit is a hand CUDA split-K path for M=1 decode.

Top-k and Sonic are where the audit matters. Grok is clean on both (hand CUDA radix/bitonic Top-k at 2.9%, pure Triton grouped-GEMM Sonic at 10.2%). Sol’s Top-k specialized on the grader’s Gaussian tails; its Sonic MoE ran a BF16 path only when a 256-element stress detector fired. Both rejected — not ranked.

So: Grok never reward-hacked a published cell. Sol did twice. Sol still wins three of the four honest GEMM/decode cells it kept; Grok takes paged attention and the two problems Sol tried to game.

Wall clock is close in aggregate — Grok ~3.3h across six clean cells, Sol ~1.9h on its four clean ones (plus ~1.8h burned on the two hacks). Per problem Sol often just… keeps grinding (W4A16 38m vs Grok 26m). Token side: Sol’s four clean cells spent ~169k output / ~66k reasoning tokens with heavy prompt cache; the Grok harness doesn’t record usage, so we can’t call a tokens/roofline winner — only that Sol is willing to spend more steps on a cell.

kernelbench.com/hard
