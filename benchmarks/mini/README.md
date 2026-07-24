# KernelBench-Mini

Small-model (< 200B) kernel bench: capped 30-minute sessions, 5 repeats per
cell, fresh 4-problem deck, canonical GPU = Lambda H100 SXM5.

| NN | problem | language | note |
| --- | --- | --- | --- |
| 01 | `01_dequant_gemv` | Triton ok | vibe-check prompt; int4 gated GEMV, group size 96 |
| 02 | `02_segmented_decay_scan` | Triton ok | decay scan with per-token resets |
| 03 | `03_topp_mask` | CUDA-only | sort-free nucleus mask, exact fp64-oracle grading, ms-anchored |
| 04 | `04_flash_attention` | CUDA-only | full causal flash forward, S up to 16384 |

See `SPEC.md` for methodology, `DEVLOG.md` for the journey, and the repo-root
`AGENTS.md` for operator commands.

```bash
./scripts/sweep_mini.sh opencode <model>   # 4 problems x 5 repeats
```
