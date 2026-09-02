# KernelBench

Frontier coding agents write GPU kernels. Each session is one autonomous
agent, graded against a roofline (or ms/speedup) ceiling, then reward-hack
audited before anything is published. Live: [kernelbench.com](https://kernelbench.com).

This monorepo is the website **and** the evals. GPU sessions launch to
Lambda / Brev (or another remote worker). Operator workflow: `AGENTS.md`.
Methodology and history: each bench's `SPEC.md` and `DEVLOG.md`.

## Benches

| bench | path | what | site |
| --- | --- | --- | --- |
| **hard** | `benchmarks/hard/` | per-op kernels (CUDA or Triton), roofline-graded | [/hard](https://kernelbench.com/hard) |
| **mega** | `benchmarks/mega/` | full fused megakernels | [/mega](https://kernelbench.com/mega) |
| **cuda** | `benchmarks/cuda/` | CUDA-only writing deck (Triton/DSL fail) | [/cuda](https://kernelbench.com/cuda) |
| **mini** | `benchmarks/mini/` | small-model (<200B) deck, capped + 5-repeat (WIP) | homepage scroll category on `/` when debuted (not `/mini`) |
| **multi** | `benchmarks/multi/` | 4×H100 NVLink multi-GPU (WIP, frontier roster) | unpublished |
| **v3** | `benchmarks/v3/` | offline archive (separate harness) | not on site |

Hard / mega / cuda share harness machinery and run unlimited wall-clock
(`BUDGET_SECONDS=0`). Mini is capped; multi is sequential on a 4-GPU node.

## Website (local)

Next.js 16 + Tailwind. Package manager is **bun** (`bun.lock`):

```bash
bun install
bun dev          # http://localhost:3000
bun run build
```

Site data is baked at build time from `benchmarks/*/results/`
(`app/_lib/data.ts`). Publish/deploy: `kb publish` then `kb deploy` — see
`AGENTS.md`.

## Layout

```
app/ public/         the website; app/_lib/data.ts bakes benchmark data at build time
benchmarks/<bench>/  problems, src (eval, hardware, viewer), scripts, results/, outputs/runs/ (gitignored archives)
scripts/lib/         shared single-GPU runner (run_harness.sh) that hard/cuda/mini wrap
kbtool/              the `kb` CLI (uv package); bin/kb shims it
docs/                REMOTE, HARNESSES, ENV, TORCH, POST, ARTICLE
environments/        Prime Intellect `verifiers` mirrors (kernel_hard / kernel_mega / kernel_v3)
media/               tracked chart generators (kbh_theme.py, make_*.py, thumb_card.py); PNGs gitignored
runs/                gitignored HF staging filled by kb publish
```

## Docs map

- `AGENTS.md` — entrypoint: rules, publish gates, pointers (under 10 KB)
- `docs/REMOTE.md` — rented GPU workers (Lambda, Brev, Verda)
- `docs/HARNESSES.md`, `docs/ENV.md`, `docs/TORCH.md` — harness, env var, torch references
- `docs/POST.md`, `docs/ARTICLE.md` — posting results
- `benchmarks/<bench>/SPEC.md` — methodology
- `benchmarks/<bench>/DEVLOG.md` — design history
- `benchmarks/<bench>/README.md` — short human entry for that deck
