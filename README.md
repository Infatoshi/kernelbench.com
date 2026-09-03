# KernelBench

Frontier coding agents write GPU kernels. Each session is one autonomous
agent, graded against a roofline (or ms/speedup) ceiling, then reward-hack
audited before anything is published. Live: [kernelbench.com](https://kernelbench.com).

This monorepo is the website **and** the evals. GPU sessions launch to
Lambda / Brev (or another remote worker). Operator rules: `AGENTS.md`, which
points at a specialized `AGENTS.md` in each directory. Methodology and
history: each bench's `SPEC.md` and `DEVLOG.md`.

## Benches

| bench | path | what | site |
| --- | --- | --- | --- |
| **hard** | `benchmarks/hard/` | per-op kernels (CUDA or Triton), roofline-graded | [/hard](https://kernelbench.com/hard) |
| **mega** | `benchmarks/mega/` | full fused megakernels | [/mega](https://kernelbench.com/mega) |
| **cuda** | `benchmarks/cuda/` | CUDA-only writing deck (Triton/DSL fail) | [/cuda](https://kernelbench.com/cuda) |
| **mini** | `benchmarks/mini/` | small-model (<200B) deck, capped + 5-repeat (WIP) | homepage scroll category on `/` when debuted (not `/mini`) |
| **multi** | `benchmarks/multi/` | 4×H100 NVLink multi-GPU (WIP, frontier roster) | unpublished |

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
`app/AGENTS.md`.

## Layout

```
app/ public/         the website (app/AGENTS.md); app/_lib/data.ts bakes benchmark data at build time
benchmarks/<bench>/  AGENTS.md, SPEC.md, DEVLOG.md, problems, src, scripts, results/, outputs/runs/ (gitignored archives)
scripts/             shared runner (lib/run_harness.sh), workers, publish helpers
kbtool/              the `kb` CLI (uv package); `uv tool install -e ./kbtool` puts `kb` on PATH
media/               tracked chart generators (media/AGENTS.md); PNGs gitignored
runs/                gitignored HF staging from `kb push-runs` (not a third archive)
```

This list is closed. A new thing goes in the directory that owns it. `kbtool/tests` fails on a new top-level dir.

Bench virtualenvs exist only on the GPU boxes that run the benchmark; the
Mac checkout carries `kbtool/.venv` alone.
