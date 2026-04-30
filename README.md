# kernelbench.com

GPU kernel engineering benchmarks for autonomous LLM coding agents.

- **v-Hard** (latest, 2026-04) — single Blackwell SM120, 7 hand-designed problems, 12 frontier models, real coding-agent CLI harnesses, forensic audit of every high-peak run with rubric leaks documented inline. Source: [github.com/Infatoshi/KernelBench-Hard](https://github.com/Infatoshi/KernelBench-Hard)
- **v3** (archive, 2026-02) — 43-58 problems per GPU across RTX 3090 / H100 / B200, 10 models, 4 difficulty levels, 1500+ evaluations.

## Architecture

Next.js 16 + React 19 + Tailwind v4. Hacker-theme aesthetic (phosphor green on near-black, monospace everywhere, subtle CRT scanlines).

- `app/page.tsx` — landing
- `app/v3/page.tsx` — v3 leaderboard with client-side filterable explorer (CSV under `public/data/v3/`)
- `app/v-hard/page.tsx` — v-hard leaderboard, fetches `leaderboard.json` and YAML annotations from the [KernelBench-Hard repo](https://github.com/Infatoshi/KernelBench-Hard) at build time with hourly ISR
- `lib/data.ts` — data loaders + tiny YAML subset parser

## Local development

```bash
npm install
npm run dev
```

## Deploy

Vercel native GitHub integration auto-deploys on push to `main`. No GHA workflow needed.
