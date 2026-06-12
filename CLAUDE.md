# kernelbench.com — operator guide

This is the **canonical monorepo** for the KernelBench website AND the eval
benchmarks. It lives on Anvil at `~/kernelbench.com` (the GPU box, where evals
run). The Mac has a secondary clone you `git pull` when working there; Anvil is
canonical. Deploys go out from Anvil.

## Layout

```
app/ lib/ public/          the website (Next.js 16, Tailwind v4)
  public/data/v3/results.csv   /v3 reads this
benchmarks/hard/           KernelBench-Hard eval — runs here
  results/leaderboard.json     /hard reads this (v2 site-shaped data)
  results/annotations/*.yaml   per-cell reward-hack / clean verdicts
  outputs/runs/                run archives (gitignored; ~186G)
  scripts/  src/  problems/    eval code
benchmarks/v3/             KernelBench-v3 eval
justfile                   `just` recipes (see below)
```

## Run a sweep (the common task) — use the `kb` CLI (on PATH, runs from any cwd)

```
kb sweep kimi-claude kimi-k2.7-code       # all 6 problems, parallel containers, 2700s
kb publish                                # rebuild leaderboard + viewers from archives
kb deploy "bench kimi k2.7"               # publish + commit + push (Vercel auto-builds)
```
Other commands: `kb run <harness> <model> <problem>` (one problem), `kb dev`
(preview, view from Mac via Tailscale anvil:3000), `kb build`, `kb audit <run_id>`,
`kb help`. The CLI lives at `bin/kb` (symlinked to ~/.local/bin/kb).

- API keys live in `~/.env_vars` (KIMI_API_KEY, ZAI_API_KEY, MINIMAX_API_KEY,
  OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, CLAUDE_CODE_OAUTH_TOKEN).
  To bench a new model: drop its key in `~/.env_vars`, then `kb sweep`.
- **If `kb sweep` prints `STOP: ... needs $X_API_KEY`**, the key is missing —
  ask the human for it, append `export X_API_KEY=...` to `~/.env_vars`, rerun.
  `kb` preflights the key before launching so you get one clear message, not
  six failed runs.
- Benching a **brand-new provider** (no harness yet) needs a harness branch:
  copy the `kimi-claude` block in `scripts/run_hard.sh` (Claude-Code → the
  provider's Anthropic-compatible endpoint) and add the row. Not a one-liner.
- Before a GPU sweep: `nvidia-smi` (the box is shared).

## Harnesses (run via `uv run kbh run <harness> <model> <problem> [effort]`)

- Native CLIs: `claude`, `codex`, `cursor`, `gemini`, `grok`, `opencode`.
- Claude-Code-routed providers (most reliable): `zai-claude` (GLM via
  api.z.ai/api/anthropic), `minimax-claude`, `kimi-claude` (Kimi via
  api.moonshot.ai/anthropic, model `kimi-k2.7-code`), `deepseek-claude`
  (DeepSeek via api.deepseek.com/anthropic, model `deepseek-v4-pro` or
  `deepseek-v4-flash`), `qwen-claude` (Qwen via DashScope Model Studio
  Intl, dashscope-intl.aliyuncs.com/apps/anthropic, model `qwen3-max` —
  needs DASHSCOPE_API_KEY, which we do not have yet). These mirror each other;
  to add one, copy the `kimi-claude` branch in `scripts/run_hard.sh`.
  Rationale: opencode is a strong harness but its `@ai-sdk/openai-compatible`
  transport stalls intermittently (~1/3-1/2 of sessions); routing these models
  through Claude Code to the provider Anthropic endpoint bypasses that adapter.
- Always container mode (`KBH_AGENT_CONTAINER=1`): isolated per-run workspace,
  native GPU, sessions overlap while GPU commands serialize through the lock.

## Hard-won gotchas

- **Commit email MUST be `elliot@arledge.net`** or Vercel silently fails the
  build verification. The repo sets it locally; new clones must `git config
  user.email elliot@arledge.net`.
- The publish pipeline regenerates `benchmarks/hard/results/leaderboard.json`
  (site data) and `public/runs/*.html` (transcript viewers) from the archives.
  `just publish` does it; don't hand-edit the leaderboard.
- Reward-hack verdicts come from `results/annotations/<run_id>.yaml`; every
  passing/failing headline cell should be audited (read the solution.py) before
  publishing. The template-mutation guard auto-flags grader tampering.
- `kb lint <run_id|--all>` is a static reward-hack TRIPWIRE: scans solution.py
  for the patterns we have caught by hand (input-identity memoization,
  stack/check.py sniffing, backend mutation, zero-kernel wrappers) plus surfaces
  template_mutated from result.json. HACK = near-certain, FLAG = review. It is a
  pointer for a human audit, NOT an auto-reject. Run it before publishing.
- `04_kahan_softmax` was removed from the deck (rewarded skipping Kahan); do not
  re-add.
- See `benchmarks/hard/DEVLOG.md` for the full journey and `SPEC.md` for
  methodology.
