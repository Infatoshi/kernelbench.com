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
- **Transcript / reasoning extraction lives in-repo at
  `scripts/transcript-extraction/`** (vendored complete extractor; see its
  `VENDORED.md`). Use it as the canonical reference when working on the
  agent-timeline viewers — it pulls full conversations (messages, tool use,
  diffs, reasoning) across every harness format (codex / claude-code / cursor /
  gemini / opencode / …), more complete than the per-bench
  `src/viewer/parsers/*` (which under-extract — an opus transcript with 216
  thinking blocks, but 107 are signature-only). The viewer renders reasoning
  untruncated (`src/viewer/html.py` reasoning-block). **What's actually
  exposable depends on the provider, not the parser:** native `claude` (opus)
  and `codex` (gpt-5.5) ENCRYPT their chain-of-thought — the transcript carries
  empty `thinking` blocks with a `signature` (Anthropic extended thinking) or
  sparse summaries (codex), so there is nothing to render. The `*-claude` routes
  to open providers (glm/zai, kimi, deepseek, minimax) return FULL thinking text
  and now render in full. So a near-empty reasoning trace for opus/codex is the
  API encrypting it, not a viewer bug.
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

## Publishing results: charts + write-ups (REQUIRED format)

When you post benchmark results (X posts, blog, threads), two rules are not
optional. They are what makes a post read as signal, not slop.

- **Charts MUST use the website NVIDIA palette.** Import the shared theme,
  never hardcode colors: `from kbh_theme import C, SERIES, apply` (module at
  `x-article-images/kbh_theme.py`, mirrored on Mac at
  `~/dev/sites/kernelbench.com/x-article-images/`). It copies the `:root`
  tokens from `app/globals.css`: bg `#111111`, accent (NVIDIA green) `#76b900`,
  fg `#eeeeee`/`#999999`, warn `#fbbf24`, bad `#fb7185`, grid `#242424`. Lead
  bars with the green accent (the ceiling/subject); rose `#fb7185` = reward
  hack (hatched), amber = warn, grey = fail, faded+dotted = real kernel that
  bugged/timed out. `x-article-images/make_glm52_4way.py` is the canonical
  example. If `globals.css` changes, update `kbh_theme.py` to match. Charts are
  generated on Mac (matplotlib) and dragged into posts; PNGs are gitignored,
  the `.py` scripts are tracked.
- **Write-ups lead with the unique/interesting/inconsistent, not adjectives.**
  "4/6 clean, strong showing, solid 2nd place" is filler. Before writing,
  actually read 2-3 transcripts/solutions for the headline cells and surface
  concrete findings: behavioral shifts (e.g. a model that stopped reward-hacking
  the fp8 cell its predecessor cheated), metric artifacts (topk's ~0.02 ceiling
  is launch-overhead-bound, not weakness - true for every model), what the
  winning kernel actually did, profiling discipline that tracks the one win,
  suspicious cross-model convergence. The qualitative read a leaderboard cell
  cannot show is the whole point.
