# Handoff: ship the KernelBench Prime verifiers environment

You're taking over the **Prime Intellect / verifiers** side of KernelBench. Ship
this env, then return to your own work. The other agent (the one that wrote this)
is handling the near-term non-Prime fix (sandboxing the existing `run_hard.sh`)
in parallel — don't touch that.

## Why this exists (the contamination finding)

KernelBench's old harness has **no filesystem sandbox**: agents get bash +
absolute paths and can read the shared `outputs/runs/` archive — every prior
winning solution — and reverse-engineer a known answer. Audit (2026-06-19) found
**mega-published 7/24 cells contaminated** (the glm-5.2 17.4x / MiniMax 16.5x
"beat opus" headline was FAKE — glm's genuine clean score is 7.3x; opus 14-19x is
real/clean) and **hard-published 0/53** (its published set is a curated clean
generation; 107/403 hard *archive* runs are contaminated but unpublished). Tools
added: `scripts/audit_contamination.py` + tripwires in both leaderboard builders.
Memory: `cross-run-contamination.md`. **The proper fix is a sandboxed harness —
that's this env.**

## What's built (and what isn't)

`environments/kernelbench_decode/` — a `verifiers` env on `CliAgentEnv`:
- `kernelbench_decode.py`: `KernelDecodeEnv(CliAgentEnv)` runs the agent binary
  (codex) in an isolated Prime GPU sandbox. `post_sandbox_setup` uploads ONLY
  the problem files (`problem/` = reference/baseline/check/benchmark/problem.yaml/
  shapes/sota + src/eval + src/hardware) — NEVER the run archive. Reward =
  decode speedup over baseline (parsed from `benchmark.py` `peak_fraction:`),
  gated on `check.py` printing `PASS`. Optional GLM-5.2-via-OpenRouter judge.
- `problem/` — the bundled W4A16 Kimi-Linear decode task (KernelBench-Mega
  problem 03). 19 files, NO solution.py.
- **Status: LOADS end-to-end** (`prime env install kernelbench_decode -p
  ./environments` works; `load_environment()` constructs with dataset + reward +
  judge). **NOT yet run against a live sandbox** — the integration points below
  are unvalidated.

## The integration findings that change the plan (read before running)

1. **`prime login` is done** (auth works; `prime env list` returns results).
2. **prime-sandboxes is cloud-only** — no local backend. Sandboxes run on Prime's
   GPUs. (So "the 3090" really means "cheapest Prime GPU.")
3. **codex uses the OpenAI Responses API; CliAgentEnv interception is
   chat_completions/anthropic only** — so codex does NOT drop into the
   interception cleanly, and Prime inference doesn't host codex's gpt-5.5.
4. **KEY INSIGHT — interception is OPTIONAL; the contamination fix is the SANDBOX
   ISOLATION, not the interception.** The simplest correct path: run codex with
   its OWN auth (`~/.codex/auth.json`) inside the isolated sandbox — it talks to
   real OpenAI/gpt-5.5 directly, no interception, no Responses problem, no
   Prime-model limit. Upload codex's auth in `post_sandbox_setup`. Same for
   claude-code (its Anthropic creds). You only need interception if you later
   want to capture rollout tokens for **RL training** — not for eval.

## To ship (recommended order)

1. **Decide auth mode.** For an eval/contamination-fix smoke, use **own-creds in
   sandbox** (upload `~/.codex/auth.json`, set `OPENAI_BASE_URL`/keys; drop the
   interception). Only wire interception if RL training is the goal.
2. **Live smoke** (costs a Prime GPU sandbox): `prime eval run kernelbench_decode
   -p ./environments -m <model/own-creds> -n 1 -r 1 --save-results --debug`.
   Validate the 3 unvalidated integration points:
   - `post_sandbox_setup`: node+codex+torch install in the `pytorch/pytorch:
     2.6.0-cuda12.4` image; `upload_bytes` paths; torch sees the GPU.
   - codex actually runs + writes `solution.py` in the sandbox.
   - reward rubric: `execute_command` running `check.py`/`benchmark.py` and
     parsing `peak_fraction:` from stdout.
   Debug these live; they're best-effort against the API.
3. For Blackwell sm_120 later, pass a cu128 image via `image=` (the default
   cu124 image covers Ampere/Hopper).
4. **After it's green**: re-run the 7 contaminated mega models sandboxed
   (glm-5.2, MiniMax-M3, deepseek-v4-pro, composer) to get their GENUINE scores,
   then republish (`benchmarks/mega/scripts/build_mega_leaderboard.py` already
   auto-excludes contaminated runs; the sandbox makes future runs clean).
5. Push to Hub when stable: `prime env push kernelbench_decode --visibility
   PRIVATE` (ask the owner PUBLIC vs PRIVATE first).

## Pointers
- Judge model: `z-ai/glm-5.2` via OpenRouter (owner's call, not gpt-4.1-mini).
- Owner keys in `~/.env_vars` (OPENROUTER/OPENAI/etc.); `vf.ensure_keys([...])`.
- Repo CLAUDE.md/AGENTS.md "CROSS-RUN CONTAMINATION" bullet documents all of this.
- There's an existing `anushkad/tritonbench` GPU-kernel verifiers env on the Hub
  worth comparing for the GPU-sandbox + reward pattern.
