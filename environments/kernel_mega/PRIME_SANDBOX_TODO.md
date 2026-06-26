# TODO: Prime-sandbox execution mode for kernel_mega (contamination-fix)

`kernel_mega` today runs the native harness locally as a multi-turn ToolEnv (the policy iterates with
write_solution/run_check/run_benchmark over a shared workspace). This TODO is the alternative
EXECUTION mode that closes the cross-run contamination hole, folded in from the former standalone
`kernelbench_decode` env (which was just mega problem `02_kimi_linear_decode` wrapped separately).

## Why (the contamination finding)
KernelBench's old harness has no filesystem sandbox: an eval agent gets bash + absolute paths and can
read the shared `outputs/runs/` archive — every prior winning solution — and reverse-engineer a known
answer. Audit (2026-06-19): mega-published 7/24 cells were contaminated (the glm-5.2 "beats opus"
headline was fake; opus 14-19x is the real winner). The fix is a sandboxed harness.

## The approach (was `environments/kernelbench_decode/`)
Run the agent (codex) in an ISOLATED Prime GPU sandbox; upload ONLY the problem files
(reference/baseline/check/benchmark/problem.yaml/shapes/sota + src), NEVER the run archive. Reward =
peak_fraction from the native `benchmark.py`, gated on `check.py` PASS. Codex uses its own
`~/.codex/auth.json` uploaded into the sandbox (own-creds mode; the verifiers interception only serves
chat/anthropic, not codex's Responses API). Optional GLM-5.2-via-OpenRouter judge.

## Status: code-complete but BLOCKED on a Prime account permission
The implementation (a `CliAgentEnv`-based env, own-creds + gpu_type fixes applied) lives in git history
at the removed `environments/kernelbench_decode/` (see `git log -- environments/kernelbench_decode`).
Two account-level blockers, both needing the owner (not code):
1. `HTTP 403: VM sandboxes are currently only available for beta users` — GPU sandboxes need Prime
   beta access on the account.
2. (when publishing) `Collaborators cannot create new environments` — the account can't create Hub
   envs yet.

## Exact eval rerun (once VM-sandbox beta is granted)
```
cd ~/kernelbench.com && set -a; . ~/.env_vars; set +a
prime eval run kernelbench_decode --env-dir-path ./environments \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --env-args '{"gpu_type": "RTX_PRO_6000", "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"}' \
  --num-examples 1 --rollouts-per-example 1 --save-results --debug
```

## How to fold in (the intended design)
Add a `sandbox="native"|"prime"` kwarg to `kernel_mega.load_environment`: `native` = the current
local ToolEnv; `prime` = the codex-in-Prime-sandbox flow above. The 3 unvalidated integration points
(post_sandbox_setup install, codex writing solution.py, reward parsing peak_fraction) are validatable
once the beta unblocks. The reward/roofline/problem deck are already shared.
