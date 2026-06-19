# TODO: finish shipping kernelbench_decode (BLOCKED on Prime VM-sandbox beta access)

Status as of 2026-06-19. The env is correct and reaches `CreateSandboxRequest` cleanly; the only
thing stopping the live smoke is an account gate, not code.

## Hard blocker (owner action, not code)
`prime eval run` fails at sandbox creation with:
`HTTP 403: VM sandboxes are currently only available for beta users`
GPU sandboxes require a VM, which is gated. **Elliot must request VM-sandbox (GPU sandbox) beta
access on the Prime Intellect account.** Once granted, rerun the smoke below — no code changes expected.

## What was fixed (all applied to kernelbench_decode.py)
1. **Own-creds mode** (interception doesn't serve codex's Responses API): `build_env_vars` no longer
   sets `OPENAI_BASE_URL` to the interception tunnel; `post_sandbox_setup` uploads
   `~/.codex/{auth.json,config.toml}` into `/root/.codex/` via `_upload_codex_auth` + `mkdir -p
   /root/.codex`. codex talks to real gpt-5.5 directly.
2. **gpu_type**: `CliAgentEnv.get_sandbox_resources` returns `gpu_type=None` -> 422. Added a
   `get_sandbox_resources` override + a `gpu_type` kwarg on `KernelDecodeEnv`/`load_environment`.
3. **Model placeholder**: `gpt-5.5` is not in Prime's registry; pass a cheap dormant placeholder
   (`meta-llama/Llama-3.2-1B-Instruct`) since codex generates via own-creds. The Prime SANDBOX API
   only accepts `H200` / `RTX_PRO_6000` (NOT the full pod GPU list). Use `RTX_PRO_6000` + a cu128
   image (default cu124 image can't run sm_120 Blackwell).

## Exact rerun command (once beta access is granted)
```
cd ~/kernelbench.com && set -a; . ~/.env_vars; set +a
prime eval run kernelbench_decode --env-dir-path ./environments \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --env-args '{"gpu_type": "RTX_PRO_6000", "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"}' \
  --num-examples 1 --rollouts-per-example 1 --save-results --debug
```

## The 3 integration points still to validate live (need a sandbox)
- `post_sandbox_setup`: node+codex+torch install in the cu128 image; `upload_bytes` paths; torch
  sees the RTX PRO 6000.
- codex actually runs + writes `/workspace/problem/solution.py`.
- reward rubric: `execute_command` runs `check.py` (expects `PASS`) then `benchmark.py`, parses
  `peak_fraction:` from stdout.

## After green
- Push to Hub: `prime env push --path environments/kernelbench_decode --visibility {PUBLIC|PRIVATE}`
  (ask Elliot).
- Re-run the 7 contaminated mega models sandboxed (glm-5.2, MiniMax-M3, deepseek-v4-pro, composer)
  for their genuine clean scores; republish.
- Judge: `z-ai/glm-5.2` via OpenRouter (`use_judge=True`); keys in `~/.env_vars`.
