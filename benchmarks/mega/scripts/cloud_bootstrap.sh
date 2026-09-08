#!/usr/bin/env bash
# Bootstrap a fresh brev H100 (driver R570 / CUDA 12.x) to run KernelBench-Mega.
#
# Run this ON the instance AFTER the repo is rsynced to ~/mega and the auth
# files are in place (~/.codex/auth.json, ~/.claude/.credentials.json,
# ~/.env_vars). Idempotent. Uses cu128 torch -- R570-safe, no painful R580/cu130
# driver upgrade -- because the decode problems are portable bf16/int4 (no
# Blackwell-only tcgen05). For Blackwell-only problems use an R580 image instead.
#
# Reproducible cloud run (from anvil), provision in a few minutes:
#   brev create claude-mega-h100 --gpu H100      # hyperstack ~$2.28/hr
#   rsync benchmarks/mega + ~/.codex/auth.json + ~/.claude/.credentials.json + ~/.env_vars
#   ssh <box> bash ~/mega/scripts/cloud_bootstrap.sh
#   ssh <box> "cd ~/mega && BUDGET_SECONDS=10800 ./scripts/run_hard.sh codex gpt-5.5 problems/02_kimi_linear_decode"
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# uv
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }

# bubblewrap: run_hard.sh sandboxes the agent under bwrap (hides outputs/runs so
# agents cannot read prior solutions). REQUIRED or cloud runs re-contaminate.
command -v bwrap >/dev/null 2>&1 || sudo apt-get install -y -qq bubblewrap >/dev/null 2>&1

# node + agent CLIs (codex and claude harnesses)
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1
  sudo apt-get install -y nodejs >/dev/null 2>&1
fi
# codex is PINNED. 0.141+ (verified on 0.153.2, still true through 0.153.4) sends no
# inline `tools` array to third-party Responses providers (OpenRouter): its exec tool is
# a server-side namespaced tool, so every call fails with "tool exec invoked with
# incompatible payload" and the run ends no_solution. 0.140.0 still sends the 10 inline
# function tools. Raise KB_CODEX_VERSION only after a logging-proxy check shows `tools`
# in the request body (kbtool/AGENTS.md, codex row). Native api.openai.com is unaffected.
KB_CODEX_VERSION="${KB_CODEX_VERSION:-0.140.0}"
codex --version 2>/dev/null | grep -q "codex-cli ${KB_CODEX_VERSION}$" || sudo npm i -g "@openai/codex@${KB_CODEX_VERSION}" >/dev/null 2>&1
command -v claude >/dev/null 2>&1 || sudo npm i -g @anthropic-ai/claude-code >/dev/null 2>&1
# gemini CLI (gemini-3.5-flash harness; key via GEMINI_API_KEY in ~/.env_vars)
command -v gemini >/dev/null 2>&1 || sudo npm i -g @google/gemini-cli >/dev/null 2>&1
# cursor agent CLI (composer harness); auth rsynced separately as ~/.config/cursor + ~/.cursor
command -v cursor-agent >/dev/null 2>&1 || command -v agent >/dev/null 2>&1 || \
  curl -fsS https://cursor.com/install -fsS 2>/dev/null | bash >/dev/null 2>&1 || true

# mega project with cu128 torch (R570-compatible)
cd ~/mega
if ! grep -q pytorch-cu128 pyproject.toml; then
  cat >> pyproject.toml <<'TOML'

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
TOML
  rm -f uv.lock
fi
uv sync >/dev/null 2>&1

echo "BOOTSTRAP OK"
uv run python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo "codex: $(codex --version 2>/dev/null | head -1)"
