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
#   ssh <box> "cd ~/mega && BUDGET_SECONDS=10800 ./scripts/run_hard.sh codex gpt-5.5 problems/03_kimi_linear_decode"
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# uv
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }

# node + agent CLIs (codex and claude harnesses)
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1
  sudo apt-get install -y nodejs >/dev/null 2>&1
fi
command -v codex >/dev/null 2>&1 || sudo npm i -g @openai/codex >/dev/null 2>&1
command -v claude >/dev/null 2>&1 || sudo npm i -g @anthropic-ai/claude-code >/dev/null 2>&1

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
