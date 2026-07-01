#!/usr/bin/env bash
# Runs ON the 8xGPU node. Staged by rsync to ~/kbm then executed over ssh.
# Phase order is fail-fast: topology gate -> torch -> 8-GPU sanity -> sweep.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> ~/kbm/benchmarks/multi  (after rsync layout)

echo "===== TOPO ====="
nvidia-smi topo -m
echo "===== GPUS ====="
nvidia-smi -L

# Topology gate: require NVSwitch (every GPU pair NV##, no PHB/PIX/SYS between GPUs).
if nvidia-smi topo -m | grep -E '^GPU[0-7]' | grep -qE '\b(PHB|PIX|SYS|NODE)\b'; then
  echo "TOPO_FAIL: non-NVLink path between GPUs (bridged/PCIe). Aborting sweep."
  exit 42
fi
echo "TOPO_OK: full NVLink mesh"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Match torch wheel to the node's CUDA driver (cu128 default; override via KBM_CU).
CU="${KBM_CU:-cu128}"
echo "===== INSTALL torch ($CU) ====="
uv venv --python 3.12 .venv >/dev/null 2>&1 || true
. .venv/bin/activate
uv pip install --quiet --index-url "https://download.pytorch.org/whl/${CU}" "torch==2.8.0"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

echo "===== 8-GPU all-reduce preflight ====="
torchrun --nproc_per_node=8 - <<'PY'
import os,torch,torch.distributed as dist
dist.init_process_group("nccl")
l=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(l)
x=torch.ones(1024,1024,device=f"cuda:{l}")
dist.all_reduce(x)
assert int(x[0,0].item())==dist.get_world_size()
if dist.get_rank()==0:print("PREFLIGHT_OK world",dist.get_world_size())
dist.destroy_process_group()
PY

echo "===== NCCL CEILING SWEEP ====="
KBM_WARMUP="${KBM_WARMUP:-200}" KBM_ITERS="${KBM_ITERS:-50}" \
  torchrun --nproc_per_node=8 scripts/nccl_ceiling.py
echo "===== SWEEP_DONE ====="
