#!/usr/bin/env bash
# One-shot H100 (brev/Hyperstack) box bring-up for KernelBench-Hard.
# Run on the box (via brev exec / ssh). Idempotent-ish. Two phases around a
# reboot: `pre` (driver+toolkit+disk+container-mode), then reboot, then `post`
# is done from the control plane (rsync binaries/auth + image pull).
# Usage on box:  bash h100_box_setup.sh pre
set -uo pipefail
PHASE="${1:?pre|verify}"

pre() {
  echo "=== [1/5] driver R580 ==="
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    -o Dpkg::Options::=--force-overwrite cuda-drivers-580 || true
  # clear any half-removed older branch that blocks the transition
  sudo dpkg --remove --force-all libnvidia-extra-570 libnvidia-gl-570 2>/dev/null || true
  sudo dpkg --configure -a || true
  sudo apt-get install -f -y -o Dpkg::Options::=--force-overwrite || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    -o Dpkg::Options::=--force-overwrite cuda-drivers-580
  sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y \
    'nvidia-fabricmanager-570' 'libnvidia-common-570' 'nvidia-firmware-570' 2>/dev/null || true
  sudo apt-get autoremove -y || true

  echo "=== [2/5] CUDA toolkit 13.2 + uv ==="
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-2
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

  echo "=== [3/5] relocate docker + containerd to /ephemeral ==="
  sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
  sudo mkdir -p /ephemeral/docker /ephemeral/containerd
  echo '{ "data-root": "/ephemeral/docker" }' | sudo tee /etc/docker/daemon.json
  sudo rm -rf /var/lib/docker/* 2>/dev/null || true
  sudo rm -rf /var/lib/containerd
  sudo ln -sfn /ephemeral/containerd /var/lib/containerd

  echo "=== [4/5] nvidia-container-toolkit legacy mode (so --gpus all works) ==="
  sudo nvidia-ctk config --set nvidia-container-runtime.mode=legacy --in-place

  echo "=== [5/5] done. REBOOT now, then run image pull + binary sync from control plane ==="
}

verify() {
  echo "driver:"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "nvcc:"; /usr/local/cuda-13.2/bin/nvcc --version | tail -1
  echo "uv:"; ~/.local/bin/uv --version 2>/dev/null || uv --version
  echo "docker root:"; docker info 2>/dev/null | grep -i "Docker Root"
  echo "ctk mode:"; grep -E '^\s*mode' /etc/nvidia-container-runtime/config.toml
  echo "gpus-all test:"; docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.1 nvidia-smi -L 2>&1 | tail -1
  echo "df:"; df -h / /ephemeral | grep -vE Filesystem
}

case "$PHASE" in
  pre) pre ;;
  verify) verify ;;
  *) echo "usage: $0 pre|verify" >&2; exit 2 ;;
esac
