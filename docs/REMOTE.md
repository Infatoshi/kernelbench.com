# Rented GPU workers (Lambda, Brev, Verda)

GPU eval sessions run on rented workers, not on anvil. This file is the runbook: bring a node up, make it able to run torch and ncu, run cells, pull archives back, tear it down. Rules that gate a published number live in `AGENTS.md`.

## Lambda Cloud

Zach / Lambda sponsored $10k of Cloud credits (2026-07) for Hard / Mega / CUDA / Multi (RTX PRO 6000, H100, B200). Credits show only in the console: Settings -> Billing -> Credits (account `elliot@arledge.net`). Tag Lambda on X when posting runs from it.

- Auth: `LAMBDA_API_KEY` in `~/.env_vars` (keep the legacy typo `LAMDBA_API_KEY` in sync; kimi-sweep reads it). Mint at https://cloud.lambda.ai/api-keys/cloud-api. Keep Mac and anvil `~/.env_vars` in sync.
- SSH keys on the account: `macbook` and `anvil` (each host's `~/.ssh/id_ed25519.pub`). `lambda_worker.sh up` attaches ONE key, the current host's, because the launch API rejects more than one (observed 2026-07-21). Override with `KB_LAMBDA_SSH_KEYS`.
- The worker scripts use the Cloud API via curl; nothing needs brew. Optional Mac-only community CLI: `brew install strand-ai/tap/lambda-cli`.

```
kb lambda list                         # capacity by type
kb lambda ls                           # running instances
kb lambda up <name> [type] [region]    # default type gpu_1x_h100_sxm5
kb lambda sync <name>                  # thin bench + allowlisted keys (preserves the node's torch-index patch)
kb lambda bootstrap <name> [--agents]  # uv + torch; --agents = agent CLIs; ncu on PATH + NVreg_RestrictProfilingToAdminUsers=0
kb lambda run <name> <harness> <model> <problem> [effort]
kb lambda pull <name>                  # -> benchmarks/hard/outputs/runs-lambda-<name>/ (excludes .venv)
kb lambda regrade <name> <run_id> [runs_dir]   # sequential isolated re-grade on the node
kb lambda down <name>                  # terminate + poll until gone
kb lambda ssh <name> [cmd...]
```

Or `./scripts/lambda_worker.sh ...` from the repo root. Env overrides: `KB_LAMBDA_TYPE`, `KB_LAMBDA_REGION`, `KB_LAMBDA_SSH_KEYS`, `KB_LAMBDA_PROBLEMS_ROOT` (default `problems-h100`; `problems-h100x4` when `KB_LAMBDA_BENCH=multi`), `KB_LAMBDA_BENCH` (default `hard`; `multi`, `cuda`, `mega` point the worker at another bench).

Multi-GPU / NVLink work can use Lambda `gpu_8x_h100_sxm5` / `gpu_8x_b200_sxm6` when `kb lambda list` shows capacity, or Brev (below).

## Bootstrap order on a fresh node

1. A node can pass `nvcc` checks and still not run torch. Lambda's stock image ships driver 570 and no NVIDIA CUDA apt repo, so `apt-get install cuda-toolkit-13-0` returns rc=100 while the driver install succeeds from Lambda's own archive. The node then has a driver but no `/usr/local/cuda-13.0`, or a toolkit with a driver too old for the cu130 wheel. Both failures are silent. Add the repo first (`cuda-keyring_1.1-1_all.deb` from `developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/`), install `cuda-toolkit-13-0` and `nvidia-driver-595`, reboot.
2. Gate every launch on `torch.cuda.is_available()` after every `uv` command. `nvcc --version` is not a CUDA probe. `nvcc` must compile `#include <cuda_runtime.h>` with `cudafe++` next to it.
3. Hyperstack / Shadeform 8xH100 nodes ship driver CUDA 12.8; the default `uv pip install torch` pulls a cu130 wheel that cannot see the GPUs ("driver too old"). Install the matched build: `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0`. Bake uv, the repo, and this wheel into a prebaked image so node time is not spent on reinstalls. `kb lambda sync` preserves the node's patched pyproject/uv.lock (a re-sync once shipped the Mac's cu130 lock over the node's cu128 one and every later graded env died at check time, 2026-08-01).
4. Stock Lambda SXM5 image has `nvcc` and no `ncu` until `apt-get install nsight-compute`.

## ncu on rented VMs (closed 2026-08-24)

`ERR_NVGPUCTRPERM` is an admin gate, not missing bare metal. Lambda `gpu_1x_h100_pcie`, Lambda `gpu_1x_h100_sxm5`, and Verda `4RTXPRO6000.120V` are all KVM VMs with `RmProfilingAdminOnly: 1`. Root `ncu` wrote a real `smsp__cycles_elapsed.avg` on all three; the Lambda `ubuntu` user has passwordless `sudo -n`. Lambda DeepTalk (Hayden, 2025-10-01) saying Nsight is unsupported on Cloud VMs is stale for these SKUs; bare metal is not required. Brev -> Lambda should match; Brev -> AWS/GCP can block counters even as root, so probe before trusting.

Do not give the agent blanket sudo. Default remote sessions use `KBH_AGENT_CONTAINER=1`: `ncu` runs inside Docker as uid 1000 with `--cap-add CAP_PERFMON`, `--user $(id -u):$(id -g)`, and `--security-opt no-new-privileges`. `--cap-add CAP_PERFMON` alone does not clear the error. On every rented box, before the first agent session: (1) `ncu` on PATH, (2) `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-ncu.conf`, then reboot or reload `nvidia` with zero GPU users. `kb lambda bootstrap` does both. Host mode (`KBH_AGENT_CONTAINER=0`, some hy3, mega bwrap) may wrap only `ncu`/`nsys` with `sudo -n` via a NOPASSWD sudoers line limited to those binaries, never `/bin/bash`. Anvil and gamer are local compute; do not change their driver policy for this.

## Running and pulling back

- Bootstrap long work with `nohup` or `systemd-run` on the node; the laptop SSH is a probe. A live tmux, an SSH 255, or a nohup that inits then exits is not "launched". Wave complete means a `result.json` on the Mac for every cell.
- Incremental pullback every 10 minutes, excluding `.venv`. Before any `rsync`/`scp`, `du -sb` the remote paths and print full vs tiny. Tiny set = `result.json`, `solution.py`, `gpu`, `check.log`, `benchmark.log`, sidecar `*.cu`/`*.cuh`/`kernels.py`; that is the audit, draft, and `kb publish` input. Do not pull `transcript.jsonl`, `agent_home/`, `.venv/`, `repo/`, or `cache/` until the run is being converted for HF. Over 20 MB, state the size and wait; over 1 GB, refuse until the user has seen the number. `ssh HOST cat .../result.json` when one file answers. (August 2026 wave: 21 full dirs = 12.7 GiB, tiny set = 0.5 MB, tiny + transcripts = 1.0 GiB, one DeepSeek TopK jsonl was 241 MB.)
- Point every worker's lock, log, and archive paths at in-repo locations and `kb lambda pull` its archives back into `outputs/runs/` before teardown. An archive stranded outside the repo is invisible to `kb publish`, `kb contamination`, and re-grades.
- Quiet GPU for regrade means zero compute PIDs, not "0% util with leftover VRAM". An `nvidia-smi` timeout means a wedged driver: stop.
- Contamination scan must rebuild grok streaming-json and read `~/.grok/sessions/*/chat_history.jsonl`. An empty `outputs/runs` is not a clean box while a CLI session store remains.

## Teardown

- Always `kb lambda down <name>` when done; idle nodes bill the credits. Confirm `kb lambda ls` no longer lists it.
- Kill by pidfile, never `pkill -f` (it matches its own ssh argv and kills the session with exit 255).
- Brev: `brev delete <name>` has a hidden interactive confirmation that silently hangs with no TTY, and `brev stop` / `yes | brev delete` no-op. Teardown goes through `scripts/brev_teardown.sh <name>`, which gives brev a pseudo-TTY, feeds it `y`, and polls `brev ls` until the instance is gone (it branches to `expect` on macOS because `script -qec ... <<< y` silently does nothing there). A forgotten 8xH100 node bills about $23/hr.

## Multi (4xH100 NVLink) specifics

The graded SKU is 4xH100 SXM behind NVSwitch: every GPU pair shows `NV18` in `nvidia-smi topo -m`. A PCIe or switchless node produces meaningless busbw numbers; `scripts/remote_ceiling.sh` has a topology gate that enforces this. The temporary poseidon/hades nodes matched this SKU. Validate correctness for free on a single GPU first via gloo+cpu (`KBM_BACKEND=gloo KBM_DEVICE=cpu KBM_WORLD_SIZE=4 python check.py`); the rented node should never see a correctness bug for the first time.
