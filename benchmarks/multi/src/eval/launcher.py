"""Spawn a torchrun job for one problem and interpret its output.

Per-problem check.py / benchmark.py are thin wrappers around this. The launcher
does the forbidden-op grep (the bare-collective tripwire) before spending a
torchrun launch, then runs the worker at the problem's world_size.

Env overrides for local single-GPU validation:
    KBM_BACKEND=gloo KBM_DEVICE=cpu KBM_WORLD_SIZE=4
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # benchmarks/multi
WORKER = REPO_ROOT / "src" / "eval" / "worker.py"


def _world_size(meta: dict) -> int:
    return int(os.environ.get("KBM_WORLD_SIZE", meta.get("world_size", 8)))


def _forbidden_check(problem_dir: Path, meta: dict) -> str | None:
    if os.environ.get("KBM_SKIP_FORBIDDEN") == "1":
        return None
    sol = problem_dir / "solution.py"
    if not sol.exists():
        return "no solution.py"
    src = sol.read_text()
    for pat in meta.get("forbidden", []):
        if re.search(re.escape(pat), src):
            return f"forbidden op used: {pat}"
    return None


def _torchrun_cmd(mode: str, problem_dir: Path, world: int) -> list[str]:
    port = os.environ.get("KBM_MASTER_PORT", "29571")
    return [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={world}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(WORKER),
        "--mode", mode,
        "--problem-dir", str(problem_dir),
    ]


def _run(mode: str, problem_dir: Path) -> int:
    meta = yaml.safe_load((problem_dir / "problem.yaml").read_text())
    world = _world_size(meta)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = _torchrun_cmd(mode, problem_dir, world)
    print(f"[launcher] {mode} world_size={world} backend={env.get('KBM_BACKEND', 'nccl')}", flush=True)
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def run_check(problem_dir: Path) -> int:
    problem_dir = Path(problem_dir).resolve()
    meta = yaml.safe_load((problem_dir / "problem.yaml").read_text())
    forbidden = _forbidden_check(problem_dir, meta)
    if forbidden:
        print(f"FAIL: {forbidden}", flush=True)
        return 1
    return _run("check", problem_dir)


def run_benchmark(problem_dir: Path) -> int:
    return _run("benchmark", Path(problem_dir).resolve())
