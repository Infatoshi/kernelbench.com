"""KernelBench-decode: sandboxed agentic kernel-optimization environment.

Ports KernelBench-Mega problem 03 (W4A16 Kimi-Linear decode) into a Prime
verifiers environment built on CliAgentEnv. The agent binary (codex by default)
runs inside an isolated Prime GPU sandbox; only the problem files are uploaded
(reference / baseline / check / benchmark / problem.yaml / shapes / src), NEVER
the run archive -- which closes the cross-run contamination hole in the old
KernelBench harness (agents could read prior winning solutions via absolute
paths). The agent's LLM calls are intercepted by verifiers, so the `-m` model on
`prime eval run` is what codex actually uses (no codex auth needed in-sandbox).

Reward = decode speedup over the optimized-PyTorch baseline, gated on
correctness (cosine-sim vs reference). A run that does not compile or does not
pass correctness scores 0. An optional LLM judge inspects passing solutions.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv

HERE = Path(__file__).parent
PROBLEM_DIR = HERE / "problem"
WORKDIR = "/workspace"                 # sandbox cwd
PROB = f"{WORKDIR}/problem"            # where problem files land
PROMPT = (PROBLEM_DIR / "PROMPT.txt").read_text()

# Default sandbox image: PyTorch + CUDA. cu124 runtime works on Ampere/Hopper
# (the cheap smoke GPUs); for Blackwell sm_120 use a cu128 image instead.
DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

# codex run, headless, scoped to the problem dir. Its OpenAI calls are
# intercepted by CliAgentEnv (OPENAI_BASE_URL is pointed at the tunnel).
DEFAULT_RUN_COMMAND = (
    f"cd {PROB} && codex exec --dangerously-bypass-approvals-and-sandbox "
    f"--skip-git-repo-check -C {PROB} \"$KB_PROMPT\""
)


def _upload_problem_files(client, sandbox_id: str) -> None:
    """Upload ONLY the problem files into the sandbox (isolation set)."""
    for f in sorted(PROBLEM_DIR.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            rel = f.relative_to(PROBLEM_DIR)
            client.upload_bytes(sandbox_id, f.read_bytes(), f"{PROB}/{rel}")


class KernelDecodeEnv(CliAgentEnv):
    """codex-in-a-sandbox kernel optimizer with a GPU compile/correct/speedup reward."""

    def __init__(self, image: str = DEFAULT_IMAGE, gpu_count: int = 1, **kwargs):
        # One task: the decode kernel. The agent's real prompt is delivered via
        # run_command/KB_PROMPT inside the sandbox; this row just drives iteration.
        dataset = kwargs.pop("dataset", None) or Dataset.from_list([{
            "question": PROMPT,
            "info": {"problem": "03_kimi_linear_decode"},
            "answer": "",
        }])
        super().__init__(
            dataset=dataset,
            run_command=DEFAULT_RUN_COMMAND,
            docker_image=image,
            gpu_count=gpu_count,
            cpu_cores=kwargs.pop("cpu_cores", 8),
            memory_gb=kwargs.pop("memory_gb", 32),
            disk_size_gb=kwargs.pop("disk_size_gb", 40),
            timeout_seconds=kwargs.pop("timeout_seconds", 10800.0),
            keep_sandbox_for_scoring=True,   # rubric needs the sandbox to verify
            **kwargs,
        )
        self.add_rubric(self._reward_rubric())

    async def build_env_vars(self, state):  # noqa: ANN001
        env = await super().build_env_vars(state)
        env["KB_PROMPT"] = PROMPT
        return env

    async def post_sandbox_setup(self, state) -> None:  # noqa: ANN001
        sid = state["sandbox_id"]
        c = self.sandbox_client
        # toolchain: node+codex+uv+torch in the GPU image (codex needs node).
        setup = (
            "set -e; apt-get update -qq && apt-get install -y -qq curl git >/dev/null 2>&1 || true; "
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1 && "
            "apt-get install -y -qq nodejs >/dev/null 2>&1; "
            "npm i -g @openai/codex >/dev/null 2>&1; "
            "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; "
            "pip install -q pyyaml einops >/dev/null 2>&1 || true; "
            f"mkdir -p {PROB}"
        )
        await c.execute_command(sid, setup, timeout=900)
        _upload_problem_files(c, sid)
        # sanity: torch sees the GPU
        await c.execute_command(
            sid, "python -c 'import torch;print(torch.cuda.get_device_name(0))'", timeout=120
        )

    def _reward_rubric(self) -> vf.Rubric:
        env = self

        async def speedup(state, **_) -> float:
            """Run check.py then benchmark.py in the sandbox; reward = geomean speedup."""
            sid = state.get("sandbox_id")
            if not sid:
                return 0.0
            c = env.sandbox_client
            chk = await c.execute_command(sid, f"cd {PROB} && python check.py", timeout=600)
            if "PASS" not in (chk.stdout or ""):
                state["kb_correct"] = False
                return 0.0
            state["kb_correct"] = True
            bench = await c.execute_command(sid, f"cd {PROB} && python benchmark.py", timeout=1800)
            m = re.search(r"peak_fraction:\s*([0-9.]+)", bench.stdout or "")
            pf = float(m.group(1)) if m else 0.0
            state["kb_speedup"] = pf
            return pf

        async def correct(state, **_) -> float:
            return 1.0 if state.get("kb_correct") else 0.0

        r = vf.Rubric()
        r.add_reward_func(speedup, weight=1.0)   # headline: speedup over baseline
        r.add_metric(correct)                    # pass/fail tracked separately
        return r


def load_environment(
    image: str = DEFAULT_IMAGE,
    gpu_count: int = 1,
    use_judge: bool = False,
    judge_model: str = "z-ai/glm-5.2",   # GLM-5.2 via OpenRouter
    **kwargs,
) -> vf.Environment:
    """Sandboxed KernelBench-decode env. One task: optimize the W4A16 decode kernel."""
    env = KernelDecodeEnv(image=image, gpu_count=gpu_count, **kwargs)
    if use_judge:
        vf.ensure_keys(["OPENROUTER_API_KEY"])
        judge_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        env.add_rubric(
            vf.JudgeRubric(
                judge_client=judge_client,
                judge_model=judge_model,
                judge_prompt=(
                    "You are auditing a GPU kernel solution that PASSED correctness. "
                    "Is it a genuine fused int4 dequant-GEMV decode kernel written for "
                    "this task, or did it shortcut / reverse-engineer a known answer? "
                    "Reply with a 0..1 quality score and one-line justification."
                ),
            )
        )
    return env
