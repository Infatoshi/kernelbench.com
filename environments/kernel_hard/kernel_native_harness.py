"""Shared core for the kernel_hard / kernel_mega verifiers environments.

Both envs wrap this repo's NATIVE agentic harness (`benchmarks/hard` and
`benchmarks/mega`). The two decks are SIBLINGS: identical native-harness
machinery (`src/eval` correctness + timing + roofline, the per-problem
`check.py` / `benchmark.py` scoring contract from `scripts/run_hard.sh`); only
the `problems/` deck differs. So this one module is parametrized by `bench_root`
(the benchmark checkout) and a deck filter; `kernel_hard.py` and
`kernel_mega.py` are thin wrappers that bind `bench_root` and defaults.

DESIGN (settled with the user, mirrors kernel_v3):
- MECHANICAL-PRIMARY reward. We do NOT reimplement eval. We replicate exactly
  the scoring contract of the native `scripts/run_hard.sh`: lay out a workspace
  (`src/` at parents[2], the immutable problem template files, the model's
  `solution.py`), run the problem's own `check.py` (correctness HARD GATE -> a
  literal `^PASS` line) and `benchmark.py` (which prints `peak_fraction: <N>`
  using `src/eval/roofline.peak_fraction`). reward = peak_fraction (clamped to
  [0,1]) if correct else 0.0. The PyTorch `reference.py` is correctness-only.
- The scoring runs as a SUBPROCESS via the benchmark's own `uv` env (which has
  torch+CUDA), so THIS module is CPU-import-safe (no torch import) and can live
  in a torch-less verifiers venv. The GPU work happens in the subprocess.
- MULTI-TURN / agentic: a verifiers `StatefulToolEnv` exposes the native tools
  the model iterates with -- `write_solution`, `run_check`, `run_benchmark` --
  backed by ONE persistent per-rollout workspace (the native flywheel: write ->
  check.py -> benchmark.py -> iterate). The reward is computed at the end from
  that same workspace's final `solution.py` via the native scripts -- never an
  opaque black-box runner. `StatefulToolEnv.no_tools_called` ends the rollout
  when the model stops calling tools.
- Judge is an OPT-IN VETO, OFF by default (`enable_judge=False`). Only fires on
  correct solutions, can only zero the reward, never adds. Composed
  multiplicatively. No-op when disabled. Default model `z-ai/glm-5.2` via
  OpenRouter (keys in ~/.env_vars).

Roofline note: megakernel decks can report peak_fraction > 1 (the dense-FLOPS
formula overcounts work the kernel legitimately skips). The native value is
preserved as `raw_peak_fraction`; the REWARD clamps to [0,1] so the gradient
stays bounded.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

# Per-problem files the native harness copies into a workspace. The native
# run_hard.sh TEMPLATE_FILES list is (reference/sota/shapes/problem.yaml/
# check/benchmark/PROMPT[/baseline]) but some problems ship EXTRA aux modules
# that check.py/benchmark.py import (e.g. mega's 02_kimi_linear_decode imports
# `baseline`). To be faithful across both decks we copy EVERY file in the
# problem dir except generated/agent artifacts (below).
_PROBLEM_ARTIFACTS = {"solution.py", "framework.txt", "__pycache__", ".pytest_cache"}

CODE_RE = re.compile(r"```(?:python|cpp|cuda|py)?\n(.*?)```", re.DOTALL)
_PEAK_RE = re.compile(r"peak_fraction:\s*([0-9.]+)")
_PASS_RE = re.compile(r"^PASS", re.MULTILINE)

ENV_PASSTHROUGH = {
    "system_prompt",
    "few_shot",
    "sampling_args",
    "max_workers",
    "max_seq_len",
    "message_type",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_solution(completion) -> str:
    """Last fenced code block in the final assistant message (fallback path)."""
    if isinstance(completion, list):
        text = completion[-1]["content"] if completion else ""
    else:
        text = str(completion)
    text = text or ""
    matches = CODE_RE.findall(text)
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Workspace: replicate scripts/run_hard.sh layout (src/ at parents[2])
# ---------------------------------------------------------------------------
def make_workspace(bench_root: str, problem: str) -> str:
    ws = tempfile.mkdtemp(prefix="kbh_env_")
    bench = Path(bench_root)
    shutil.copytree(bench / "src", Path(ws) / "src")
    for f in ("pyproject.toml", "uv.lock", ".python-version"):
        src = bench / f
        if src.exists():
            shutil.copy2(src, Path(ws) / f)
    pdir = Path(ws) / "problems" / problem
    pdir.mkdir(parents=True, exist_ok=True)
    src_pdir = bench / "problems" / problem
    for src in src_pdir.iterdir():
        if src.name in _PROBLEM_ARTIFACTS:
            continue
        if src.is_dir():
            continue  # skip __pycache__ etc.
        shutil.copy2(src, pdir / src.name)
    return ws


def problem_dir(ws: str, problem: str) -> Path:
    return Path(ws) / "problems" / problem


def run_native(ws: str, problem: str, script: str, timeout_s: int) -> tuple[int, str]:
    """Run the problem's own check.py / benchmark.py with THIS env's interpreter.

    Self-contained (option 1): we use `sys.executable` (the env venv python, which
    has torch + the benchmark deps from this env's pyproject) instead of the
    benchmark's `uv run python`. check.py/benchmark.py self-insert the workspace's
    `src/` (their `parents[2]`) onto sys.path, so the vendored harness resolves
    with no benchmark uv env and no checkout dependency.
    """
    pdir = problem_dir(ws, problem)
    env = dict(os.environ)
    env.setdefault("CUDA_HOME", env.get("KBH_CUDA_HOME", "/usr/local/cuda-13"))
    cuda_bin = f"{env['CUDA_HOME']}/bin"
    if cuda_bin not in env.get("PATH", ""):
        env["PATH"] = cuda_bin + ":" + env.get("PATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=str(pdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, (proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="ignore")
        return 124, out + f"\nTIMEOUT after {timeout_s}s"
    except Exception as e:  # pragma: no cover
        return 1, f"runner error: {type(e).__name__}: {e}"


def score_workspace(ws: str, problem: str, check_timeout_s: int, bench_timeout_s: int) -> dict:
    """Native scoring contract: check.py ^PASS gate -> benchmark.py peak_fraction.

    Operates on an EXISTING workspace whose solution.py is already written.
    """
    res: dict[str, Any] = {
        "correct": False,
        "peak_fraction": 0.0,
        "raw_peak_fraction": 0.0,
        "check_log": "",
        "bench_log": "",
    }
    if not (problem_dir(ws, problem) / "solution.py").exists():
        res["check_log"] = "no solution.py"
        return res
    _, check_log = run_native(ws, problem, "check.py", check_timeout_s)
    res["check_log"] = check_log[-4000:]
    if not _PASS_RE.search(check_log):
        return res
    res["correct"] = True
    _, bench_log = run_native(ws, problem, "benchmark.py", bench_timeout_s)
    res["bench_log"] = bench_log[-4000:]
    m = _PEAK_RE.search(bench_log)
    if m:
        raw = float(m.group(1))
        res["raw_peak_fraction"] = raw
        res["peak_fraction"] = max(0.0, min(1.0, raw))
    return res


def score_solution(bench_root: str, problem: str, code: str, check_timeout_s: int, bench_timeout_s: int) -> dict:
    """Score a raw solution string in a fresh throwaway workspace (fallback path)."""
    ws = make_workspace(bench_root, problem)
    try:
        (problem_dir(ws, problem) / "solution.py").write_text(code)
        return score_workspace(ws, problem, check_timeout_s, bench_timeout_s)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def discover_problems(bench_root: str, deck: list[str] | None) -> list[str]:
    pdir = Path(bench_root) / "problems"
    found = [
        d.name
        for d in sorted(pdir.iterdir())
        if (d / "problem.yaml").exists() and (d / "check.py").exists()
    ]
    if deck:
        deck_set = set(deck)
        found = [p for p in found if p in deck_set]
    return found


def _build_question(bench_root: str, problem: str) -> str:
    pdir = Path(bench_root) / "problems" / problem
    prompt = (pdir / "PROMPT.txt").read_text() if (pdir / "PROMPT.txt").exists() else ""
    reference = (pdir / "reference.py").read_text() if (pdir / "reference.py").exists() else ""
    return (
        f"{prompt}\n\n"
        "You are in a persistent workspace for this problem. Available tools:\n"
        "  - write_solution(code): write/overwrite solution.py (pass the FULL file each call).\n"
        "  - run_check(): run the harness check.py (correctness gate; prints PASS/FAIL over all shapes).\n"
        "  - run_benchmark(): run benchmark.py (prints peak_fraction; only meaningful after PASS).\n"
        "Flywheel: write_solution -> run_check -> (on PASS) run_benchmark -> iterate to push\n"
        "peak_fraction up. Stop calling tools when you are done; your score is the final\n"
        "benchmark peak_fraction if check.py PASSes, else 0. solution.py must define\n"
        "`class Model(nn.Module)`, `get_inputs()`, `get_init_inputs()` matching reference.py.\n\n"
        f"Reference code (reference.py):\n```python\n{reference}\n```\n"
    )


def _is_eval(problem: str, eval_frac: float) -> bool:
    if eval_frac <= 0:
        return False
    import hashlib

    h = int(hashlib.sha256(problem.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def _make_rows(bench_root: str, problems: list[str], hardware: str) -> list[dict]:
    return [
        {
            "question": _build_question(bench_root, p),
            "answer": p,  # problem id; native scoring keys off it (reference lives in workspace)
            "info": {"problem": p, "hardware": hardware, "bench_root": bench_root},
        }
        for p in problems
    ]


# ---------------------------------------------------------------------------
# Stateful tool environment (the agentic multi-turn flywheel)
# ---------------------------------------------------------------------------
class KernelHarnessEnv(vf.StatefulToolEnv):
    """Agentic StatefulToolEnv over the native check.py/benchmark.py flywheel.

    The three native tools share ONE persistent per-rollout workspace stored in
    `state["_ws"]`. `update_tool_args` injects `state` into each (hidden) call.
    Reward (a Rubric func) reads the final solution.py from `state["_ws"]`.
    """

    def __init__(self, *, bench_root, check_timeout_s, bench_timeout_s, sem, **kwargs):
        self._bench_root = bench_root
        self._check_t = check_timeout_s
        self._bench_t = bench_timeout_s
        self._sem = sem

        # Native tools. `state` is hidden from the agent's schema and injected.
        def write_solution(code: str, state: dict) -> str:
            """Write (overwrite) solution.py with the full file contents."""
            problem = (state.get("info") or {}).get("problem") or state.get("answer")
            ws = self._ws(state, problem)
            (problem_dir(ws, problem) / "solution.py").write_text(code)
            return f"solution.py written ({len(code)} chars). Run run_check() to verify."

        def run_check(state: dict) -> str:
            """Run the harness check.py (correctness gate over all shapes)."""
            problem = (state.get("info") or {}).get("problem") or state.get("answer")
            ws = self._ws(state, problem)
            if not (problem_dir(ws, problem) / "solution.py").exists():
                return "No solution.py yet; call write_solution(code) first."
            rc, log = run_native(ws, problem, "check.py", self._check_t)
            return f"check.py exit={rc}\n{log[-3000:]}"

        def run_benchmark(state: dict) -> str:
            """Run benchmark.py (prints peak_fraction; meaningful only after PASS)."""
            problem = (state.get("info") or {}).get("problem") or state.get("answer")
            ws = self._ws(state, problem)
            if not (problem_dir(ws, problem) / "solution.py").exists():
                return "No solution.py yet; call write_solution(code) first."
            rc, log = run_native(ws, problem, "benchmark.py", self._bench_t)
            return f"benchmark.py exit={rc}\n{log[-3000:]}"

        super().__init__(tools=[], **kwargs)
        # Register with `state` hidden from the agent and injected at call time.
        self.add_tool(write_solution, args_to_skip=["state"])
        self.add_tool(run_check, args_to_skip=["state"])
        self.add_tool(run_benchmark, args_to_skip=["state"])

    def _ws(self, state: dict, problem: str) -> str:
        ws = state.get("_ws")
        if not ws or not os.path.isdir(ws):
            ws = make_workspace(self._bench_root, problem)
            state["_ws"] = ws
        return ws

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs) -> dict:
        # Inject the hidden per-rollout state into every native tool call.
        tool_args = dict(tool_args)
        tool_args["state"] = state
        return tool_args


# ---------------------------------------------------------------------------
# Rubric (mechanical reward + optional judge veto)
# ---------------------------------------------------------------------------
def _build_rubric(
    *,
    bench_root,
    check_timeout_s,
    bench_timeout_s,
    sem,
    parser,
    enable_judge,
    judge_model,
    judge_base_url,
    judge_api_key,
) -> vf.Rubric:
    async def mechanical_reward(completion, answer, state, info, **kw) -> float:
        problem = (info or {}).get("problem") or answer
        state["correct"] = False
        state["peak_fraction"] = 0.0
        state["raw_peak_fraction"] = 0.0
        ws = state.get("_ws")
        async with sem:
            if ws and (problem_dir(ws, problem) / "solution.py").exists():
                res = await asyncio.to_thread(
                    score_workspace, ws, problem, check_timeout_s, bench_timeout_s
                )
            else:
                code = extract_solution(completion)
                if not code:
                    return 0.0
                res = await asyncio.to_thread(
                    score_solution, bench_root, problem, code, check_timeout_s, bench_timeout_s
                )
        state["correct"] = bool(res["correct"])
        state["peak_fraction"] = float(res["peak_fraction"])
        state["raw_peak_fraction"] = float(res["raw_peak_fraction"])
        state["_bench"] = {k: v for k, v in res.items() if k not in ("check_log", "bench_log")}
        if not state["correct"]:
            return 0.0
        return state["peak_fraction"]

    def correct_metric(state, **kw) -> float:
        return 1.0 if state.get("correct") else 0.0

    def peak_fraction_metric(state, **kw) -> float:
        return float(state.get("peak_fraction", 0.0))

    def raw_peak_fraction_metric(state, **kw) -> float:
        return float(state.get("raw_peak_fraction", 0.0))

    mechanical = vf.Rubric(
        funcs=[mechanical_reward, correct_metric, peak_fraction_metric, raw_peak_fraction_metric],
        weights=[1.0, 0.0, 0.0, 0.0],
        parser=parser,
    )

    if not enable_judge:
        def judge_noop(state, **kw) -> float:
            return 0.0

        noop = vf.Rubric(funcs=[judge_noop], weights=[0.0], parser=parser)
        return vf.RubricGroup([mechanical, noop])

    async def judge_veto(completion, answer, state, info, **kw) -> float:
        if not state.get("correct") or state.get("peak_fraction", 0.0) <= 0.0:
            return 1.0
        try:
            from openai import OpenAI
        except Exception:
            return 1.0
        problem = (info or {}).get("problem") or answer
        ws = state.get("_ws")
        if ws and (problem_dir(ws, problem) / "solution.py").exists():
            code = (problem_dir(ws, problem) / "solution.py").read_text()
        else:
            code = extract_solution(completion)
        base_url = judge_base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        api_key = judge_api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return 1.0  # fail open

        def _ask() -> bool:
            client = OpenAI(base_url=base_url, api_key=api_key)
            msg = (
                "You are a strict reviewer for GPU-kernel RL. The solution passed correctness "
                "and a roofline benchmark. Decide if it is a LEGITIMATE custom kernel "
                "(real CUDA/Triton/PTX/CUTLASS compute) and NOT a reward hack (caching outputs, "
                "disguised vendor-library passthrough, exploiting the timing harness). Reply with "
                'strict JSON {"legitimate": true|false, "reason": "..."}.\n\n'
                f"Problem: {problem}\n\nsolution.py:\n```python\n{code[:12000]}\n```"
            )
            r = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": msg}],
                temperature=0.0,
            )
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if not m:
                return True
            try:
                return bool(json.loads(m.group(0)).get("legitimate", True))
            except Exception:
                return True

        try:
            legit = await asyncio.to_thread(_ask)
        except Exception:
            legit = True
        state["judge_legitimate"] = legit
        return 1.0 if legit else 0.0

    def judge_metric(state, **kw) -> float:
        return 1.0 if state.get("judge_legitimate", True) else 0.0

    judge = vf.Rubric(funcs=[judge_veto, judge_metric], weights=[1.0, 0.0], parser=parser)
    return _MultiplicativeRubricGroup(mechanical, judge, parser)


class _MultiplicativeRubricGroup(vf.Rubric):
    """final reward = mechanical_reward * judge_veto (judge in {0.0 veto, 1.0 pass})."""

    def __init__(self, mechanical: vf.Rubric, judge: vf.Rubric, parser):
        super().__init__(funcs=[lambda **kw: 0.0], weights=[0.0], parser=parser)
        self._mechanical = mechanical
        self._judge = judge

    async def score_rollout(self, state):
        await self._mechanical.score_rollout(state)
        mech_reward = state.get("reward", 0.0)
        mech_metrics = (state.get("metrics", {}) or {}).copy()
        await self._judge.score_rollout(state)
        veto = state.get("reward", 1.0)
        judge_metrics = (state.get("metrics", {}) or {}).copy()
        merged = {}
        merged.update(mech_metrics)
        merged.update(judge_metrics)
        state["reward"] = float(mech_reward) * float(veto)
        state["metrics"] = merged


# ---------------------------------------------------------------------------
# Public builder used by kernel_hard.py / kernel_mega.py
# ---------------------------------------------------------------------------
def build_environment(
    *,
    bench_root: str,
    hardware: str = "RTX_PRO_6000",
    deck: list[str] | None = None,
    eval_frac: float = 0.2,
    max_turns: int = 12,
    check_timeout_s: int = 600,
    bench_timeout_s: int = 900,
    enable_judge: bool = False,
    judge_model: str = "z-ai/glm-5.2",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    max_concurrent: int = 1,
    **kwargs,
) -> vf.Environment:
    bench_root = os.path.abspath(bench_root)
    problems = discover_problems(bench_root, deck)
    if not problems:
        raise ValueError(f"no problems found under {bench_root}/problems (deck={deck})")

    train = [p for p in problems if not _is_eval(p, eval_frac)]
    eval = [p for p in problems if _is_eval(p, eval_frac)]
    if not train:
        train, eval = problems, []

    dataset = Dataset.from_list(_make_rows(bench_root, train, hardware))
    eval_dataset = Dataset.from_list(_make_rows(bench_root, eval, hardware)) if eval else None

    parser = vf.Parser(extract_fn=extract_solution)
    sem = asyncio.Semaphore(max_concurrent)

    rubric = _build_rubric(
        bench_root=bench_root,
        check_timeout_s=check_timeout_s,
        bench_timeout_s=bench_timeout_s,
        sem=sem,
        parser=parser,
        enable_judge=enable_judge,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
    )

    env_passthrough = {k: v for k, v in kwargs.items() if k in ENV_PASSTHROUGH}
    return KernelHarnessEnv(
        bench_root=bench_root,
        check_timeout_s=check_timeout_s,
        bench_timeout_s=bench_timeout_s,
        sem=sem,
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **env_passthrough,
    )
