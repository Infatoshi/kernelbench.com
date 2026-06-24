"""kernel_v3: verifiers/Prime RL environment wrapping the hardened v3 kernel-benchmark harness.

Design (settled with user):
- Reward is MECHANICAL-PRIMARY. Write the model's solution to a v3 LocalSandbox, call v3's
  `run_benchmark`, and set reward = pct_of_peak (% of roofline, normalized to [0,1]) if the
  solution is CORRECT, else 0.0. Correctness (v3's multi-seed + determinism/race + guardrails)
  is a HARD GATE. The PyTorch reference is correctness-only; we do NOT use speedup-vs-eager as
  the reward.
- Judge is an OPT-IN VETO, OFF by default. Only fires when state.correct AND pct_of_peak>0; it
  can only zero the reward (never adds). Implemented as a weight-1 reward func that returns 1.0
  (pass-through) normally and 0.0 when the v3 judge says illegitimate; composed multiplicatively
  with the mechanical reward via vf.RubricGroup. When disabled it is a no-op (weight 0, returns 0).
- correct + pct_of_peak surfaced as weight-0 metric reader funcs.

CPU-import-safe: torch / v3 eval are imported lazily inside the reward, never at module top.

SELF-CONTAINED (option 1): v3's eval (src/eval), sandbox (src/agent), config (src/config) and
the problems/ tree are VENDORED into this env dir at `_v3/` (listed in [tool.hatch.build].include).
We add the vendored `_v3/` dir to sys.path so the original `from src.config...` / `from src.eval...`
imports resolve against the vendored copy. The reward runs via the ENV's own python+torch: the
vendored LocalSandbox.run_command rewrites bare `python` to sys.executable so GPU subprocesses use
this venv's torch (no benchmark uv env, no checkout). torch + benchmark deps are in pyproject.
TODO(modal): sandbox="modal" path is stubbed -- wire src/agent/modal_sandbox.py (Modal B200) for
the Hub/remote-GPU path; today only sandbox="local" is implemented.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

# ---------------------------------------------------------------------------
# v3 location: the VENDORED copy bundled in this env dir (`_v3/`). Self-contained;
# NO checkout dependency. (`KERNEL_V3_ROOT` can still override for local dev, but
# the default is the vendored dir and nothing falls back to a repo checkout.)
# ---------------------------------------------------------------------------
V3_ROOT = os.environ.get(
    "KERNEL_V3_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v3"),
)

# v3's gpu_sku for the local Blackwell box is "RTX_PRO_6000", but
# src/config/precision_matrix.py ships NO peak-TFLOPS row for it, so
# compute_percent_of_peak() returns None for every problem (pct_of_peak reward
# would be dead). We additively inject a peak row (NVIDIA RTX PRO 6000 Blackwell
# Workstation datasheet, dense tensor TFLOPS; fp32 = non-tensor peak). This is a
# DATA patch, not eval-logic -- v3's compute_percent_of_peak is reused verbatim.
# fp32 entry is REQUIRED: the default level-1 matmuls run at fp32 precision.
# NOTE on the fp32 entry (calibration / anti-saturation):
# A torch fp32 matmul on Blackwell actually executes on TF32 TENSOR cores (the
# known-good Triton kernels use input_precision="tf32"), measured ~122 TFLOPS.
# v3 keys the peak lookup by precision="fp32". If we set fp32 -> 125 (the FP32
# CUDA-core FMA ceiling), a STOCK matmul already reads ~97% of roofline and the
# 122->~250 TFLOPS optimization range clips flat into [0.97, 1.0] -> a SATURATED
# reward with no perf gradient. To keep headroom for RL, we set the fp32-key peak
# to the TF32 TENSOR ceiling (~252), so a stock matmul reads ~0.48 and a fully
# optimized tf32 matmul approaches 1.0. Flip to 125.0 if you want literal
# fp32-CUDA-core roofline semantics (saturated). See README.
RTX_PRO_6000_PEAK_TFLOPS = {
    "fp32": 252.0,   # TF32 tensor dense ceiling (anti-saturation; see note above)
    "tf32": 252.0,   # TF32 tensor dense (= fp16/2, standard Blackwell ratio)
    "fp16": 504.0,   # FP16 tensor dense
    "bf16": 504.0,   # BF16 tensor dense
    "fp8": 1008.0,   # FP8 tensor dense
    "fp4": 2016.0,   # FP4 tensor dense
}
RTX_PRO_6000_PRECISIONS = ["fp4", "fp8", "fp16", "bf16", "fp32"]

CODE_RE = re.compile(r"```(?:python|cpp|cuda|py)?\n(.*?)```", re.DOTALL)

# v3's compute_tflops() only has a COMPUTE roofline (non-None pct_of_peak) for
# gemm + attention. All other ops (softmax, layernorm, reduction, conv,
# elementwise, fused, model) are MEMORY-BOUND and return pct_of_peak=None; the
# env scores them via a MEMORY-BANDWIDTH roofline (_memory_roofline_fraction), so
# the full op set now scores. Default dataset to ALL ops; subset via op_types
# kwarg (None = all).
ROOFLINE_OP_TYPES = None  # None -> include every op_type
COMPUTE_ROOFLINE_OP_TYPES = ("gemm", "attention")  # ops v3 scores via pct_of_peak

# Peak HBM bandwidth (GB/s) per NVIDIA arch, keyed by the env's `hardware` kwarg
# (v3 gpu_sku). Used for the memory-bound roofline: peak_fraction = achieved/peak.
# RTX_PRO_6000 (Blackwell Workstation, sm_120) datasheet ~1792 GB/s -> 1800.
# RTX3090 (sm_86) 936, B200 (sm_100) 8000 HBM3e, H100 (sm_90) 3350 SXM HBM3.
PEAK_BANDWIDTH_GBPS = {
    "RTX_PRO_6000": 1800.0,
    "RTX3090": 936.0,
    "B200": 8000.0,
    "H100": 3350.0,
}


# ---------------------------------------------------------------------------
# Lazy v3 bootstrap (no torch / v3 imports at module top)
# ---------------------------------------------------------------------------
def _ensure_v3_on_path() -> None:
    import sys

    if V3_ROOT not in sys.path:
        sys.path.insert(0, V3_ROOT)


def _patch_peak_table() -> None:
    """Additively inject RTX_PRO_6000 peak data. Safe: never mutates existing keys.

    benchmark.py binds the same dict objects it imports from precision_matrix, so
    setdefault on those objects is seen by run_benchmark at call time.
    """
    _ensure_v3_on_path()
    from src.config import precision_matrix as pm

    pm.HARDWARE_PEAK_TFLOPS.setdefault("RTX_PRO_6000", dict(RTX_PRO_6000_PEAK_TFLOPS))
    pm.HARDWARE_PRECISIONS.setdefault("RTX_PRO_6000", list(RTX_PRO_6000_PRECISIONS))


# ---------------------------------------------------------------------------
# Memory-bandwidth roofline (for ops v3 has no compute roofline for)
# ---------------------------------------------------------------------------
# Sandbox-side script: load the v3 reference, sum input bytes (read once) + output
# bytes (write once) = minimal HBM traffic for one forward. Runs with the SAME
# `python` (torch-equipped) that ran the benchmark, so the env process stays
# torch-free. Prints a one-line JSON `{"bytes_moved": N}`.
_BYTES_MOVED_SCRIPT = r'''
import importlib.util, json, sys
import torch
def _bytes(o):
    if isinstance(o, torch.Tensor):
        return o.element_size() * o.nelement()
    if isinstance(o, (list, tuple)):
        return sum(_bytes(v) for v in o)
    return 0
try:
    spec = importlib.util.spec_from_file_location("kb_reference", "reference.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    with torch.no_grad():
        inputs = m.get_inputs()
        in_bytes = _bytes(inputs)
        model = m.Model(*m.get_init_inputs()).eval()
        out = model(*inputs)
    out_bytes = _bytes(out)
    print(json.dumps({"bytes_moved": int(in_bytes + out_bytes)}))
except Exception as e:
    print(json.dumps({"bytes_moved": None, "error": str(e)}))
'''


def _compute_bytes_moved_in_sandbox(sandbox) -> float | None:
    """Run _BYTES_MOVED_SCRIPT in the live sandbox; return total bytes or None."""
    import json

    try:
        sandbox.write_file("_bytes_moved.py", _BYTES_MOVED_SCRIPT)
        result = sandbox.run_command("python _bytes_moved.py", timeout=120)
        for line in result.get("stdout", "").split("\n"):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                bm = data.get("bytes_moved")
                return float(bm) if bm else None
    except Exception:
        return None
    return None


def _memory_roofline_fraction(res: dict, hardware: str) -> float | None:
    """peak_fraction = achieved_GBps / peak_BW_GBps, in [0,1]. None if undeterminable.

    Uses kernel time from the v3 result dict key `sol_ms` (solution median ms) and
    `bytes_moved` (stashed by _run_v3_benchmark from the live sandbox).
    """
    sol_ms = res.get("sol_ms")
    if sol_ms is None or sol_ms <= 0:
        return None
    peak_bw = PEAK_BANDWIDTH_GBPS.get(hardware)
    if peak_bw is None or peak_bw <= 0:
        return None
    bytes_moved = res.get("bytes_moved")
    if bytes_moved is None or bytes_moved <= 0:
        return None
    achieved_gbps = bytes_moved / (sol_ms / 1000.0) / 1e9
    return max(0.0, min(1.0, achieved_gbps / peak_bw))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _extract(completion) -> str:
    if isinstance(completion, list):
        text = completion[-1]["content"] if completion else ""
    else:
        text = str(completion)
    matches = CODE_RE.findall(text)
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def _read_problem_metadata(reference_code: str) -> dict:
    _ensure_v3_on_path()
    from src.eval.context import extract_reference_metadata

    return extract_reference_metadata(reference_code)


def _build_question(problem_name: str, level: int, hardware: str, reference_code: str, metadata: dict) -> str:
    """Self-contained task prompt (does not need a live sandbox).

    Mirrors v3's context.py pieces: API reference, template solution, task context,
    and the reference code -- so the prompt matches what v3 expects the model to solve.
    """
    _ensure_v3_on_path()
    from src.eval import context as ctx

    api_ref = ctx._build_api_reference("nvidia")
    template = ctx._build_template_solution("nvidia")
    task_ctx = ctx.build_task_context(problem_name, level, hardware, hardware, metadata)
    return (
        f"Optimize the benchmark task and produce a `solution.py` for NVIDIA GPU `{hardware}`.\n\n"
        "Your `solution.py` MUST define `class Model(nn.Module)` (same interface as the reference),\n"
        "plus `get_inputs()` and `get_init_inputs()`. Use CUDA C++ (`load_inline`) or Triton.\n"
        "No PyTorch operator fallbacks (torch.matmul, F.linear, ...). Return ONE ```python``` block.\n\n"
        f"{task_ctx}\n"
        f"{api_ref}\n"
        f"[TEMPLATE_solution.py]\n```python\n{template}\n```\n\n"
        f"Reference code:\n```python\n{reference_code}\n```\n"
    )


def _level_dirs(levels) -> list[int]:
    if levels is None:
        return [1, 2, 3, 4]
    if isinstance(levels, str):
        return [int(x) for x in re.split(r"[,\s]+", levels.strip()) if x]
    if isinstance(levels, int):
        return [levels]
    return [int(x) for x in levels]


def _is_eval(problem_id: str, eval_frac: float) -> bool:
    """Deterministic hash-based held-out split. Disjoint train/eval."""
    if eval_frac <= 0:
        return False
    h = int(hashlib.sha256(problem_id.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def _build_rows(levels, hardware: str, op_types) -> list[dict]:
    import glob

    rows: list[dict] = []
    op_filter = set(op_types) if op_types else None
    for level in _level_dirs(levels):
        pat = os.path.join(V3_ROOT, "problems", f"level{level}", "*.py")
        for path in sorted(glob.glob(pat)):
            problem_name = os.path.basename(path)
            reference_code = open(path).read()
            metadata = _read_problem_metadata(reference_code)
            op_type = str(metadata.get("op_type", "unknown")).lower()
            if op_filter is not None and op_type not in op_filter:
                continue
            problem_id = f"level{level}/{problem_name}"
            rows.append(
                {
                    "question": _build_question(problem_name, level, hardware, reference_code, metadata),
                    "answer": reference_code,
                    "info": {
                        "problem": problem_name,
                        "problem_id": problem_id,
                        "level": level,
                        "op_type": op_type,
                        "hardware": hardware,
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Mechanical evaluation (the reward core). Synchronous; called via the rubric.
# ---------------------------------------------------------------------------
def _run_v3_benchmark(solution_code: str, reference_code: str, hardware: str, level: int) -> dict:
    """Write solution to a v3 LocalSandbox and call v3's run_benchmark verbatim."""
    _ensure_v3_on_path()
    _patch_peak_table()
    from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig
    from src.eval.benchmark import run_benchmark

    sb = LocalSandbox(reference_code, LocalSandboxConfig(timeout=600))
    sb.start()
    try:
        sb.write_file("solution.py", solution_code)
        res = run_benchmark(sb, "solution.py", hardware, level, False)
        # Stash minimal HBM traffic (read inputs + write output once) for the
        # memory-bandwidth roofline. Only needed when v3 has no compute roofline.
        if res.get("correct") and res.get("pct_of_peak") is None:
            res["bytes_moved"] = _compute_bytes_moved_in_sandbox(sb)
        return res
    finally:
        sb.stop()


def _reward_and_roofline(res: dict, hardware: str) -> tuple[float, str]:
    """Reward in [0,1] + which roofline produced it ("compute"|"memory"|"none").

    Correctness is a hard gate. When v3 supplies a COMPUTE roofline (pct_of_peak,
    gemm/attention) use it. Otherwise the op is memory-bound -> compute a memory-
    bandwidth peak_fraction from sol_ms + bytes_moved.
    """
    if not res.get("correct"):
        return 0.0, "none"
    pct = res.get("pct_of_peak")
    if pct is not None:
        return max(0.0, min(1.0, float(pct) / 100.0)), "compute"
    frac = _memory_roofline_fraction(res, hardware)
    if frac is not None:
        return frac, "memory"
    return 0.0, "none"


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------
def load_environment(
    *,
    levels: Any = "1,2,3,4",
    hardware: str = "RTX_PRO_6000",
    sandbox: str = "local",
    op_types: Any = ROOFLINE_OP_TYPES,
    eval_frac: float = 0.2,
    enable_judge: bool = False,
    judge_model: str = "claude-opus-4-6",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    max_concurrent: int = 1,
    **kwargs,
) -> vf.Environment:
    """Build the kernel_v3 verifiers environment.

    Args:
        levels: "1,2,3,4" or list/int. Which v3 problem levels to include.
        hardware: v3 gpu_sku passed to run_benchmark. Default RTX_PRO_6000 (anvil/sm_120).
        sandbox: "local" (implemented) or "modal" (TODO).
        op_types: restrict to these op_types (default None = ALL 42 problems). gemm/attention
            score via v3's compute roofline; every other op scores via the env's memory-
            bandwidth roofline. Pass e.g. ("softmax",) to subset.
        eval_frac: fraction held out into a deterministic, disjoint eval split (hash-based).
        enable_judge: OFF by default. When True, the judge can VETO (zero) a correct, pct>0 reward.
        judge_model / judge_base_url / judge_api_key: judge config (v3 judge_solution model key).
        max_concurrent: cap concurrent GPU benchmarks (run_benchmark also holds an internal GPU lock).
    """
    if sandbox not in ("local", "modal"):
        raise ValueError(f"sandbox must be 'local' or 'modal', got {sandbox!r}")
    if sandbox == "modal":
        raise NotImplementedError("sandbox='modal' not yet wired; use sandbox='local'. See module TODO.")

    if isinstance(op_types, str):
        op_types = [x for x in re.split(r"[,\s]+", op_types.strip()) if x]

    all_rows = _build_rows(levels, hardware, op_types)
    train_rows = [r for r in all_rows if not _is_eval(r["info"]["problem_id"], eval_frac)]
    eval_rows = [r for r in all_rows if _is_eval(r["info"]["problem_id"], eval_frac)]
    # If eval_frac would empty train (tiny level sets), keep all in train.
    if not train_rows:
        train_rows, eval_rows = all_rows, []

    dataset = Dataset.from_list(train_rows) if train_rows else None
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    parser = vf.Parser(extract_fn=_extract)

    import asyncio

    sem = asyncio.Semaphore(max_concurrent)

    async def mechanical_reward(completion, answer, state, info, **kw) -> float:
        solution_code = _extract(completion)
        state["correct"] = False
        state["pct_of_peak"] = 0.0
        state["peak_fraction"] = 0.0
        state["roofline_kind"] = "none"
        state["speedup"] = 0.0
        if not solution_code:
            return 0.0
        hw = info.get("hardware", hardware)
        async with sem:
            res = await asyncio.to_thread(
                _run_v3_benchmark,
                solution_code,
                answer,
                hw,
                int(info.get("level", 1)),
            )
        state["correct"] = bool(res.get("correct", False))
        pct = res.get("pct_of_peak")
        state["pct_of_peak"] = float(pct) if pct is not None else 0.0
        sp = res.get("speedup")
        state["speedup"] = float(sp) if sp is not None else 0.0
        state["_bench"] = res
        reward, kind = _reward_and_roofline(res, hw)
        state["peak_fraction"] = float(reward)
        state["roofline_kind"] = kind
        return reward

    def correct_metric(state, **kw) -> float:
        return 1.0 if state.get("correct") else 0.0

    def pct_of_peak_metric(state, **kw) -> float:
        return float(state.get("pct_of_peak", 0.0))

    def peak_fraction_metric(state, **kw) -> float:
        return float(state.get("peak_fraction", 0.0))

    mechanical_rubric = vf.Rubric(
        funcs=[mechanical_reward, correct_metric, pct_of_peak_metric, peak_fraction_metric],
        weights=[1.0, 0.0, 0.0, 0.0],
        parser=parser,
    )

    # --- Judge veto rubric (opt-in). Multiplicative: 1.0 pass-through, 0.0 = veto. ---
    if enable_judge:

        async def judge_veto(completion, answer, state, info, **kw) -> float:
            # Only fires on a correct, scored solution; can only zero the reward.
            if not state.get("correct") or state.get("peak_fraction", 0.0) <= 0.0:
                return 1.0
            solution_code = _extract(completion)
            _ensure_v3_on_path()
            from src.eval.judge import judge_solution

            if judge_base_url:
                os.environ.setdefault("OPENAI_BASE_URL", judge_base_url)
            if judge_api_key:
                os.environ.setdefault("OPENAI_API_KEY", judge_api_key)
            verdict = await asyncio.to_thread(
                judge_solution,
                judge_model,
                answer,
                solution_code,
                info.get("problem", ""),
                state.get("_bench", {}),
            )
            state["judge_legitimate"] = bool(verdict.get("legitimate", True))
            state["judge_reason"] = verdict.get("reason", "")
            # On judge_error, v3 returns legitimate=True -> no veto (fail open).
            return 1.0 if verdict.get("legitimate", True) else 0.0

        def judge_metric(state, **kw) -> float:
            return 1.0 if state.get("judge_legitimate", True) else 0.0

        judge_rubric = vf.Rubric(
            funcs=[judge_veto, judge_metric],
            weights=[1.0, 0.0],
            parser=parser,
        )
        # RubricGroup sums sub-rubric rewards by default; we need MULTIPLICATIVE veto.
        # Compose by wrapping: final reward = mechanical * judge_veto. Implemented as a
        # single combined rubric so the weighted SUM stays correct (judge is pass/veto).
        rubric = _MultiplicativeRubricGroup(mechanical_rubric, judge_rubric, parser)
    else:
        # No-op judge rubric (weight 0, returns 0) so the rubric shape is stable.
        def judge_noop(state, **kw) -> float:
            return 0.0

        noop_rubric = vf.Rubric(funcs=[judge_noop], weights=[0.0], parser=parser)
        rubric = vf.RubricGroup([mechanical_rubric, noop_rubric])

    env_kwargs = {k: v for k, v in kwargs.items() if k in _ENV_PASSTHROUGH}
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        **env_kwargs,
    )


_ENV_PASSTHROUGH = {
    "system_prompt",
    "few_shot",
    "sampling_args",
    "max_workers",
    "max_seq_len",
    "message_type",
}


class _MultiplicativeRubricGroup(vf.Rubric):
    """Combines a mechanical rubric and a judge-veto rubric multiplicatively.

    verifiers' score_rollout mutates state["reward"]/state["metrics"] in place
    (returns None). RubricGroup SUMS sub-rubric rewards; we need the judge to be a
    MULTIPLICATIVE veto (judge reward in {0.0 veto, 1.0 pass}), so we override.
    final reward = mechanical_reward * judge_veto. Metrics from both are merged.
    """

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
