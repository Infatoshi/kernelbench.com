# Build-Ready Spec: Kernel-Code-Gen RL Environments for verifiers + prime-rl (8xB200)

Verified against live source June 18, 2026. All signatures below are quoted from cloned repos, not from a summarizer.

## Versions verified (pin these)
| Component | Version | Date | Source |
|---|---|---|---|
| `verifiers` (PyPI) | **0.1.14** (main also tags `0.1.15.dev18`, 2026-06-01) | 2026-05-07 | https://pypi.org/project/verifiers/ |
| `prime` CLI (PyPI) | **0.6.15** | 2026-06-18 | https://pypi.org/project/prime/ |
| `prime-rl` (GitHub) | **v0.5.0** | 2026-03-30 | https://github.com/PrimeIntellect-ai/prime-rl/releases |
| Python | `>=3.10,<3.14` (use `>=3.11` for kernel envs, matching kernelbench) | | |

REPO MOVED: `github.com/willccbb/verifiers` -> `github.com/PrimeIntellect-ai/verifiers`. Community envs: `github.com/PrimeIntellect-ai/prime-environments`.

---

## (a) TIGHT SPEC

### 1. Environment classes (verbatim, current main)
Hierarchy: `SingleTurnEnv -> MultiTurnEnv -> Environment(ABC)`. Also `ToolEnv`, `StatefulToolEnv`, `SandboxEnv`, `PythonEnv`, `EnvGroup`.

```python
class SingleTurnEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(max_turns=1, **kwargs)
```
Real surface is the base:
```python
class Environment(ABC):
    def __init__(self, dataset=None, eval_dataset=None, system_prompt=None,
                 few_shot=None, parser=None, rubric=None, sampling_args=None,
                 message_type=_MESSAGE_TYPE_UNSET, tool_defs=None,
                 max_workers=512, env_id=None, env_args=None, map_kwargs={},
                 max_seq_len=None, score_rollouts=True, pass_threshold=0.5, **kwargs):
```
`import verifiers as vf` then `vf.SingleTurnEnv`, `vf.Rubric`, `vf.Parser`, etc.

**Dataset format/columns.** A HuggingFace `datasets.Dataset` (or a zero-arg builder callable returning one — both work; kernelbench/gsm8k pass `eval_dataset=build_fn`). Column contract:
- Required: **`question`** OR **`prompt`** (chat list `[{"role","content"}]`). `question_key` default `"question"`.
- **`answer`** optional, `answer_key` default `"answer"`. (Kernel envs put the PyTorch reference source here.)
- **`info`** optional dict, passed through to rewards (put `problem_id`, shapes, dtype, arch here).
- **`task`** optional, for multi-task / `EnvGroup` routing.
- `example_id` auto-added if missing.

For kernel envs: `question` = problem prompt, `answer` = reference PyTorch module source, `info` = `{problem_id, level, ...}`.

### 2. Reward / Rubric API (verbatim, current main `verifiers/rubrics/rubric.py`)
```python
class Rubric:
    def __init__(self, funcs=None, weights=None, parser=None): ...
    def add_reward_func(self, func, weight=1.0): ...
    async def score_rollout(self, state):   # individual funcs (current name)
    async def score_group(self, states):    # group funcs, asyncio.gather
```
- Construct: `vf.Rubric(funcs=[f1, f2], weights=[1.0, 0.0], parser=parser)`.
- **Reward fn signature: introspected by parameter name.** Available kwargs the rubric injects: **`prompt, completion, answer, state, info, parser, task, **kwargs`**. Declare only the ones you want (or accept `**kwargs`).
- **RETURN TYPE = `float`.** Current source coerces: `ans = float(await maybe_await(func, **allowed))` (lines 201/210). A reward func returning a `vf.RolloutScore` will break on current main (that is the STALE 0.1.3 kernelbench API — do NOT copy it).
- **Async supported** (`maybe_await` handles sync and `async def`). Errors are caught -> 0.0.
- **Final reward** = weighted sum: `sum(reward*weight)`.
- **Metrics**: `score_rollout` auto-records `state["metrics"][func.__name__]` per reward func. To surface extra metrics (speedup, fast tiers), stash them in `state[...]` from your main reward, then add weight-0 reward funcs that read them back (kernelbench `make_state_metric_extractor` pattern).
- **Group rewards**: `GroupRewardFunc`/`RubricGroup`/`score_group` score across a whole rollout group (return `list[float]`).
- **NO per-reward timeout exists in the Rubric.** You must enforce timeout in the reward (sandbox `timeout_per_command_seconds`, or `asyncio.wait_for`).

### 3. Packaging convention (verbatim from real envs)
```
environments/<my_env>/
  <my_env>.py        # module named after env; defines load_environment(...)
  pyproject.toml
  README.md
  utils/ ...          # optional
```
`pyproject.toml` (from kernelbench, current):
```toml
[project]
name = "kernelbench"                 # hub name, hyphenated
version = "0.1.5"                    # semver
description = "..."
tags = ["gpu","cuda","single-turn","sandbox","coding","eval"]
requires-python = ">=3.11"
dependencies = ["verifiers>=0.1.3", "datasets", "prime>=0.5.0", "modal", "torch"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["kernelbench.py", "utils", "prompts"]

[tool.verifiers.eval]                # optional defaults for vf-eval/prime eval
num_examples = 5
rollouts_per_example = 3
```
**Entrypoint is a CONVENTION, not an entry-point table:** a top-level function named exactly `load_environment(...)` in `<my_env>.py`. The Hub/prime-rl import the module and call it.

`load_environment` contract (TWO coexisting forms; build the classic one):
- **Classic (USE THIS):** `def load_environment(**flat_kwargs) -> vf.Environment:` — flat keyword args with defaults; returns a constructed `vf.SingleTurnEnv` (or other Environment). This is what prime-rl `args = {...}` maps onto. Verbatim kernelbench: `def load_environment(levels=None, ..., gpu="L40S", ...) -> vf.SingleTurnEnv:`.
- **v1 (opt-in, `prime env init --v1`):** `def load_environment(config: vf.EnvConfig) -> vf.Env:` with `Taskset`/`Harness`/`@vf.reward`. New, not required. The math_python env ships BOTH (`math_python.py` classic wraps `math_python_v1.py`).

### 4. Hub discovery / install / train consumption
**CLI (`prime` package):**
```
prime env init <name>            # scaffold (--v1 for v1 layout)
prime env install <owner/name>   # install from Hub
prime env push                   # publish current env dir; --auto-bump bumps version; --team <t>
prime env list / info / inspect
vf-install <env>                 # verifiers-native installer (equivalent path)
vf-eval <env>                    # run eval rollouts vs an API model
```
Hub name = `owner/name`; version = semver in `pyproject.toml`. Metadata source = `pyproject.toml` (no separate manifest).

**prime-rl training (3 async processes, each its own TOML):**
```
uv run inference    @ infer.toml     # vLLM, OpenAI-compatible server
uv run orchestrator @ orch.toml      # runs the verifiers env rollouts vs vLLM
uv run trainer      @ train.toml     # FSDP2 + GRPO
```
**Env reference in TOML (VERIFIED verbatim from prime-rl configs):**
```toml
[[orchestrator.train.env]]
id   = "my-kernel-env"          # installed env name
name = "kernel"                  # label
args = { gpu = "B200", sandbox_client_max_workers = 128, num_perf_trials = 50 }
                                 # args == kwargs passed straight to your load_environment(**args)
```
GPU split lives in `[deployment]`: `num_train_gpus = 4`, `num_infer_gpus = 4` (for 8xB200, e.g. 4 train + 4 infer; tune).

**Division of labor:**
- ENV AUTHOR provides: the env package (dataset/`question`+`answer`+`info`, parser, rubric/reward, `load_environment(**kwargs)`), published to Hub. Plus every concurrency/timeout knob as a `load_environment` kwarg (see below).
- TRAINER provides: model, GPU layout, vLLM config, GRPO/FSDP hyperparams, the 3 TOMLs that name your env.

veRL <-> verifiers: NOT confirmed as a first-class consumer; prime-rl is the verified trainer. Treat veRL support as unverified.

### 5. Long-running / code-exec GPU rewards (THE core design decision)
**Every real GPU-kernel env runs the compile+timing on a GPU that is NOT a training GPU.** On 8xB200 with the unchanged stack, vLLM + FSDP saturate the GPUs; timing a kernel under that contention is meaningless. So the build-ready answer is: reward GPU != training GPU. Two compliant ways:

- **(a) Remote sandbox-as-a-service (the unchanged-stack default).**
  - **Modal** (kernelbench, backend_bench): a `modal.Function` on an `nvidia/cuda` image compiles the kernel (torch C++/CUDA extension), runs correctness vs reference, times speedup. Reward calls it via `await asyncio.to_thread(modal_eval_kernel(...))` so the event loop keeps scheduling. Set the Modal GPU to `"B200"`.
  - **Prime sandboxes** (`vf.SandboxEnv`/`PythonEnv`, math_python): `CreateSandboxRequest(docker_image=..., gpu_count=1, timeout_minutes=..., timeout_per_command_seconds=...)`. `SandboxEnv.__init__` exposes `gpu_count`, `cpu_cores`, `memory_gb`, `disk_size_gb`, `timeout_per_command_seconds`, `sandbox_client_max_workers`. Thread these to `load_environment` kwargs.
  - Reward call is per-completion and `async`; concurrency is bounded by your sandbox client worker pool, NOT a global semaphore.

- **(b) Held-out LOCAL GPU via subprocess.** Reserve 1+ of the 8 B200s, NOT given to vLLM/trainer. Reward shells out (`asyncio.create_subprocess_exec`) with `CUDA_VISIBLE_DEVICES` pinned to the held-out GPU, `torch.utils.cpp_extension.load_inline` inside, `asyncio.wait_for` timeout. Outside the standard pattern; you own isolation; shrinks training pool. "Stacks UNCHANGED" points at (a).

**Concurrency / timeout knobs (current, verified):**
- prime-rl REMOVED `orchestrator.max_concurrent` and its global semaphore (CHANGELOG 2026-04-05). Concurrency is now bounded by ENV-AUTHOR-DEFINED `args` knobs, e.g. `sandbox_client_max_workers`, `math_verify_max_workers`, `rubric_max_workers` (these are kwargs YOUR `load_environment` defines and uses — not built-ins). Real configs use `sandbox_client_max_workers = 128/256`.
- `Environment(max_workers=512)` is a sync-bridge ThreadPool default, not a reward-rate cap — for compile-heavy rewards cap concurrency yourself via your sandbox worker pool / an `asyncio.Semaphore` you create in `load_environment`.
- Per-command timeout: sandbox `timeout_per_command_seconds` (Prime) / `timeout=` in `modal.Function`/`Sandbox.exec.aio` (Modal). No Rubric-level timeout.

---

## (b) MINIMAL COMPLETE COMPLIANT ENV (current API, reward shells out to compile+run)

`environments/kernel_cuda/kernel_cuda.py`:
```python
import asyncio, json, re
import verifiers as vf
from datasets import Dataset

CODE_RE = re.compile(r"```(?:python|cpp|cuda)?\n(.*?)```", re.DOTALL)

def _extract(completion) -> str:
    text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
    m = CODE_RE.findall(text)
    return m[-1].strip() if m else ""

async def _run_eval(candidate_src: str, ref_src: str, gpu_id: int, timeout_s: int) -> dict:
    """Compile candidate + reference in a subprocess on a held-out GPU; return metrics."""
    payload = json.dumps({"candidate": candidate_src, "ref": ref_src}).encode()
    proc = await asyncio.create_subprocess_exec(
        "python", "eval_worker.py",
        env={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(payload), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill(); return {"correct": 0.0, "speedup": 0.0}
    try:
        return json.loads(out.decode().strip().splitlines()[-1])
    except Exception:
        return {"correct": 0.0, "speedup": 0.0}

def load_environment(
    dataset_name: str = "my/kernel-dataset",
    gpu_id: int = 7,                 # held-out B200 not given to vLLM/trainer
    reward_timeout_s: int = 120,
    max_concurrent_compiles: int = 4,
    speedup_cap: float = 4.0,
    **kwargs,
) -> vf.Environment:
    rows = [{
        "question": "Write a fused CUDA kernel for the reference module. Return one ```cuda``` block.",
        "answer": "import torch\nclass Model(torch.nn.Module): ...",   # PyTorch reference source
        "info": {"problem_id": "demo-1"},
    }]
    dataset = Dataset.from_list(rows)

    sem = asyncio.Semaphore(max_concurrent_compiles)   # cap concurrent compiles ourselves
    parser = vf.Parser(extract_fn=_extract)

    async def kernel_reward(completion, answer, state, info, **kw) -> float:
        cand = _extract(completion)
        if not cand:
            state["correct"] = 0.0; state["speedup"] = 0.0
            return 0.0
        async with sem:
            res = await _run_eval(cand, answer, gpu_id, reward_timeout_s)
        state["correct"] = float(res.get("correct", 0.0))
        state["speedup"] = float(res.get("speedup", 0.0))
        if not state["correct"]:
            return 0.0
        return min(state["speedup"], speedup_cap) / speedup_cap   # in [0,1]

    def correct_metric(state, **kw) -> float:  return state.get("correct", 0.0)
    def speedup_metric(state, **kw) -> float:  return state.get("speedup", 0.0)

    rubric = vf.Rubric(
        funcs=[kernel_reward, correct_metric, speedup_metric],
        weights=[1.0, 0.0, 0.0],          # only kernel_reward affects the gradient
        parser=parser,
    )
    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
```
For the unchanged-stack default, swap `_run_eval`'s subprocess for `await asyncio.to_thread(modal_eval_kernel(..., gpu="B200"))` (copy structure from prime-environments/kernelbench `utils/modal_sandbox.py`).

pyproject.toml: as in section 3, `name="kernel-cuda"`, deps `["verifiers>=0.1.14","datasets","torch"]` (+`"modal"` for option a).

---

## (c) GOTCHAS

**Compile-heavy GPU rewards:**
1. Reward GPU must NOT be a training GPU. On 8xB200, either split `[deployment]` and hold one out (local subprocess), or offload to Modal/Prime sandbox with `gpu="B200"`. Contended timings are garbage.
2. Reward returns a `float` on current main (0.1.14). `vf.RolloutScore` return = stale 0.1.3 API; it will break. Surface extra metrics by stashing in `state[...]` + weight-0 reader reward funcs.
3. NO per-reward timeout in Rubric. Enforce it yourself: `asyncio.wait_for` (subprocess) or sandbox `timeout_per_command_seconds` (Prime) / `timeout=` (Modal). Always return 0.0 on timeout/exception, never raise.
4. prime-rl REMOVED `orchestrator.max_concurrent` (2026-04-05). Cap concurrent compiles via an `asyncio.Semaphore` you create in `load_environment`, exposed as a kwarg (e.g. `max_concurrent_compiles`) so the trainer tunes it via `args = {...}`. Don't rely on `Environment(max_workers=512)` — that's a thread-bridge default, not a rate cap.
5. Cache the baseline timing (kernelbench `baseline_cache.py`) — don't re-time the PyTorch reference every rollout.
6. Gate the expensive perf pass on correctness (backend_bench pattern): only time speedup if numerically correct first.
7. `load_inline`/nvcc build dirs must be unique per candidate (hash of source) or parallel compiles collide. kernelbench uses `build_dir=/tmp/kb_build/{sha256[:20]}`.
8. Make the module CPU-import-safe (lazy-import torch/modal inside the reward) so `prime env push` / Hub indexing on a CPU box doesn't fail.

**Installable/discoverable on the Hub:**
9. Module filename, `pyproject` `name`, and `[tool.hatch.build] include` must agree; `include` must list every file the reward imports (utils/, prompts/), or the published archive is broken.
10. `load_environment` must take ONLY keyword args with defaults — prime-rl calls it as `load_environment(**args)` from the TOML `args` table. No positional/required params.
11. Bump `version` (semver) on every `prime env push`, or use `--auto-bump`. Pin `verifiers>=0.1.14` to avoid the score_rollouts->score_rollout rename and the float-return change biting you.
12. Build backend = hatchling; deps must include `verifiers` plus everything the reward imports (torch, modal, etc.).

## Source files read (cloned, byte-exact)
- verifiers: `verifiers/envs/{environment,singleturn_env,multiturn_env,tool_env,sandbox_env,python_env}.py`, `verifiers/rubrics/rubric.py`, `verifiers/types.py`
- prime-environments: `environments/kernelbench/{kernelbench.py,pyproject.toml}`, math_python, backend_bench
- prime-rl: `configs/math_python/math_python.toml`, `configs/env_mix/env_mix.toml`, `CHANGELOG.md`
- PyPI: verifiers 0.1.14 (2026-05-07), prime 0.6.15
- Links: https://github.com/PrimeIntellect-ai/verifiers · https://github.com/PrimeIntellect-ai/prime-environments · https://github.com/PrimeIntellect-ai/prime-rl · https://docs.primeintellect.ai · https://app.primeintellect.ai/dashboard/environments
