"""Context building for agent workspace: reference metadata, self-check, environment."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict


def safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_reference_metadata(reference_code: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "op_type": "unknown",
        "supported_precisions": [],
        "hardware_required": [],
        "has_model_class": False,
        "has_get_inputs": False,
        "has_get_init_inputs": False,
    }
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return metadata

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            metadata["has_model_class"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            metadata["has_get_inputs"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            metadata["has_get_init_inputs"] = True
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "OP_TYPE":
                value = safe_literal_eval(node.value)
                if isinstance(value, str):
                    metadata["op_type"] = value
            elif target.id == "SUPPORTED_PRECISIONS":
                value = safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["supported_precisions"] = [str(x) for x in value]
            elif target.id == "HARDWARE_REQUIRED":
                value = safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["hardware_required"] = [str(x) for x in value]
    return metadata


def self_check_command(is_metal: bool) -> str:
    """Correctness self-check that compares solution output against reference."""
    if is_metal:
        return (
            'python -c "import mlx.core as mx, reference, solution, numpy as np; '
            "ref_m = reference.Model(*reference.get_init_inputs()); "
            "inputs = reference.get_inputs(); "
            "mlx_inputs = [mx.array(x.numpy()) if hasattr(x, 'numpy') else x for x in inputs]; "
            "ref_out = ref_m(*[x.to('mps') if hasattr(x, 'to') else x for x in inputs]).cpu(); "
            "sol_out = solution.solution(*mlx_inputs); mx.eval(sol_out); "
            "diff = float(np.max(np.abs(np.array(sol_out) - ref_out.numpy()))); "
            "print(f'max_diff={diff:.6f}'); "
            "print('PASS' if diff < 0.01 else 'FAIL')\""
        )
    return (
        'python -c "import reference, solution, torch; '
        "ref_m = reference.Model(*reference.get_init_inputs()).cuda().eval(); "
        "sol_m = solution.Model(*reference.get_init_inputs()).cuda().eval(); "
        "sol_m.load_state_dict(ref_m.state_dict(), strict=False); "
        "inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]; "
        "ref_out = ref_m(*inputs); sol_out = sol_m(*inputs); "
        "diff = torch.max(torch.abs(ref_out - sol_out)).item(); "
        "print(f'max_diff={diff:.6f}'); "
        "print('PASS' if torch.allclose(ref_out, sol_out, atol=1e-2, rtol=1e-2) else 'FAIL')\""
    )


def augment_system_prompt(system_prompt: str, is_metal: bool) -> str:
    check = self_check_command(is_metal)
    return (
        system_prompt
        + f"""

EXECUTION ENVIRONMENT:
- Isolated benchmark sandbox. Working directory: `/workspace`.
- Files: `reference.py`, `ENVIRONMENT.md`, `BACKEND_API.md`, `TEMPLATE_solution.py`, `TASK_CONTEXT.md`.
- No internet. No package installation.

REQUIRED CORRECTNESS CHECK (before submitting):
Run:
`{check}`
- If PASS: submit immediately.
- If FAIL: fix the numerical error and recheck. Do NOT submit until PASS.
"""
    )


def inject_workspace_context(system_prompt: str, context_bundle: Dict[str, str]) -> str:
    environment_md = context_bundle.get("environment_md", "").strip()
    backend_api_md = context_bundle.get("backend_api_md", "").strip()
    task_context_md = context_bundle.get("task_context_md", "").strip()
    template_solution_py = context_bundle.get("template_solution_py", "").strip()
    return (
        system_prompt
        + "\n\nWORKSPACE CONTEXT:\n"
        + "\n[ENVIRONMENT.md]\n" + environment_md
        + "\n\n[BACKEND_API.md]\n" + backend_api_md
        + "\n\n[TASK_CONTEXT.md]\n" + task_context_md
        + "\n\n[TEMPLATE_solution.py]\n```python\n" + template_solution_py + "\n```\n"
    )


def collect_runtime_environment(sandbox, hardware_name: str, gpu_name: str, vram_gb: int, level: int) -> str:
    probe_cmd = """python - <<'PY'
import importlib, json, platform, sys
modules = {}
for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
    try:
        mod = importlib.import_module(name)
        modules[name] = getattr(mod, "__version__", "unknown")
    except Exception:
        modules[name] = None
print(json.dumps({"python_version": sys.version.split()[0], "platform": platform.platform(), "modules": modules}))
PY"""
    probe_result = sandbox.run_command(probe_cmd, timeout=45)
    payload: Dict[str, Any] = {}
    if probe_result.get("returncode") == 0:
        for line in reversed((probe_result.get("stdout") or "").splitlines()):
            if line.strip().startswith("{"):
                try:
                    payload = json.loads(line.strip())
                    break
                except json.JSONDecodeError:
                    continue

    modules = payload.get("modules", {})
    lines = [
        "# Environment",
        f"- hardware: `{hardware_name}`",
        f"- gpu: `{gpu_name}` ({vram_gb}GB)",
        f"- level: `{level}`",
        f"- python: `{payload.get('python_version', 'unknown')}`",
        "",
        "## Available Frameworks",
    ]
    for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
        version = modules.get(name)
        lines.append(f"- {name}: `{version if version else 'unavailable'}`")
    return "\n".join(lines)


def build_task_context(problem_name: str, level: int, gpu_name: str, hardware_name: str, metadata: Dict[str, Any]) -> str:
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    return f"""# Task Context
- problem: `{problem_name}`
- level: `{level}`
- hardware: `{hardware_name}`
- gpu: `{gpu_name}`
- op_type: `{metadata.get("op_type", "unknown")}`
- supported_precisions: `{precisions}`
"""


def prepare_workspace_context(
    hardware_name: str, gpu_name: str, vram_gb: int, level: int,
    problem_name: str, metadata: Dict[str, Any], sandbox, is_metal: bool = False,
) -> Dict[str, str]:
    if is_metal:
        api_ref = _build_api_reference("metal")
        template = _build_template_solution("metal")
    else:
        api_ref = _build_api_reference("nvidia")
        template = _build_template_solution("nvidia")

    return {
        "environment_md": collect_runtime_environment(sandbox, hardware_name, gpu_name, vram_gb, level),
        "backend_api_md": api_ref,
        "template_solution_py": template,
        "task_context_md": build_task_context(problem_name, level, gpu_name, hardware_name, metadata),
    }


def seed_workspace_context(sandbox, context_bundle: Dict[str, str]) -> None:
    try:
        sandbox.write_file("ENVIRONMENT.md", context_bundle.get("environment_md", ""))
        sandbox.write_file("BACKEND_API.md", context_bundle.get("backend_api_md", ""))
        sandbox.write_file("TEMPLATE_solution.py", context_bundle.get("template_solution_py", ""))
        sandbox.write_file("TASK_CONTEXT.md", context_bundle.get("task_context_md", ""))
    except Exception as exc:
        print(f"Warning: failed to seed workspace context: {exc}", flush=True)


def build_initial_user_message(
    hardware_name: str, problem_name: str, level: int, gpu_name: str,
    max_turns: int, reference_code: str, metadata: Dict[str, Any],
) -> str:
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    return f"""Optimize the benchmark task and produce `/workspace/solution.py`.

- hardware: `{hardware_name}`
- problem: `{problem_name}`
- level: `{level}`
- gpu: `{gpu_name}`
- op_type: `{metadata.get("op_type", "unknown")}`
- precisions: `{precisions}`

Take as many turns as you need. Run the correctness self-check and only submit after PASS.

Reference code:
```python
{reference_code}
```
"""


def _build_api_reference(platform: str) -> str:
    if platform == "metal":
        return """# API Reference: Metal (MLX)
- Required: `import mlx.core as mx`
- Implement `def solution(*inputs)` accepting MLX arrays.
- No torch, no triton, no CUDA extensions.
"""
    return """# API Reference: NVIDIA GPU
- You may use CUDA C++ (`torch.utils.cpp_extension.load_inline`) or Triton (`@triton.jit`).
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- No PyTorch operator fallbacks (torch.matmul, F.linear, etc.).
"""


def _build_template_solution(platform: str) -> str:
    if platform == "metal":
        return """import mlx.core as mx

def solution(*inputs):
    return mx.matmul(inputs[0], inputs[1])
"""
    return """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx];
}

torch::Tensor my_op(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  my_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
'''

ext = load_inline(
    name='my_ext',
    cpp_sources='torch::Tensor my_op(torch::Tensor);',
    cuda_sources=cuda_source,
    functions=['my_op'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.my_op(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
"""
