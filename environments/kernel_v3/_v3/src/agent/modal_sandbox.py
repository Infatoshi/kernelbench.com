"""
Modal Sandbox for Agentic Kernel Optimization

Provides GPU-accelerated execution environment on Modal cloud.
Implements same interface as LocalSandbox for drop-in replacement.
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass

import modal

# Modal image with ML dependencies and CUDA 13.1 dev tools
# CUDA 13.1 provides: cuTile DSL, full Blackwell (sm_100) support, improved CuTe
# Includes git/cmake for CUTLASS/CuTe DSL support
GPU_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("ninja-build", "build-essential", "git", "cmake", "wget")
    .run_commands("git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass")
    .env({"CUTLASS_PATH": "/opt/cutlass/include"})
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "triton",
        "flash-linear-attention",
        "cuda-tile",
    )
)

# Get or create the Modal app
APP = modal.App.lookup("kernelbench-eval", create_if_missing=True)

GPUType = Literal["L40S", "A100", "H100", "B200"]


@dataclass
class ModalSandboxConfig:
    """Configuration for Modal sandbox."""
    gpu: GPUType = "H100"
    timeout: int = 300  # Per-command timeout
    sandbox_timeout: int = 3600  # Total sandbox lifetime


class ModalSandbox:
    """
    Modal-based sandbox for GPU kernel optimization.

    Implements same interface as LocalSandbox:
    - start() / stop()
    - run_command(command, timeout)
    - write_file(path, content)
    - read_file(path)
    - file_exists(path)
    """

    def __init__(self, problem_code: str, config: Optional[ModalSandboxConfig] = None):
        self.problem_code = problem_code
        self.config = config or ModalSandboxConfig()
        self._sandbox = None
        self._started = False
        self._gpu_info = None

    def start(self):
        """Start the Modal sandbox."""
        if self._started:
            return

        # Create Modal sandbox with specified GPU
        self._sandbox = modal.Sandbox.create(
            "sleep", "infinity",  # Keep alive
            app=APP,
            image=GPU_IMAGE,
            gpu=self.config.gpu,
            timeout=self.config.sandbox_timeout,
            workdir="/workspace",
        )

        self._started = True

        # Create workspace and write reference code
        self._exec("mkdir -p /workspace")
        self._write_file_raw("/workspace/reference.py", self.problem_code)

        # Verify CuTe headers are available in the image.
        cute_check = self._exec("test -f /opt/cutlass/include/cute/tensor.hpp && echo 'CuTe OK' || echo 'CuTe MISSING'")
        if "CuTe OK" not in cute_check["stdout"]:
            raise RuntimeError("CuTe headers missing at /opt/cutlass/include/cute/tensor.hpp")

        # Verify CuTile Python package availability (used by CuTileBench).
        cutile_check = self._exec(
            "python -c \"import importlib.metadata as md; import cuda.tile as ct; print(md.version('cuda-tile'))\""
        )
        if cutile_check["returncode"] == 0:
            print(f"CuTile Python available: {cutile_check['stdout'].strip()}", flush=True)
        else:
            print(f"Warning: CuTile Python unavailable: {cutile_check['stderr'].strip()}", flush=True)

        # Get GPU info
        result = self._exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        self._gpu_info = result["stdout"].strip()

    def stop(self):
        """Stop and clean up the Modal sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
        self._started = False

    def _exec(self, command: str, timeout: int = 300) -> dict:
        """Execute command directly in sandbox."""
        try:
            process = self._sandbox.exec("bash", "-c", command)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.wait()
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": process.returncode,
                "timed_out": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timed_out": "timeout" in str(e).lower()
            }

    def _write_file_raw(self, path: str, content: str):
        """Write file using heredoc (internal use)."""
        # Use base64 to handle special characters safely
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        self._exec(f"echo '{encoded}' | base64 -d > {path}")

    def run_command(self, command: str, timeout: Optional[int] = None) -> dict:
        """Execute a command in the sandbox."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        timeout = timeout or self.config.timeout
        return self._exec(f"cd /workspace && {command}", timeout)

    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file in the sandbox."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        full_path = f"/workspace/{path}" if not path.startswith("/") else path

        # Create parent directories
        parent = os.path.dirname(full_path)
        if parent:
            self._exec(f"mkdir -p {parent}")

        # Write file using base64 encoding for safety
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        result = self._exec(f"echo '{encoded}' | base64 -d > {full_path}")
        return result["returncode"] == 0

    def read_file(self, path: str) -> Optional[str]:
        """Read content from a file in the sandbox."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        full_path = f"/workspace/{path}" if not path.startswith("/") else path
        result = self._exec(f"cat {full_path}")

        if result["returncode"] == 0:
            return result["stdout"]
        return None

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the sandbox."""
        if not self._started:
            return False

        full_path = f"/workspace/{path}" if not path.startswith("/") else path
        result = self._exec(f"test -f {full_path} && echo exists")
        return "exists" in result["stdout"]

    def get_gpu_info(self) -> str:
        """Get GPU information."""
        return self._gpu_info or "Unknown"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_modal_sandbox(problem_code: str, gpu: GPUType = "H100", **kwargs) -> ModalSandbox:
    """Factory function to create Modal sandbox."""
    config = ModalSandboxConfig(gpu=gpu, **kwargs)
    return ModalSandbox(problem_code, config)
