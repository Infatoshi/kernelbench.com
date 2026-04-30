"""
Local Sandbox for Agentic Kernel Optimization.

Provides local execution with the same interface as ModalSandbox.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LocalSandboxConfig:
    """Configuration for local sandbox."""
    timeout: int = 300
    workdir: Optional[str] = None
    cleanup: bool = True


class LocalSandbox:
    """
    Local sandbox for GPU kernel optimization.

    Implements the same interface as ModalSandbox:
    - start() / stop()
    - run_command(command, timeout)
    - write_file(path, content)
    - read_file(path)
    - file_exists(path)
    """

    def __init__(self, problem_code: str, config: Optional[LocalSandboxConfig] = None):
        self.problem_code = problem_code
        self.config = config or LocalSandboxConfig()
        self._workspace: Optional[Path] = None
        self._owns_workspace = False
        self._started = False
        self._gpu_info = "Unknown"

    def start(self):
        """Start local sandbox."""
        if self._started:
            return

        if self.config.workdir:
            self._workspace = Path(self.config.workdir).expanduser().resolve()
            self._workspace.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False
        else:
            self._workspace = Path(tempfile.mkdtemp(prefix="kernelbench_local_"))
            self._owns_workspace = True

        self.write_file("reference.py", self.problem_code)
        result = self._exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=30)
        info = result["stdout"].strip()
        if info:
            self._gpu_info = info
        self._started = True

    def stop(self):
        """Stop and clean up local sandbox."""
        if self._owns_workspace and self.config.cleanup and self._workspace:
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace = None
        self._owns_workspace = False
        self._started = False

    def _replace_workspace_path(self, text: str) -> str:
        """Map Modal-style /workspace paths to local workspace."""
        if not self._workspace:
            return text
        workspace = str(self._workspace)
        text = text.replace("/workspace/", f"{workspace}/")
        text = text.replace("/workspace", workspace)
        return text

    def _resolve_path(self, path: str) -> Path:
        if not self._workspace:
            raise RuntimeError("Sandbox not started")

        if path == "/workspace":
            return self._workspace
        if path.startswith("/workspace/"):
            return self._workspace / path[len("/workspace/"):]
        if path.startswith("/"):
            return Path(path)
        return self._workspace / path

    def _exec(self, command: str, timeout: Optional[int] = None) -> dict:
        if not self._workspace:
            return {
                "stdout": "",
                "stderr": "Sandbox not started",
                "returncode": -1,
                "timed_out": False,
            }

        command = self._replace_workspace_path(command)
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
                check=False,
            )
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout or "",
                "stderr": e.stderr or f"Command timed out after {timeout or self.config.timeout}s",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timed_out": False,
            }

    def run_command(self, command: str, timeout: Optional[int] = None) -> dict:
        """Execute command in local sandbox."""
        if not self._started:
            raise RuntimeError("Sandbox not started")
        return self._exec(command, timeout)

    def write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        if not self._started and path != "reference.py":
            raise RuntimeError("Sandbox not started")

        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return True

    def read_file(self, path: str) -> Optional[str]:
        """Read file content."""
        if not self._started:
            raise RuntimeError("Sandbox not started")
        target = self._resolve_path(path)
        if not target.exists():
            return None
        return target.read_text()

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        if not self._started:
            return False
        return self._resolve_path(path).is_file()

    def get_gpu_info(self) -> str:
        """Get GPU information."""
        return self._gpu_info

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_local_sandbox(problem_code: str, **kwargs) -> LocalSandbox:
    """Factory function to create local sandbox."""
    config = LocalSandboxConfig(**kwargs)
    return LocalSandbox(problem_code, config)
