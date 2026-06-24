"""
Metal Sandbox for Agentic Kernel Optimization.

Provides SSH-based execution on a remote Apple Silicon machine with MLX/Metal.
Implements the same interface as LocalSandbox for drop-in replacement.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetalSandboxConfig:
    """Configuration for Metal sandbox."""

    ssh_host: str = "macbook"
    timeout: int = 300
    cleanup: bool = True
    workdir: Optional[str] = None


class MetalSandbox:
    """
    SSH-based sandbox for Metal kernel optimization via MLX.

    Implements the same interface as LocalSandbox:
    - start() / stop()
    - run_command(command, timeout)
    - write_file(path, content)
    - read_file(path)
    - file_exists(path)
    """

    def __init__(self, problem_code: str, config: Optional[MetalSandboxConfig] = None):
        self.problem_code = problem_code
        self.config = config or MetalSandboxConfig()
        if self.config.ssh_host == "macbook" and platform.system() == "Darwin":
            # When running MetalBench directly on the macbook, resolve to local SSH.
            self.config.ssh_host = "localhost"
        self._local_mode = self.config.ssh_host in {"localhost", "127.0.0.1"}
        self._workspace: Optional[str] = None
        self._started = False
        self._gpu_info = "Unknown"

    def _ssh_cmd(self, command: str) -> list[str]:
        bootstrap = (
            "export PATH=$PATH:/opt/homebrew/bin:/usr/local/bin; "
            "if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then "
            "python() { python3 \"$@\"; }; "
            "fi; "
        )
        remote_command = f"bash --noprofile --norc -lc {shlex.quote(bootstrap + command)}"
        return ["ssh", self.config.ssh_host, remote_command]

    def _exec(self, command: str, timeout: Optional[int] = None, stdin: Optional[str] = None) -> dict:
        bootstrap = (
            "export PATH=$PATH:/opt/homebrew/bin:/usr/local/bin; "
            "if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then "
            "python() { python3 \"$@\"; }; "
            "fi; "
        )
        try:
            if self._local_mode:
                env = os.environ.copy()
                venv_bin = os.path.join(os.getcwd(), ".venv", "bin")
                if os.path.isdir(venv_bin):
                    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
                completed = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-lc", bootstrap + command],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=timeout or self.config.timeout,
                    check=False,
                    env=env,
                )
                return {
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                    "timed_out": False,
                }

            completed = subprocess.run(
                self._ssh_cmd(command),
                input=stdin,
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

    def _replace_workspace_path(self, text: str) -> str:
        if not self._workspace:
            return text
        text = text.replace("/workspace/", f"{self._workspace}/")
        text = text.replace("/workspace", self._workspace)
        return text

    def _resolve_path(self, path: str) -> str:
        if not self._workspace:
            raise RuntimeError("Sandbox not started")

        if path == "/workspace":
            return self._workspace
        if path.startswith("/workspace/"):
            return f"{self._workspace}/{path[len('/workspace/') :]}"
        if path.startswith("/"):
            return path
        return f"{self._workspace}/{path}"

    def _detect_gpu_info(self) -> str:
        chip_result = self._exec("system_profiler SPHardwareDataType | awk -F': ' '/Chip/{print $2; exit}'")
        mem_result = self._exec("system_profiler SPHardwareDataType | awk -F': ' '/Memory/{print $2; exit}'")

        chip = chip_result["stdout"].strip()
        mem = mem_result["stdout"].strip()

        if chip and mem:
            return f"{chip}, {mem} unified memory (MPS)"
        if chip:
            return f"{chip} (MPS)"
        return "Apple Silicon (MPS)"

    def start(self):
        """Start remote Metal sandbox."""
        if self._started:
            return

        if self.config.workdir:
            self._workspace = self.config.workdir
            self._exec(f"mkdir -p {shlex.quote(self._workspace)}", timeout=30)
        else:
            result = self._exec("mktemp -d /tmp/kernelbench_metal_XXXXXX", timeout=30)
            workspace = result["stdout"].strip()
            if not workspace or result["returncode"] != 0:
                raise RuntimeError(f"Failed to create remote workspace: {result['stderr']}")
            self._workspace = workspace

        self._started = True
        self.write_file("reference.py", self.problem_code)
        self._gpu_info = self._detect_gpu_info()

        # Verify MLX is available in the remote execution environment.
        mlx_check = self._exec(
            "python -c \"import mlx.core as mx; print(mx.default_device())\"",
            timeout=120,
        )
        if mlx_check["returncode"] != 0:
            raise RuntimeError(f"MLX is not available on remote host: {mlx_check['stderr']}")

    def stop(self):
        """Stop and optionally clean up remote workspace."""
        if self._workspace and self.config.cleanup and not self.config.workdir:
            self._exec(f"rm -rf {shlex.quote(self._workspace)}", timeout=30)

        self._workspace = None
        self._started = False

    def run_command(self, command: str, timeout: Optional[int] = None) -> dict:
        """Execute command in remote workspace."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        mapped = self._replace_workspace_path(command)
        full_command = f"cd {shlex.quote(self._workspace)} && {mapped}"
        return self._exec(full_command, timeout=timeout)

    def write_file(self, path: str, content: str) -> bool:
        """Write file to remote workspace."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        remote_path = self._resolve_path(path)
        parent = os.path.dirname(remote_path)
        if parent:
            self._exec(f"mkdir -p {shlex.quote(parent)}")

        result = self._exec(f"cat > {shlex.quote(remote_path)}", stdin=content)
        return result["returncode"] == 0

    def read_file(self, path: str) -> Optional[str]:
        """Read file from remote workspace."""
        if not self._started:
            raise RuntimeError("Sandbox not started")

        remote_path = self._resolve_path(path)
        result = self._exec(f"cat {shlex.quote(remote_path)}")
        if result["returncode"] == 0:
            return result["stdout"]
        return None

    def file_exists(self, path: str) -> bool:
        """Check if file exists in remote workspace."""
        if not self._started:
            return False

        remote_path = self._resolve_path(path)
        result = self._exec(f"test -f {shlex.quote(remote_path)} && echo exists")
        return "exists" in result["stdout"]

    def get_gpu_info(self) -> str:
        """Get remote GPU info string."""
        return self._gpu_info

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_metal_sandbox(problem_code: str, **kwargs) -> MetalSandbox:
    """Factory function to create Metal sandbox."""

    config = MetalSandboxConfig(**kwargs)
    return MetalSandbox(problem_code, config)
