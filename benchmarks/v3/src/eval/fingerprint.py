"""Hardware fingerprinting — captures GPU/system metadata for every run."""

from __future__ import annotations

from typing import Any, Dict


def fingerprint_nvidia(sandbox) -> Dict[str, Any]:
    """Capture NVIDIA GPU metadata via nvidia-smi inside the sandbox."""
    fp: Dict[str, Any] = {"platform": "nvidia"}
    cmd = (
        "nvidia-smi --query-gpu="
        "gpu_name,driver_version,memory.total,power.limit,ecc.mode.current,persistence_mode"
        " --format=csv,noheader,nounits 2>/dev/null"
    )
    result = sandbox.run_command(cmd, timeout=10)
    if result["returncode"] == 0 and result["stdout"].strip():
        parts = [p.strip() for p in result["stdout"].strip().split(",")]
        if len(parts) >= 6:
            fp["gpu_name"] = parts[0]
            fp["driver_version"] = parts[1]
            fp["memory_total_mb"] = int(float(parts[2]))
            fp["power_cap_w"] = parts[3]
            fp["ecc_mode"] = parts[4]
            fp["persistence_mode"] = parts[5]

    cuda_cmd = "nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//'"
    cuda_result = sandbox.run_command(cuda_cmd, timeout=10)
    if cuda_result["returncode"] == 0 and cuda_result["stdout"].strip():
        fp["cuda_version"] = cuda_result["stdout"].strip()

    return fp


def fingerprint_apple() -> Dict[str, Any]:
    """Capture Apple Silicon metadata via sysctl and system_profiler."""
    import platform
    import subprocess

    fp: Dict[str, Any] = {"platform": "apple_silicon"}

    fp["macos_version"] = platform.mac_ver()[0]
    fp["arch"] = platform.machine()

    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5
        ).decode().strip()
        fp["chip_model"] = chip
    except Exception:
        pass

    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=5
        ).decode().strip())
        fp["total_memory_gb"] = round(mem_bytes / (1024 ** 3))
    except Exception:
        pass

    try:
        gpu_cores = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.core_count"], timeout=5
        ).decode().strip()
        fp["cpu_core_count"] = int(gpu_cores)
    except Exception:
        pass

    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], timeout=10
        ).decode()
        for line in sp.splitlines():
            line = line.strip()
            if "Model Name" in line:
                fp["form_factor"] = line.split(":")[-1].strip()
            elif "Model Identifier" in line:
                fp["model_identifier"] = line.split(":")[-1].strip()
            elif "Chip" in line and "chip_model" not in fp:
                fp["chip_model"] = line.split(":")[-1].strip()
            elif "GPU" in line and "core" in line.lower():
                fp["gpu_cores"] = line.split(":")[-1].strip()
    except Exception:
        pass

    return fp


def get_fingerprint(hardware_target, sandbox=None) -> Dict[str, Any]:
    """Dispatch to platform-specific fingerprinting."""
    if hardware_target.is_metal:
        return fingerprint_apple()
    if sandbox is not None:
        return fingerprint_nvidia(sandbox)
    return {"platform": "unknown"}
