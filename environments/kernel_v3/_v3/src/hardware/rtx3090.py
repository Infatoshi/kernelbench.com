"""RTX 3090 hardware target — local GPU, CUDA/Triton approaches."""

from src.hardware import HardwareTarget, register


@register("rtx3090")
class RTX3090Target(HardwareTarget):
    name = "rtx3090"
    display_name = "RTX 3090"
    gpu_sku = "RTX3090"
    vram_gb = 24
    problem_dirs = ["level1", "level2", "level3", "level4", "graphics"]
    exclude_problems = ["9_FP4_BlockScaled_Matmul.py"]

    def create_sandbox(self, problem_code: str):
        from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig
        return LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
