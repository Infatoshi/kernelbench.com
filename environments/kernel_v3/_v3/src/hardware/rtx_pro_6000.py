"""RTX PRO 6000 Blackwell Workstation — local GPU, SM120."""

from src.hardware import HardwareTarget, register


@register("rtx_pro_6000")
class RTXPro6000Target(HardwareTarget):
    name = "rtx_pro_6000"
    display_name = "RTX PRO 6000 Blackwell Workstation"
    gpu_sku = "RTX_PRO_6000"
    vram_gb = 96
    # Reuses the RTX 3090 problem layout; 4 and 9 (FP8/FP4) are runnable here.
    problem_dirs = ["level1", "level2", "level3", "level4", "graphics"]
    exclude_problems: list[str] = []

    def create_sandbox(self, problem_code: str):
        from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig
        return LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
