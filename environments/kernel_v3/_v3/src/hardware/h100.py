"""H100 hardware target — Modal cloud, CUDA/Triton/CUTLASS/CuTe approaches."""

from src.hardware import HardwareTarget, register


@register("h100")
class H100Target(HardwareTarget):
    name = "h100"
    display_name = "H100"
    gpu_sku = "H100"
    vram_gb = 80
    problem_dirs = ["level1", "level2", "level3", "level4", "tile_specialized"]
    exclude_problems = ["9_FP4_BlockScaled_Matmul.py"]

    def create_sandbox(self, problem_code: str):
        from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig
        return ModalSandbox(problem_code, ModalSandboxConfig(gpu="H100", timeout=300, sandbox_timeout=3600))
