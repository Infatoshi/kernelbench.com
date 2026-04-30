"""B200 hardware target — Modal cloud, CUDA/Triton/CUTLASS/CuTe/CuTile approaches."""

from src.hardware import HardwareTarget, register


@register("b200")
class B200Target(HardwareTarget):
    name = "b200"
    display_name = "B200"
    gpu_sku = "B200"
    vram_gb = 192
    problem_dirs = ["level1", "level2", "level3", "level4", "tile_specialized", "cutile"]
    exclude_problems = []

    def create_sandbox(self, problem_code: str):
        from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig
        return ModalSandbox(problem_code, ModalSandboxConfig(gpu="B200", timeout=300, sandbox_timeout=3600))
