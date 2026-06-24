"""M4 Max hardware target — local macOS, MLX approach."""

from src.hardware import HardwareTarget, register

_METAL_EXCLUDE = {
    "4_FP8_Matmul.py",
    "6_INT4_Quantized_GEMM.py",
    "7_GatedDeltaNet.py",
    "8_KimiDeltaAttention.py",
    "9_FP4_BlockScaled_Matmul.py",
}


@register("m4max")
class M4MaxTarget(HardwareTarget):
    name = "m4max"
    display_name = "M4"
    gpu_sku = "M4MAX"
    vram_gb = 128
    problem_dirs = [
        "level1", "level2", "level3", "level4",
        "metal_level1", "metal_level2", "metal_level3", "metal_level4",
    ]
    exclude_problems = list(_METAL_EXCLUDE)
    is_metal = True

    def create_sandbox(self, problem_code: str):
        from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig
        return MetalSandbox(problem_code, MetalSandboxConfig())
