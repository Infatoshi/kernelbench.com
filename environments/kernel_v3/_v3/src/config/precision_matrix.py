"""Precision capability matrix and helpers for hardware/operation combinations."""

from __future__ import annotations


HARDWARE_PRECISIONS = {
    "B200": ["fp4", "fp8", "bf16"],
    "H100": ["fp8", "fp16"],
    "A100": ["fp16", "bf16", "fp32"],
    "RTX3090": ["fp16", "bf16", "fp32"],
    "M4MAX": ["fp16", "fp32"],
}


OP_PRECISION_VALIDITY = {
    "gemm": ["fp4", "fp8", "fp16", "bf16", "fp32"],
    "softmax": ["fp16", "bf16", "fp32"],
    "layernorm": ["fp16", "bf16", "fp32"],
    "attention": ["fp8", "fp16", "bf16", "fp32"],
    "conv2d": ["fp16", "bf16", "fp32"],
    "reduction": ["fp16", "bf16", "fp32"],
}


HARDWARE_PEAK_TFLOPS = {
    "B200": {"fp4": 9000.0, "fp8": 4500.0, "bf16": 2250.0, "fp16": 2250.0},
    "H100": {"fp8": 1979.0, "fp16": 989.0, "bf16": 989.0},
    "A100": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "RTX3090": {"fp16": 71.0, "bf16": 71.0, "fp32": 35.6},
    "M4MAX": {"fp16": 27.5, "fp32": 13.8},
}


def get_valid_precisions(hardware: str, op_type: str) -> list[str]:
    """Return intersection of hardware-supported and op-valid precision modes."""
    hw_precs = set(HARDWARE_PRECISIONS.get(hardware, ["fp32"]))
    op_precs = set(OP_PRECISION_VALIDITY.get(op_type, ["fp32"]))
    return sorted(hw_precs & op_precs)

