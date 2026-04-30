import torch
import torch.nn as nn

OP_TYPE = "elementwise"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Premultiplied alpha over-operator compositing of two RGBA layers."""

    def __init__(self):
        super().__init__()

    def forward(self, foreground: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        fg_rgb = foreground[:, :3]
        fg_a = foreground[:, 3:4]
        bg_rgb = background[:, :3]
        bg_a = background[:, 3:4]

        out_a = fg_a + bg_a * (1.0 - fg_a)
        out_rgb = fg_rgb + bg_rgb * (1.0 - fg_a)
        return torch.cat([out_rgb, out_a], dim=1)


def get_inputs():
    fg = torch.rand(8, 4, 1024, 1024)
    bg = torch.rand(8, 4, 1024, 1024)
    return [fg, bg]


def get_init_inputs():
    return []
