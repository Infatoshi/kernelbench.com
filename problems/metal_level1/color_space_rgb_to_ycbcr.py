import torch
import torch.nn as nn

OP_TYPE = "elementwise"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """RGB to YCbCr color space conversion (BT.601)."""

    def __init__(self):
        super().__init__()
        transform = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=torch.float32,
        )
        offset = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("transform", transform)
        self.register_buffer("offset", offset)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = image.shape
        pixels = image.permute(0, 2, 3, 1).reshape(-1, 3)
        ycbcr = pixels @ self.transform.T + self.offset
        return ycbcr.reshape(B, H, W, 3).permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand(8, 3, 1024, 1024)]


def get_init_inputs():
    return []
