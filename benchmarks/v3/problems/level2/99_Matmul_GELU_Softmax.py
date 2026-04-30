import torch
import torch.nn as nn


OP_TYPE = "fused"
SUPPORTED_PRECISIONS = ['fp16', 'bf16', 'fp32']
HARDWARE_REQUIRED = ['RTX3090', 'H100', 'B200']

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

batch_size = 128
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]