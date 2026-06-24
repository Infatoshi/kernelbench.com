import torch
import torch.nn as nn


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ['fp4', 'fp8', 'fp16', 'bf16', 'fp32']
HARDWARE_REQUIRED = ['RTX3090', 'H100', 'B200']

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed