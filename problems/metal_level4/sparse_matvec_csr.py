import torch
import torch.nn as nn

OP_TYPE = "sparse"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Sparse matrix-vector multiplication using COO-to-dense-then-matvec baseline."""

    def __init__(self, N: int = 8192, nnz_per_row: int = 32):
        super().__init__()
        self.N = N
        self.nnz_per_row = nnz_per_row
        torch.manual_seed(42)
        rows = torch.arange(N).repeat_interleave(nnz_per_row)
        cols = torch.randint(0, N, (N * nnz_per_row,))
        vals = torch.randn(N * nnz_per_row)
        indices = torch.stack([rows, cols])
        sparse = torch.sparse_coo_tensor(indices, vals, (N, N)).coalesce()
        self.register_buffer("sparse_matrix", sparse.to_dense())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparse_matrix @ x


def get_inputs():
    return [torch.randn(8192, 1)]


def get_init_inputs():
    return [8192, 32]
