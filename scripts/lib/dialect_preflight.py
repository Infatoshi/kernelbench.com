"""Exercise the Python CUDA dialects in the immutable grading environment."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import cuda.tile as ct
import cutlass
import cutlass.cute as cute


@triton.jit
def _triton_add_one(x, size: tl.constexpr):
    offsets = tl.arange(0, size)
    tl.store(x + offsets, tl.load(x + offsets) + 1)


@ct.kernel()
def _cutile_add_one(x, size: ct.Constant[int]):
    tile = ct.load(x, index=(ct.bid(0),), shape=(size,))
    ct.store(x, index=(ct.bid(0),), tile=tile + 1)


@cute.jit
def _cute_scalar_add(a: cutlass.Int32, b: cutlass.Int32):
    cute.printf("immutable CuTe DSL preflight: {}\n", a + b)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the immutable grading environment")

    triton_value = torch.zeros(32, device="cuda", dtype=torch.float32)
    _triton_add_one[(1,)](triton_value, triton_value.numel())
    torch.cuda.synchronize()
    torch.testing.assert_close(triton_value, torch.ones_like(triton_value))

    cutile_value = torch.zeros(32, device="cuda", dtype=torch.float32)
    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        _cutile_add_one,
        (cutile_value, cutile_value.numel()),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(cutile_value, torch.ones_like(cutile_value))

    cute_add = cute.compile(_cute_scalar_add, 1, 2)
    cute_add(1, 2)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
