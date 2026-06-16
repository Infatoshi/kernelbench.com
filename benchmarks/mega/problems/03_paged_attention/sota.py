"""SOTA reference for paged-attention decode.

Tries, in order:
  1. FlashInfer's BatchDecodeWithPagedKVCacheWrapper (preferred -- portable,
     supports SM120, GQA, arbitrary head_dim).
  2. vLLM's paged_attention_v2 CUDA op (requires its KV-cache layout, more
     finicky; we adapt the layout on the fly when possible).

If neither is importable, is_available() returns False and the benchmark just
reports eager + compiled + solution.

Agents are FORBIDDEN from importing these in solution.py (see problem.yaml).
This file is only for the benchmark's reference line.
"""
from __future__ import annotations

import torch


def _try_flashinfer(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> torch.Tensor | None:
    try:
        import flashinfer  # noqa: F401
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    except Exception:
        return None

    B, H, D = query.shape
    # FlashInfer expects K and V as separate (num_blocks, page_size, num_kv_heads, head_dim) tensors.
    # Our reference packs [K|V] on the last dim -- split here.
    k_cache = kv_cache[..., :D].contiguous()
    v_cache = kv_cache[..., D:].contiguous()

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # Build the indptr / indices / last_page_len schedule.
    pages_per_seq = ((seq_lens + page_size - 1) // page_size).int()
    indptr = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)
    indices_list = [block_table[b, : int(pages_per_seq[b].item())] for b in range(B)]
    indices = torch.cat(indices_list).int()
    last_page_len = ((seq_lens - 1) % page_size + 1).int()

    wrapper.plan(
        indptr, indices, last_page_len,
        num_qo_heads=H,
        num_kv_heads=num_kv_heads,
        head_dim=D,
        page_size=page_size,
        data_type=query.dtype,
    )
    return wrapper.run(query, (k_cache, v_cache))


def sota_forward(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> torch.Tensor:
    out = _try_flashinfer(query, kv_cache, block_table, seq_lens, num_kv_heads, head_dim, page_size)
    if out is not None:
        return out
    raise RuntimeError("No SOTA backend available (flashinfer not installed)")


def is_available() -> bool:
    try:
        import flashinfer  # noqa: F401
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: F401
        return True
    except Exception:
        return False
