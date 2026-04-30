"""Naive PyTorch paged-attention decode reference (correctness oracle, not SOTA).

Single-query decode: each batch element has a query of shape (num_heads, head_dim)
and attends over a KV cache of `seq_len[b]` tokens stored as fixed-size pages in
a global pool. Pages for batch element b are listed in `block_table[b]`.

The reference performs the slow path:
  1. Gather pages -> contiguous (seq_len, num_kv_heads, head_dim) per batch element.
  2. Repeat KV heads for grouped-query (broadcast num_kv_heads -> num_heads).
  3. Manual softmax(QK^T / sqrt(d)) @ V in fp32, cast back to bf16.

This avoids torch.nn.functional.scaled_dot_product_attention (which is on the
forbidden list) so the agent cannot dispatch through SDPA either.
"""
import math

import torch
import torch.nn as nn

OP_TYPE = "attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


# --- Shape knobs (overridden by check.py / benchmark.py from shapes.py) ----
BATCH = 8
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 1024
PAGE_SIZE = 16


class Model(nn.Module):
    """Single-query paged attention decode.

    Forward inputs (all on device):
      query:       (batch, num_heads, head_dim)               bf16
      kv_cache:    (num_blocks, page_size, num_kv_heads, head_dim * 2)
                   Layout: last dim packs [K | V] so a single gather pulls both.
                   Stored as bf16.
      block_table: (batch, max_blocks)                        int32
      seq_lens:    (batch,)                                   int32

    Output:
      attn_out:    (batch, num_heads, head_dim)               bf16
    """

    def __init__(
        self,
        batch: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        page_size: int,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.page_size = page_size
        self.group_size = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)

        # No learned parameters: everything flows through get_inputs(). We keep
        # an empty buffer so state_dict() round-trips trivially between reference
        # and solution.
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, H, D = query.shape
        Hkv = self.num_kv_heads
        G = self.group_size
        P = self.page_size

        out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)

        for b in range(B):
            L = int(seq_lens[b].item())
            num_pages = (L + P - 1) // P
            pages = block_table[b, :num_pages].long()
            # Gather: (num_pages, page_size, num_kv_heads, 2*head_dim)
            kv = kv_cache.index_select(0, pages)
            kv = kv.reshape(num_pages * P, Hkv, 2 * D)
            kv = kv[:L]                          # mask trailing padded slots
            k = kv[..., :D]                      # (L, Hkv, D)
            v = kv[..., D:]                      # (L, Hkv, D)

            # Broadcast KV heads to query heads (GQA): (L, H, D)
            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)

            q = query[b]                          # (H, D)
            # Attention in fp32 for the oracle.
            qf = q.float()
            kf = k.float()
            vf = v.float()
            # scores: (H, L) = (H, D) @ (L, H, D) -> per-head dot
            scores = torch.einsum("hd,lhd->hl", qf, kf) * self.scale
            probs = torch.softmax(scores, dim=-1)
            # out: (H, D) = sum_l probs[h, l] * v[l, h, :]
            o = torch.einsum("hl,lhd->hd", probs, vf)
            out[b] = o.to(query.dtype)

        return out


def get_inputs():
    """Build random paged inputs for the current module-level shape knobs."""
    B = BATCH
    H = NUM_HEADS
    Hkv = NUM_KV_HEADS
    D = HEAD_DIM
    L = SEQ_LEN
    P = PAGE_SIZE

    pages_per_seq = (L + P - 1) // P
    # Keep the global pool larger than strictly needed and shuffle assignments
    # so the block_table actually exercises non-contiguous gather.
    total_pages = max(B * pages_per_seq + 8, 64)

    query = torch.randn(B, H, D, dtype=torch.bfloat16) * 0.1
    kv_cache = torch.randn(total_pages, P, Hkv, 2 * D, dtype=torch.bfloat16) * 0.1

    perm = torch.randperm(total_pages)[: B * pages_per_seq].reshape(B, pages_per_seq).int()
    # Pad to pages_per_seq columns; for fixed-seq-len shapes this is exact.
    block_table = perm.contiguous()
    seq_lens = torch.full((B,), L, dtype=torch.int32)

    return [query, kv_cache, block_table, seq_lens]


def get_init_inputs():
    return [BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN, PAGE_SIZE]
