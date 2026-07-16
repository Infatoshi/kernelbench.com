"""Qwen3-0.6B-geometry decode reference (eager PyTorch).

One block = RMSNorm → QKV → Q/K RMSNorm → RoPE → causal GQA attn → O
→ residual → RMSNorm → SwiGLU MLP → residual.

Stacks `num_layers` blocks (default 4) of MegaQwen / Qwen3-0.6B geometry.

Protocol
--------
prefill(model, ctx_len, seed) -> (hidden, k_caches, v_caches)
  Untimed. Builds a real KV cache of length ctx_len by sequential decode
  steps at positions 0..ctx_len-1 (same numerics as a chunked prefill).

decode_step(model, hidden, k_caches, v_caches, position) -> ...
  Single decode step at `position` (must equal current cache length).

run(ctx_len, decode_steps, seed, model=None) -> dict
  prefill then decode_steps decode steps. Returns last_hidden for numeric
  check. No tokenizer, no tokens — pure tensors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "megaqwen_decode"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

HIDDEN = 1024
INTERMEDIATE = 3072
NUM_Q = 16
NUM_KV = 8
HEAD_DIM = 128
NUM_LAYERS = 4
EPS = 1e-6


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + EPS) * w


def _rope(q: torch.Tensor, k: torch.Tensor, pos: int):
    D = q.shape[-1]
    half = D // 2
    inv = 1.0 / (
        10000
        ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    )
    t = torch.tensor([pos], device=q.device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos, sin = freqs.cos(), freqs.sin()

    def apply(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return apply(q), apply(k)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        H, I, D = HIDDEN, INTERMEDIATE, HEAD_DIM
        self.input_ln = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.q_proj = nn.Parameter(torch.empty(NUM_Q * D, H, dtype=torch.bfloat16))
        self.k_proj = nn.Parameter(torch.empty(NUM_KV * D, H, dtype=torch.bfloat16))
        self.v_proj = nn.Parameter(torch.empty(NUM_KV * D, H, dtype=torch.bfloat16))
        self.q_norm = nn.Parameter(torch.ones(D, dtype=torch.bfloat16))
        self.k_norm = nn.Parameter(torch.ones(D, dtype=torch.bfloat16))
        self.o_proj = nn.Parameter(torch.empty(H, NUM_Q * D, dtype=torch.bfloat16))
        self.post_ln = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.gate_proj = nn.Parameter(torch.empty(I, H, dtype=torch.bfloat16))
        self.up_proj = nn.Parameter(torch.empty(I, H, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(H, I, dtype=torch.bfloat16))
        for p in self.parameters():
            if p is self.input_ln or p is self.post_ln or p is self.q_norm or p is self.k_norm:
                continue
            nn.init.normal_(p, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        position: int,
    ):
        residual = x.float()
        h = _rmsnorm(residual, self.input_ln.float())
        q = (h @ self.q_proj.float().T).view(NUM_Q, HEAD_DIM)
        k = (h @ self.k_proj.float().T).view(NUM_KV, HEAD_DIM)
        v = (h @ self.v_proj.float().T).view(NUM_KV, HEAD_DIM)
        q = _rmsnorm(q, self.q_norm.float())
        k = _rmsnorm(k, self.k_norm.float())
        q, k = _rope(q, k, position)
        k_cache = k_cache.clone()
        v_cache = v_cache.clone()
        k_cache[:, position, :] = k.to(k_cache.dtype)
        v_cache[:, position, :] = v.to(v_cache.dtype)
        k_all = k_cache[:, : position + 1, :].float()
        v_all = v_cache[:, : position + 1, :].float()
        rep = NUM_Q // NUM_KV
        k_all = k_all.repeat_interleave(rep, dim=0)
        v_all = v_all.repeat_interleave(rep, dim=0)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = torch.einsum("hd,hld->hl", q, k_all) * scale
        att = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum("hl,hld->hd", att, v_all).reshape(-1)
        attn_out = attn_out @ self.o_proj.float().T
        h = residual + attn_out
        residual = h
        h = _rmsnorm(h, self.post_ln.float())
        gate = h @ self.gate_proj.float().T
        up = h @ self.up_proj.float().T
        h = F.silu(gate) * up
        h = h @ self.down_proj.float().T
        y = residual + h
        return y.to(torch.bfloat16), k_cache, v_cache


class Model(nn.Module):
    def __init__(self, num_layers: int = NUM_LAYERS, max_seq: int = 131072):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq = max_seq
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        position: int,
    ):
        h = x
        new_k, new_v = [], []
        for i, block in enumerate(self.blocks):
            h, kc, vc = block(h, k_caches[i], v_caches[i], position)
            new_k.append(kc)
            new_v.append(vc)
        return h, new_k, new_v


def empty_caches(num_layers: int, max_seq: int, device, dtype=torch.bfloat16):
    k = [
        torch.zeros(NUM_KV, max_seq, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(num_layers)
    ]
    v = [
        torch.zeros(NUM_KV, max_seq, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(num_layers)
    ]
    return k, v


def _seeded_hidden(seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(HIDDEN, generator=g, dtype=torch.bfloat16).to(device)


@torch.no_grad()
def prefill(
    model: Model,
    ctx_len: int,
    seed: int,
    device: torch.device | None = None,
):
    """Build KV of length ctx_len. NOT timed in benchmark."""
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    assert ctx_len <= model.max_seq
    h = _seeded_hidden(seed, device)
    k_caches, v_caches = empty_caches(model.num_layers, model.max_seq, device)
    # Stream of "inputs" for positions 0..ctx_len-1 (deterministic, not tokens).
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    for t in range(ctx_len):
        # Fresh activation each step so prefill is not a trivial identity path;
        # residual stream still flows through the block stack.
        x_t = torch.randn(HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
        # Mix previous hidden lightly so layers stay coupled.
        x_t = (0.5 * x_t + 0.5 * h).to(torch.bfloat16)
        h, k_caches, v_caches = model(x_t, k_caches, v_caches, t)
    return h, k_caches, v_caches


@torch.no_grad()
def decode_steps(
    model: Model,
    hidden: torch.Tensor,
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    start_pos: int,
    n_steps: int,
    seed: int,
):
    """Run n_steps decode steps starting at start_pos. Timed in benchmark."""
    device = hidden.device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 2)
    h = hidden
    for i in range(n_steps):
        pos = start_pos + i
        x_t = torch.randn(HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
        x_t = (0.5 * x_t + 0.5 * h).to(torch.bfloat16)
        h, k_caches, v_caches = model(x_t, k_caches, v_caches, pos)
    return h, k_caches, v_caches


def run(
    ctx_len: int,
    n_decode: int,
    seed: int,
    model: Model | None = None,
    max_seq: int | None = None,
) -> dict:
    """Prefill then decode. Returns last_hidden for numeric correctness.

    Parameter is `n_decode` (not decode_steps) so it does not shadow the
    decode_steps() function.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_seq = max_seq or max(ctx_len + n_decode, 512)
    if model is None:
        model = Model(NUM_LAYERS, max_seq)
    else:
        if getattr(model, "max_seq", 0) < ctx_len + n_decode:
            raise ValueError(
                f"model.max_seq={getattr(model, 'max_seq', None)} too small for "
                f"ctx_len={ctx_len}+n_decode={n_decode}"
            )
    model = model.to(device).eval()
    h, k_caches, v_caches = prefill(model, ctx_len, seed, device=device)
    h, k_caches, v_caches = decode_steps(
        model, h, k_caches, v_caches, start_pos=ctx_len, n_steps=n_decode, seed=seed
    )
    return {
        "last_hidden": h.detach(),
        "ctx_len": ctx_len,
        "decode_steps": n_decode,
    }


def get_init_inputs():
    return [NUM_LAYERS, 131072]


def get_inputs():
    return []
