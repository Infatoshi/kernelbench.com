"""PyTorch reference for one Qwen3-0.6B decode-block step."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "qwen3_decode_block"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
EPS = 1.0e-6

seq_len = 32


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (xf * scale * weight.float()).to(torch.bfloat16)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    cosf = cos.float()
    sinf = sin.float()
    even = xf[..., 0::2]
    odd = xf[..., 1::2]
    cos_e = cosf[0::2]
    sin_e = sinf[0::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = even * cos_e - odd * sin_e
    out[..., 1::2] = odd * cos_e + even * sin_e
    return out.to(torch.bfloat16)


class Model(nn.Module):
    def __init__(self, seq_len_: int = 32):
        super().__init__()
        self.seq_len = int(seq_len_)
        self.cache_len = self.seq_len - 1
        self.input_norm = nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))
        self.post_attn_norm = nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))
        self.q_norm = nn.Parameter(torch.ones(HEAD_DIM, dtype=torch.bfloat16))
        self.k_norm = nn.Parameter(torch.ones(HEAD_DIM, dtype=torch.bfloat16))
        self.q_proj = nn.Parameter(torch.empty(Q_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16))
        self.k_proj = nn.Parameter(torch.empty(KV_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16))
        self.v_proj = nn.Parameter(torch.empty(KV_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16))
        self.o_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, Q_SIZE, dtype=torch.bfloat16))
        self.gate_proj = nn.Parameter(torch.empty(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16))
        self.up_proj = nn.Parameter(torch.empty(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ]:
            nn.init.normal_(p, mean=0.0, std=0.0125)

    def forward(
        self,
        hidden: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        x = _rms_norm(hidden, self.input_norm)
        q = F.linear(x.float(), self.q_proj.float()).to(torch.bfloat16).view(NUM_Q_HEADS, HEAD_DIM)
        k = F.linear(x.float(), self.k_proj.float()).to(torch.bfloat16).view(NUM_KV_HEADS, HEAD_DIM)
        v = F.linear(x.float(), self.v_proj.float()).to(torch.bfloat16).view(NUM_KV_HEADS, HEAD_DIM)
        q = _rms_norm(q, self.q_norm)
        k = _rms_norm(k, self.k_norm)
        q = _apply_rope(q, cos[-1], sin[-1])
        k = _apply_rope(k, cos[-1], sin[-1])
        full_k = torch.cat([k_cache, k[:, None, :]], dim=1)
        full_v = torch.cat([v_cache, v[:, None, :]], dim=1)
        attn_heads = []
        scale = 1.0 / math.sqrt(HEAD_DIM)
        for qh in range(NUM_Q_HEADS):
            kvh = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
            scores = (full_k[kvh].float() @ q[qh].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out = probs @ full_v[kvh].float()
            attn_heads.append(out.to(torch.bfloat16))
        attn = torch.stack(attn_heads, dim=0).reshape(Q_SIZE)
        attn_out = F.linear(attn.float(), self.o_proj.float()).to(torch.bfloat16)
        h = residual + attn_out
        mlp_in = _rms_norm(h, self.post_attn_norm)
        gate = F.linear(mlp_in.float(), self.gate_proj.float())
        up = F.linear(mlp_in.float(), self.up_proj.float())
        fused = F.silu(gate) * up
        down = F.linear(fused, self.down_proj.float()).to(torch.bfloat16)
        return h + down


def get_inputs():
    hidden = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16) * 0.25
    cache_len = seq_len - 1
    k_cache = torch.randn(NUM_KV_HEADS, cache_len, HEAD_DIM, dtype=torch.bfloat16) * 0.125
    v_cache = torch.randn(NUM_KV_HEADS, cache_len, HEAD_DIM, dtype=torch.bfloat16) * 0.125
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0)
    inv_freq = 1.0 / (10000.0 ** (dim / HEAD_DIM))
    angles = pos * inv_freq
    cos = torch.cos(angles).to(torch.bfloat16)
    sin = torch.sin(angles).to(torch.bfloat16)
    return [hidden, k_cache, v_cache, cos, sin]


def get_init_inputs():
    return [seq_len]
