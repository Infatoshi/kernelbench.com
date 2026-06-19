"""PyTorch reference for a Kimi-Linear W4A16 hybrid decode unit (batch=1).

Correctness oracle, NOT the fast baseline (baseline.py) or the SOTA ceiling.
One repeating motif of Kimi-Linear-48B-A3B -- 3 KDA (gated-delta linear attn)
layers + 1 MLA (multi-head latent attention) layer, each + a 64-expert MoE FFN
-- decoded one token at a time, with the big projection weights stored as
**W4A16** (int4 weights, group-128 asymmetric, bf16 accumulation; the AWQ/GPTQ
format that open-source inference actually ships).

Why W4A16 and not fp8: at batch-1 decode the bottleneck is *weight memory
traffic*. int4 weights are 0.25x the bytes of bf16 (4x less traffic), and a
fused dequant-GEMV realizes that win in the memory-bound regime where decode
lives -- unlike fp8 tensor-core matmul, which has fixed overhead and loses at
M=1. int4/bf16-acc also needs no special tensor-core format, so it runs on any
bf16-capable GPU (Ampere, Hopper, Blackwell).

The two decode states are unchanged: KDA carries a fixed-size recurrent state
S[H,Dk,Dv] + short-conv window; MLA carries a growing compressed latent KV
cache. `step(hidden, state) -> (hidden, state)` consumes one token.

This reference dequantizes naively (unpack -> bf16 matrix -> matmul). The whole
problem is to FUSE the int4 unpack+dequant into the GEMV and never materialize
the bf16 weight. Pack format is identical to KernelBench-Hard problem 07.

Contract (reference.py / baseline.py / solution.py all expose it):
    Model(cfg); Model.step(hidden, state) -> (hidden, state)
    build_config(shape); init_state(cfg, ctx, seed); init_token(cfg, seed)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "kimi_linear_w4a16_decode"
HARDWARE_REQUIRED = ["RTX_PRO_6000"]
EPS = 1.0e-6
GROUP_SIZE = 128


@dataclass(frozen=True)
class Config:
    hidden: int = 2304
    kda_heads: int = 32
    kda_head_dim: int = 128
    short_conv: int = 4
    mla_heads: int = 32
    kv_lora: int = 512
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    rope_theta: float = 10000.0
    n_experts: int = 64
    n_active: int = 8
    n_shared: int = 1
    moe_inter: int = 1024
    routed_scaling: float = 2.446
    group: int = 128
    pattern: tuple = ("K", "K", "K", "M")
    dtype: torch.dtype = field(default=torch.bfloat16)


def build_config(shape: dict) -> Config:
    return Config(n_experts=int(shape.get("n_experts", 64)))


# --------------------------------------------------------------------------- #
# W4A16 quantization (AWQ/GPTQ-style, identical to Hard problem 07)
# --------------------------------------------------------------------------- #
def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """(K, N) uint8 in [0,15] -> (K//2, N): even-K low nibble, odd-K high."""
    lo = w_q[0::2] & 0xF
    hi = w_q[1::2] & 0xF
    return (lo | (hi << 4)).contiguous()


def _unpack_int4(w_packed: torch.Tensor, K: int) -> torch.Tensor:
    out = torch.empty((K, w_packed.shape[1]), dtype=torch.uint8, device=w_packed.device)
    out[0::2] = w_packed & 0xF
    out[1::2] = (w_packed >> 4) & 0xF
    return out


def quantize(w_io: torch.Tensor, group: int = GROUP_SIZE):
    """Quantize an (in, out) weight along `in`, per-group asymmetric int4.

    Returns (w_q[in//2, out] uint8, scales[in//g, out] bf16, zeros[..] bf16).
    """
    K, N = w_io.shape
    ng = K // group
    wg = w_io.view(ng, group, N).float()
    wmin = wg.min(dim=1, keepdim=True).values
    wmax = wg.max(dim=1, keepdim=True).values
    scales = (wmax - wmin).clamp_min(1e-8) / 15.0
    zeros = (-wmin / scales).round().clamp(0, 15)
    w_q = ((wg / scales) + zeros).round().clamp(0, 15).to(torch.uint8).view(K, N)
    return _pack_int4(w_q), scales.squeeze(1).to(torch.bfloat16), zeros.squeeze(1).to(torch.bfloat16)


def dequant(w_q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, K: int, group: int) -> torch.Tensor:
    """(w_q, scales, zeros) -> (K, N) bf16 weight (naive; materializes full matrix)."""
    wu = _unpack_int4(w_q, K).to(torch.bfloat16)
    s = scales.repeat_interleave(group, dim=0)
    z = zeros.repeat_interleave(group, dim=0)
    return (wu - z) * s


class QuantLinear(nn.Module):
    """W4A16 linear: y = x @ dequant(w_q). Weights are int4 buffers (carried in
    state_dict). Reference forward accumulates in fp32 (oracle quality)."""

    def __init__(self, in_f: int, out_f: int, group: int = GROUP_SIZE):
        super().__init__()
        assert in_f % group == 0 and in_f % 2 == 0
        self.in_f, self.out_f, self.group = in_f, out_f, group
        ng = in_f // group
        self.register_buffer("w_q", torch.zeros(in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(ng, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(ng, out_f, dtype=torch.bfloat16))

    def init_random(self, gen: torch.Generator, std: float = 0.02) -> None:
        w = torch.randn(self.in_f, self.out_f, generator=gen) * std
        wq, s, z = quantize(w, self.group)
        self.w_q.copy_(wq)
        self.scales.copy_(s)
        self.zeros.copy_(z)

    def weight_bf(self) -> torch.Tensor:
        return dequant(self.w_q, self.scales, self.zeros, self.in_f, self.group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.weight_bf().float()).to(torch.bfloat16)


class QuantExperts(nn.Module):
    """E independent W4A16 expert weights, each (in, out). Dequant-on-gather."""

    def __init__(self, n: int, in_f: int, out_f: int, group: int = GROUP_SIZE):
        super().__init__()
        self.n, self.in_f, self.out_f, self.group = n, in_f, out_f, group
        ng = in_f // group
        self.register_buffer("w_q", torch.zeros(n, in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(n, ng, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(n, ng, out_f, dtype=torch.bfloat16))

    def init_random(self, gen: torch.Generator, std: float = 0.02) -> None:
        for e in range(self.n):
            w = torch.randn(self.in_f, self.out_f, generator=gen) * std
            wq, s, z = quantize(w, self.group)
            self.w_q[e].copy_(wq)
            self.scales[e].copy_(s)
            self.zeros[e].copy_(z)

    def weight_bf(self, e: int) -> torch.Tensor:
        return dequant(self.w_q[e], self.scales[e], self.zeros[e], self.in_f, self.group)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * w.float()).to(x.dtype)


def _rope_cossin(pos: int, dim: int, theta: float, device):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    ang = pos * inv
    return torch.cos(ang), torch.sin(ang)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    even, odd = xf[..., 0::2], xf[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# layers (math identical to the bf16 version; weights are now W4A16)
# --------------------------------------------------------------------------- #
class KDA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, Dk, d = cfg.kda_heads, cfg.kda_head_dim, cfg.hidden
        self.q_proj = QuantLinear(d, H * Dk, cfg.group)
        self.k_proj = QuantLinear(d, H * Dk, cfg.group)
        self.v_proj = QuantLinear(d, H * Dk, cfg.group)
        self.g_proj = QuantLinear(d, H * Dk, cfg.group)
        self.beta_proj = nn.Linear(d, H, bias=False, dtype=cfg.dtype)   # tiny, bf16
        self.conv_w = nn.Parameter(torch.empty(3, H * Dk, cfg.short_conv, dtype=cfg.dtype))
        self.o_proj = QuantLinear(H * Dk, d, cfg.group)
        self.scale = Dk ** -0.5

    def _short_conv(self, val, prev, idx):
        win = torch.cat([prev, val[None]], dim=0)
        w = self.conv_w[idx].float().transpose(0, 1)
        out = (win.float() * w).sum(0)
        return F.silu(out).to(val.dtype), win[1:]

    def step(self, x, st):
        H, Dk = self.cfg.kda_heads, self.cfg.kda_head_dim
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, st["cq"] = self._short_conv(q, st["cq"], 0)
        k, st["ck"] = self._short_conv(k, st["ck"], 1)
        v, st["cv"] = self._short_conv(v, st["cv"], 2)
        q = q.view(H, Dk).float() * self.scale
        k = k.view(H, Dk).float()
        v = v.view(H, Dk).float()
        g = (-F.softplus(self.g_proj(x).float())).view(H, Dk)
        beta = torch.sigmoid(self.beta_proj(x).float())
        S = st["S"] * g.exp()[:, :, None]
        pred = (S * k[:, :, None]).sum(1)
        S = S + beta[:, None, None] * k[:, :, None] * (v - pred)[:, None, :]
        o = (S * q[:, :, None]).sum(1)
        st["S"] = S
        return self.o_proj(o.reshape(H * Dk).to(torch.bfloat16))


class MLA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, d = cfg.mla_heads, cfg.hidden
        self.q_proj = QuantLinear(d, H * (cfg.qk_nope + cfg.qk_rope), cfg.group)
        self.kv_a = QuantLinear(d, cfg.kv_lora + cfg.qk_rope, cfg.group)
        self.kv_b = QuantLinear(cfg.kv_lora, H * (cfg.qk_nope + cfg.v_head), cfg.group)
        self.o_proj = QuantLinear(H * cfg.v_head, d, cfg.group)
        self.scale = (cfg.qk_nope + cfg.qk_rope) ** -0.5

    def step(self, x, st):
        cfg = self.cfg
        H = cfg.mla_heads
        pos = st["c_kv"].shape[0]
        q = self.q_proj(x).view(H, cfg.qk_nope + cfg.qk_rope)
        q_nope = q[:, : cfg.qk_nope].float()
        q_rope = q[:, cfg.qk_nope :]
        kv = self.kv_a(x)
        c_kv = kv[: cfg.kv_lora]
        k_rope = kv[cfg.kv_lora :]
        cos, sin = _rope_cossin(pos, cfg.qk_rope, cfg.rope_theta, x.device)
        q_rope = _apply_rope(q_rope, cos, sin).float()
        k_rope = _apply_rope(k_rope, cos, sin)
        st["c_kv"] = torch.cat([st["c_kv"], c_kv[None]], 0)
        st["k_rope"] = torch.cat([st["k_rope"], k_rope[None]], 0)
        kvb = self.kv_b(st["c_kv"]).view(-1, H, cfg.qk_nope + cfg.v_head).float()
        k_nope = kvb[..., : cfg.qk_nope]
        v = kvb[..., cfg.qk_nope :]
        scores = (torch.einsum("hd,lhd->lh", q_nope, k_nope)
                  + torch.einsum("hd,ld->lh", q_rope, st["k_rope"].float())) * self.scale
        p = torch.softmax(scores, dim=0)
        o = torch.einsum("lh,lhd->hd", p, v)
        return self.o_proj(o.reshape(H * cfg.v_head).to(torch.bfloat16))


class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, m, E = cfg.hidden, cfg.moe_inter, cfg.n_experts
        self.router = nn.Linear(d, E, bias=False, dtype=cfg.dtype)   # tiny, bf16
        self.gate = QuantExperts(E, d, m, cfg.group)
        self.up = QuantExperts(E, d, m, cfg.group)
        self.down = QuantExperts(E, m, d, cfg.group)
        self.s_gate = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_up = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_down = QuantExperts(cfg.n_shared, m, d, cfg.group)

    def _ffn(self, x, experts_g, experts_u, experts_d, e):
        h = F.silu(x.float() @ experts_g.weight_bf(e).float()) * (x.float() @ experts_u.weight_bf(e).float())
        return h @ experts_d.weight_bf(e).float()

    def step(self, x):
        cfg = self.cfg
        probs = torch.softmax(self.router(x).float(), dim=-1)
        w, idx = torch.topk(probs, cfg.n_active)
        w = w / (w.sum() + 1e-9) * cfg.routed_scaling
        out = x.new_zeros(cfg.hidden, dtype=torch.float32)
        for j in range(cfg.n_active):
            out = out + w[j] * self._ffn(x, self.gate, self.up, self.down, int(idx[j]))
        for s in range(cfg.n_shared):
            out = out + self._ffn(x, self.s_gate, self.s_up, self.s_down, s)
        return out.to(torch.bfloat16)


class Block(nn.Module):
    def __init__(self, cfg: Config, kind: str):
        super().__init__()
        self.kind = kind
        self.attn_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.moe_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.attn = KDA(cfg) if kind == "K" else MLA(cfg)
        self.moe = MoE(cfg)

    def step(self, x, st):
        h = x + self.attn.step(_rmsnorm(x, self.attn_norm), st)
        return h + self.moe.step(_rmsnorm(h, self.moe_norm))


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)
        self.reset_parameters()

    def reset_parameters(self):
        g = torch.Generator(device="cpu").manual_seed(1234)
        for mod in self.modules():
            if isinstance(mod, (QuantLinear, QuantExperts)):
                mod.init_random(g)
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0.0, 0.02, generator=g)
            elif isinstance(mod, KDA):
                nn.init.normal_(mod.conv_w, 0.0, 0.1, generator=g)

    def step(self, hidden, state):
        for i, blk in enumerate(self.blocks):
            hidden = blk.step(hidden, state[i])
        return hidden, state


# --------------------------------------------------------------------------- #
# state / inputs
# --------------------------------------------------------------------------- #
def init_state(cfg: Config, context_len: int, seed: int) -> list:
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(seed)
    H, Dk = cfg.kda_heads, cfg.kda_head_dim
    C = H * Dk
    state = []
    for kind in cfg.pattern:
        if kind == "K":
            state.append({
                "S": torch.randn(H, Dk, Dk, device=dev, generator=g) * 0.05,
                "cq": torch.randn(cfg.short_conv - 1, C, device=dev, generator=g, dtype=cfg.dtype) * 0.1,
                "ck": torch.randn(cfg.short_conv - 1, C, device=dev, generator=g, dtype=cfg.dtype) * 0.1,
                "cv": torch.randn(cfg.short_conv - 1, C, device=dev, generator=g, dtype=cfg.dtype) * 0.1,
            })
        else:
            state.append({
                "c_kv": torch.randn(context_len, cfg.kv_lora, device=dev, generator=g, dtype=cfg.dtype) * 0.1,
                "k_rope": torch.randn(context_len, cfg.qk_rope, device=dev, generator=g, dtype=cfg.dtype) * 0.1,
            })
    return state


def init_token(cfg: Config, seed: int) -> torch.Tensor:
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    return torch.randn(cfg.hidden, device=dev, generator=g, dtype=cfg.dtype) * 0.25


if __name__ == "__main__":
    cfg = build_config({"n_experts": 64})
    m = Model(cfg).cuda().eval()
    st = init_state(cfg, context_len=2048, seed=0)
    h = init_token(cfg, seed=0)
    with torch.no_grad():
        for _ in range(4):
            h, st = m.step(h, st)
    torch.cuda.synchronize()
    print(f"ok: out {tuple(h.shape)} finite {torch.isfinite(h).all().item()} | MLA cache {st[3]['c_kv'].shape[0]}")
    nbytes = sum(b.numel() * b.element_size() for b in m.buffers())
    print(f"int4 weight buffers: {nbytes/1e6:.1f} MB (vs ~{nbytes*4/1e6:.0f} MB bf16)")
