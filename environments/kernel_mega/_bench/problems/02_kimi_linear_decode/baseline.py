"""Optimized-PyTorch W4A16 baseline for the Kimi-Linear decode unit.

The floor the solution must beat -- NOT the oracle (reference.py). Same int4
weights, same math, written the competent pure-PyTorch way: bf16 accumulation
and batched MoE expert dequant instead of the reference's fp32 per-expert loop.

It still *materializes* each dequantized bf16 weight before the matmul (int4
read + bf16 write + bf16 read = ~9x the traffic of a fused path). That is the
whole point left on the table: stock PyTorch cannot fuse the int4 unpack into
the GEMV, so beating this baseline requires a custom fused dequant-GEMV kernel.
`score = baseline_latency / solution_latency`.
"""
from __future__ import annotations

import reference as R
import torch
import torch.nn as nn
import torch.nn.functional as F

Config = R.Config
build_config = R.build_config
init_state = R.init_state
init_token = R.init_token
_rmsnorm = R._rmsnorm
_rope_cossin = R._rope_cossin
_apply_rope = R._apply_rope


def _wbf(ql) -> torch.Tensor:
    return ql.weight_bf()  # [in, out] bf16


class KDA(R.KDA):
    def step(self, x, st):
        H, Dk = self.cfg.kda_heads, self.cfg.kda_head_dim
        q = x @ _wbf(self.q_proj)
        k = x @ _wbf(self.k_proj)
        v = x @ _wbf(self.v_proj)
        q, st["cq"] = self._short_conv(q, st["cq"], 0)
        k, st["ck"] = self._short_conv(k, st["ck"], 1)
        v, st["cv"] = self._short_conv(v, st["cv"], 2)
        q = q.view(H, Dk).float() * self.scale
        k = k.view(H, Dk).float()
        v = v.view(H, Dk).float()
        g = (-F.softplus((x @ _wbf(self.g_proj)).float())).view(H, Dk)
        beta = torch.sigmoid(self.beta_proj(x).float())
        S = st["S"] * g.exp()[:, :, None]
        pred = (S * k[:, :, None]).sum(1)
        S = S + beta[:, None, None] * k[:, :, None] * (v - pred)[:, None, :]
        o = (S * q[:, :, None]).sum(1)
        st["S"] = S
        return (o.reshape(H * Dk).to(torch.bfloat16)) @ _wbf(self.o_proj)


class MLA(R.MLA):
    def step(self, x, st):
        cfg = self.cfg
        H = cfg.mla_heads
        pos = st["c_kv"].shape[0]
        q = (x @ _wbf(self.q_proj)).view(H, cfg.qk_nope + cfg.qk_rope)
        q_nope = q[:, : cfg.qk_nope]
        q_rope = q[:, cfg.qk_nope :]
        kv = x @ _wbf(self.kv_a)
        c_kv = kv[: cfg.kv_lora]
        k_rope = kv[cfg.kv_lora :]
        cos, sin = _rope_cossin(pos, cfg.qk_rope, cfg.rope_theta, x.device)
        q_rope = _apply_rope(q_rope, cos, sin)
        k_rope = _apply_rope(k_rope, cos, sin)
        st["c_kv"] = torch.cat([st["c_kv"], c_kv[None]], 0)
        st["k_rope"] = torch.cat([st["k_rope"], k_rope[None]], 0)
        kvb = (st["c_kv"] @ _wbf(self.kv_b)).view(-1, H, cfg.qk_nope + cfg.v_head)
        k_nope = kvb[..., : cfg.qk_nope]
        v = kvb[..., cfg.qk_nope :]
        scores = (torch.einsum("hd,lhd->lh", q_nope, k_nope).float()
                  + torch.einsum("hd,ld->lh", q_rope, st["k_rope"]).float()) * self.scale
        p = torch.softmax(scores, dim=0).to(x.dtype)
        o = torch.einsum("lh,lhd->hd", p, v)
        return (o.reshape(H * cfg.v_head)) @ _wbf(self.o_proj)


def _batch_wbf(qe: R.QuantExperts, idx: torch.Tensor) -> torch.Tensor:
    """Dequant the selected experts to [k, in, out] bf16 (still materialized)."""
    K = qe.in_f
    wu = torch.empty((idx.numel(), K, qe.out_f), dtype=torch.uint8, device=qe.w_q.device)
    wp = qe.w_q[idx]
    wu[:, 0::2] = wp & 0xF
    wu[:, 1::2] = (wp >> 4) & 0xF
    s = qe.scales[idx].repeat_interleave(qe.group, dim=1)
    z = qe.zeros[idx].repeat_interleave(qe.group, dim=1)
    return (wu.to(torch.bfloat16) - z) * s


class MoE(R.MoE):
    def step(self, x):
        cfg = self.cfg
        probs = torch.softmax(self.router(x).float(), dim=-1)
        w, idx = torch.topk(probs, cfg.n_active)
        w = (w / (w.sum() + 1e-9) * cfg.routed_scaling).to(x.dtype)
        g = _batch_wbf(self.gate, idx)            # [k, d, m]
        u = _batch_wbf(self.up, idx)
        d = _batch_wbf(self.down, idx)            # [k, m, d]
        hh = F.silu(torch.einsum("e,kem->km", x, g)) * torch.einsum("e,kem->km", x, u)
        routed = torch.einsum("km,kmd->kd", hh, d)
        out = (w[:, None] * routed).sum(0)
        sidx = torch.arange(cfg.n_shared, device=x.device)
        sg = _batch_wbf(self.s_gate, sidx)
        su = _batch_wbf(self.s_up, sidx)
        sd = _batch_wbf(self.s_down, sidx)
        sh = F.silu(torch.einsum("e,kem->km", x, sg)) * torch.einsum("e,kem->km", x, su)
        out = out + torch.einsum("km,kmd->kd", sh, sd).sum(0)
        return out


class Block(nn.Module):
    def __init__(self, cfg, kind):
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)

    def step(self, hidden, state):
        for i, blk in enumerate(self.blocks):
            hidden = blk.step(hidden, state[i])
        return hidden, state
