"""Naive multi-axis RoPE pre-attention reference (correctness only).

Mirrors the Qwen2-VL apply_multimodal_rotary_pos_emb semantics: three position
axes (temporal, height, width), each contributing a slice of the head_dim via
mrope_section. Standard rotate-half on q and k once per-axis cos/sin have been
mixed.

Layout convention:
  q: (B, T, Hq, D)   bf16
  k: (B, T, Hkv, D)  bf16
  pos_t, pos_h, pos_w: (B, T) int64

Output: q_rot, k_rot in (B, H, T, D) — i.e. the layout an attention kernel
expects (head dim second). The transpose is part of the "pre-attention prep".
"""
import torch
import torch.nn as nn

OP_TYPE = "rope"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]


def _build_inv_freq(D: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Model(nn.Module):
    def __init__(self, B: int, T: int, Hq: int, Hkv: int, D: int,
                 mrope_section: tuple[int, int, int], max_pos: int):
        super().__init__()
        assert sum(mrope_section) == D // 2, \
            f"mrope_section must sum to D/2, got {mrope_section} sum={sum(mrope_section)} D/2={D//2}"
        self.B, self.T = B, T
        self.Hq, self.Hkv, self.D = Hq, Hkv, D
        self.mrope_section = tuple(mrope_section)
        self.max_pos = max_pos

        inv_freq = _build_inv_freq(D)
        pos = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)             # (max_pos, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (max_pos, D)
        self.register_buffer("cos_cache", emb.cos().to(torch.bfloat16))
        self.register_buffer("sin_cache", emb.sin().to(torch.bfloat16))

    def _mix_axes(self, c_t: torch.Tensor, c_h: torch.Tensor, c_w: torch.Tensor) -> torch.Tensor:
        """Pick per-axis values according to mrope_section, mirrored across the half boundary."""
        D = self.D
        s0, s1, s2 = self.mrope_section
        out = torch.empty_like(c_t)
        # First half [0:D/2]: pairs 0..D/2-1 split across t/h/w
        out[..., :s0]                 = c_t[..., :s0]
        out[..., s0:s0+s1]            = c_h[..., s0:s0+s1]
        out[..., s0+s1:s0+s1+s2]      = c_w[..., s0+s1:s0+s1+s2]
        # Mirror into second half [D/2:D] with the same axis assignments
        h = D // 2
        out[..., h:h+s0]              = c_t[..., h:h+s0]
        out[..., h+s0:h+s0+s1]        = c_h[..., h+s0:h+s0+s1]
        out[..., h+s0+s1:h+s0+s1+s2]  = c_w[..., h+s0+s1:h+s0+s1+s2]
        return out

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor,
        pos_t: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gather per-axis cos/sin: (B, T, D)
        cos_t = self.cos_cache[pos_t]
        sin_t = self.sin_cache[pos_t]
        cos_h = self.cos_cache[pos_h]
        sin_h = self.sin_cache[pos_h]
        cos_w = self.cos_cache[pos_w]
        sin_w = self.sin_cache[pos_w]

        cos = self._mix_axes(cos_t, cos_h, cos_w)  # (B, T, D)
        sin = self._mix_axes(sin_t, sin_h, sin_w)

        cos = cos.unsqueeze(2)  # (B, T, 1, D) -> broadcast over heads
        sin = sin.unsqueeze(2)

        q_rot = (q * cos) + (_rotate_half(q) * sin)  # (B, T, Hq, D)
        k_rot = (k * cos) + (_rotate_half(k) * sin)

        # Permute to attention layout (B, H, T, D)
        return q_rot.transpose(1, 2).contiguous(), k_rot.transpose(1, 2).contiguous()


# Module-level shape shims (overwritten per-shape by check.py / benchmark.py).
B = 1
T = 4096
Hq = 32
Hkv = 8
D = 128
MROPE_SECTION = (16, 24, 24)
MAX_POS = 32768


def get_inputs():
    q = torch.randn(B, T, Hq, D, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, T, Hkv, D, dtype=torch.bfloat16) * 0.5
    pos_t = torch.randint(0, MAX_POS, (B, T), dtype=torch.int64)
    pos_h = torch.randint(0, MAX_POS, (B, T), dtype=torch.int64)
    pos_w = torch.randint(0, MAX_POS, (B, T), dtype=torch.int64)
    return [q, k, pos_t, pos_h, pos_w]


def get_init_inputs():
    return [B, T, Hq, Hkv, D, MROPE_SECTION, MAX_POS]
