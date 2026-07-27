"""Kimi-Linear W4A8 hybrid decode fused into ONE persistent Triton megakernel.

One `step(hidden, state)` == one `_mega[(P,)](...)` launch.  The whole per-token
forward lives inside that launch: 3x(KDA + MoE) + 1x(MLA + MoE), every int4
dequant-GEMV, the short causal conv, the KDA recurrent state update, the MLA
latent-cache attention (weight-absorbed), the MoE router / top-8 / expert GEMVs,
all 8 RMSNorms and all 8 residual adds.  The grid is persistent (two CTAs per SM)
and the 33 dataflow dependencies between "stages" are resolved by software
grid-wide barriers (release atomic arrival + volatile poll of a monotonically
increasing target), so no kernel boundary is ever needed.

Decode at batch 1 is pure weight streaming: 230 MB of int4 projections and
routed experts per token (plus 2-19 MB of MLA cache over the scored context
range), which is a 127 us floor at the 1818 GB/s this kernel's GEMV loop
sustains.  Every design choice below is about keeping that stream saturated and
about not paying for anything else.

Design notes that matter for reading the kernel:

* GEMV, W4A8 with dp4a.  Weights are repacked once (at first step) into a
  task-major int32 arena W32[tile][kchunk][word][col]: the 8 nibbles of one
  int32 are 8 consecutive contraction elements of ONE output column, ordered so
  that `w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F` are each a 4-lane uint8
  vector aligned with 4 *consecutive* int8 activations.  A task is 512 output
  columns x 128 contraction elements == 16 int32 words x 512 columns == 32 KB
  (x3 for the two stages with split-k to spare, see KM below),
  and each word is retired by two `dp4a.u32.s32` against a *uniform* int32 of
  packed activations -- 4 weights per instruction, exact in int32.  The scales
  come out of the sum:
      sum_k (v-z)*s*x == s*xs*(sum_k v*xq - z*sum_k xq)
  so the whole epilogue is two FFMAs per output column, once per 32 KB.
  A standalone replica of this loop runs at 1818 GB/s against a 1861 GB/s
  load-only ceiling for the same access pattern.
* Two CTAs per SM.  The same replica reaches only 1533 GB/s with one CTA per SM:
  a single 4-warp CTA cannot keep enough loads in flight to cover DRAM latency.
* Residual adds ride the GEMV epilogue.  The four attention o_projs and the four
  MoE down_projs write their split-k partials with `red.global.add.f32` directly
  onto a residual stream that an earlier stage seeded, with the MoE router
  weight folded in.  That removes 8 reduce-and-add stages, 8 grid barriers and
  ~8 MB/token of partial traffic.  Because L2 atomics do not invalidate L1, the
  four addresses involved are read back with `.cg`.
* MLA absorption.  kv_b is folded into the query (qa[h,c] = sum_d q_nope[h,d]
  Wb[c,h,d]) and into the output (o[h,d] = sum_c pc[h,c] Wb[c,h,128+d]), so the
  cache pass costs O(L*512) instead of the reference's O(L*512*8192).  Softmax
  is flash-decode style: 114 l-chunks x 2 head-groups == 228 programs, each with
  a running max, then a rescaling combine.
* Cross-CTA memory model.  Apart from the four `.cg` residual addresses, every
  shared scratch address is written at most once per launch, every write is
  >=32 B (sector) aligned, and no program reads an address before the barrier
  that follows its write -- so plain L1-cached loads and stores are safe (L1
  dirty state is per-32 B-sector, and the release fence in the barrier writes
  dirty sectors back to L2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# config mirror (names/shapes must match reference.py for load_state_dict)
#
# Every constant the kernel reads is a tl.constexpr: triton requires that of
# jit-visible globals, and constexpr still behaves like an int on the host
# (__index__), so the packing code below shares one set of definitions.
# --------------------------------------------------------------------------- #
_c = tl.constexpr

D = _c(2304)            # hidden
H = _c(32)              # heads (kda and mla)
DK = _c(128)            # kda head dim
C4 = _c(H * DK)         # 4096
KVL = _c(512)           # kv lora rank
QKN = _c(128)           # qk nope
QKR = _c(64)            # qk rope
VH = _c(128)            # v head
QH = _c(QKN + QKR)      # 192
KVBW = _c(QKN + VH)     # 256
MI = _c(1024)           # moe intermediate
NEXP = _c(64)
NBLK = _c(4)

NSM = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 114
# Two CTAs per SM, not one.  A standalone replica of the _gemv inner loop tops
# out at 1533 GB/s with 114 CTAs and 1818 GB/s with 228 (the pure weight stream
# reaches 1761 and 1861), so the second resident CTA buys ~19% on every int4
# GEMV.  It costs ~9 us on the grid barriers, which have twice as many arrivals
# to gather; the net at ctx 2048 is -13 us.
P = 2 * NSM                # persistent grid: two CTAs per SM (host-side int)
NWARP = 4
NBAR = 33                  # barriers per launch
NT = _c(512)               # GEMV tile width (output columns)
NTB = _c(128)              # small tile width (kv_b)
RU = _c(4)                 # word loads per pipelined iteration
NSTG = _c(4)               # pipeline depth
GA = _c(2)                 # mla head groups
NCH = _c(P // GA)          # mla l-chunks
NCHP = _c(triton.next_power_of_2(int(NCH)))
BL = _c(16)                # mla rows per dot
QXW = _c(C4 // 4)          # per-program int8-activation words (max 4096 elements)
QSW = _c(8 * (C4 // 128))  # per-program (scale, sum-of-int8) pairs, stride 8

# per-stage GEMV task shapes: NTILE tiles of <stage> NT cols x KS k-chunks of 128.
#
# DNT is the tile width of the two stages whose output width is D=2304, which
# tiles badly: 2304 is 4.5 tiles of 512, so 11% of every weight byte loaded is
# padding and the final round runs on 46 of the 114 CTAs.  256 divides 2304
# exactly and removes both -- and is 8 us *slower*, because 256 columns over
# 128 threads is an 8-byte-per-thread load and the weight stream needs 16 to
# reach full bandwidth.  Keeping it as a named knob so the answer stays visible.
DNT = _c(512)                                               # o_proj / down width
#
# KM is how many 128-groups one task contracts over, so KS*KM == KF == the true
# k-chunk count of the weight arena and KM>1 is a pure reduction of split-k
# partial traffic (and of the reduce depth in the consumer stage).  It is free
# only while NTILE*KS*nslot still covers the grid: qkvg goes 576 tasks in 3
# rounds -> 192 in 1, both 84.2% efficient, and gate/up 648/3 -> 216/1, both
# 94.7%.  qkva (252 -> 126) and the two o_projs would lose occupancy, so they
# stay at KM=1 and KS == KF.
QKVG_NTILE, QKVG_KF, QKVG_KM = _c(4 * C4 // NT), _c(D // 128), _c(3)
QKVG_KS = _c(QKVG_KF // QKVG_KM)                            # 32, 18, 3 -> 6
OP_NTILE, OP_KS, OP_KM = _c((D + DNT - 1) // DNT), _c(C4 // 128), _c(1)
QKVA_NTILE, QKVA_KS, QKVA_KM = _c(14), _c(D // 128), _c(1)  # 14, 18
GU_NTILE, GU_KF, GU_KM = _c(2 * MI // NT), _c(D // 128), _c(3)
GU_KS = _c(GU_KF // GU_KM)                                  # 4, 18, 3 -> 6
DN_NTILE, DN_KS, DN_KM = _c((D + DNT - 1) // DNT), _c(MI // 128), _c(1)
KVBN_NTILE = _c(H * (KVL // 128))                           # 128
KVBV_NTILE, KVBV_KS = _c(H), _c(KVL // 128)                 # 32, 4

QKVA_NPAD = _c(QKVA_NTILE * NT)   # 7168

# ---- int32 weight arena ----
W_QKVG = _c(QKVG_NTILE * QKVG_KF * 16 * NT)
W_OP = _c(OP_NTILE * OP_KS * 16 * DNT)
W_KDA = _c(W_QKVG + W_OP)
W_QKVA = _c(QKVA_NTILE * QKVA_KS * 16 * NT)
W_KVBN = _c(KVBN_NTILE * 16 * NTB)
W_KVBV = _c(KVBV_NTILE * KVBV_KS * 16 * NTB)
W_MLA = _c(W_QKVA + W_KVBN + W_KVBV + W_OP)
W_GU = _c(GU_NTILE * GU_KF * 16 * NT)
W_DN = _c(DN_NTILE * DN_KS * 16 * DNT)
W_EXP = _c(W_GU + W_DN)
W_MOE = _c((NEXP + 1) * W_EXP)
A_KDA = _c(0)
A_MLA = _c(3 * W_KDA)
A_MOE = _c(A_MLA + W_MLA)
W_TOT = _c(A_MOE + NBLK * W_MOE)

# ---- parallel scale arena (bf16, holds s/8) and zero arena (int8) ----
S_QKVG = _c(QKVG_NTILE * QKVG_KF * NT)
S_OP = _c(OP_NTILE * OP_KS * DNT)
S_KDA = _c(S_QKVG + S_OP)
S_QKVA = _c(QKVA_NTILE * QKVA_KS * NT)
S_KVBV = _c(KVBV_NTILE * KVBV_KS * NTB)
S_MLA = _c(S_QKVA + S_KVBV + S_OP)
S_GU = _c(GU_NTILE * GU_KF * NT)
S_DN = _c(DN_NTILE * DN_KS * DNT)
S_EXP = _c(S_GU + S_DN)
S_MOE = _c((NEXP + 1) * S_EXP)
B_KDA = _c(0)
B_MLA = _c(3 * S_KDA)
B_MOE = _c(B_MLA + S_MLA)
S_TOT = _c(B_MOE + NBLK * S_MOE)

# ---- bf16 side arena for the small dense weights ----
WB_CW = _c(0)                                  # [3][3][4][4096]  conv_w tap-major
WB_BW = _c(WB_CW + 3 * 3 * 4 * C4)             # [3][32][2304]    beta_proj
WB_RW = _c(WB_BW + 3 * H * D)                  # [4][64][2304]    router
WB_NRM = _c(WB_RW + NBLK * NEXP * D)           # [4][2][2304]     norm weights
WB_TOT = _c(WB_NRM + NBLK * 2 * D)

# ---- bf16 scratch ----
CB_QA = _c(0)                                  # [32][512]       absorbed query
CB_QR = _c(CB_QA + H * KVL)                    # [32][64]        rope'd query
CB_APC = _c(CB_QR + H * QKR)                   # [NCH][32][512]  pc partials
CB_TOT = _c(CB_APC + NCH * H * KVL)

# --------------------------------------------------------------------------- #
# fp32 scratch layout
# --------------------------------------------------------------------------- #
_off = 0


def _a(n):
    global _off
    o = _off
    _off += (int(n) + 63) // 64 * 64   # keep every region 256 B aligned
    return _c(o)


SC_INVF = _a(QKR // 2)                    # rope inverse frequencies
SC_EZ = _a(8)                             # stays zero: EPTR for non-MoE gemvs
SC_SN8 = _a(H * 4 * QKN)                  # kv_b nope scales, s          [h][g][d]
SC_SNZ = _a(H * 4 * QKN)                  # kv_b nope scales, s*z
SC_XNP = _a(P * D)                        # private normed activation
SC_TOP = _a(P * 32)                       # private [9 expert ids | 8 probs]
SC_XQP = _a(P * KVL)                      # private scalar operands (kv_b gemvs)
SC_QX = _a(P * QXW)                       # private int8 activation, 4 per word
SC_QS = _a(P * QSW)                       # private (scale, sum-of-int8) pairs
SC_QHH = _a(NBLK * 72 * 32)               # shared int8 moe hidden   [slot*8+g][32]
SC_QHS = _a(NBLK * 72 * 8)                # shared (scale, sum) pairs    (stride 8)
SC_PQ = _a(3 * QKVG_KS * 4 * C4)          # q/k/v/g partials
SC_QKV = _a(3 * 4 * C4)                   # q, k, v, exp(g)
SC_BETA = _a(3 * H * 8)                   # beta                     (stride 8)
SC_O1 = _a(3 * C4)                        # kda attention output
SC_HA = _a(NBLK * D)                      # hidden after attn sublayer
SC_HM = _a(NBLK * D)                      # hidden after moe sublayer
SC_RLOG = _a(NBLK * NEXP * 8)             # router logits            (stride 8)
SC_PGU = _a(NBLK * 9 * GU_KS * 2 * MI)    # gate/up partials
SC_HH = _a(NBLK * 9 * MI)                 # silu(gate)*up
SC_PQA = _a(QKVA_KS * QKVA_NPAD)          # mla q_proj|kv_a partials
SC_AM = _a(NCH * H)                       # per-chunk running max
SC_AL = _a(NCH * H)                       # per-chunk sumexp
SC_O1M = _a(C4)                           # mla attention output
SC_TOT = _c(_off)

# strides that appear inside the kernel as `<runtime> * <stride>`
PQ_SLOT = _c(QKVG_KS * 4 * C4)            # per-kda-block q/k/v/g partial slot
PGU_SLOT = _c(9 * GU_KS * 2 * MI)         # per-moe-block gate/up partial slot
S_SLOT = _c(H * DK * DK)                  # per-kda-block recurrent state
CV_SLOT = _c(3 * C4)                      # one conv state (3 taps x 4096)

MLA_SCALE = _c(float(QH) ** -0.5)
KDA_SCALE = _c(float(DK) ** -0.5)
ROUTED_SCALE = _c(2.446)


# --------------------------------------------------------------------------- #
# module skeletons: identical parameter/buffer names to reference.py
# --------------------------------------------------------------------------- #
class QuantLinear(nn.Module):
    def __init__(self, in_f, out_f, group=128):
        super().__init__()
        self.in_f, self.out_f, self.group = in_f, out_f, group
        self.register_buffer("w_q", torch.zeros(in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(in_f // group, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(in_f // group, out_f, dtype=torch.bfloat16))


class QuantExperts(nn.Module):
    def __init__(self, n, in_f, out_f, group=128):
        super().__init__()
        self.n, self.in_f, self.out_f, self.group = n, in_f, out_f, group
        self.register_buffer("w_q", torch.zeros(n, in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(n, in_f // group, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(n, in_f // group, out_f, dtype=torch.bfloat16))


class KDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, c = cfg.hidden, cfg.kda_heads * cfg.kda_head_dim
        self.q_proj = QuantLinear(d, c, cfg.group)
        self.k_proj = QuantLinear(d, c, cfg.group)
        self.v_proj = QuantLinear(d, c, cfg.group)
        self.g_proj = QuantLinear(d, c, cfg.group)
        self.beta_proj = nn.Linear(d, cfg.kda_heads, bias=False, dtype=cfg.dtype)
        self.conv_w = nn.Parameter(torch.empty(3, c, cfg.short_conv, dtype=cfg.dtype))
        self.o_proj = QuantLinear(c, d, cfg.group)


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.q_proj = QuantLinear(d, cfg.mla_heads * (cfg.qk_nope + cfg.qk_rope), cfg.group)
        self.kv_a = QuantLinear(d, cfg.kv_lora + cfg.qk_rope, cfg.group)
        self.kv_b = QuantLinear(cfg.kv_lora, cfg.mla_heads * (cfg.qk_nope + cfg.v_head), cfg.group)
        self.o_proj = QuantLinear(cfg.mla_heads * cfg.v_head, d, cfg.group)


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, m, E = cfg.hidden, cfg.moe_inter, cfg.n_experts
        self.router = nn.Linear(d, E, bias=False, dtype=cfg.dtype)
        self.gate = QuantExperts(E, d, m, cfg.group)
        self.up = QuantExperts(E, d, m, cfg.group)
        self.down = QuantExperts(E, m, d, cfg.group)
        self.s_gate = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_up = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_down = QuantExperts(cfg.n_shared, m, d, cfg.group)


class Block(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.attn_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.moe_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.attn = KDA(cfg) if kind == "K" else MLA(cfg)
        self.moe = MoE(cfg)


# --------------------------------------------------------------------------- #
# device code
# --------------------------------------------------------------------------- #
@triton.jit
def _dp4a(a, b, c):
    """c + dot(4 uint8 lanes of a, 4 int8 lanes of b), exact in int32."""
    return tl.inline_asm_elementwise("dp4a.u32.s32 $0, $1, $2, $3;", "=r,r,r,r",
                                     [a, b, c], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _rq(y):
    """round-to-nearest-even for |y| <= 127, without a libdevice call."""
    return (y + 12582912.0) - 12582912.0


@triton.jit
def _quant(SCF, src, dq, ds, NG: tl.constexpr, QG: tl.constexpr,
           CG: tl.constexpr = 0):
    """int8-quantize NG groups of 128 fp32 activations for the dp4a GEMVs.

    Per group: 32 packed int32 words (4 int8 lanes each, bitcast into the fp32
    arena so no extra kernel argument is needed) plus the pair (scale,
    sum-of-int8) that the GEMV epilogue needs for the zero-point correction.

    QG groups are reduced per pass.  A group is 128 elements == one element per
    thread, so its max and its sum are both *whole-CTA* reductions (shared
    memory, two barriers); doing one group at a time cost ~0.7 us each, and at
    272 groups per token that was a quarter of the whole step.  Widening the
    tile amortises the two reductions over QG groups at no extra traffic.

    QG must be a power of two and is chosen per call site: rounding NG up to a
    multiple of QG costs masked-off lanes, so a caller with NG=1 passes QG=1.
    CG=1 bypasses L1 on the source, for the one caller whose input arrived by
    L2 atomics (see the mla o_proj).
    """
    r32 = tl.arange(0, 32)[None, :, None]
    c4 = tl.arange(0, 4)[None, None, :]
    for g0 in tl.static_range(0, NG, QG):
        g = g0 + tl.arange(0, QG)
        m = g < NG
        if CG != 0:
            xt = tl.load(SCF + src + g[:, None, None] * 128 + r32 * 4 + c4,
                         mask=m[:, None, None], other=0.0, cache_modifier=".cg")
        else:
            xt = tl.load(SCF + src + g[:, None, None] * 128 + r32 * 4 + c4,
                         mask=m[:, None, None], other=0.0)
        mx = tl.max(tl.max(tl.abs(xt), axis=2), axis=1)
        q = _rq(xt * (127.0 / (mx + 1.0e-30))[:, None, None]).to(tl.int32)
        tl.store(SCF + dq + g[:, None] * 32 + tl.arange(0, 32)[None, :],
                 tl.sum((q & 0xFF) << (c4 * 8), axis=2).to(tl.float32, bitcast=True),
                 mask=m[:, None])
        tl.store(SCF + ds + 8 * g, mx * (1.0 / 127.0), mask=m)
        tl.store(SCF + ds + 8 * g + 1,
                 tl.sum(tl.sum(q, axis=2), axis=1).to(tl.float32), mask=m)
    tl.debug_barrier()


@triton.jit
def _bf(x):
    """round to bf16 precision, keep the fp32 container (matches reference casts)."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _bar(BAR, tgt, pid):
    """grid-wide barrier on a single monotonically increasing arrival counter.

    `tgt` is the cumulative arrival count over the whole launch (and across
    launches), so the counter is never reset and no sense reversal is needed.

    Two variants measured slower and are worth not re-inventing:
      * spreading arrivals over 8 counters 64 B apart and polling their sum
        (+0.25 us/barrier: 8x the poll traffic, and if the sum is taken with
        `tl.sum` over a tile it also puts a shared-memory reduction in the
        spin loop);
      * having the last arriver publish a release flag on a separate line so
        waiters spin somewhere that is not being atomically updated
        (+0.4 us/barrier: the publisher's own atomic has to complete a full
        L2 round trip before the flag store can even be issued).

    Cross-CTA data is left in L1: the release on the arrival makes it visible,
    and no shared address is written twice within a launch, so no waiter can be
    holding a stale line for it.
    """
    tl.debug_barrier()
    tl.atomic_add(BAR, 1, sem="release", scope="gpu")
    while tl.load(BAR, volatile=True) < tgt:
        pass
    tl.debug_barrier()


@triton.jit
def _cpy(dst, src, n, pid, P: tl.constexpr, W: tl.constexpr):
    for i in range(pid * W, n, P * W):
        tl.store(dst + i + tl.arange(0, W), tl.load(src + i + tl.arange(0, W)))


# A whole-hidden pass is walked in DNC chunks of DCH.  Two constraints pull in
# opposite directions: a 128-element chunk is one element per thread, so *every*
# `tl.sum` over it is a full shared-memory CTA reduction (~0.7 us) -- but a
# single 4096-wide tile is 32 live registers per thread and pushes this kernel,
# already at 255 registers, into heavy spilling.  DCH=1024 keeps 8 elements per
# thread while still reducing only once per pass, at the cost of 3 iterations.
DCH = _c(1024)
DNC = _c(3)


@triton.jit
def _dotd(SCF, aoff, WB, woff):
    """<x, w> over the hidden vector: x fp32 in scratch, w bf16 in the weight blob.

    Partials stay in a DCH-wide vector so the cross-thread reduction happens once
    at the end instead of once per chunk.
    """
    acc = tl.zeros((DCH,), tl.float32)
    for i in tl.static_range(0, DNC):
        o = i * DCH + tl.arange(0, DCH)
        m = o < D
        acc += (tl.load(SCF + aoff + o, mask=m, other=0.0)
                * tl.load(WB + woff + o, mask=m, other=0.0).to(tl.float32))
    return tl.sum(acc)


@triton.jit
def _nld(SCF, HB, hoff, use_bf, o):
    """load a slice of the residual stream, bf16-rounded like the reference."""
    if use_bf != 0:
        v = tl.load(HB + o).to(tl.float32)
    else:
        v = _bf(tl.load(SCF + hoff + o, cache_modifier=".cg"))
    return v


@triton.jit
def _norm(WB, SCF, HB, hoff, woff, use_bf, pid):
    """rmsnorm(h)*w -> the program-private activation slot XNP[pid], bf16 rounded.

    Deliberately redundant across programs: the result is program-private, so the
    GEMV that consumes it needs no barrier and hits in its own L1.

    The hidden is read exactly once and held in registers across the CTA
    reduction (18 floats per thread).  Re-reading it for the rescale pass was
    cheaper while it came back out of L1, but the residual stream is now written
    by L2 atomics and has to be read `.cg`, so a second pass would be a second
    round trip -- 4 MB of it, all P programs pulling the same 9 KB.
    2304 == 2048 + 256, so two exact power-of-two tiles cover it with no mask.
    """
    oa = tl.arange(0, 2048)
    ob = 2048 + tl.arange(0, 256)
    ha = _nld(SCF, HB, hoff, use_bf, oa)
    hb = _nld(SCF, HB, hoff, use_bf, ob)
    rs = tl.rsqrt((tl.sum(ha * ha) + tl.sum(hb * hb)) / 2304.0 + 1.0e-6)
    tl.store(SCF + SC_XNP + pid * D + oa,
             _bf(ha * rs * tl.load(WB + woff + oa).to(tl.float32)))
    tl.store(SCF + SC_XNP + pid * D + ob,
             _bf(hb * rs * tl.load(WB + woff + ob).to(tl.float32)))
    tl.debug_barrier()


@triton.jit
def _gemv(WA, SA, ZA, SCF, EPTR, wbase, sbase, pbase, xqb, xsb, xqstr, xsstr, estw, ests,
          ntile, ks, nslot, npad, atom, wptr, wnorm, pid,
          KM: tl.constexpr, NTW: tl.constexpr, P: tl.constexpr):
    """fused int4 dequant-GEMV stage, W4A8.

    nslot*ntile*ks tasks; one task is NTW output columns x KM*128 contraction
    elements (== KM*16 int32 weight words per column, one scale group each).
    EPTR[slot] picks the expert (a zero float for the non-MoE stages).

    With atom=0 each task drops its split-k partial in its own scratch slot and a
    later stage reduces them.  With atom!=0 the partial is instead red.add-ed
    straight into the residual stream at `pbase` (pre-seeded with the residual by
    an earlier stage, `npad` output columns wide, `wptr[slot]*wnorm` weighting the
    first nslot-1 slots).  That deletes the reduce+residual stage that would
    otherwise follow -- and, worth far more, the grid barrier in front of it.

    Both operands are integers: `w & 0x0F0F0F0F` is a 4-lane uint8 vector of the
    nibbles for 4 consecutive contraction indices (see _pack), so one dp4a
    against one *uniform* int32 of packed activations retires 4 weights, and the
    128-element dot product is exact in int32.  The group scale, the activation
    scale and the zero-point correction are applied once per task:
        sum_k (v-z)*s*x == s*xs*(sum_k v*xq - z*sum_k xq)
    That is ~4.5x fewer instructions per weight byte than dequantising to fp32,
    which is what this loop is actually bound by.

    The task shape is passed as *runtime* values on purpose: the five GEMV
    stages then share two compiled copies of this heavily unrolled inner loop
    (one per distinct KM), not five, which cuts both compile time and
    instruction footprint.  The extra integer math is amortised over 32 KB of
    weight loads per task.
    """
    nl = tl.arange(0, NTW)
    nk = ntile * ks
    for t in range(pid, nslot * nk, P):
        s = t // nk
        r = t - s * nk
        k = r // ntile
        i = r - k * ntile
        e = tl.load(EPTR + s).to(tl.int32)
        po = pbase + (s * ks + k) * npad + i * NTW
        g0 = (i * ks + k) * KM      # first of this task's KM 128-groups
        val = tl.zeros((NTW,), tl.float32)
        for j in tl.static_range(KM):
            wo = wbase + e * estw + (g0 + j) * (16 * NTW)
            so = sbase + e * ests + (g0 + j) * NTW
            qo = xqb + s * xqstr + (k * KM + j) * 32
            zo = xsb + (s * xsstr + k * KM + j) * 8
            # Issued before the accumulation loop, not after it: the epilogue
            # operands are independent of `a`, so hoisting them lets their
            # ~500-cycle latency hide under the dp4a chain instead of extending
            # it.  They are kept in their narrow load types (bf16 scale, int8
            # zero) so the live set across the loop is 3 registers per thread
            # rather than 8.
            sc = tl.load(SA + so + nl)
            zv = tl.load(ZA + so + nl)
            xs = tl.load(SCF + zo)
            xz = tl.load(SCF + zo + 1)
            a = tl.zeros((NTW,), tl.int32)
            for m in tl.range(0, 16, RU, num_stages=NSTG):
                for u in tl.static_range(RU):
                    w = tl.load(WA + wo + (m + u) * NTW + nl)
                    xl = tl.load(SCF + qo + 2 * (m + u)).to(tl.int32, bitcast=True)
                    xh = tl.load(SCF + qo + 2 * (m + u) + 1).to(tl.int32, bitcast=True)
                    a = _dp4a(w & 0x0F0F0F0F, xl, a)
                    a = _dp4a((w >> 4) & 0x0F0F0F0F, xh, a)
            val += ((sc.to(tl.float32) * (8.0 * xs))
                    * (a.to(tl.float32) - zv.to(tl.float32) * xz))
        if atom == 0:
            tl.store(SCF + po + nl, val)
        else:
            if s < nslot - 1:
                wj = tl.load(SCF + wptr + s) * wnorm
            else:
                wj = 1.0
            # red.global.add.f32 at L2.  The mask drops the tail of the last
            # NTW-tile (D is not a multiple of NTW).  Readers of pbase must not
            # have touched it earlier in the launch -- atomics land in L2 and do
            # not invalidate a stale L1 line -- which holds here: the stage that
            # seeds pbase with the residual only stores to it.
            oi = i * NTW + nl
            tl.atomic_add(SCF + pbase + oi, val * wj, mask=oi < npad,
                          sem="relaxed")


@triton.jit
def _redp(SCF, pbase, off, KS: tl.constexpr, NPAD: tl.constexpr, n: tl.constexpr):
    """sum the KS split-k partials of n contiguous output columns."""
    acc = tl.zeros((n,), tl.float32)
    for k in tl.static_range(KS):
        acc += tl.load(SCF + pbase + k * NPAD + off + tl.arange(0, n))
    return acc


@triton.jit
def _gemv_small(WA, SCF, wo, qo, NTW: tl.constexpr):
    """dp4a GEMV over exactly 128 contraction elements; scales applied outside."""
    acc = tl.zeros((NTW,), tl.int32)
    nl = tl.arange(0, NTW)
    for m in tl.range(0, 16, RU, num_stages=NSTG):
        for u in tl.static_range(RU):
            w = tl.load(WA + wo + (m + u) * NTW + nl)
            acc = _dp4a(w & 0x0F0F0F0F,
                        tl.load(SCF + qo + 2 * (m + u)).to(tl.int32, bitcast=True), acc)
            acc = _dp4a((w >> 4) & 0x0F0F0F0F,
                        tl.load(SCF + qo + 2 * (m + u) + 1).to(tl.int32, bitcast=True), acc)
    return acc


@triton.jit
def _moe(WB, WA, SA, ZA, SCF, BAR, b, nb, pid,
         sid: tl.constexpr, SSTOP: tl.constexpr, P: tl.constexpr):
    """4 stages: norm+router | top8+gate/up | silu*up+seed | down (accumulating)."""
    if sid + 1 <= SSTOP:
        _norm(WB, SCF, WB, SC_HA + b * D, WB_NRM + b * 2 * D + D, 0, pid)
        _quant(SCF, SC_XNP + pid * D, SC_QX + pid * QXW, SC_QS + pid * QSW, 18, 16)
        for e in range(pid, NEXP, P):
            tl.store(SCF + SC_RLOG + (b * NEXP + e) * 8,
                     _bf(_dotd(SCF, SC_XNP + pid * D, WB, WB_RW + (b * NEXP + e) * D)))
    nb += P
    _bar(BAR, nb, pid)

    if sid + 2 <= SSTOP:
        ee = tl.arange(0, NEXP)
        lg = tl.load(SCF + SC_RLOG + b * NEXP * 8 + ee * 8)
        mx = tl.max(lg)
        den = tl.sum(tl.exp(lg - mx))
        rk = tl.sum(((lg[None, :] > lg[:, None])
                     | ((lg[None, :] == lg[:, None]) & (ee[None, :] < ee[:, None]))).to(tl.int32),
                    axis=1)
        tl.store(SCF + SC_TOP + pid * 32 + rk, ee.to(tl.float32), mask=rk < 8)
        tl.store(SCF + SC_TOP + pid * 32 + 16 + rk, tl.exp(lg - mx) / den, mask=rk < 8)
        tl.store(SCF + SC_TOP + pid * 32 + 8, NEXP + 0.0)
        tl.debug_barrier()
        _gemv(WA, SA, ZA, SCF, SCF + SC_TOP + pid * 32,
              A_MOE + b * W_MOE, B_MOE + b * S_MOE, SC_PGU + b * PGU_SLOT,
              SC_QX + pid * QXW, SC_QS + pid * QSW, 0, 0, W_EXP, S_EXP,
              GU_NTILE, GU_KS, 9, 2 * MI, 0, 0, 0.0, pid, GU_KM, NT, P)
    nb += P
    _bar(BAR, nb, pid)

    if sid + 3 <= SSTOP:
        # Seed the block output with the residual; the down_proj GEMV of the next
        # stage red.adds the expert contributions on top of it.  A separate
        # buffer from SC_HA is required, not merely tidy: SC_HA was *read* by the
        # rmsnorm two stages ago, so the reading CTAs hold it in L1, and L2
        # atomics would not invalidate those lines.
        for c in range(pid, 72, P):
            o = c * 32 + tl.arange(0, 32)
            tl.store(SCF + SC_HM + b * D + o,
                     _bf(tl.load(SCF + SC_HA + b * D + o, cache_modifier=".cg")))
        pg = SC_PGU + b * PGU_SLOT
        for t in range(pid, 72, P):
            s = t // 8
            g = t % 8
            base = pg + s * (GU_KS * 2 * MI)
            gv = _redp(SCF, base, g * 128, GU_KS, 2 * MI, 128)
            uv = _redp(SCF, base, MI + g * 128, GU_KS, 2 * MI, 128)
            hv = (gv * tl.sigmoid(gv)) * uv
            ho = SC_HH + b * 9 * MI + s * MI + g * 128
            tl.store(SCF + ho + tl.arange(0, 128), hv)
            # one task == exactly one 128-element quantisation group, so the
            # int8 form of the down_proj input is produced here (once) instead
            # of redundantly in all P programs.  Per-block slot: like every
            # other shared region this must be written at most once per launch,
            # or a reader's L1 can still hold the previous block's line.
            tl.debug_barrier()
            _quant(SCF, ho, SC_QHH + (b * 72 + t) * 32, SC_QHS + (b * 72 + t) * 8, 1, 1)
    nb += P
    _bar(BAR, nb, pid)

    if sid + 4 <= SSTOP:
        # The router weight of each of the 8 routed slots (the 9th is the shared
        # expert, weight 1) folds into the GEMV epilogue, so the 9 x 8 split-k
        # partials never reach memory: they red.add straight into SC_HM.
        wsum = tl.sum(tl.load(SCF + SC_TOP + pid * 32 + 16 + tl.arange(0, 8)))
        _gemv(WA, SA, ZA, SCF, SCF + SC_TOP + pid * 32,
              A_MOE + b * W_MOE + W_GU, B_MOE + b * S_MOE + S_GU,
              SC_HM + b * D, SC_QHH + b * 72 * 32, SC_QHS + b * 72 * 8,
              MI // 4, MI // 128, W_EXP, S_EXP,
              DN_NTILE, DN_KS, 9, D, 1, SC_TOP + pid * 32 + 16,
              ROUTED_SCALE / (wsum + 1.0e-9), pid, DN_KM, DNT, P)
    nb += P
    _bar(BAR, nb, pid)         # HM is consumed by the next block's rmsnorm


@triton.jit(do_not_specialize=["bar0", "L"])
def _mega(HBF, HOUT, WA, SA, ZA, WB, SCF, SCB, BAR, bar0,
          SB, CVI, CVO, CKV, KRP,
          S0, S1, S2, CQ0, CK0, CV0, CQ1, CK1, CV1, CQ2, CK2, CV2, CKVS, KRPS,
          L, DOCOPY, SSTOP: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    nb = bar0

    # ---------------- adopt a freshly fed state into our own buffers ---------
    # statically unrolled, so no later stage needs a runtime-selected pointer
    if DOCOPY != 0:
        _cpy(SB + 0 * S_SLOT, S0, S_SLOT, pid, P, 512)
        _cpy(SB + 1 * S_SLOT, S1, S_SLOT, pid, P, 512)
        _cpy(SB + 2 * S_SLOT, S2, S_SLOT, pid, P, 512)
        _cpy(CVI + 0 * CV_SLOT, CQ0, CV_SLOT, pid, P, 128)
        _cpy(CVI + 1 * CV_SLOT, CK0, CV_SLOT, pid, P, 128)
        _cpy(CVI + 2 * CV_SLOT, CV0, CV_SLOT, pid, P, 128)
        _cpy(CVI + 3 * CV_SLOT, CQ1, CV_SLOT, pid, P, 128)
        _cpy(CVI + 4 * CV_SLOT, CK1, CV_SLOT, pid, P, 128)
        _cpy(CVI + 5 * CV_SLOT, CV1, CV_SLOT, pid, P, 128)
        _cpy(CVI + 6 * CV_SLOT, CQ2, CV_SLOT, pid, P, 128)
        _cpy(CVI + 7 * CV_SLOT, CK2, CV_SLOT, pid, P, 128)
        _cpy(CVI + 8 * CV_SLOT, CV2, CV_SLOT, pid, P, 128)
        _cpy(CKV, CKVS, L * KVL, pid, P, 512)
        _cpy(KRP, KRPS, L * QKR, pid, P, 64)

    # ========================= 3 x (KDA + MoE) =============================
    for b in range(0, 3):
        # ---- norm + q/k/v/g projection ----
        if 1 <= SSTOP:
            _norm(WB, SCF, HBF, SC_HM + tl.maximum(b - 1, 0) * D, WB_NRM + b * 2 * D,
                  tl.where(b == 0, 1, 0), pid)
            _quant(SCF, SC_XNP + pid * D, SC_QX + pid * QXW, SC_QS + pid * QSW, 18, 16)
            _gemv(WA, SA, ZA, SCF, SCF + SC_EZ, A_KDA + b * W_KDA, B_KDA + b * S_KDA,
                  SC_PQ + b * PQ_SLOT, SC_QX + pid * QXW, SC_QS + pid * QSW, 0, 0, 0, 0,
                  QKVG_NTILE, QKVG_KS, 1, 4 * C4, 0, 0, 0.0, pid, QKVG_KM, NT, P)
        nb += P
        _bar(BAR, nb, pid)

        # ---- short causal conv + silu, gate decay, beta ----
        if 2 <= SSTOP:
            # seed the attention residual stream for the o_proj two stages down
            for c in range(pid, 72, P):
                o = c * 32 + tl.arange(0, 32)
                if b == 0:
                    hv = tl.load(HBF + o).to(tl.float32)
                else:
                    hv = _bf(tl.load(SCF + SC_HM + (b - 1) * D + o,
                                     cache_modifier=".cg"))
                tl.store(SCF + SC_HA + b * D + o, hv)
            pq = SC_PQ + b * PQ_SLOT
            for c in range(pid, 128, P):
                mi = c // 32
                col = (c % 32) * 128
                nlq = col + tl.arange(0, 128)
                acc = _redp(SCF, pq + mi * C4, col, QKVG_KS, 4 * C4, 128)
                if mi == 3:
                    # exp(-softplus(g)) == sigmoid(-g)
                    tl.store(SCF + SC_QKV + b * 4 * C4 + 3 * C4 + nlq, tl.sigmoid(-_bf(acc)))
                else:
                    val = _bf(acc)
                    ci = CVI + (b * 3 + mi) * CV_SLOT + nlq
                    p0 = tl.load(ci).to(tl.float32)
                    p1 = tl.load(ci + C4).to(tl.float32)
                    p2 = tl.load(ci + 2 * C4).to(tl.float32)
                    cw = WB + WB_CW + (b * 3 + mi) * 4 * C4 + nlq
                    o = (p0 * tl.load(cw).to(tl.float32)
                         + p1 * tl.load(cw + C4).to(tl.float32)
                         + p2 * tl.load(cw + 2 * C4).to(tl.float32)
                         + val * tl.load(cw + 3 * C4).to(tl.float32))
                    o = _bf(o * tl.sigmoid(o))
                    o = tl.where(mi == 0, o * KDA_SCALE, o)
                    tl.store(SCF + SC_QKV + b * 4 * C4 + mi * C4 + nlq, o)
                    co = CVO + (b * 3 + mi) * CV_SLOT + nlq
                    tl.store(co, p1.to(tl.bfloat16))
                    tl.store(co + C4, p2.to(tl.bfloat16))
                    tl.store(co + 2 * C4, val.to(tl.bfloat16))
            for h in range(pid, H, P):
                tl.store(SCF + SC_BETA + b * H * 8 + h * 8,
                         tl.sigmoid(_bf(_dotd(SCF, SC_XNP + pid * D,
                                              WB, WB_BW + (b * H + h) * D))))
        nb += P
        _bar(BAR, nb, pid)

        # ---- gated delta rule: decay, predict, correct, read out ----
        if 3 <= SSTOP:
            qb = SC_QKV + b * 4 * C4
            dk = tl.arange(0, 128)
            dv = tl.arange(0, 8)
            for t in range(pid, H * 16, P):
                h = t // 16
                d0 = (t % 16) * 8
                ge = tl.load(SCF + qb + 3 * C4 + h * 128 + dk)
                kk = tl.load(SCF + qb + C4 + h * 128 + dk)
                qq = tl.load(SCF + qb + h * 128 + dk)
                vv = tl.load(SCF + qb + 2 * C4 + h * 128 + d0 + dv)
                bt = tl.load(SCF + SC_BETA + b * H * 8 + h * 8)
                sadr = SB + b * S_SLOT + h * (DK * DK) + dk[:, None] * DK + d0 + dv[None, :]
                sg = tl.load(sadr) * ge[:, None]
                pred = tl.sum(sg * kk[:, None], axis=0)
                aa = tl.sum(sg * qq[:, None], axis=0)
                dvv = bt * (vv - pred)
                tl.store(sadr, sg + kk[:, None] * dvv[None, :])
                tl.store(SCF + SC_O1 + b * C4 + h * 128 + d0 + dv,
                         _bf(aa + dvv * tl.sum(kk * qq)))
        nb += P
        _bar(BAR, nb, pid)

        # ---- kda o_proj, accumulated onto the residual ----
        if 4 <= SSTOP:
            _quant(SCF, SC_O1 + b * C4, SC_QX + pid * QXW, SC_QS + pid * QSW, OP_KS, 16)
            _gemv(WA, SA, ZA, SCF, SCF + SC_EZ, A_KDA + b * W_KDA + W_QKVG,
                  B_KDA + b * S_KDA + S_QKVG, SC_HA + b * D,
                  SC_QX + pid * QXW, SC_QS + pid * QSW, 0, 0, 0, 0,
                  OP_NTILE, OP_KS, 1, D, 1, 0, 0.0, pid, OP_KM, DNT, P)
        nb += P
        _bar(BAR, nb, pid)

        _moe(WB, WA, SA, ZA, SCF, BAR, b, nb, pid, 4, SSTOP, P)
        nb += 4 * P

    # ============================ MLA + MoE ================================
    # ---- norm + q_proj | kv_a ----
    if 31 <= SSTOP:
        _norm(WB, SCF, WB, SC_HM + 2 * D, WB_NRM + 3 * 2 * D, 0, pid)
        _quant(SCF, SC_XNP + pid * D, SC_QX + pid * QXW, SC_QS + pid * QSW, 18, 16)
        _gemv(WA, SA, ZA, SCF, SCF + SC_EZ, A_MLA, B_MLA, SC_PQA,
              SC_QX + pid * QXW, SC_QS + pid * QSW, 0, 0, 0, 0,
              QKVA_NTILE, QKVA_KS, 1, QKVA_NPAD, 0, 0, 0.0, pid, QKVA_KM, NT, P)
    nb += P
    _bar(BAR, nb, pid)

    # ---- absorbed query (kv_b nope half), rope, cache append ----
    if 32 <= SSTOP:
        d128 = tl.arange(0, 128)
        for t in range(pid, KVBN_NTILE, P):
            h = t // 4
            g = t % 4
            qn = _bf(_redp(SCF, SC_PQA, h * QH, QKVA_KS, QKVA_NPAD, 128)) * MLA_SCALE
            sn8 = tl.load(SCF + SC_SN8 + (h * 4 + g) * 128 + d128)
            snz = tl.load(SCF + SC_SNZ + (h * 4 + g) * 128 + d128)
            cst = tl.sum(qn * snz)          # exact zero-point term, from fp32 q
            tl.store(SCF + SC_XQP + pid * KVL + d128, qn * sn8)
            tl.debug_barrier()
            _quant(SCF, SC_XQP + pid * KVL, SC_QX + pid * QXW, SC_QS + pid * QSW, 1, 1)
            acc = _gemv_small(WA, SCF, (A_MLA + W_QKVA) + t.to(tl.int64) * (16 * NTB),
                              SC_QX + pid * QXW, NTB)
            xs = tl.load(SCF + SC_QS + pid * QSW)
            tl.store(SCB + CB_QA + h * KVL + g * 128 + tl.arange(0, NTB),
                     (xs * acc.to(tl.float32) - cst).to(tl.bfloat16))
            tl.debug_barrier()
        i32 = tl.arange(0, 32)
        ang = L.to(tl.float32) * tl.load(SCF + SC_INVF + i32)
        cs = tl.cos(ang)
        sn = tl.sin(ang)
        ev = tl.arange(0, 64)[None, :] == (2 * i32)[:, None]
        od = tl.arange(0, 64)[None, :] == (2 * i32 + 1)[:, None]
        if pid == 0:
            ck = _bf(_redp(SCF, SC_PQA, H * QH, QKVA_KS, QKVA_NPAD, KVL))
            tl.store(CKV + L.to(tl.int64) * KVL + tl.arange(0, KVL), ck.to(tl.bfloat16))
        if pid == 1:
            kr = _bf(_redp(SCF, SC_PQA, H * QH + KVL, QKVA_KS, QKVA_NPAD, 64))
            ke = tl.sum(tl.where(ev, kr[None, :], 0.0), axis=1)
            ko = tl.sum(tl.where(od, kr[None, :], 0.0), axis=1)
            tl.store(KRP + L.to(tl.int64) * QKR + 2 * i32, _bf(ke * cs - ko * sn).to(tl.bfloat16))
            tl.store(KRP + L.to(tl.int64) * QKR + 2 * i32 + 1,
                     _bf(ko * cs + ke * sn).to(tl.bfloat16))
        for zc in range(pid, 32, P):       # zero the stage-34 accumulator
            tl.store(SCF + SC_O1M + zc * 128 + tl.arange(0, 128),
                     tl.zeros((128,), tl.float32))
        for hh in range(pid - 2, H, P):
            if hh >= 0:
                qr = _bf(_redp(SCF, SC_PQA, hh * QH + QKN, QKVA_KS, QKVA_NPAD, 64))
                qe = tl.sum(tl.where(ev, qr[None, :], 0.0), axis=1)
                qo = tl.sum(tl.where(od, qr[None, :], 0.0), axis=1)
                tl.store(SCB + CB_QR + hh * QKR + 2 * i32, _bf(qe * cs - qo * sn).to(tl.bfloat16))
                tl.store(SCB + CB_QR + hh * QKR + 2 * i32 + 1,
                         _bf(qo * cs + qe * sn).to(tl.bfloat16))
    nb += P
    _bar(BAR, nb, pid)

    # ---- latent attention: flash-decode over (l-chunk, head-group) ----
    if 33 <= SSTOP:
        hg = pid % GA
        ci = pid // GA
        if ci < NCH:
            h0 = hg * 16
            hr = tl.arange(0, 16)
            cl = tl.arange(0, KVL)
            qa = tl.trans(tl.load(SCB + CB_QA + (h0 + hr)[:, None] * KVL + cl[None, :]))
            qr = tl.trans(tl.load(SCB + CB_QR + (h0 + hr)[:, None] * QKR
                                  + tl.arange(0, QKR)[None, :]))
            cw = (L + NCH) // NCH
            l0 = ci * cw
            l1 = tl.minimum(l0 + cw, L + 1)
            mx = tl.full((16,), -1.0e30, tl.float32)
            ls = tl.zeros((16,), tl.float32)
            pc = tl.zeros((16, KVL), tl.float32)
            for lb in range(l0, l1, BL):
                rows = lb + tl.arange(0, BL)
                msk = rows < l1
                ck = tl.load(CKV + rows[:, None].to(tl.int64) * KVL + cl[None, :],
                             mask=msk[:, None], other=0.0)
                kr = tl.load(KRP + rows[:, None].to(tl.int64) * QKR
                             + tl.arange(0, QKR)[None, :], mask=msk[:, None], other=0.0)
                s = tl.dot(ck, qa) + tl.dot(kr, qr) * MLA_SCALE
                s = tl.where(msk[:, None], s, -1.0e30)
                mn = tl.maximum(mx, tl.max(s, axis=0))
                al = tl.exp(mx - mn)
                pe = tl.exp(s - mn[None, :])
                ls = ls * al + tl.sum(pe, axis=0)
                pc = pc * al[:, None] + tl.dot(tl.trans(pe).to(tl.bfloat16), ck)
                mx = mn
            tl.store(SCF + SC_AM + ci * H + h0 + hr, mx)
            tl.store(SCF + SC_AL + ci * H + h0 + hr, ls)
            tl.store(SCB + CB_APC + (ci * H + h0 + hr)[:, None] * KVL + cl[None, :],
                     pc.to(tl.bfloat16))
    nb += P
    _bar(BAR, nb, pid)

    # ---- softmax combine + absorbed output projection (kv_b v half) ----
    if 34 <= SSTOP:
        # One task per (head, 128-wide slice of the latent), not per head: with
        # 32 heads only 32 of the P programs had anything to do here and the
        # stage ran at 14% occupancy while pulling 3.7 MB of partial contexts.
        # The four slices of a head red.add into the same 128 outputs.
        nl = tl.arange(0, NTB)
        cc = tl.arange(0, NCHP)
        cm = cc < NCH
        for t in range(pid, H * KVBV_KS, P):
            h = t // KVBV_KS
            g = t % KVBV_KS
            am = tl.load(SCF + SC_AM + cc * H + h, mask=cm, other=-1.0e30)
            al = tl.load(SCF + SC_AL + cc * H + h, mask=cm, other=0.0)
            sc = tl.exp(am - tl.max(am))
            pcv = tl.sum(sc[:, None] * tl.load(
                SCB + CB_APC + (cc * H + h)[:, None] * KVL + g * 128 + nl[None, :],
                mask=cm[:, None], other=0.0).to(tl.float32), axis=0) / tl.sum(sc * al)
            tl.store(SCF + SC_XQP + pid * KVL + nl, pcv)
            tl.debug_barrier()
            _quant(SCF, SC_XQP + pid * KVL, SC_QX + pid * QXW, SC_QS + pid * QSW, 1, 1)
            a = _gemv_small(WA, SCF, (A_MLA + W_QKVA + W_KVBN) + t.to(tl.int64) * (16 * NTB),
                            SC_QX + pid * QXW, NTB)
            so = (B_MLA + S_QKVA) + t.to(tl.int64) * NTB
            qs = SC_QS + pid * QSW
            f = tl.load(SA + so + nl).to(tl.float32) * (8.0 * tl.load(SCF + qs))
            z = tl.load(ZA + so + nl).to(tl.float32)
            tl.atomic_add(SCF + SC_O1M + h * VH + nl,
                          f * (a.to(tl.float32) - z * tl.load(SCF + qs + 1)),
                          sem="relaxed")
            tl.debug_barrier()
        # seed the attention residual stream for the o_proj of the next stage
        for c in range(pid, 72, P):
            o = c * 32 + tl.arange(0, 32)
            tl.store(SCF + SC_HA + 3 * D + o,
                     _bf(tl.load(SCF + SC_HM + 2 * D + o, cache_modifier=".cg")))
    nb += P
    _bar(BAR, nb, pid)

    # ---- mla o_proj, accumulated onto the residual ----
    if 35 <= SSTOP:
        _quant(SCF, SC_O1M, SC_QX + pid * QXW, SC_QS + pid * QSW, OP_KS, 16, 1)
        _gemv(WA, SA, ZA, SCF, SCF + SC_EZ, A_MLA + W_QKVA + W_KVBN + W_KVBV,
              B_MLA + S_QKVA + S_KVBV, SC_HA + 3 * D,
              SC_QX + pid * QXW, SC_QS + pid * QSW, 0, 0, 0, 0,
              OP_NTILE, OP_KS, 1, D, 1, 0, 0.0, pid, OP_KM, DNT, P)
    nb += P
    _bar(BAR, nb, pid)

    _moe(WB, WA, SA, ZA, SCF, BAR, 3, nb, pid, 35, SSTOP, P)

    # ---- emit the bf16 next-token hidden (same program that wrote HM) ----
    for c in range(pid, 72, P):
        o = c * 32 + tl.arange(0, 32)
        tl.store(HOUT + o, tl.load(SCF + SC_HM + 3 * D + o,
                                   cache_modifier=".cg").to(tl.bfloat16))


# --------------------------------------------------------------------------- #
# host-side weight repacking
# --------------------------------------------------------------------------- #
NIB = (0, 2, 4, 6, 1, 3, 5, 7)
"""nibble slot of contraction element j inside a packed word.

Element j lands in byte j%4, low nibble for j<4 and high nibble for j>=4, so
`w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F` are each a 4-lane uint8 vector
lined up with 4 *consecutive* int8 activations -- one dp4a each.
"""


def _pack(wq, scales, zeros, K, N, ntw, ntile):
    """(.., K//2, N) uint8 -> task-major int32 arena, plus (s/8) bf16 and z int8."""
    lead = tuple(wq.shape[:-2])
    ng = K // 128
    npad = ntile * ntw
    dev = wq.device
    v = torch.zeros(lead + (K, npad), dtype=torch.int32, device=dev)
    v[..., 0::2, :N] = (wq & 0xF).to(torch.int32)
    v[..., 1::2, :N] = ((wq >> 4) & 0xF).to(torch.int32)
    v = v.view(lead + (ng, 16, 8, ntile, ntw))
    w32 = torch.zeros(lead + (ng, 16, ntile, ntw), dtype=torch.int32, device=dev)
    for j in range(8):
        w32 |= v.select(-3, j) << (4 * NIB[j])
    w32 = w32.movedim(-2, -4).contiguous()                    # [.., ntile, ng, 16, ntw]
    sp = torch.zeros(lead + (ng, npad), dtype=torch.float32, device=dev)
    sp[..., :N] = scales.float() / 8.0
    s8 = sp.view(lead + (ng, ntile, ntw)).movedim(-2, -3).contiguous().to(torch.bfloat16)
    zp = torch.zeros(lead + (ng, npad), dtype=torch.float32, device=dev)
    zp[..., :N] = zeros.float()
    z8 = zp.view(lead + (ng, ntile, ntw)).movedim(-2, -3).contiguous().to(torch.int8)
    return w32.reshape(-1), s8.reshape(-1), z8.reshape(-1)


def _unpack_kvb(wq):
    v = torch.zeros((KVL, H * KVBW), dtype=torch.int32, device=wq.device)
    v[0::2] = (wq & 0xF).to(torch.int32)
    v[1::2] = ((wq >> 4) & 0xF).to(torch.int32)
    return v.view(KVL, H, KVBW)


def _pack_kvbn(wq):
    """kv_b nope half, transposed: [h][g][word][c]; 8 nibbles == 8 consecutive d."""
    vn = _unpack_kvb(wq)[:, :, :QKN].permute(1, 0, 2).contiguous()   # (h, c, d)
    vn = vn.view(H, 4, NTB, 16, 8)
    w32 = torch.zeros((H, 4, NTB, 16), dtype=torch.int32, device=wq.device)
    for j in range(8):
        w32 |= vn[..., j] << (4 * NIB[j])
    return w32.permute(0, 1, 3, 2).contiguous().reshape(-1)


def _pack_kvbv(wq, scales, zeros):
    """kv_b v half, natural orientation: [h][g][word][dv], contraction over c."""
    vv = _unpack_kvb(wq)[:, :, QKN:].reshape(4, 16, 8, H, VH)
    w32 = torch.zeros((4, 16, H, VH), dtype=torch.int32, device=wq.device)
    for j in range(8):
        w32 |= vv[:, :, j] << (4 * NIB[j])
    w32 = w32.permute(2, 0, 1, 3).contiguous()
    s = (scales.float() / 8.0).view(4, H, KVBW)[:, :, QKN:].permute(1, 0, 2).contiguous()
    z = zeros.float().view(4, H, KVBW)[:, :, QKN:].permute(1, 0, 2).contiguous()
    return w32.reshape(-1), s.reshape(-1).to(torch.bfloat16), z.reshape(-1).to(torch.int8)


# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)
        self._ready = False
        self._sstop = 39

    # ------------------------------------------------------------------ #
    def _build(self):
        dev = self.blocks[0].attn_norm.device
        cfg = self.cfg
        wa = torch.zeros(W_TOT, dtype=torch.int32, device=dev)
        sa = torch.zeros(S_TOT, dtype=torch.bfloat16, device=dev)
        za = torch.zeros(S_TOT, dtype=torch.int8, device=dev)
        wb = torch.zeros(WB_TOT, dtype=torch.bfloat16, device=dev)
        scf = torch.zeros(SC_TOT, dtype=torch.float32, device=dev)

        def put(off, soff, packed):
            w32, s8, z8 = packed
            wa[off:off + w32.numel()] = w32
            if s8 is not None:
                sa[soff:soff + s8.numel()] = s8
                za[soff:soff + z8.numel()] = z8

        for b in range(3):
            k = self.blocks[b].attn
            put(A_KDA + b * W_KDA, B_KDA + b * S_KDA,
                _pack(torch.cat([k.q_proj.w_q, k.k_proj.w_q, k.v_proj.w_q, k.g_proj.w_q], 1),
                      torch.cat([k.q_proj.scales, k.k_proj.scales,
                                 k.v_proj.scales, k.g_proj.scales], 1),
                      torch.cat([k.q_proj.zeros, k.k_proj.zeros,
                                 k.v_proj.zeros, k.g_proj.zeros], 1),
                      D, 4 * C4, NT, QKVG_NTILE))
            put(A_KDA + b * W_KDA + W_QKVG, B_KDA + b * S_KDA + S_QKVG,
                _pack(k.o_proj.w_q, k.o_proj.scales, k.o_proj.zeros, C4, D, DNT, OP_NTILE))
            wb[WB_CW + b * 12 * C4: WB_CW + (b + 1) * 12 * C4] = \
                k.conv_w.detach().permute(0, 2, 1).reshape(-1)
            wb[WB_BW + b * H * D: WB_BW + (b + 1) * H * D] = k.beta_proj.weight.detach().reshape(-1)

        m = self.blocks[3].attn
        wq = torch.zeros(D // 2, QKVA_NPAD, dtype=torch.uint8, device=dev)
        sc = torch.zeros(D // 128, QKVA_NPAD, dtype=torch.bfloat16, device=dev)
        ze = torch.zeros(D // 128, QKVA_NPAD, dtype=torch.bfloat16, device=dev)
        wq[:, :H * QH] = m.q_proj.w_q
        sc[:, :H * QH] = m.q_proj.scales
        ze[:, :H * QH] = m.q_proj.zeros
        wq[:, H * QH:H * QH + KVL + QKR] = m.kv_a.w_q
        sc[:, H * QH:H * QH + KVL + QKR] = m.kv_a.scales
        ze[:, H * QH:H * QH + KVL + QKR] = m.kv_a.zeros
        put(A_MLA, B_MLA, _pack(wq, sc, ze, D, QKVA_NPAD, NT, QKVA_NTILE))
        put(A_MLA + W_QKVA, 0, (_pack_kvbn(m.kv_b.w_q), None, None))
        put(A_MLA + W_QKVA + W_KVBN, B_MLA + S_QKVA,
            _pack_kvbv(m.kv_b.w_q, m.kv_b.scales, m.kv_b.zeros))
        put(A_MLA + W_QKVA + W_KVBN + W_KVBV, B_MLA + S_QKVA + S_KVBV,
            _pack(m.o_proj.w_q, m.o_proj.scales, m.o_proj.zeros, C4, D, DNT, OP_NTILE))
        s = m.kv_b.scales.float().view(4, H, KVBW)[:, :, :QKN].permute(1, 0, 2).contiguous()
        z = m.kv_b.zeros.float().view(4, H, KVBW)[:, :, :QKN].permute(1, 0, 2).contiguous()
        scf[SC_SN8:SC_SN8 + H * 4 * QKN] = s.reshape(-1)
        scf[SC_SNZ:SC_SNZ + H * 4 * QKN] = (s * z).reshape(-1)
        qkr = int(QKR)          # torch needs a real int, not a constexpr
        scf[SC_INVF:SC_INVF + qkr // 2] = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, qkr, 2, device=dev, dtype=torch.float32) / qkr))

        for b in range(NBLK):
            mo = self.blocks[b].moe
            for e in range(NEXP + 1):
                if e < NEXP:
                    g, u, dn, i = mo.gate, mo.up, mo.down, e
                else:
                    g, u, dn, i = mo.s_gate, mo.s_up, mo.s_down, 0
                off = A_MOE + b * W_MOE + e * W_EXP
                soff = B_MOE + b * S_MOE + e * S_EXP
                put(off, soff, _pack(torch.cat([g.w_q[i], u.w_q[i]], 1),
                                     torch.cat([g.scales[i], u.scales[i]], 1),
                                     torch.cat([g.zeros[i], u.zeros[i]], 1),
                                     D, 2 * MI, NT, GU_NTILE))
                put(off + W_GU, soff + S_GU,
                    _pack(dn.w_q[i], dn.scales[i], dn.zeros[i], MI, D, DNT, DN_NTILE))
            wb[WB_RW + b * NEXP * D: WB_RW + (b + 1) * NEXP * D] = \
                mo.router.weight.detach().reshape(-1)
            wb[WB_NRM + b * 2 * D: WB_NRM + b * 2 * D + D] = self.blocks[b].attn_norm.detach()
            wb[WB_NRM + b * 2 * D + D: WB_NRM + (b + 1) * 2 * D] = self.blocks[b].moe_norm.detach()

        self._wa, self._sa, self._za, self._wb, self._scf = wa, sa, za, wb, scf
        self._scb = torch.zeros(CB_TOT, dtype=torch.bfloat16, device=dev)
        self._bar = torch.zeros(8 * 64, dtype=torch.int32, device=dev)
        self._barbase = 0
        self._hout = torch.zeros(D, dtype=torch.bfloat16, device=dev)
        self._sbuf = torch.zeros(3, H, DK, DK, dtype=torch.float32, device=dev)
        self._cv = torch.zeros(2, 9, 3, C4, dtype=torch.bfloat16, device=dev)
        self._par = 0
        self._ckv = None
        self._krp = None
        self._cap = 0
        self._ready = True

    def _cache(self, ckv):
        n = ckv.shape[0]
        if self._ckv is not None and ckv.data_ptr() == self._ckv.data_ptr() and n < self._cap:
            return 0
        self._cap = n + 512
        self._ckv = torch.empty(self._cap, KVL, dtype=torch.bfloat16, device=ckv.device)
        self._krp = torch.empty(self._cap, QKR, dtype=torch.bfloat16, device=ckv.device)
        return 1

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, hidden, state):
        if not self._ready:
            self._build()
        ki = [i for i, k in enumerate(self.cfg.pattern) if k == "K"]
        mi = self.cfg.pattern.index("M")
        docopy = self._cache(state[mi]["c_kv"])
        if state[ki[0]]["S"].data_ptr() != self._sbuf.data_ptr():
            docopy = 1
        L = int(state[mi]["c_kv"].shape[0])
        cs = [state[i][n] for i in ki for n in ("cq", "ck", "cv")]
        _mega[(P,)](
            hidden, self._hout, self._wa, self._sa, self._za, self._wb, self._scf,
            self._scb, self._bar, self._barbase,
            self._sbuf, self._cv[self._par], self._cv[1 - self._par], self._ckv, self._krp,
            state[ki[0]]["S"], state[ki[1]]["S"], state[ki[2]]["S"],
            cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8],
            state[mi]["c_kv"], state[mi]["k_rope"],
            L, docopy, self._sstop, P, num_warps=NWARP,
        )
        self._barbase += NBAR * P
        out = self._cv[1 - self._par]
        self._par = 1 - self._par
        for j, i in enumerate(ki):
            state[i]["S"] = self._sbuf[j]
            state[i]["cq"] = out[j * 3 + 0]
            state[i]["ck"] = out[j * 3 + 1]
            state[i]["cv"] = out[j * 3 + 2]
        state[mi]["c_kv"] = self._ckv[:L + 1]
        state[mi]["k_rope"] = self._krp[:L + 1]
        return self._hout, state
