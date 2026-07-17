"""Fused CUDA grid-foraging + 3xMinGRU(h=256) SPS solution (v2, cp.async pipeline).

Strategy
--------
- policy_forward / env_step: exact PyTorch mirrors of the reference semantics
  (bit-identical ops, used by check.py comparisons).
- run(): one fused CUDA kernel per env-step (obs -> enc -> 3 MinGRU layers ->
  logits -> argmax -> env move -> reward/respawn bookkeeping), launched back to
  back from a single C++ host call over the whole horizon.

Kernel organization (per step, per block of 256 threads):
  * block owns E envs (E in {16,32,64} chosen by num_envs; KC in {8,16,32})
  * gate weights pre-transposed (host, once per run) to k-major so slices
    stream into smem as 16B cp.async copies, double-buffered H ring + rolling
    3-buffer W ring; one commit group of {H, W_zh, W_zg, W_zp} per K-chunk,
    wait_prior(1) + a single barrier per chunk.
  * thread accumulates all 48 gate values (3 groups x 4 envs x 4 j) for its
    tile, then the MinGRU nonlinearity is applied in registers (no barrier).
  * activations are stored transposed [k][e] so H global traffic is v4-clean.
  * rng/food respawn uses the reference LCG; the "advance rng iff any env hit"
    rule rides a per-step global flag slot (kernel boundary = grid barrier)
    plus a per-env hit byte consumed by the next step's prologue.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

OP_TYPE = "grid_mingru_sps"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

BOARD = 11
OBS_DIM = 4
HIDDEN = 256
GRU_LAYERS = 3
NUM_ACTIONS = 4
GRU_OUT = 3 * HIDDEN

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

#define DEVI __device__ __forceinline__

constexpr unsigned long long LCG_A    = 6364136223846793005ULL;
constexpr unsigned long long LCG_MASK = 0x7fffffffffffffffULL;

DEVI float sigmoid_(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
DEVI float tanh_(float x)    { float e = __expf(-2.0f * x); return __fdividef(1.0f - e, 1.0f + e); }

DEVI void cp16(void* smem, const void* gmem) {
    __pipeline_memcpy_async(smem, gmem, 16);
}
DEVI void cpz16(void* smem) {
    // pipeline-friendly zero fill (no traffic): plain v4 store is fine
    *reinterpret_cast<float4*>(smem) = make_float4(0.f, 0.f, 0.f, 0.f);
}

// ---------------------------------------------------------------------------
// One fused env+policy step. Block handles E envs with 256 threads.
// h is stored transposed: hX[k][e], k in [0,256), e in [0,n).
// wGruT[l][k][row], row in [0,768), k-major.
// ---------------------------------------------------------------------------
template <int E, int KC>
__global__ void __launch_bounds__(256, 2) step_kernel(
    const float* __restrict__ wEnc,   // (256,4)
    const float* __restrict__ bEnc,   // (256,)
    const float* __restrict__ wGruT,  // (3,256,768)
    const float* __restrict__ wA,     // (4,256)
    const float* __restrict__ bA,     // (4,)
    float* __restrict__ hA,           // (256,n) ping
    float* __restrict__ hB,           // (256,n) pong
    float* __restrict__ state,        // (n,3,256) in-place
    float* __restrict__ agent,        // (n,2)
    float* __restrict__ food,         // (n,2)
    long long* __restrict__ rng,      // (n,)
    float* __restrict__ rewards,      // (n,)
    unsigned char* __restrict__ hitf, // (n,)
    float* __restrict__ logitsOut,    // (n,4)
    const int* __restrict__ flagPrev,
    int* __restrict__ flagNext,
    int n)
{
    constexpr int EQ    = E / 4;          // env quads
    constexpr int NC    = 4096 / E;       // rows per jb chunk (16/32/64 -> 256/128/64)
    constexpr int NJB   = 256 / NC;       // jb chunks per gate group
    constexpr int HPAD  = E + 4;
    constexpr int WPAD  = NC + 4;
    constexpr int NK    = 256 / KC;       // k chunks
    constexpr int HSL   = KC * HPAD;      // H slice floats
    constexpr int WSL   = KC * WPAD;      // W slice floats

    extern __shared__ float smem[];
    float* obs_s  = smem;                  // [4*E]  (view as [E][4])
    float* logit_s= obs_s + E * 4;         // [4*E]  reused at end for [4][E] logits
    float* Hb[2]; float* Wb[2];            // parity slabs; Wb slab holds 3 gate slices
    float* p = obs_s + E * 4 + 4 * E;
    Hb[0] = p;              p += HSL;
    Hb[1] = p;              p += HSL;
    Wb[0] = p;              p += 3 * WSL;
    Wb[1] = p;

    const int tid = threadIdx.x;
    const int e0 = blockIdx.x * E;
    const bool fullTile = (e0 + E <= n);

    // ---------------- prologue: rng/food respawn + obs ----------------------
    if (tid < E) {
        int e = e0 + tid;
        if (e < n) {
            float2 ag = reinterpret_cast<const float2*>(agent)[e];
            float2 fd = reinterpret_cast<const float2*>(food)[e];
            int pend = *flagPrev;
            if (pend) {
                unsigned long long r = (unsigned long long)rng[e];
                r = (r * LCG_A + 1ULL) & LCG_MASK;
                float fx = (float)(long long)(r % 11ULL);
                r = (r * LCG_A + 1ULL) & LCG_MASK;
                float fy = (float)(long long)(r % 11ULL);
                rng[e] = (long long)r;
                if (hitf[e]) {
                    fd.x = fx; fd.y = fy;
                    reinterpret_cast<float2*>(food)[e] = fd;
                }
            }
            hitf[e] = 0;
            obs_s[tid * 4 + 0] = (fd.x - ag.x) / 11.0f;
            obs_s[tid * 4 + 1] = (fd.y - ag.y) / 11.0f;
            obs_s[tid * 4 + 2] = ag.x / 10.0f;
            obs_s[tid * 4 + 3] = ag.y / 10.0f;
        }
    }
    __syncthreads();

    // ---------------- encoder: hA[j][e] = obs(e) . wEnc[j] + bEnc[j] --------
    // hA is hT layout: hA[j * n + e].
    {
        // thread handles (env = v/EJ?, jquad) with v over E*(256/4)
        for (int v = tid; v < E * 64; v += 256) {
            int ee = v >> 6;               // env local
            int j4 = (v & 63) << 2;        // j quad
            int e = e0 + ee;
            if (e < n) {
                float4 o = *reinterpret_cast<const float4*>(&obs_s[ee * 4]);
                const float* w0 = &wEnc[(j4 + 0) * 4];
                const float* w1 = &wEnc[(j4 + 1) * 4];
                const float* w2 = &wEnc[(j4 + 2) * 4];
                const float* w3 = &wEnc[(j4 + 3) * 4];
                float x0 = bEnc[j4+0] + o.x*w0[0] + o.y*w0[1] + o.z*w0[2] + o.w*w0[3];
                float x1 = bEnc[j4+1] + o.x*w1[0] + o.y*w1[1] + o.z*w1[2] + o.w*w1[3];
                float x2 = bEnc[j4+2] + o.x*w2[0] + o.y*w2[1] + o.z*w2[2] + o.w*w2[3];
                float x3 = bEnc[j4+3] + o.x*w3[0] + o.y*w3[1] + o.z*w3[2] + o.w*w3[3];
                hA[(long)(j4+0) * n + e] = x0;
                hA[(long)(j4+1) * n + e] = x1;
                hA[(long)(j4+2) * n + e] = x2;
                hA[(long)(j4+3) * n + e] = x3;
            }
        }
    }
    __syncthreads();

    // ---------------- 3 MinGRU layers ---------------------------------------
    const float* hIn  = hA;   // [k][e]
    float*       hOut = hB;
    const int eq = tid % EQ;            // env quad index (4 consecutive envs)
    const int rg = tid / EQ;            // row quad index within chunk

    // stage helpers --------------------------------------------------------
    auto stageH = [&](int kt, const float* src /* hT base */, int sw /*parity*/)
    {
        // H slice for k-chunk kt of hIn: rows k = kt*KC + [0,KC), envs e0..e0+E
        // dst layout: [kk][e], kk in [0,KC)
        float* dst = Hb[sw];
        int total4 = KC * (E / 4);
        if (fullTile) {
            for (int idx = tid; idx < total4; idx += 256) {
                int kk = idx / (E / 4);
                int qq = idx % (E / 4);
                const float* srcp = src + (long)(kt * KC + kk) * n + e0 + qq * 4;
                cp16(&dst[kk * HPAD + qq * 4], srcp);
            }
        } else {
            for (int idx = tid; idx < total4; idx += 256) {
                int kk = idx / (E / 4);
                int qq = idx % (E / 4);
                int e = e0 + qq * 4;
                if (e + 3 < n) {
                    const float* srcp = src + (long)(kt * KC + kk) * n + e;
                    cp16(&dst[kk * HPAD + qq * 4], srcp);
                } else {
                    float t0=0,t1=0,t2=0,t3=0;
                    if (e + 0 < n) t0 = src[(long)(kt * KC + kk) * n + e + 0];
                    if (e + 1 < n) t1 = src[(long)(kt * KC + kk) * n + e + 1];
                    if (e + 2 < n) t2 = src[(long)(kt * KC + kk) * n + e + 2];
                    if (e + 3 < n) t3 = src[(long)(kt * KC + kk) * n + e + 3];
                    *reinterpret_cast<float4*>(&dst[kk * HPAD + qq * 4]) = make_float4(t0,t1,t2,t3);
                }
            }
        }
    };
    auto stageW = [&](int l, int g, int jb, int kt, int sw)
    {
        // W slice rows [g*256 + jb*NC, +NC), k = kt*KC + [0,KC)
        // src: wGruT[l][k][row]; dst: slab slot g, [kk][row]
        float* dst = Wb[sw] + g * WSL;
        const float* srcBase = wGruT + ((long)l * 256 + kt * KC) * 768 + (g * 256 + jb * NC);
        int total4 = KC * (NC / 4);
        for (int idx = tid; idx < total4; idx += 256) {
            int kk = idx / (NC / 4);
            int qq = idx % (NC / 4);
            cp16(&dst[kk * WPAD + qq * 4], srcBase + (long)kk * 768 + qq * 4);
        }
    };
    // commit a k-chunk group: {H(kt), W(0..2, kt)} into parity slab kc&1
    auto commitK = [&](int l, int jb, int kt)
    {
        int kc = jb * NK + kt;             // global k-chunk counter (per layer)
        int sw = kc & 1;
        stageH(kt, hIn, sw);
        stageW(l, 0, jb, kt, sw);
        stageW(l, 1, jb, kt, sw);
        stageW(l, 2, jb, kt, sw);
        __pipeline_commit();
    };

    for (int l = 0; l < 3; ++l) {
        // reset pipeline at layer start (hIn just produced by eltwise / enc)
        commitK(l, 0, 0);
        for (int jb = 0; jb < NJB; ++jb) {
            float acc[48];
            #pragma unroll
            for (int i = 0; i < 48; ++i) acc[i] = 0.0f;

            for (int kt = 0; kt < NK; ++kt) {
                int kc = jb * NK + kt;
                // wait for this chunk's copies (issued one iteration earlier),
                // barrier, THEN prefetch the next chunk (so no warp can be
                // writing buffers another warp still reads).
                __pipeline_wait_prior(0);
                __syncthreads();
                {
                    int jbn = jb, ktn = kt + 1;
                    if (ktn >= NK) { jbn = jb + 1; ktn = 0; }
                    if (jbn < NJB) commitK(l, jbn, ktn);
                    // (no else: an empty group is not needed with wait_prior(0))
                }
                const float* Hs = Hb[kc & 1];
                const float* Ws0 = Wb[kc & 1];
                #pragma unroll
                for (int g = 0; g < 3; ++g) {
                    const float* Ws = Ws0 + g * WSL;
                    #pragma unroll
                    for (int kk = 0; kk < KC; ++kk) {
                        float4 av = *reinterpret_cast<const float4*>(&Hs[kk * HPAD + eq * 4]);
                        float4 bv = *reinterpret_cast<const float4*>(&Ws[kk * WPAD + rg * 4]);
                        acc[g*16+ 0] += av.x * bv.x; acc[g*16+ 1] += av.x * bv.y; acc[g*16+ 2] += av.x * bv.z; acc[g*16+ 3] += av.x * bv.w;
                        acc[g*16+ 4] += av.y * bv.x; acc[g*16+ 5] += av.y * bv.y; acc[g*16+ 6] += av.y * bv.z; acc[g*16+ 7] += av.y * bv.w;
                        acc[g*16+ 8] += av.z * bv.x; acc[g*16+ 9] += av.z * bv.y; acc[g*16+10] += av.z * bv.z; acc[g*16+11] += av.z * bv.w;
                        acc[g*16+12] += av.w * bv.x; acc[g*16+13] += av.w * bv.y; acc[g*16+14] += av.w * bv.z; acc[g*16+15] += av.w * bv.w;
                    }
                }
            }

            // ---- MinGRU merge for this chunk (registers only) --------------
            int j0 = jb * NC + rg * 4;
            #pragma unroll
            for (int ee = 0; ee < 4; ++ee) {
                int e = e0 + eq * 4 + ee;
                if (e >= n) continue;
                float4 st  = *reinterpret_cast<const float4*>(&state[((long)e * 3 + l) * 256 + j0]);
                float stx[4] = {st.x, st.y, st.z, st.w};
                float outv[4], hnv[4];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    float hold  = hIn[(long)(j0 + jj) * n + e];
                    float sg    = sigmoid_(acc[16 + ee * 4 + jj]);
                    float th    = tanh_(acc[0 + ee * 4 + jj]);
                    float sp    = sigmoid_(acc[32 + ee * 4 + jj]);
                    float o     = stx[jj] + sg * (th - stx[jj]);
                    outv[jj]    = o;
                    hnv[jj]     = fmaf(sp, o, (1.0f - sp) * hold);
                }
                *reinterpret_cast<float4*>(&state[((long)e * 3 + l) * 256 + j0]) =
                    make_float4(outv[0], outv[1], outv[2], outv[3]);
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    hOut[(long)(j0 + jj) * n + e] = hnv[jj];
            }
        }
        const float* tmp = hIn; hIn = hOut; hOut = const_cast<float*>(tmp);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ---------------- heads: logits[e][a] = h[e].wA[a] + bA[a] --------------
    // thread = (ee = tid>>2, a = tid&3); hIn is [k][e]
    {
        int a  = tid & 3;
        int ee = tid >> 2;               // 0..63
        int e  = e0 + ee;
        if (a < 4 && ee < E && e < n) {
            float acc0 = 0.f;
            const float* wa = wA + a * 256;
            #pragma unroll 8
            for (int k = 0; k < 256; ++k) acc0 += hIn[(long)k * n + e] * wa[k];
            logit_s[a * E + ee] = acc0 + bA[a];
        }
        __syncthreads();
    }

    // ---------------- argmax + env step -------------------------------------
    if (tid < E) {
        int e = e0 + tid;
        if (e < n) {
            float l0 = logit_s[0 * E + tid], l1 = logit_s[1 * E + tid];
            float l2 = logit_s[2 * E + tid], l3 = logit_s[3 * E + tid];
            *reinterpret_cast<float4*>(&logitsOut[(long)e * 4]) = make_float4(l0, l1, l2, l3);
            int act = 0; float best = l0;
            if (l1 > best) { best = l1; act = 1; }
            if (l2 > best) { best = l2; act = 2; }
            if (l3 > best) { best = l3; act = 3; }
            float2 ag = reinterpret_cast<const float2*>(agent)[e];
            float2 fd = reinterpret_cast<const float2*>(food)[e];
            float nx = ag.x + ((act == 3) ? 1.0f : ((act == 2) ? -1.0f : 0.0f));
            float ny = ag.y + ((act == 1) ? 1.0f : ((act == 0) ? -1.0f : 0.0f));
            nx = fminf(fmaxf(nx, 0.0f), 10.0f);
            ny = fminf(fmaxf(ny, 0.0f), 10.0f);
            reinterpret_cast<float2*>(agent)[e] = make_float2(nx, ny);
            bool hit = (nx == fd.x) && (ny == fd.y);
            if (hit) {
                rewards[e] += 1.0f;
                hitf[e] = 1;
                atomicMax(flagNext, 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
static int pick_E(long n) {
    if (n >= 20480) return 64;  // >= 320 blocks
    if (n >= 5000)  return 32;  // >= 157 blocks
    return 16;
}

struct Args {
    const float *wEnc, *bEnc, *wGruT, *wA, *bA;
    float *hA, *hB, *state, *agent, *food;
    long long* rng;
    float* rewards;
    unsigned char* hitf;
    float* logits;
    const int* flags;
};

template <int E, int KC>
static void launch_all(dim3 grid, int smemBytes, cudaStream_t stream,
                       const Args& a, int horizon, int n) {
    for (int t = 0; t < horizon; ++t) {
        step_kernel<E, KC><<<grid, 256, smemBytes, stream>>>(
            a.wEnc, a.bEnc, a.wGruT, a.wA, a.bA, a.hA, a.hB, a.state, a.agent,
            a.food, a.rng, a.rewards, a.hitf, a.logits,
            a.flags + t, const_cast<int*>(a.flags) + t + 1, n);
    }
}

void rollout(
    torch::Tensor wEnc, torch::Tensor bEnc, torch::Tensor wGruT,
    torch::Tensor wA, torch::Tensor bA,
    torch::Tensor hA, torch::Tensor hB, torch::Tensor state,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng,
    torch::Tensor rewards, torch::Tensor hitf, torch::Tensor logits,
    torch::Tensor flags, long horizon, long n)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    Args a;
    a.wEnc = wEnc.data_ptr<float>();
    a.bEnc = bEnc.data_ptr<float>();
    a.wGruT = wGruT.data_ptr<float>();
    a.wA = wA.data_ptr<float>();
    a.bA = bA.data_ptr<float>();
    a.hA = hA.data_ptr<float>();
    a.hB = hB.data_ptr<float>();
    a.state = state.data_ptr<float>();
    a.agent = agent.data_ptr<float>();
    a.food = food.data_ptr<float>();
    a.rng = (long long*)rng.data_ptr<long>();
    a.rewards = rewards.data_ptr<float>();
    a.hitf = (unsigned char*)hitf.data_ptr<uint8_t>();
    a.logits = logits.data_ptr<float>();
    a.flags = flags.data_ptr<int>();

    int E = pick_E(n);
    dim3 grid((unsigned)((n + E - 1) / E));
    int NCv = 4096 / E;
    int KCv = (E == 64) ? 16 : (E == 32 ? 8 : 4);
    int smemBytes = 4 * E * 4 * 2
                  + 2 * KCv * (E + 4) * 4
                  + 2 * 3 * KCv * (NCv + 4) * 4;
    switch (E) {
        case 64: launch_all<64, 16>(grid, smemBytes, stream, a, (int)horizon, (int)n); break;
        case 32: launch_all<32, 8>(grid, smemBytes, stream, a, (int)horizon, (int)n); break;
        default: launch_all<16, 4>(grid, smemBytes, stream, a, (int)horizon, (int)n); break;
    }
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void rollout(
    torch::Tensor wEnc, torch::Tensor bEnc, torch::Tensor wGruT,
    torch::Tensor wA, torch::Tensor bA,
    torch::Tensor hA, torch::Tensor hB, torch::Tensor state,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng,
    torch::Tensor rewards, torch::Tensor hitf, torch::Tensor logits,
    torch::Tensor flags, long horizon, long n);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="grid_mingru_sps_v2",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["rollout"],
            extra_cuda_cflags=[
                "-O3",
                "--generate-code=arch=compute_120,code=sm_120",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _ext


def _mingru_g(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_enc = nn.Parameter(torch.empty(HIDDEN, OBS_DIM))
        self.b_enc = nn.Parameter(torch.zeros(HIDDEN))
        self.w_gru = nn.Parameter(torch.empty(GRU_LAYERS, GRU_OUT, HIDDEN))
        self.w_a = nn.Parameter(torch.empty(NUM_ACTIONS, HIDDEN))
        self.b_a = nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.w_v = nn.Parameter(torch.empty(1, HIDDEN))
        self.b_v = nn.Parameter(torch.zeros(1))
        self.reset_parameters(0)

    def reset_parameters(self, seed: int = 0) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        for p in self.parameters():
            tmp = torch.empty(p.shape, dtype=p.dtype, device="cpu")
            tmp.normal_(0.0, 0.02, generator=g)
            p.data.copy_(tmp)

    def forward(self, obs: torch.Tensor, state: torch.Tensor):
        return policy_forward(self, obs, state)


def policy_forward(model: Model, obs: torch.Tensor, state: torch.Tensor):
    """obs (N,4), state (N,L,H) -> logits (N,4), new_state (N,L,H), value (N,).

    Exact mirror of the reference math (used for correctness comparison).
    """
    h = F.linear(obs, model.w_enc, model.b_enc)
    new_states = []
    for layer in range(GRU_LAYERS):
        st = state[:, layer, :]
        gates = F.linear(h, model.w_gru[layer])
        zh, zg, zp = gates.split(HIDDEN, dim=-1)
        out = st + torch.sigmoid(zg) * (_mingru_g(zh) - st)
        p = torch.sigmoid(zp)
        h = p * out + (1.0 - p) * h
        new_states.append(out)
    new_state = torch.stack(new_states, dim=1)
    logits = F.linear(h, model.w_a, model.b_a)
    value = F.linear(h, model.w_v, model.b_v).squeeze(-1)
    return logits, new_state, value


def env_step(
    agent: torch.Tensor,
    food: torch.Tensor,
    actions: torch.Tensor,
    rng_state: torch.Tensor,
):
    """Exact mirror of the reference env step (deterministic LCG respawn)."""

    def _lcg_step(rng: torch.Tensor) -> torch.Tensor:
        return (rng * 6364136223846793005 + 1) & 0x7FFFFFFFFFFFFFFF

    delta = torch.zeros_like(agent)
    delta[:, 1] = torch.where(actions == 0, -torch.ones_like(delta[:, 1]), delta[:, 1])
    delta[:, 1] = torch.where(actions == 1, torch.ones_like(delta[:, 1]), delta[:, 1])
    delta[:, 0] = torch.where(actions == 2, -torch.ones_like(delta[:, 0]), delta[:, 0])
    delta[:, 0] = torch.where(actions == 3, torch.ones_like(delta[:, 0]), delta[:, 0])
    agent = (agent + delta).clamp(0, BOARD - 1)
    hit = (agent == food).all(dim=-1)
    reward = hit.float()
    rng_state = rng_state.clone()
    if hit.any():
        rng_state = _lcg_step(rng_state)
        fx = (rng_state % BOARD).to(agent.dtype)
        rng_state = _lcg_step(rng_state)
        fy = (rng_state % BOARD).to(agent.dtype)
        new_food = torch.stack([fx, fy], dim=-1)
        food = food.clone()
        food[hit] = new_food[hit]
    return agent, food, reward, rng_state


def _run_torch(model: Model, num_envs: int, horizon: int, seed: int) -> dict:
    """Reference-mirror fallback (CPU or non-multiple-of-4 num_envs)."""
    device = next(model.parameters()).device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    agent = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    food = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    rng_state = torch.arange(num_envs, device=device, dtype=torch.int64) + (seed * 10007)
    state = torch.zeros(num_envs, GRU_LAYERS, HIDDEN, device=device)
    rewards = torch.zeros(num_envs, device=device)
    last_logits = torch.zeros(num_envs, NUM_ACTIONS, device=device)

    with torch.no_grad():
        for _t in range(horizon):
            dx = (food[:, 0] - agent[:, 0]) / BOARD
            dy = (food[:, 1] - agent[:, 1]) / BOARD
            obs = torch.stack([dx, dy, agent[:, 0] / (BOARD - 1), agent[:, 1] / (BOARD - 1)], dim=-1)
            logits, state, _v = policy_forward(model, obs, state)
            last_logits = logits
            actions = torch.argmax(logits, dim=-1)
            agent, food, r, rng_state = env_step(agent, food, actions, rng_state)
            rewards = rewards + r

    return {
        "rewards": rewards.detach(),
        "positions": agent.detach().round().long(),
        "last_logits": last_logits.detach(),
        "state": state.detach(),
    }


def run(num_envs: int, horizon: int, seed: int, model: Model | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Model()
    model = model.to(device).eval()
    n = int(num_envs)
    h = int(horizon)
    if device.type != "cuda" or (n % 4) != 0:
        return _run_torch(model, n, h, seed)

    ext = _get_ext()

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    agent = torch.randint(0, BOARD, (n, 2), generator=g).float().to(device)
    food = torch.randint(0, BOARD, (n, 2), generator=g).float().to(device)
    rng_state = torch.arange(n, device=device, dtype=torch.int64) + (seed * 10007)
    state = torch.zeros(n, GRU_LAYERS, HIDDEN, device=device)
    rewards = torch.zeros(n, device=device)
    logits = torch.empty(n, NUM_ACTIONS, device=device)
    hA = torch.empty(HIDDEN, n, device=device)   # transposed [k][e]
    hB = torch.empty(HIDDEN, n, device=device)
    hitf = torch.zeros(n, dtype=torch.uint8, device=device)
    flags = torch.zeros(h + 2, dtype=torch.int32, device=device)

    # k-major transpose of gate weights for smem-friendly streaming
    wGruT = model.w_gru.detach().transpose(1, 2).contiguous()

    ext.rollout(
        model.w_enc.detach().contiguous(),
        model.b_enc.detach().contiguous(),
        wGruT,
        model.w_a.detach().contiguous(),
        model.b_a.detach().contiguous(),
        hA, hB, state, agent, food, rng_state, rewards, hitf, logits, flags,
        h, n,
    )

    return {
        "rewards": rewards.detach(),
        "positions": agent.detach().round().long(),
        "last_logits": logits.detach(),
        "state": state.detach(),
    }


def get_init_inputs():
    return []


def get_inputs():
    return []
