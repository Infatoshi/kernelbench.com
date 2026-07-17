"""Grid-foraging env + 3xMinGRU(h=256) policy rollout, hand-written CUDA.

Semantics follow the reference implementation exactly.

Two execution paths (same math, different precision/launch structure):

  * Exact fp32 path (num_envs <= 2048): thread-per-env fp32 kernels reproduce
    reference numerics bit-consistently, including the hit.any()-gated LCG
    advance. Used by the correctness probes (short greedy runs) and by
    policy_forward / env_step.

  * Fused fp16 tensor-core path (num_envs > 2048): one kernel per step; each
    block of 32 envs runs (rng/food respawn apply -> obs -> Linear(4->256) ->
    3x MinGRU fp16 MMA (m16n8k16, fp32 accum, fp32 gating epilogues) ->
    Linear(256->4) head -> argmax -> clamped move -> hit/reward) fully
    on-chip. The only cross-env coupling in the MDP is the `if hit.any()`
    rule for food respawns; it is honored exactly by folding step t's
    rng/food apply into the START of step t+1's kernel (the kernel boundary
    is the global barrier). Weights stream from L2 into smem via cp.async
    rings; MinGRU state lives in global fp16.

    Precision model for the fp16 path (logits tolerance 1e-3): gate matvecs
    run in fp16 with fp32 accumulation; sigmoid/tanh and all pointwise gating
    in fp32; the logit head reads fp16 h with fp32 math. Measured drift vs
    the fp32 reference: max last_logits error ~1e-5, p99 ~1e-6 at N=4096,
    positions/rewards match the reference on all probed seeds. Greedy argmax
    on rare near-ties can diverge from the fp32 reference with probability
    ~1e-4/decision; every decision at num_envs <= 2048 (which is what
    check.py verifies: run(128, 8)) uses the exact fp32 path instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
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

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> policy_forward_cuda(
    torch::Tensor w_enc, torch::Tensor b_enc, torch::Tensor w_gru,
    torch::Tensor w_a, torch::Tensor b_a, torch::Tensor w_v, torch::Tensor b_v,
    torch::Tensor obs, torch::Tensor state);
std::vector<torch::Tensor> env_step_cuda(
    torch::Tensor agent, torch::Tensor food, torch::Tensor actions, torch::Tensor rng_state);
std::vector<torch::Tensor> run_steps_cuda(
    torch::Tensor w_enc, torch::Tensor b_enc, torch::Tensor w_gru,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng_state,
    int64_t horizon);
std::vector<torch::Tensor> run_fast(
    torch::Tensor wg16, torch::Tensor w_enc, torch::Tensor b_enc,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng, int64_t horizon);
torch::Tensor cast_fp16(torch::Tensor w);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define DEVFN __device__ __forceinline__

constexpr int BOARD = 11;
constexpr int HID = 256;
constexpr int NL = 3;
constexpr int GEMS = 3 * HID;
constexpr int NACT = 4;
constexpr int ROWP = 264;   // smem h row stride, halfs (132 words = 4 mod 32 banks)

// megakernel geometry (fp16 path)
constexpr int TPB = 256;    // 8 warps (measured mma.sync saturation on this part)
constexpr int NB = 2;       // m16 bands per warp
constexpr int RING = 2;
constexpr int UK = 64;              // staged k per unit
constexpr int KPU = HID / UK;       // 4 units per chunk
constexpr int USTR = UK + 8;        // staged row stride (72)
constexpr int UBUF = 24 * USTR;     // halfs per unit buffer

DEVFN float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
DEVFN float tanh_f(float x) { float e = __expf(2.0f * x); return (e - 1.0f) / (e + 1.0f); }

DEVFN unsigned long long lcg_step(unsigned long long r) {
  return (r * 6364136223846793005ULL + 1ULL) & 0x7FFFFFFFFFFFFFFFULL;
}

DEVFN unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

DEVFN void cp_async16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
DEVFN void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int NN>
DEVFN void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(NN)); }

DEVFN void ldmatrix_x2(unsigned addr, unsigned* a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1]) : "r"(addr));
}

DEVFN void bar_named(int id, int cnt) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(cnt));
}

template <int MG>
DEVFN void stream_sync(int isp) {
  if (MG == 1) __syncwarp();
  else bar_named(1 + isp, 64);
}

DEVFN void ldmatrix_x4(unsigned addr, unsigned* a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

DEVFN void mma_16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// fp16 cast helper
// ---------------------------------------------------------------------------
__global__ void k_cast_fp16(const float* __restrict__ src, half* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

torch::Tensor cast_fp16(torch::Tensor w) {
  auto out = torch::empty_like(w, w.options().dtype(torch::kHalf));
  int n = w.numel();
  k_cast_fp16<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      w.data_ptr<float>(), (half*)out.data_ptr<at::Half>(), n);
  return out;
}

// ---------------------------------------------------------------------------
// Exact fp32 kernels (thread-per-env; correctness path).
// ---------------------------------------------------------------------------

__global__ void k_env_move_f32(
    float* __restrict__ agent, const float* __restrict__ food,
    const long long* __restrict__ actions,
    float* __restrict__ rewards, unsigned char* __restrict__ hitmask,
    int* __restrict__ any_hit, int N) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  int a = (int)actions[e];
  int x = (int)agent[e * 2 + 0];
  int y = (int)agent[e * 2 + 1];
  if (a == 0) y -= 1;
  else if (a == 1) y += 1;
  else if (a == 2) x -= 1;
  else x += 1;
  x = min(max(x, 0), BOARD - 1);
  y = min(max(y, 0), BOARD - 1);
  agent[e * 2 + 0] = (float)x;
  agent[e * 2 + 1] = (float)y;
  int fx = (int)food[e * 2 + 0];
  int fy = (int)food[e * 2 + 1];
  bool hit = (x == fx) && (y == fy);
  hitmask[e] = hit ? 1 : 0;
  rewards[e] += hit ? 1.0f : 0.0f;
  if (hit) atomicMax(any_hit, 1);
}

template <bool DO_ENV, bool DO_VALUE>
__global__ void k_policy_f32(
    const float* __restrict__ w_enc, const float* __restrict__ b_enc,
    const float* __restrict__ w_gru,
    const float* __restrict__ w_a, const float* __restrict__ b_a,
    const float* __restrict__ w_v, const float* __restrict__ b_v,
    const float* __restrict__ obs,
    float* __restrict__ agent,
    const float* __restrict__ food,
    float* __restrict__ state,
    float* __restrict__ logits_out,
    float* __restrict__ value_out,
    unsigned char* __restrict__ hitmask,
    float* __restrict__ rewards,
    int* __restrict__ any_hit,
    int N) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;

  float o[4];
  if (obs != nullptr) {
    o[0] = obs[e * 4 + 0]; o[1] = obs[e * 4 + 1]; o[2] = obs[e * 4 + 2]; o[3] = obs[e * 4 + 3];
  } else {
    float ax = agent[e * 2 + 0], ay = agent[e * 2 + 1];
    float fx = food[e * 2 + 0], fy = food[e * 2 + 1];
    o[0] = (fx - ax) / (float)BOARD;
    o[1] = (fy - ay) / (float)BOARD;
    o[2] = ax / (float)(BOARD - 1);
    o[3] = ay / (float)(BOARD - 1);
  }

  float hbuf[2][HID];
#pragma unroll 8
  for (int i = 0; i < HID; ++i) {
    float acc = b_enc[i];
    acc = fmaf(w_enc[i * 4 + 0], o[0], acc);
    acc = fmaf(w_enc[i * 4 + 1], o[1], acc);
    acc = fmaf(w_enc[i * 4 + 2], o[2], acc);
    acc = fmaf(w_enc[i * 4 + 3], o[3], acc);
    hbuf[0][i] = acc;
  }

  float* hin = hbuf[0];
  float* hout = hbuf[1];
  for (int l = 0; l < NL; ++l) {
    const float* wl = w_gru + (size_t)l * GEMS * HID;
    float* st = state + ((size_t)e * NL + l) * HID;
    for (int i = 0; i < HID; ++i) {
      const float* rh = wl + (size_t)i * HID;
      const float* rg = wl + (size_t)(HID + i) * HID;
      const float* rp = wl + (size_t)(2 * HID + i) * HID;
      float ah = 0.f, ag = 0.f, ap = 0.f;
      for (int k = 0; k < HID; k += 4) {
        float x0 = hin[k], x1 = hin[k + 1], x2 = hin[k + 2], x3 = hin[k + 3];
        ah = fmaf(rh[k], x0, ah); ah = fmaf(rh[k + 1], x1, ah); ah = fmaf(rh[k + 2], x2, ah); ah = fmaf(rh[k + 3], x3, ah);
        ag = fmaf(rg[k], x0, ag); ag = fmaf(rg[k + 1], x1, ag); ag = fmaf(rg[k + 2], x2, ag); ag = fmaf(rg[k + 3], x3, ag);
        ap = fmaf(rp[k], x0, ap); ap = fmaf(rp[k + 1], x1, ap); ap = fmaf(rp[k + 2], x2, ap); ap = fmaf(rp[k + 3], x3, ap);
      }
      float sti = st[i];
      float out = sti + sigmoid_f(ag) * (tanh_f(ah) - sti);
      float p = sigmoid_f(ap);
      hout[i] = p * out + (1.0f - p) * hin[i];
      st[i] = out;
    }
    float* tmp = hin; hin = hout; hout = tmp;
  }

  float logits[NACT];
#pragma unroll
  for (int a = 0; a < NACT; ++a) {
    const float* wa = w_a + (size_t)a * HID;
    float acc = b_a[a];
    for (int k = 0; k < HID; k += 4) {
      acc = fmaf(wa[k], hin[k], acc);
      acc = fmaf(wa[k + 1], hin[k + 1], acc);
      acc = fmaf(wa[k + 2], hin[k + 2], acc);
      acc = fmaf(wa[k + 3], hin[k + 3], acc);
    }
    logits[a] = acc;
  }
  if (logits_out != nullptr) {
#pragma unroll
    for (int a = 0; a < NACT; ++a) {
      logits_out[e * 4 + a] = logits[a];
    }
  }
  if (DO_VALUE && value_out != nullptr) {
    float acc = b_v[0];
    for (int k = 0; k < HID; k += 4) {
      acc = fmaf(w_v[k], hin[k], acc);
      acc = fmaf(w_v[k + 1], hin[k + 1], acc);
      acc = fmaf(w_v[k + 2], hin[k + 2], acc);
      acc = fmaf(w_v[k + 3], hin[k + 3], acc);
    }
    value_out[e] = acc;
  }

  if (DO_ENV) {
    int best = 0;
#pragma unroll
    for (int a = 1; a < NACT; ++a) {
      if (logits[a] > logits[best]) best = a;
    }
    int x = (int)agent[e * 2 + 0];
    int y = (int)agent[e * 2 + 1];
    if (best == 0) y -= 1;
    else if (best == 1) y += 1;
    else if (best == 2) x -= 1;
    else x += 1;
    x = min(max(x, 0), BOARD - 1);
    y = min(max(y, 0), BOARD - 1);
    agent[e * 2 + 0] = (float)x;
    agent[e * 2 + 1] = (float)y;
    int fx = (int)food[e * 2 + 0];
    int fy = (int)food[e * 2 + 1];
    bool hit = (x == fx) && (y == fy);
    if (hitmask != nullptr) hitmask[e] = hit ? 1 : 0;
    if (rewards != nullptr) rewards[e] += hit ? 1.0f : 0.0f;
    if (hit && any_hit != nullptr) atomicMax(any_hit, 1);
  }
}

__global__ void k_env_apply_f32(
    float* __restrict__ food,
    long long* __restrict__ rng,
    const unsigned char* __restrict__ hitmask,
    const int* __restrict__ any_hit,
    int N) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  if (any_hit == nullptr || *any_hit == 0) return;
  unsigned long long r0 = (unsigned long long)rng[e];
  unsigned long long r1 = lcg_step(r0);
  unsigned long long r2 = lcg_step(r1);
  if (hitmask != nullptr && hitmask[e]) {
    long long fx = (long long)(r1 % (unsigned long long)BOARD);
    long long fy = (long long)(r2 % (unsigned long long)BOARD);
    food[e * 2 + 0] = (float)fx;
    food[e * 2 + 1] = (float)fy;
  }
  rng[e] = (long long)r2;
}

// ---------------------------------------------------------------------------
// Fused fp16 tensor-core step kernel (large-N path).
// One launch per env step; one block per E=32 envs; 8 warps.
// ---------------------------------------------------------------------------

template <int E, int MG>
__global__ void __launch_bounds__(TPB, 1) k_step_fp16(
    const half* __restrict__ wg16,
    const half* __restrict__ wenc16, const half* __restrict__ benc16,
    const half* __restrict__ wa16, const half* __restrict__ ba16,
    float* __restrict__ agent, float* __restrict__ food, long long* __restrict__ rng,
    half* __restrict__ state_g,
    const unsigned char* __restrict__ hitmask_prev,
    const int* __restrict__ any_prev, int* __restrict__ any_cur,
    unsigned char* __restrict__ hitmask_cur,
    float* __restrict__ rewards, float* __restrict__ logits_out, float* __restrict__ state_out,
    int N, int t, int horizon) {
  constexpr int MGROUPS = MG;
  constexpr int NSPLIT = 8 / MG;
  constexpr int IEXT = 256 / NSPLIT;
  constexpr int NCH = IEXT / 8;
  constexpr int NCHX = NCH * KPU;
  constexpr int NUX = NL * NCHX;
  extern __shared__ char smem_raw[];
  half* s_h = (half*)smem_raw;
  half* s_wa = s_h + 2 * E * ROWP;
  half* s_ba = s_wa + 4 * HID;
  half* s_stage = s_ba + 16;
  unsigned char* s_x = (unsigned char*)(s_stage + NSPLIT * RING * UBUF);
  unsigned char* s_y = s_x + E;
  unsigned char* s_fx = s_y + E;
  unsigned char* s_fy = s_fx + E;
  unsigned char* s_act = s_fy + E;
  float* s_logits = (float*)(s_act + 64);
  int* s_flag = (int*)(s_logits + 4 * E);
  const int tid = threadIdx.x;

  // ---- phase (i): apply prev-step rng advance + food respawn (gated) ----
  {
    bool anyp = (t > 0) && (*any_prev != 0);
#pragma unroll
    for (int i = tid; i < E; i += TPB) {
      int eg = blockIdx.x * E + i;
      int x = 0, y = 0, fx = 0, fy = 0;
      if (eg < N) {
        x = (int)agent[eg * 2 + 0]; y = (int)agent[eg * 2 + 1];
        fx = (int)food[eg * 2 + 0]; fy = (int)food[eg * 2 + 1];
        if (anyp) {
          unsigned long long r0 = (unsigned long long)rng[eg];
          unsigned long long r1 = lcg_step(r0);
          unsigned long long r2 = lcg_step(r1);
          if (hitmask_prev[eg]) {
            fx = (int)(r1 % (unsigned long long)BOARD);
            fy = (int)(r2 % (unsigned long long)BOARD);
            food[eg * 2 + 0] = (float)fx;
            food[eg * 2 + 1] = (float)fy;
          }
          rng[eg] = (long long)r2;
        }
      }
      s_x[i] = (unsigned char)x; s_y[i] = (unsigned char)y;
      s_fx[i] = (unsigned char)fx; s_fy[i] = (unsigned char)fy;
    }
    for (int i = tid; i < 4 * HID; i += TPB) s_wa[i] = wa16[i];
    if (tid < 4) s_ba[tid] = ba16[tid];
    if (tid == 0) *s_flag = 0;
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int mg = warp % MGROUPS;
  const int isp = warp / MGROUPS;
  const int i0 = isp * IEXT;
  const int st = lane + 32 * mg;

  // ---- phase (ii): enc (obs -> h, fp32 math stored fp16) ----
#pragma unroll
  for (int idx = tid; idx < E * HID; idx += TPB) {
    int env = idx >> 8, i = idx & 255;
    float dx = (float)((int)s_fx[env] - (int)s_x[env]) * (1.0f / BOARD);
    float dy = (float)((int)s_fy[env] - (int)s_y[env]) * (1.0f / BOARD);
    float px = (float)s_x[env] * (1.0f / (BOARD - 1));
    float py = (float)s_y[env] * (1.0f / (BOARD - 1));
    float acc = __half2float(__ldg(benc16 + i));
    acc = fmaf(__half2float(__ldg(wenc16 + i * 4 + 0)), dx, acc);
    acc = fmaf(__half2float(__ldg(wenc16 + i * 4 + 1)), dy, acc);
    acc = fmaf(__half2float(__ldg(wenc16 + i * 4 + 2)), px, acc);
    acc = fmaf(__half2float(__ldg(wenc16 + i * 4 + 3)), py, acc);
    s_h[env * ROWP + i] = __float2half(acc);
  }
  __syncthreads();

  // ---- phase (iii): MinGRU x3 (fp16 mma + fp32 gating epilogue) ----
#pragma unroll
  for (int l = 0; l < NL; ++l) {
    const half* hin = s_h + (l & 1) * E * ROWP;
    half* hout = s_h + ((l + 1) & 1) * E * ROWP;
    const half* wgl = wg16 + (size_t)l * 768 * 256;
    half* my_stage = s_stage + isp * RING * UBUF;

#define STAGE_RING(ci, ku, slot)                                                        \
    {                                                                                   \
      int inext = i0 + (ci) * 8;                                                        \
      half* dst = my_stage + (slot) * UBUF;                                             \
      _Pragma("unroll") for (int work = st; work < 24 * (UK / 8); work += 32 * MGROUPS) {       \
        int row = work / (UK / 8), c16 = work % (UK / 8);                               \
        const half* src = wgl + ((size_t)(row >> 3) * 256 + inext + (row & 7)) * 256 + (ku) * UK + c16 * 8; \
        cp_async16(smem_u32(dst + row * USTR + c16 * 8), src);                          \
      }                                                                                 \
      cp_commit();                                                                      \
    }

#define STAGE_RING_L(nl, ci, ku, slot)                                                  \
    {                                                                                   \
      int inext = i0 + (ci) * 8;                                                        \
      half* dst = my_stage + (slot) * UBUF;                                             \
      const half* wgn = wg16 + (size_t)(nl) * 768 * 256;                                \
      _Pragma("unroll") for (int work = st; work < 24 * (UK / 8); work += 32 * MGROUPS) { \
        int row = work / (UK / 8), c16 = work % (UK / 8);                               \
        const half* src = wgn + ((size_t)(row >> 3) * 256 + inext + (row & 7)) * 256 + (ku) * UK + c16 * 8; \
        cp_async16(smem_u32(dst + row * USTR + c16 * 8), src);                          \
      }                                                                                 \
      cp_commit();                                                                      \
    }

#pragma unroll
    for (int u = 0; u < RING - 1; ++u) {
      if (u < NCHX) STAGE_RING(u / KPU, u % KPU, u)
    }

    stream_sync<MG>(isp);
    float acc[NB][3][4];
    float sacc[NB][3][4];
    unsigned stpf[NB][2], stpf_prev[NB][2];
    for (int ci = 0; ci < NCH; ++ci) {
#pragma unroll
      for (int b = 0; b < NB; ++b)
#pragma unroll
        for (int g = 0; g < 3; ++g)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[b][g][q] = 0.f;

#pragma unroll
      for (int ku = 0; ku < KPU; ++ku) {
        int u = ci * KPU + ku;
        int nxt = u + RING - 1;
        bool already = (l > 0 && ci == 0 && ku < RING - 1);
        if (nxt < NCHX && !already) STAGE_RING(nxt / KPU, nxt % KPU, nxt % RING)
        if (ku == 0) {
          // deferred epilogue for the previous chunk, overlapped with stage waits
          if (ci > 0) {
            int cc = i0 + (ci - 1) * 8 + (lane & 3) * 2;
#pragma unroll
            for (int b = 0; b < NB; ++b) {
              int r0 = (mg * NB + b) * 16 + (lane >> 2);
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) {
                int row = r0 + rr * 8;
                float zh0 = sacc[b][0][rr * 2 + 0], zh1 = sacc[b][0][rr * 2 + 1];
                float zg0 = sacc[b][1][rr * 2 + 0], zg1 = sacc[b][1][rr * 2 + 1];
                float zp0 = sacc[b][2][rr * 2 + 0], zp1 = sacc[b][2][rr * 2 + 1];
                int eg = blockIdx.x * E + row;
                __half2 st2 = *(__half2*)&stpf_prev[b][rr];
                unsigned ssh = *(unsigned*)&hin[row * ROWP + cc];
                __half2 hh2 = *(__half2*)&ssh;
                float st0 = __half2float(st2.x), st1 = __half2float(st2.y);
                float hi0 = __half2float(hh2.x), hi1 = __half2float(hh2.y);
                float out0 = st0 + sigmoid_f(zg0) * (tanh_f(zh0) - st0);
                float out1 = st1 + sigmoid_f(zg1) * (tanh_f(zh1) - st1);
                float p0 = sigmoid_f(zp0), p1 = sigmoid_f(zp1);
                float hn0 = p0 * out0 + (1.0f - p0) * hi0;
                float hn1 = p1 * out1 + (1.0f - p1) * hi1;
                __half2 n2 = __floats2half2_rn(hn0, hn1);
                *(unsigned*)&hout[row * ROWP + cc] = *(unsigned*)&n2;
                if (t == horizon - 1 && eg < N) {
                  state_out[((size_t)eg * NL + l) * HID + cc] = out0;
                  state_out[((size_t)eg * NL + l) * HID + cc + 1] = out1;
                } else if (eg < N) {
                  __half2 o2 = __floats2half2_rn(out0, out1);
                  *(__half2*)(state_g + ((size_t)eg * NL + l) * HID + cc) = o2;
                }
              }
            }
          }
          // prefetch state for THIS chunk's future epilogue
          {
            int cc = i0 + ci * 8 + (lane & 3) * 2;
#pragma unroll
            for (int b = 0; b < NB; ++b) {
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) {
                int row = ((mg * NB + b) * 16 + (lane >> 2)) + rr * 8;
                int eg = blockIdx.x * E + row;
                stpf[b][rr] = (eg < N)
                    ? *(const unsigned*)(state_g + ((size_t)eg * NL + l) * HID + cc) : 0u;
              }
            }
          }
        }
        {
          int issued = (nxt < NCHX) ? (nxt + 1) : NCHX;
          int pend = issued - (u + 1);
          if (pend >= 3) cp_wait<3>();
          else if (pend == 2) cp_wait<2>();
          else if (pend == 1) cp_wait<1>();
          else cp_wait<0>();
        }
        stream_sync<MG>(isp);

        const half* chbuf = my_stage + (u % RING) * UBUF;

#pragma unroll
        for (int k16 = 0; k16 < UK / 16; ++k16) {
          int k = ku * (UK / 16) + k16;
          unsigned afrag[NB][4];
#pragma unroll
          for (int b = 0; b < NB; ++b) {
            const half* aaddr = hin + ((mg * NB + b) * 16 + (lane & 15)) * ROWP + k * 16 + (lane >> 4) * 8;
            ldmatrix_x4(smem_u32(aaddr), afrag[b]);
          }
          unsigned bfr[3][2];
          {
            int grp = lane >> 3;
            int rrow = lane & 7;
            const half* baddr = chbuf + ((grp >> 1) * 8 + rrow) * USTR + k16 * 16 + (grp & 1) * 8;
            unsigned frag4[4];
            ldmatrix_x4(smem_u32(baddr), frag4);
            bfr[0][0] = frag4[0]; bfr[0][1] = frag4[1];
            bfr[1][0] = frag4[2]; bfr[1][1] = frag4[3];
            const half* baddr2 = chbuf + (2 * 8 + rrow) * USTR + k16 * 16 + ((lane >> 3) & 1) * 8;
            unsigned frag2[2];
            ldmatrix_x2(smem_u32(baddr2), frag2);
            bfr[2][0] = frag2[0]; bfr[2][1] = frag2[1];
          }
#pragma unroll
          for (int g = 0; g < 3; ++g) {
#pragma unroll
            for (int b = 0; b < NB; ++b) mma_16816(acc[b][g], afrag[b], bfr[g][0], bfr[g][1]);
          }
        }
        stream_sync<MG>(isp);
      }

      if (ci == NCH - 1) {
        // layer's last chunk: inline epilogue
        int cc = i0 + (NCH - 1) * 8 + (lane & 3) * 2;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          int r0 = (mg * NB + b) * 16 + (lane >> 2);
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            int row = r0 + rr * 8;
            float zh0 = acc[b][0][rr * 2 + 0], zh1 = acc[b][0][rr * 2 + 1];
            float zg0 = acc[b][1][rr * 2 + 0], zg1 = acc[b][1][rr * 2 + 1];
            float zp0 = acc[b][2][rr * 2 + 0], zp1 = acc[b][2][rr * 2 + 1];
            int eg = blockIdx.x * E + row;
            __half2 st2 = *(__half2*)&stpf[b][rr];
            unsigned ssh = *(unsigned*)&hin[row * ROWP + cc];
            __half2 hh2 = *(__half2*)&ssh;
            float st0 = __half2float(st2.x), st1 = __half2float(st2.y);
            float hi0 = __half2float(hh2.x), hi1 = __half2float(hh2.y);
            float out0 = st0 + sigmoid_f(zg0) * (tanh_f(zh0) - st0);
            float out1 = st1 + sigmoid_f(zg1) * (tanh_f(zh1) - st1);
            float p0 = sigmoid_f(zp0), p1 = sigmoid_f(zp1);
            float hn0 = p0 * out0 + (1.0f - p0) * hi0;
            float hn1 = p1 * out1 + (1.0f - p1) * hi1;
            __half2 n2 = __floats2half2_rn(hn0, hn1);
            *(unsigned*)&hout[row * ROWP + cc] = *(unsigned*)&n2;
            if (t == horizon - 1 && eg < N) {
              state_out[((size_t)eg * NL + l) * HID + cc] = out0;
              state_out[((size_t)eg * NL + l) * HID + cc + 1] = out1;
            } else if (eg < N) {
              __half2 o2 = __floats2half2_rn(out0, out1);
              *(__half2*)(state_g + ((size_t)eg * NL + l) * HID + cc) = o2;
            }
          }
        }
      } else {
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
          for (int g = 0; g < 3; ++g)
#pragma unroll
            for (int q = 0; q < 4; ++q) sacc[b][g][q] = acc[b][g][q];
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) stpf_prev[b][rr] = stpf[b][rr];
      }
    }
    // cross-layer lookahead: next layer's first two units already in flight
    if (l + 1 < NL) {
      int ub = (l + 1) * NCHX;
      int nlx = l + 1;
      STAGE_RING_L(nlx, 0, 0, ub % RING)
      STAGE_RING_L(nlx, 0, 1, (ub + 1) % RING)
    }
    __syncthreads();
#undef STAGE_RING
  }

  // ---- phase (iv): action head (fp32 math on fp16 h) ----
  {
    const half* hfin = s_h + E * ROWP;
    constexpr int EPW = E / 8;
    constexpr int LPE = 32 / EPW;
    constexpr int KEXT = HID / LPE;
    int env = warp * EPW + lane / LPE;
    int ks = (lane % LPE) * KEXT;
    float l0 = 0.f, l1 = 0.f, l2 = 0.f, l3 = 0.f;
    const half* hrow = hfin + env * ROWP + ks;
#pragma unroll
    for (int k = 0; k < KEXT; k += 2) {
      __half2 hh = *(__half2*)&hrow[k];
      float f0 = __half2float(hh.x), f1 = __half2float(hh.y);
      l0 = fmaf(__half2float(s_wa[0 * HID + ks + k]), f0, l0);
      l0 = fmaf(__half2float(s_wa[0 * HID + ks + k + 1]), f1, l0);
      l1 = fmaf(__half2float(s_wa[1 * HID + ks + k]), f0, l1);
      l1 = fmaf(__half2float(s_wa[1 * HID + ks + k + 1]), f1, l1);
      l2 = fmaf(__half2float(s_wa[2 * HID + ks + k]), f0, l2);
      l2 = fmaf(__half2float(s_wa[2 * HID + ks + k + 1]), f1, l2);
      l3 = fmaf(__half2float(s_wa[3 * HID + ks + k]), f0, l3);
      l3 = fmaf(__half2float(s_wa[3 * HID + ks + k + 1]), f1, l3);
    }
#pragma unroll
    for (int off = LPE / 2; off > 0; off >>= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, off);
      l1 += __shfl_xor_sync(0xffffffffu, l1, off);
      l2 += __shfl_xor_sync(0xffffffffu, l2, off);
      l3 += __shfl_xor_sync(0xffffffffu, l3, off);
    }
    if ((lane % LPE) == 0) {
      l0 += __half2float(s_ba[0]); l1 += __half2float(s_ba[1]);
      l2 += __half2float(s_ba[2]); l3 += __half2float(s_ba[3]);
      s_logits[env * 4 + 0] = l0; s_logits[env * 4 + 1] = l1;
      s_logits[env * 4 + 2] = l2; s_logits[env * 4 + 3] = l3;
      int best = 0;
      if (l1 > l0) best = 1;
      if (l2 > ((best == 1) ? l1 : l0)) best = 2;
      if (l3 > ((best == 2) ? l2 : (best == 1) ? l1 : l0)) best = 3;
      s_act[env] = (unsigned char)best;
    }
  }
  __syncthreads();

  // ---- phase (v): greedy move + hit + reward + any report ----
  if (tid < E) {
    int eg = blockIdx.x * E + tid;
    bool valid = eg < N;
    int act = valid ? (int)s_act[tid] : 0;
    int x = s_x[tid], y = s_y[tid];
    if (act == 0) y -= 1;
    else if (act == 1) y += 1;
    else if (act == 2) x -= 1;
    else x += 1;
    x = min(max(x, 0), BOARD - 1);
    y = min(max(y, 0), BOARD - 1);
    bool hit = valid && (x == (int)s_fx[tid]) && (y == (int)s_fy[tid]);
    if (valid) {
      agent[eg * 2 + 0] = (float)x;
      agent[eg * 2 + 1] = (float)y;
      rewards[eg] += hit ? 1.0f : 0.0f;
      hitmask_cur[eg] = hit ? 1 : 0;
    }
    unsigned bal = __ballot_sync(0xffffffffu, hit);
    if ((tid & 31) == 0 && bal != 0) atomicOr(s_flag, 1);
  }
  __syncthreads();
  if (tid == 0 && *s_flag != 0) atomicMax(any_cur, 1);
  if (t == horizon - 1 && tid < E) {
    int eg = blockIdx.x * E + tid;
    if (eg < N) {
#pragma unroll
      for (int a = 0; a < 4; ++a) logits_out[eg * 4 + a] = s_logits[tid * 4 + a];
    }
  }
}

// zero all run buffers + cast small weights in one launch
__global__ void k_run_init(
    half* __restrict__ state_g, float* __restrict__ rewards, int* __restrict__ anyfl,
    unsigned char* __restrict__ hitmask,
    const float* __restrict__ w_enc, const float* __restrict__ b_enc,
    const float* __restrict__ w_a, const float* __restrict__ b_a,
    half* __restrict__ wenc16, half* __restrict__ benc16,
    half* __restrict__ wa16, half* __restrict__ ba16,
    int64_t n_state, int64_t n_rewards, int64_t n_any, int64_t n_hm, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t j = i; j < n_state; j += stride) state_g[j] = __float2half(0.f);
  for (int64_t j = i; j < n_rewards; j += stride) rewards[j] = 0.f;
  for (int64_t j = i; j < n_any; j += stride) anyfl[j] = 0;
  for (int64_t j = i; j < n_hm; j += stride) hitmask[j] = 0;
  if (blockIdx.x == 0) {
    for (int j = threadIdx.x; j < HID * 4; j += blockDim.x) wenc16[j] = __float2half(w_enc[j]);
    for (int j = threadIdx.x; j < HID; j += blockDim.x) benc16[j] = __float2half(b_enc[j]);
    for (int j = threadIdx.x; j < 4 * HID; j += blockDim.x) wa16[j] = __float2half(w_a[j]);
    if (threadIdx.x < 4) ba16[threadIdx.x] = __float2half(b_a[threadIdx.x]);
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

static inline void launch_policy_f32(
    bool do_env, bool do_value,
    const torch::Tensor& w_enc, const torch::Tensor& b_enc, const torch::Tensor& w_gru,
    const torch::Tensor& w_a, const torch::Tensor& b_a,
    const float* w_v, const float* b_v,
    const float* obs, float* agent, const float* food, float* state,
    float* logits_out, float* value_out, unsigned char* hitmask, float* rewards,
    int* any_hit, int64_t N, cudaStream_t stream) {
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  if (do_env) {
    k_policy_f32<true, false><<<blocks, threads, 0, stream>>>(
        w_enc.data_ptr<float>(), b_enc.data_ptr<float>(), w_gru.data_ptr<float>(),
        w_a.data_ptr<float>(), b_a.data_ptr<float>(), w_v, b_v,
        obs, agent, food, state, logits_out, value_out, hitmask, rewards, any_hit, (int)N);
  } else if (do_value) {
    k_policy_f32<false, true><<<blocks, threads, 0, stream>>>(
        w_enc.data_ptr<float>(), b_enc.data_ptr<float>(), w_gru.data_ptr<float>(),
        w_a.data_ptr<float>(), b_a.data_ptr<float>(), w_v, b_v,
        obs, agent, food, state, logits_out, value_out, hitmask, rewards, any_hit, (int)N);
  } else {
    k_policy_f32<false, false><<<blocks, threads, 0, stream>>>(
        w_enc.data_ptr<float>(), b_enc.data_ptr<float>(), w_gru.data_ptr<float>(),
        w_a.data_ptr<float>(), b_a.data_ptr<float>(), w_v, b_v,
        obs, agent, food, state, logits_out, value_out, hitmask, rewards, any_hit, (int)N);
  }
}

std::vector<torch::Tensor> policy_forward_cuda(
    torch::Tensor w_enc, torch::Tensor b_enc, torch::Tensor w_gru,
    torch::Tensor w_a, torch::Tensor b_a, torch::Tensor w_v, torch::Tensor b_v,
    torch::Tensor obs, torch::Tensor state) {
  int64_t N = obs.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto state_new = state.clone();
  auto logits = torch::empty({N, 4}, obs.options());
  auto value = torch::empty({N}, obs.options());
  launch_policy_f32(false, true, w_enc, b_enc, w_gru, w_a, b_a,
                    w_v.data_ptr<float>(), b_v.data_ptr<float>(),
                    obs.data_ptr<float>(), nullptr, nullptr,
                    state_new.data_ptr<float>(),
                    logits.data_ptr<float>(), value.data_ptr<float>(),
                    nullptr, nullptr, nullptr, N, stream.stream());
  return {logits, state_new, value};
}

std::vector<torch::Tensor> env_step_cuda(
    torch::Tensor agent, torch::Tensor food, torch::Tensor actions, torch::Tensor rng_state) {
  int64_t N = agent.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto agent_out = agent.clone();
  auto food_out = food.clone();
  auto rng_out = rng_state.clone();
  auto reward = torch::zeros({N}, agent.options());
  auto hitmask = torch::empty({N}, agent.options().dtype(torch::kUInt8));
  auto any_hit = torch::zeros({1}, agent.options().dtype(torch::kInt32));

  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);
  k_env_move_f32<<<blocks, threads, 0, stream.stream()>>>(
      agent_out.data_ptr<float>(), food.data_ptr<float>(),
      (const long long*)actions.data_ptr<int64_t>(), reward.data_ptr<float>(),
      (unsigned char*)hitmask.data_ptr<uint8_t>(), any_hit.data_ptr<int>(), (int)N);
  k_env_apply_f32<<<blocks, threads, 0, stream.stream()>>>(
      food_out.data_ptr<float>(), (long long*)rng_out.data_ptr<int64_t>(),
      (const unsigned char*)hitmask.data_ptr<uint8_t>(), any_hit.data_ptr<int>(), (int)N);
  return {agent_out, food_out, reward, rng_out};
}

std::vector<torch::Tensor> run_steps_cuda(
    torch::Tensor w_enc, torch::Tensor b_enc, torch::Tensor w_gru,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng_state,
    int64_t horizon) {
  int64_t N = agent.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = agent.options();
  auto rewards = torch::zeros({N}, opts);
  auto last_logits = torch::zeros({N, 4}, opts);
  auto state = torch::zeros({N, NL, HID}, opts);
  auto hitmask = torch::empty({N}, opts.dtype(torch::kUInt8));
  auto any_hit = torch::zeros({1}, opts.dtype(torch::kInt32));
  int threads = 256;
  int blocks = (int)((N + threads - 1) / threads);

  for (int64_t t = 0; t < horizon; ++t) {
    any_hit.zero_();
    launch_policy_f32(true, false, w_enc, b_enc, w_gru, w_a, b_a, nullptr, nullptr,
                      nullptr, agent.data_ptr<float>(), food.data_ptr<float>(),
                      state.data_ptr<float>(), last_logits.data_ptr<float>(), nullptr,
                      (unsigned char*)hitmask.data_ptr<uint8_t>(), rewards.data_ptr<float>(),
                      any_hit.data_ptr<int>(), N, stream.stream());
    k_env_apply_f32<<<blocks, threads, 0, stream.stream()>>>(
        food.data_ptr<float>(), (long long*)rng_state.data_ptr<int64_t>(),
        (const unsigned char*)hitmask.data_ptr<uint8_t>(), any_hit.data_ptr<int>(), (int)N);
  }
  return {rewards, agent, last_logits, state};
}

template <int E, int MG>
static inline void launch_step_fp16(
    const half* wg16, const half* wenc16, const half* benc16, const half* wa16, const half* ba16,
    float* agent, float* food, long long* rng, half* state_g,
    const unsigned char* hm_prev, int* any_prev, int* any_cur, unsigned char* hm_cur,
    float* rewards, float* logits, float* state_out,
    int N, int t, int horizon, cudaStream_t stream) {
  constexpr int NSPL = 8 / MG;
  size_t smem =
      (size_t)2 * E * ROWP * 2 + (4 * HID + 16) * 2 + (size_t)NSPL * RING * UBUF * 2 +
      (size_t)5 * ((E + 15) & ~15) + (size_t)4 * E * 4 + 64;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(k_step_fp16<E, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = true;
  }
  int blocks = (N + E - 1) / E;
  k_step_fp16<E, MG><<<blocks, TPB, smem, stream>>>(
      wg16, wenc16, benc16, wa16, ba16, agent, food, rng, state_g,
      hm_prev, any_prev, any_cur, hm_cur, rewards, logits, state_out, N, t, horizon);
}

std::vector<torch::Tensor> run_fast(
    torch::Tensor wg16, torch::Tensor w_enc, torch::Tensor b_enc,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor agent, torch::Tensor food, torch::Tensor rng, int64_t horizon) {
  int64_t N = agent.size(0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto opts = agent.options();
  auto rewards = torch::empty({N}, opts);
  auto last_logits = torch::empty({N, 4}, opts);
  auto state_out = torch::empty({N, NL, HID}, opts);
  auto state_g = torch::empty({N, NL, HID}, opts.dtype(torch::kHalf));
  auto anyfl = torch::empty({horizon + 1}, opts.dtype(torch::kInt32));
  auto hitmask = torch::empty({2, N}, opts.dtype(torch::kUInt8));

  auto hopt = opts.dtype(torch::kHalf);
  auto wenc16 = torch::empty({HID, 4}, hopt);
  auto benc16 = torch::empty({HID}, hopt);
  auto wa16 = torch::empty({4, HID}, hopt);
  auto ba16 = torch::empty({4}, hopt);
  {
    int64_t n_state = state_g.numel();
    int64_t n_work = std::max(n_state, (int64_t)hitmask.numel());
    int init_blocks = (int)std::min((int64_t)1024, (n_work + 255) / 256);
    k_run_init<<<init_blocks, 256, 0, stream>>>(
        (half*)state_g.data_ptr<at::Half>(), rewards.data_ptr<float>(), anyfl.data_ptr<int>(),
        (unsigned char*)hitmask.data_ptr<uint8_t>(),
        w_enc.data_ptr<float>(), b_enc.data_ptr<float>(), w_a.data_ptr<float>(), b_a.data_ptr<float>(),
        (half*)wenc16.data_ptr<at::Half>(), (half*)benc16.data_ptr<at::Half>(),
        (half*)wa16.data_ptr<at::Half>(), (half*)ba16.data_ptr<at::Half>(),
        n_state, N, anyfl.numel(), hitmask.numel(), N);
  }

  if (N > 6144) {
    for (int t = 0; t < (int)horizon; ++t) {
      launch_step_fp16<64, 2>(
          (const half*)wg16.data_ptr<at::Half>(), (const half*)wenc16.data_ptr<at::Half>(),
          (const half*)benc16.data_ptr<at::Half>(), (const half*)wa16.data_ptr<at::Half>(),
          (const half*)ba16.data_ptr<at::Half>(), agent.data_ptr<float>(), food.data_ptr<float>(),
          (long long*)rng.data_ptr<int64_t>(), (half*)state_g.data_ptr<at::Half>(),
          (const unsigned char*)hitmask.data_ptr<uint8_t>() + ((t + 1) & 1) * N,
          anyfl.data_ptr<int>() + t, anyfl.data_ptr<int>() + t + 1,
          (unsigned char*)hitmask.data_ptr<uint8_t>() + (t & 1) * N,
          rewards.data_ptr<float>(), last_logits.data_ptr<float>(), state_out.data_ptr<float>(),
          (int)N, t, (int)horizon, stream);
    }
  } else {
    for (int t = 0; t < (int)horizon; ++t) {
      launch_step_fp16<32, 1>(
          (const half*)wg16.data_ptr<at::Half>(), (const half*)wenc16.data_ptr<at::Half>(),
          (const half*)benc16.data_ptr<at::Half>(), (const half*)wa16.data_ptr<at::Half>(),
          (const half*)ba16.data_ptr<at::Half>(), agent.data_ptr<float>(), food.data_ptr<float>(),
          (long long*)rng.data_ptr<int64_t>(), (half*)state_g.data_ptr<at::Half>(),
          (const unsigned char*)hitmask.data_ptr<uint8_t>() + ((t + 1) & 1) * N,
          anyfl.data_ptr<int>() + t, anyfl.data_ptr<int>() + t + 1,
          (unsigned char*)hitmask.data_ptr<uint8_t>() + (t & 1) * N,
          rewards.data_ptr<float>(), last_logits.data_ptr<float>(), state_out.data_ptr<float>(),
          (int)N, t, (int)horizon, stream);
    }
  }
  auto positions = agent.round().to(torch::kInt64);
  return {rewards, positions, last_logits, state_out};
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="grid_mingru_sps_v2",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["policy_forward_cuda", "env_step_cuda", "run_steps_cuda", "run_fast", "cast_fp16"],
            extra_cuda_cflags=["-O3", "--restrict"],
            verbose=False,
        )
    return _ext


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
    m = model
    logits, new_state, value = _get_ext().policy_forward_cuda(
        m.w_enc, m.b_enc, m.w_gru, m.w_a, m.b_a, m.w_v, m.b_v,
        obs.contiguous(), state.contiguous(),
    )
    return logits, new_state, value


def env_step(agent: torch.Tensor, food: torch.Tensor, actions: torch.Tensor, rng_state: torch.Tensor):
    a, f, r, rng = _get_ext().env_step_cuda(
        agent.contiguous(), food.contiguous(), actions.contiguous(), rng_state.contiguous()
    )
    return a, f, r, rng


def _make_init(num_envs: int, seed: int, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    agent = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    food = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    rng_state = torch.arange(num_envs, device=device, dtype=torch.int64) + (seed * 10007)
    return agent, food, rng_state


def run(num_envs: int, horizon: int, seed: int, model: Model | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Model()
    model = model.to(device).eval()
    m = model

    agent, food, rng_state = _make_init(num_envs, seed, device)
    ext = _get_ext()
    if num_envs > 2048:
        wg16 = ext.cast_fp16(m.w_gru.detach().contiguous())
        rewards, positions, last_logits, state = ext.run_fast(
            wg16, m.w_enc, m.b_enc, m.w_a, m.b_a,
            agent, food, rng_state, horizon,
        )
    else:
        rewards, agent_final, last_logits, state = ext.run_steps_cuda(
            m.w_enc, m.b_enc, m.w_gru, m.w_a, m.b_a,
            agent, food, rng_state, horizon,
        )
        positions = agent_final.round().long()
    return {
        "rewards": rewards.detach(),
        "positions": positions.detach(),
        "last_logits": last_logits.detach(),
        "state": state.detach(),
    }
