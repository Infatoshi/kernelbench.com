"""Vectorized grid-foraging + 3xMinGRU(h=256) policy rollout — CUDA megakernel.

run(): ONE persistent cooperative kernel executes the whole horizon.
  Default path (rollout_v2, "-2"): B=32 envs/block, 256 threads (8 warps),
  FFMA-blocked gate GEMM in fp32: per-thread tile = 8 envs x 2 comps x 3 gates
  (48 accumulators), H ping-pong in shared memory ([comp][env] padded rows),
  W staged from a pass-permuted offline transpose via a cp.async double
  buffer (pass = 128 comps x 3 gates = 384 contiguous rows per k-tile).
  env move / hit.any() reduction (per-step hit_count slot + one grid.sync) /
  exact reference LCG respawn / rewards / final positions all run in-kernel.
  Init: reference's two CPU MT19937 randint draws reproduced exactly on CPU
  into pinned staging -> H2D int64 -> fp32 cast on GPU.
  numerics are pure fp32 FFMA -> matches the eager reference to reorder
  noise (~5e-8; fits the atol 1e-6 small stress case; positions are exact).

Alternative rollout variants (0..14, KBH_ROLLOUT_VARIANT) from the tuning
sweep are kept for reference; check.py/benchmark.py run the default path.
policy_forward()/env_step() are standalone kernels used by check.py.
"""

from __future__ import annotations

import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
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

num_envs = 4096
horizon = 32


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

#define HID 256
#define NL 3
#define BOARD_D 11

__device__ __forceinline__ unsigned long long lcg_step(unsigned long long r) {
  return (r * 6364136223846793005ULL + 1ULL) & 0x7FFFFFFFFFFFFFFFULL;
}
__device__ __forceinline__ float sig(float x) { return 1.0f / (1.0f + expf(-x)); }

// streaming (evict-first) global access helpers for state: it has no L2 reuse
// (201MB stream) and must not evict the 2.36MB weight matrix from L2.
__device__ __forceinline__ float ldcs(const float* p) {
  float v;
  asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void stcs(float* p, float v) {
  asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(p), "f"(v));
}

// cp.async wrappers (inline PTX)
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
  unsigned sdst = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sdst), "l"(gmem_src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ void cp_wait2() { asm volatile("cp.async.wait_group 2;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// ============================================================ naive kernels
// (used by check.py through policy_forward / env_step; correctness only)

struct Weights {
  const float* __restrict__ w_enc;
  const float* __restrict__ b_enc;
  const float* __restrict__ w_gru;
  const float* __restrict__ w_a;
  const float* __restrict__ b_a;
  const float* __restrict__ w_v;
  const float* __restrict__ b_v;
};

__device__ __forceinline__ void policy_dtof(const Weights& W,
                                             const float* obs,
                                             const float* state,
                                             float* state_out,
                                             float* logits,
                                             float* value) {
  float h[HID], hn[HID];
#pragma unroll 8
  for (int o = 0; o < HID; ++o) {
    const float* wr = W.w_enc + o * 4;
    h[o] = W.b_enc[o] + obs[0] * wr[0] + obs[1] * wr[1] + obs[2] * wr[2] + obs[3] * wr[3];
  }
  for (int l = 0; l < NL; ++l) {
    const float* wl = W.w_gru + (size_t)l * 3 * HID * HID;
    const float* st = state + (size_t)l * HID;
    float* st_out = state_out + (size_t)l * HID;
    for (int j = 0; j < HID; ++j) {
      const float* wzh = wl + (size_t)j * HID;
      const float* wzg = wl + (size_t)(HID + j) * HID;
      const float* wzp = wl + (size_t)(2 * HID + j) * HID;
      float zh = 0.f, zg = 0.f, zp = 0.f;
#pragma unroll 8
      for (int k = 0; k < HID; ++k) {
        float hv = h[k];
        zh += wzh[k] * hv;
        zg += wzg[k] * hv;
        zp += wzp[k] * hv;
      }
      float s = st[j];
      float out = s + sig(zg) * (tanhf(zh) - s);
      float p = sig(zp);
      hn[j] = p * out + (1.0f - p) * h[j];
      st_out[j] = out;
    }
#pragma unroll 8
    for (int j = 0; j < HID; ++j) h[j] = hn[j];
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const float* wr = W.w_a + a * HID;
    float acc = W.b_a[a];
#pragma unroll 8
    for (int k = 0; k < HID; ++k) acc += wr[k] * h[k];
    logits[a] = acc;
  }
  {
    float acc = W.b_v[0];
#pragma unroll 8
    for (int k = 0; k < HID; ++k) acc += W.w_v[k] * h[k];
    *value = acc;
  }
}

__global__ void policy_forward_kernel(
    const float* __restrict__ w_enc, const float* __restrict__ b_enc,
    const float* __restrict__ w_gru,
    const float* __restrict__ w_a, const float* __restrict__ b_a,
    const float* __restrict__ w_v, const float* __restrict__ b_v,
    const float* __restrict__ obs,
    const float* __restrict__ state,
    float* __restrict__ state_out,
    float* __restrict__ logits,
    float* __restrict__ value,
    int N) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  Weights W{w_enc, b_enc, w_gru, w_a, b_a, w_v, b_v};
  policy_dtof(W, obs + (size_t)e * 4, state + (size_t)e * 3 * HID,
              state_out + (size_t)e * 3 * HID, logits + (size_t)e * 4, value + e);
}

__global__ void env_move_kernel(
    float* __restrict__ agent,
    const float* __restrict__ food,
    const long long* __restrict__ actions,
    float* __restrict__ reward,
    unsigned char* __restrict__ hit,
    int* __restrict__ any_hit,
    int N) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  float ax = agent[e * 2], ay = agent[e * 2 + 1];
  long long a = actions[e];
  float dx = (a == 2) ? -1.f : (a == 3) ? 1.f : 0.f;
  float dy = (a == 0) ? -1.f : (a == 1) ? 1.f : 0.f;
  ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
  ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
  agent[e * 2] = ax;
  agent[e * 2 + 1] = ay;
  bool h = (ax == food[e * 2]) && (ay == food[e * 2 + 1]);
  hit[e] = h ? 1 : 0;
  reward[e] = h ? 1.f : 0.f;
  if (h) atomicOr(any_hit, 1);
}

__global__ void env_respawn_kernel(
    float* __restrict__ food,
    const unsigned char* __restrict__ hit,
    long long* __restrict__ rng,
    const int* __restrict__ any_hit,
    int N) {
  if (*any_hit == 0) return;
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  unsigned long long r = (unsigned long long)rng[e];
  unsigned long long r1 = lcg_step(r);
  unsigned long long r2 = lcg_step(r1);
  rng[e] = (long long)r2;
  if (hit[e]) {
    food[e * 2] = (float)(long long)(r1 % BOARD_D);
    food[e * 2 + 1] = (float)(long long)(r2 % BOARD_D);
  }
}

// ============================================================ V2 rollout
// B envs/block, (B/8)*64 threads. Thread tile: E_TILE envs x C_TILE comps x 3 gates.
// H buffers: [256 k][B envs] float with row pad. W stages: [K_TILE k][384 rows].

#define E_TILE 8
#define C_TILE 2
#define K_TILE 8
#define PASS_ROWS 384
#define HPAD(B) ((B) + 4)

struct SmemLayout {
  float* Ha;      // 256*HPAD
  float* Hb;      // 256*HPAD
  float* Wst;     // 2*K_TILE*384
  float* obs;     // B*4
  float* agent;   // B*2
  float* food;    // B*2
  float* logits;  // B*4
  int* action;    // B
  int* flag;      // 2
  unsigned char* hitb;  // B
};

__device__ __forceinline__ SmemLayout smem_layout(float* base, int B) {
  SmemLayout s;
  int hpad = HPAD(B);
  s.Ha = base;
  s.Hb = s.Ha + 256 * hpad;
  s.Wst = s.Hb + 256 * hpad;
  s.obs = s.Wst + 2 * K_TILE * PASS_ROWS;
  s.agent = s.obs + B * 4;
  s.food = s.agent + B * 2;
  s.logits = s.food + B * 2;
  s.action = (int*)(s.logits + B * 4);
  s.flag = s.action + B;
  s.hitb = (unsigned char*)(s.flag + 8);
  return s;
}

__host__ __device__ __forceinline__ int smem_floats(int B) {
  int hpad = HPAD(B);
  int fl = 2 * 256 * hpad + 2 * K_TILE * PASS_ROWS + B * 4 + B * 2 + B * 2 + B * 4 + B + 8 + B;
  return fl + 32;
}

__device__ float DUMMY_W[768 * 256];
template <int B>
__global__ void __launch_bounds__((B / E_TILE) * 64) rollout_probe(
    const float* __restrict__ Wt,   // [3][256][768] pass-permuted transpose
    const float* __restrict__ We,   // [256][4]
    const float* __restrict__ be,   // [256]
    const float* __restrict__ Wa,   // [4][256]
    const float* __restrict__ ba,   // [4]
    float* __restrict__ agent,      // [N][2]
    float* __restrict__ food,       // [N][2]
    long long* __restrict__ rng,    // [N]
    float* __restrict__ state,      // [N][3][256]
    float* __restrict__ rewards,    // [N]
    float* __restrict__ last_logits,// [N][4]
    int* __restrict__ hit_count,    // [horizon]
    unsigned char* __restrict__ hits_g,  // [N]
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int EGRP = B / E_TILE;   // env groups
  constexpr int TPB = EGRP * 64;     // threads per block
  extern __shared__ float sm[];
  SmemLayout S = smem_layout(sm, B);
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int env_grp = tid / 64;         // 8 envs per group
  const int comp_grp = tid % 64;        // 2 comps per group
  const int e0 = env_grp * E_TILE;
  const int c0 = comp_grp * C_TILE;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      // ---------------- stage agent/food + obs + encoder into Ha
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        S.agent[tid * 2] = ax; S.agent[tid * 2 + 1] = ay;
        S.food[tid * 2] = fx; S.food[tid * 2 + 1] = fy;
        S.obs[tid * 4 + 0] = (fx - ax) / 11.f;
        S.obs[tid * 4 + 1] = (fy - ay) / 11.f;
        S.obs[tid * 4 + 2] = ax / 10.f;
        S.obs[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      // encoder: Ha[comp][env] = be[comp] + sum_c obs[env][c]*We[comp][c]
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += S.obs[e * 4 + 0] * wr[0];
        acc += S.obs[e * 4 + 1] * wr[1];
        acc += S.obs[e * 4 + 2] * wr[2];
        acc += S.obs[e * 4 + 3] * wr[3];
        S.Ha[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = S.Ha;
      float* Hout = S.Hb;

      // ---------------- 3 GRU layers
      for (int l = 0; l < 3; ++l) {
        const float* Wl = DUMMY_W;  // probe: always-tiny-hot window
        for (int p = 0; p < 2; ++p) {
          // gate GEMM: rows [p*384, p*384+384) of Wl, k in [0,256)
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[E_TILE][C_TILE][3];
#pragma unroll
          for (int ei = 0; ei < E_TILE; ++ei)
#pragma unroll
            for (int ci = 0; ci < C_TILE; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          // prologue: stage kt=0 (8 rows of 384 floats, src row stride 768)
          {
            const float4* src = (const float4*)(Wseg);
            float4* dst = (float4*)(S.Wst);
            for (int i = tid; i < K_TILE * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < 256 / K_TILE; ++kt) {
            // prefetch next
            if (kt + 1 < 256 / K_TILE) {
              const float4* src = (const float4*)(Wseg + (size_t)(kt + 1) * K_TILE * 768);
              float4* dst = (float4*)(S.Wst + ((kt + 1) & 1) * K_TILE * PASS_ROWS);
              for (int i = tid; i < K_TILE * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
            }
            cp_commit();
            cp_wait1();
            __syncthreads();
            const float* wseg_sm = S.Wst + (kt & 1) * K_TILE * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
              int k = kt * K_TILE + kk;
              const float4 a_lo = *(const float4*)&Hin[k * HP + e0];
              const float4 a_hi = *(const float4*)&Hin[k * HP + e0 + 4];
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float2 b_zh = *(const float2*)(wrow);
              const float2 b_zg = *(const float2*)(wrow + 128);
              const float2 b_zp = *(const float2*)(wrow + 256);
              float av;
              av = a_lo.x;
              acc[0][0][0] += av * b_zh.x; acc[0][1][0] += av * b_zh.y;
              acc[0][0][1] += av * b_zg.x; acc[0][1][1] += av * b_zg.y;
              acc[0][0][2] += av * b_zp.x; acc[0][1][2] += av * b_zp.y;
              av = a_lo.y;
              acc[1][0][0] += av * b_zh.x; acc[1][1][0] += av * b_zh.y;
              acc[1][0][1] += av * b_zg.x; acc[1][1][1] += av * b_zg.y;
              acc[1][0][2] += av * b_zp.x; acc[1][1][2] += av * b_zp.y;
              av = a_lo.z;
              acc[2][0][0] += av * b_zh.x; acc[2][1][0] += av * b_zh.y;
              acc[2][0][1] += av * b_zg.x; acc[2][1][1] += av * b_zg.y;
              acc[2][0][2] += av * b_zp.x; acc[2][1][2] += av * b_zp.y;
              av = a_lo.w;
              acc[3][0][0] += av * b_zh.x; acc[3][1][0] += av * b_zh.y;
              acc[3][0][1] += av * b_zg.x; acc[3][1][1] += av * b_zg.y;
              acc[3][0][2] += av * b_zp.x; acc[3][1][2] += av * b_zp.y;
              av = a_hi.x;
              acc[4][0][0] += av * b_zh.x; acc[4][1][0] += av * b_zh.y;
              acc[4][0][1] += av * b_zg.x; acc[4][1][1] += av * b_zg.y;
              acc[4][0][2] += av * b_zp.x; acc[4][1][2] += av * b_zp.y;
              av = a_hi.y;
              acc[5][0][0] += av * b_zh.x; acc[5][1][0] += av * b_zh.y;
              acc[5][0][1] += av * b_zg.x; acc[5][1][1] += av * b_zg.y;
              acc[5][0][2] += av * b_zp.x; acc[5][1][2] += av * b_zp.y;
              av = a_hi.z;
              acc[6][0][0] += av * b_zh.x; acc[6][1][0] += av * b_zh.y;
              acc[6][0][1] += av * b_zg.x; acc[6][1][1] += av * b_zg.y;
              acc[6][0][2] += av * b_zp.x; acc[6][1][2] += av * b_zp.y;
              av = a_hi.w;
              acc[7][0][0] += av * b_zh.x; acc[7][1][0] += av * b_zh.y;
              acc[7][0][1] += av * b_zg.x; acc[7][1][1] += av * b_zg.y;
              acc[7][0][2] += av * b_zp.x; acc[7][1][2] += av * b_zp.y;
            }
            __syncthreads();
          }
          cp_wait0();
          // elementwise GRU update for this pass's components
#pragma unroll
          for (int ci = 0; ci < C_TILE; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < E_TILE; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hout[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        const float* tmp = Hin; Hin = Hout; Hout = (float*)tmp;
      }

      // ---------------- heads: logits = Hf x Wa^T + ba
      // Hf == Hin after the final swap
      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Hin[k * HP + e] * wr[k];
        S.logits[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = S.logits[tid * 4], l1 = S.logits[tid * 4 + 1],
                l2 = S.logits[tid * 4 + 2], l3 = S.logits[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          S.action[tid] = act;
        }
      }
      __syncthreads();

      // ---------------- env move
      if (tid == 0) S.flag[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = S.agent[tid * 2], ay = S.agent[tid * 2 + 1];
          int act = S.action[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          S.agent[tid * 2] = ax; S.agent[tid * 2 + 1] = ay;
          bool hit = (ax == S.food[tid * 2]) && (ay == S.food[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; S.flag[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && S.flag[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    // ---------------- respawn (all my chunks)
    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}


template <int B>
__global__ void __launch_bounds__((B / E_TILE) * 64) rollout_v2(
    const float* __restrict__ Wt,   // [3][256][768] pass-permuted transpose
    const float* __restrict__ We,   // [256][4]
    const float* __restrict__ be,   // [256]
    const float* __restrict__ Wa,   // [4][256]
    const float* __restrict__ ba,   // [4]
    float* __restrict__ agent,      // [N][2]
    float* __restrict__ food,       // [N][2]
    long long* __restrict__ rng,    // [N]
    float* __restrict__ state,      // [N][3][256]
    float* __restrict__ rewards,    // [N]
    float* __restrict__ last_logits,// [N][4]
    int* __restrict__ hit_count,    // [horizon]
    unsigned char* __restrict__ hits_g,  // [N]
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int EGRP = B / E_TILE;   // env groups
  constexpr int TPB = EGRP * 64;     // threads per block
  extern __shared__ float sm[];
  SmemLayout S = smem_layout(sm, B);
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int env_grp = tid / 64;         // 8 envs per group
  const int comp_grp = tid % 64;        // 2 comps per group
  const int e0 = env_grp * E_TILE;
  const int c0 = comp_grp * C_TILE;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      // ---------------- stage agent/food + obs + encoder into Ha
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        S.agent[tid * 2] = ax; S.agent[tid * 2 + 1] = ay;
        S.food[tid * 2] = fx; S.food[tid * 2 + 1] = fy;
        S.obs[tid * 4 + 0] = (fx - ax) / 11.f;
        S.obs[tid * 4 + 1] = (fy - ay) / 11.f;
        S.obs[tid * 4 + 2] = ax / 10.f;
        S.obs[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      // encoder: Ha[comp][env] = be[comp] + sum_c obs[env][c]*We[comp][c]
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += S.obs[e * 4 + 0] * wr[0];
        acc += S.obs[e * 4 + 1] * wr[1];
        acc += S.obs[e * 4 + 2] * wr[2];
        acc += S.obs[e * 4 + 3] * wr[3];
        S.Ha[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = S.Ha;
      float* Hout = S.Hb;

      // ---------------- 3 GRU layers
      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          // gate GEMM: rows [p*384, p*384+384) of Wl, k in [0,256)
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[E_TILE][C_TILE][3];
#pragma unroll
          for (int ei = 0; ei < E_TILE; ++ei)
#pragma unroll
            for (int ci = 0; ci < C_TILE; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          // prologue: stage kt=0 (8 rows of 384 floats, src row stride 768)
          {
            const float4* src = (const float4*)(Wseg);
            float4* dst = (float4*)(S.Wst);
            for (int i = tid; i < K_TILE * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < 256 / K_TILE; ++kt) {
            // prefetch next
            if (kt + 1 < 256 / K_TILE) {
              const float4* src = (const float4*)(Wseg + (size_t)(kt + 1) * K_TILE * 768);
              float4* dst = (float4*)(S.Wst + ((kt + 1) & 1) * K_TILE * PASS_ROWS);
              for (int i = tid; i < K_TILE * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
            }
            cp_commit();
            cp_wait1();
            __syncthreads();
            const float* wseg_sm = S.Wst + (kt & 1) * K_TILE * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
              int k = kt * K_TILE + kk;
              const float4 a_lo = *(const float4*)&Hin[k * HP + e0];
              const float4 a_hi = *(const float4*)&Hin[k * HP + e0 + 4];
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float2 b_zh = *(const float2*)(wrow);
              const float2 b_zg = *(const float2*)(wrow + 128);
              const float2 b_zp = *(const float2*)(wrow + 256);
              float av;
              av = a_lo.x;
              acc[0][0][0] += av * b_zh.x; acc[0][1][0] += av * b_zh.y;
              acc[0][0][1] += av * b_zg.x; acc[0][1][1] += av * b_zg.y;
              acc[0][0][2] += av * b_zp.x; acc[0][1][2] += av * b_zp.y;
              av = a_lo.y;
              acc[1][0][0] += av * b_zh.x; acc[1][1][0] += av * b_zh.y;
              acc[1][0][1] += av * b_zg.x; acc[1][1][1] += av * b_zg.y;
              acc[1][0][2] += av * b_zp.x; acc[1][1][2] += av * b_zp.y;
              av = a_lo.z;
              acc[2][0][0] += av * b_zh.x; acc[2][1][0] += av * b_zh.y;
              acc[2][0][1] += av * b_zg.x; acc[2][1][1] += av * b_zg.y;
              acc[2][0][2] += av * b_zp.x; acc[2][1][2] += av * b_zp.y;
              av = a_lo.w;
              acc[3][0][0] += av * b_zh.x; acc[3][1][0] += av * b_zh.y;
              acc[3][0][1] += av * b_zg.x; acc[3][1][1] += av * b_zg.y;
              acc[3][0][2] += av * b_zp.x; acc[3][1][2] += av * b_zp.y;
              av = a_hi.x;
              acc[4][0][0] += av * b_zh.x; acc[4][1][0] += av * b_zh.y;
              acc[4][0][1] += av * b_zg.x; acc[4][1][1] += av * b_zg.y;
              acc[4][0][2] += av * b_zp.x; acc[4][1][2] += av * b_zp.y;
              av = a_hi.y;
              acc[5][0][0] += av * b_zh.x; acc[5][1][0] += av * b_zh.y;
              acc[5][0][1] += av * b_zg.x; acc[5][1][1] += av * b_zg.y;
              acc[5][0][2] += av * b_zp.x; acc[5][1][2] += av * b_zp.y;
              av = a_hi.z;
              acc[6][0][0] += av * b_zh.x; acc[6][1][0] += av * b_zh.y;
              acc[6][0][1] += av * b_zg.x; acc[6][1][1] += av * b_zg.y;
              acc[6][0][2] += av * b_zp.x; acc[6][1][2] += av * b_zp.y;
              av = a_hi.w;
              acc[7][0][0] += av * b_zh.x; acc[7][1][0] += av * b_zh.y;
              acc[7][0][1] += av * b_zg.x; acc[7][1][1] += av * b_zg.y;
              acc[7][0][2] += av * b_zp.x; acc[7][1][2] += av * b_zp.y;
            }
            __syncthreads();
          }
          cp_wait0();
          // elementwise GRU update for this pass's components
#pragma unroll
          for (int ci = 0; ci < C_TILE; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < E_TILE; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hout[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        const float* tmp = Hin; Hin = Hout; Hout = (float*)tmp;
      }

      // ---------------- heads: logits = Hf x Wa^T + ba
      // Hf == Hin after the final swap
      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Hin[k * HP + e] * wr[k];
        S.logits[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = S.logits[tid * 4], l1 = S.logits[tid * 4 + 1],
                l2 = S.logits[tid * 4 + 2], l3 = S.logits[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          S.action[tid] = act;
        }
      }
      __syncthreads();

      // ---------------- env move
      if (tid == 0) S.flag[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = S.agent[tid * 2], ay = S.agent[tid * 2 + 1];
          int act = S.action[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          S.agent[tid * 2] = ax; S.agent[tid * 2 + 1] = ay;
          bool hit = (ax == S.food[tid * 2]) && (ay == S.food[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; S.flag[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && S.flag[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    // ---------------- respawn (all my chunks)
    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}


// ============================================================ V3 rollout
// Unified template: B envs/block, TPB = (B/4)*64*KSP threads, E_TILE=4, C_TILE=2.
// HMODE: 0 -> Hout to per-block gmem scratch + restage into Ha each layer;
//        1 -> smem ping-pong Ha/Hb (no restage).
// WMODE: 0 -> W k-tiles staged with cp.async double buffer;
//        1 -> B-frags read straight from gmem (__ldg), no stage/syncs.
// KSP: k-range split KSP ways with __shfl_xor reduction (KSP=2 doubles threads).
// state = gmem RMW (state is too big for registers when blocks own many chunks).

template <int B, int KSP, int HMODE, int WMODE, int E_TILE_V = 4>
__global__ void __launch_bounds__((B / E_TILE_V) * 64 * KSP) rollout_v3(
    const float* __restrict__ Wt,
    const float* __restrict__ We,
    const float* __restrict__ be,
    const float* __restrict__ Wa,
    const float* __restrict__ ba,
    float* __restrict__ agent,
    float* __restrict__ food,
    long long* __restrict__ rng,
    float* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ last_logits,
    int* __restrict__ hit_count,
    unsigned char* __restrict__ hits_g,
    float* __restrict__ hscr_g,
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int KV = (KSP == 2) ? 4 : 8;
  constexpr int NSPL = KSP;
  constexpr int TPB = (B / E_TILE_V) * 64 * NSPL;
  constexpr int WSTAGE = (WMODE == 0) ? (KSP * 2 * KV * PASS_ROWS) : 0;
  constexpr int HBUF = (HMODE == 1) ? 2 : 1;
  extern __shared__ float sm[];
  float* Ha_s = sm;
  float* Hb_s = Ha_s + 256 * HP;
  float* Wst_s = Ha_s + HBUF * 256 * HP;
  float* obs_s = Wst_s + WSTAGE;
  float* agent_s = obs_s + B * 4;
  float* food_s = agent_s + B * 2;
  float* logits_s = food_s + B * 2;
  int* action_s = (int*)(logits_s + B * 4);
  int* flag_s = action_s + B;
  cg::grid_group grid = cg::this_grid();
  float* Hscr = hscr_g + (size_t)blockIdx.x * 256 * HP;

  const int tid = threadIdx.x;
  const int ksp = (NSPL == 1) ? 0 : (tid % NSPL);
  const int cg2 = (tid / NSPL) & 63;
  const int eg = tid / (NSPL * 64);
  const int e0 = eg * E_TILE_V;
  const int c0 = cg2 * 2;
  const int kbase = ksp * (256 / NSPL);
  const int KT_iters = (WMODE == 0) ? (256 / NSPL / KV) : 0;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        agent_s[tid * 2] = ax; agent_s[tid * 2 + 1] = ay;
        food_s[tid * 2] = fx; food_s[tid * 2 + 1] = fy;
        obs_s[tid * 4 + 0] = (fx - ax) / 11.f;
        obs_s[tid * 4 + 1] = (fy - ay) / 11.f;
        obs_s[tid * 4 + 2] = ax / 10.f;
        obs_s[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += obs_s[e * 4 + 0] * wr[0];
        acc += obs_s[e * 4 + 1] * wr[1];
        acc += obs_s[e * 4 + 2] * wr[2];
        acc += obs_s[e * 4 + 3] * wr[3];
        Ha_s[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = Ha_s;
      float* Hout = (HMODE == 1) ? Hb_s : Hscr;

      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[E_TILE_V][2][3];
#pragma unroll
          for (int ei = 0; ei < E_TILE_V; ++ei)
#pragma unroll
            for (int ci = 0; ci < 2; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          if (WMODE == 0) {
            // stage kt=0 for all splits
            for (int i = tid; i < NSPL * KV * (PASS_ROWS / 4); i += TPB) {
              int side = i / (KV * (PASS_ROWS / 4));
              int j = i % (KV * (PASS_ROWS / 4));
              int row = j / (PASS_ROWS / 4);
              int c4 = j % (PASS_ROWS / 4);
              const float4* src = (const float4*)(Wseg + (size_t)(side * (256 / NSPL)) * 768);
              float4* dst = (float4*)(Wst_s + side * KV * PASS_ROWS);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
#pragma unroll 1
            for (int kt = 0; kt < KT_iters; ++kt) {
              if (kt + 1 < KT_iters) {
                for (int i = tid; i < NSPL * KV * (PASS_ROWS / 4); i += TPB) {
                  int side = i / (KV * (PASS_ROWS / 4));
                  int j = i % (KV * (PASS_ROWS / 4));
                  int row = j / (PASS_ROWS / 4);
                  int c4 = j % (PASS_ROWS / 4);
                  const float4* src = (const float4*)(Wseg + (size_t)(side * (256 / NSPL) + (kt + 1) * KV) * 768);
                  float4* dst = (float4*)(Wst_s + ((kt + 1) & 1) * NSPL * KV * PASS_ROWS + side * KV * PASS_ROWS);
                  cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                             src + row * (768 / 4) + c4);
                }
              }
              cp_commit();
              cp_wait1();
              __syncthreads();
              const float* wseg_sm = Wst_s + ((kt & 1) * NSPL + ksp) * KV * PASS_ROWS;
#pragma unroll
              for (int kk = 0; kk < KV; ++kk) {
                int k = kbase + kt * KV + kk;
                const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
                const float2 b_zh = *(const float2*)(wrow);
                const float2 b_zg = *(const float2*)(wrow + 128);
                const float2 b_zp = *(const float2*)(wrow + 256);
                {
                float avs[E_TILE_V];
#pragma unroll
                for (int z = 0; z < E_TILE_V / 4; ++z) {
                  const float4 a4 = *(const float4*)&Hin[k * HP + e0 + z * 4];
                  avs[z * 4] = a4.x; avs[z * 4 + 1] = a4.y;
                  avs[z * 4 + 2] = a4.z; avs[z * 4 + 3] = a4.w;
                }
#pragma unroll
                for (int ei = 0; ei < E_TILE_V; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                }
              }
              }
              __syncthreads();
            }
            cp_wait0();
          } else {
            // WMODE==1: direct gmem B-frags, no syncthreads in k loop
#pragma unroll 1
            for (int kt = 0; kt < 256 / NSPL / 4; ++kt) {
#pragma unroll
              for (int kq = 0; kq < 4; ++kq) {
                int k = kbase + kt * 4 + kq;
                const float* wrow = Wseg + (size_t)k * 768 + c0;
                const float2 b_zh = *(const float2*)(wrow);
                const float2 b_zg = *(const float2*)(wrow + 128);
                const float2 b_zp = *(const float2*)(wrow + 256);
                {
                float avs[E_TILE_V];
#pragma unroll
                for (int z = 0; z < E_TILE_V / 4; ++z) {
                  const float4 a4 = *(const float4*)&Hin[k * HP + e0 + z * 4];
                  avs[z * 4] = a4.x; avs[z * 4 + 1] = a4.y;
                  avs[z * 4 + 2] = a4.z; avs[z * 4 + 3] = a4.w;
                }
#pragma unroll
                for (int ei = 0; ei < E_TILE_V; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                }
              }
            }
          }
          }
          if (NSPL == 2) {
#pragma unroll
            for (int ei = 0; ei < E_TILE_V; ++ei)
#pragma unroll
              for (int ci = 0; ci < 2; ++ci)
#pragma unroll
                for (int g = 0; g < 3; ++g)
                  acc[ei][ci][g] += __shfl_xor_sync(0xffffffffu, acc[ei][ci][g], 1);
          }
          // elementwise
#pragma unroll
          for (int ci = (NSPL == 2 ? ksp : 0); ci < (NSPL == 2 ? ksp + 1 : 2); ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < E_TILE_V; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hout[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        if (HMODE == 1) {
          const float* tmp = Hin; Hin = Hout; Hout = (float*)tmp;
        } else {
          __threadfence_block();
          for (int i = tid * 4; i < 256 * HP; i += TPB * 4) {
            cp_async16(&Ha_s[i], &Hscr[i]);
          }
          cp_commit();
          cp_wait0();
          __syncthreads();
        }
      }
      const float* Hfin = (HMODE == 1) ? Hin : Ha_s;

      // heads
      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Hfin[k * HP + e] * wr[k];
        logits_s[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = logits_s[tid * 4], l1 = logits_s[tid * 4 + 1],
                l2 = logits_s[tid * 4 + 2], l3 = logits_s[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          action_s[tid] = act;
        }
      }
      __syncthreads();

      if (tid == 0) flag_s[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = agent_s[tid * 2], ay = agent_s[tid * 2 + 1];
          int act = action_s[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          agent_s[tid * 2] = ax; agent_s[tid * 2 + 1] = ay;
          bool hit = (ax == food_s[tid * 2]) && (ay == food_s[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; flag_s[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && flag_s[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}



// ============================================================ rollout_pipe
// rollout_v2 geometry (B=32, E_TILE=8, TPB=256, smem ping-pong) but with a
// 3-deep cp.async W pipeline (KV=4): 3 tiles in flight covers DRAM latency.
template <int B>
__global__ void __launch_bounds__((B / 8) * 64) rollout_pipe(
    const float* __restrict__ Wt,
    const float* __restrict__ We,
    const float* __restrict__ be,
    const float* __restrict__ Wa,
    const float* __restrict__ ba,
    float* __restrict__ agent,
    float* __restrict__ food,
    long long* __restrict__ rng,
    float* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ last_logits,
    int* __restrict__ hit_count,
    unsigned char* __restrict__ hits_g,
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int TPB = (B / 8) * 64;
  constexpr int KV_P = 4;
  constexpr int NBUF = 3;
  constexpr int WSTAGE = NBUF * KV_P * PASS_ROWS;
  extern __shared__ float sm[];
  float* Ha = sm;
  float* Hb = Ha + 256 * HP;
  float* Wst = Hb + 256 * HP;
  float* obs = Wst + WSTAGE;
  float* agent_sm = obs + B * 4;
  float* food_sm = agent_sm + B * 2;
  float* logits_sm = food_sm + B * 2;
  int* action_sm = (int*)(logits_sm + B * 4);
  int* flag_sm = action_sm + B;
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int env_grp = tid / 64;
  const int comp_grp = tid % 64;
  const int e0 = env_grp * 8;
  const int c0 = comp_grp * 2;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
        food_sm[tid * 2] = fx; food_sm[tid * 2 + 1] = fy;
        obs[tid * 4 + 0] = (fx - ax) / 11.f;
        obs[tid * 4 + 1] = (fy - ay) / 11.f;
        obs[tid * 4 + 2] = ax / 10.f;
        obs[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += obs[e * 4 + 0] * wr[0];
        acc += obs[e * 4 + 1] * wr[1];
        acc += obs[e * 4 + 2] * wr[2];
        acc += obs[e * 4 + 3] * wr[3];
        Ha[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = Ha;
      float* Hout = Hb;

      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[8][2][3];
#pragma unroll
          for (int ei = 0; ei < 8; ++ei)
#pragma unroll
            for (int ci = 0; ci < 2; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          // prologue: prefetch tiles 0..NBUF-2
#pragma unroll
          for (int pf = 0; pf < NBUF - 1; ++pf) {
            for (int i = tid; i < KV_P * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              const float4* src = (const float4*)(Wseg + (size_t)pf * KV_P * 768);
              float4* dst = (float4*)(Wst + pf * KV_P * PASS_ROWS);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < 256 / KV_P; ++kt) {
            if (kt + NBUF - 1 < 256 / KV_P) {
              for (int i = tid; i < KV_P * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                const float4* src = (const float4*)(Wseg + (size_t)(kt + NBUF - 1) * KV_P * 768);
                float4* dst = (float4*)(Wst + ((kt + NBUF - 1) % NBUF) * KV_P * PASS_ROWS);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
            }
            cp_commit();
            cp_wait2();
            __syncthreads();
            const float* wseg_sm = Wst + (kt % NBUF) * KV_P * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < KV_P; ++kk) {
              int k = kt * KV_P + kk;
              const float4 a_lo = *(const float4*)&Hin[k * HP + e0];
              const float4 a_hi = *(const float4*)&Hin[k * HP + e0 + 4];
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float2 b_zh = *(const float2*)(wrow);
              const float2 b_zg = *(const float2*)(wrow + 128);
              const float2 b_zp = *(const float2*)(wrow + 256);
              {
                float avs[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                a_hi.x, a_hi.y, a_hi.z, a_hi.w};
#pragma unroll
                for (int ei = 0; ei < 8; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                }
              }
            }
            __syncthreads();
          }
          cp_wait0();
          // elementwise
#pragma unroll
          for (int ci = 0; ci < 2; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hout[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        const float* tmp = Hin; Hin = Hout; Hout = (float*)tmp;
      }

      // heads
      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Hin[k * HP + e] * wr[k];
        logits_sm[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = logits_sm[tid * 4], l1 = logits_sm[tid * 4 + 1],
                l2 = logits_sm[tid * 4 + 2], l3 = logits_sm[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          action_sm[tid] = act;
        }
      }
      __syncthreads();

      if (tid == 0) flag_sm[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = agent_sm[tid * 2], ay = agent_sm[tid * 2 + 1];
          int act = action_sm[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
          bool hit = (ax == food_sm[tid * 2]) && (ay == food_sm[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; flag_sm[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && flag_sm[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}



// ============================================================ rollout_p1
// v2 geometry (B=32, E_TILE=8, TPB=256, ping-pong) with ring pipeline:
// prefetch issued AFTER (wait + syncthreads), so one sync per k-tile is safe
// (the sync proves all warps finished the previous tile's reads).
template <int B, int NBUF, int KV_C>
__global__ void __launch_bounds__((B / 8) * 64) rollout_p1(
    const float* __restrict__ Wt,
    const float* __restrict__ We,
    const float* __restrict__ be,
    const float* __restrict__ Wa,
    const float* __restrict__ ba,
    float* __restrict__ agent,
    float* __restrict__ food,
    long long* __restrict__ rng,
    float* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ last_logits,
    int* __restrict__ hit_count,
    unsigned char* __restrict__ hits_g,
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int TPB = (B / 8) * 64;
  constexpr int WSTAGE = NBUF * KV_C * PASS_ROWS;
  constexpr int KTOT = 256 / KV_C;
  extern __shared__ float sm[];
  float* Ha = sm;
  float* Hb = Ha + 256 * HP;
  float* Wst = Hb + 256 * HP;
  float* obs = Wst + WSTAGE;
  float* agent_sm = obs + B * 4;
  float* food_sm = agent_sm + B * 2;
  float* logits_sm = food_sm + B * 2;
  int* action_sm = (int*)(logits_sm + B * 4);
  int* flag_sm = action_sm + B;
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int env_grp = tid / 64;
  const int comp_grp = tid % 64;
  const int e0 = env_grp * 8;
  const int c0 = comp_grp * 2;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
        food_sm[tid * 2] = fx; food_sm[tid * 2 + 1] = fy;
        obs[tid * 4 + 0] = (fx - ax) / 11.f;
        obs[tid * 4 + 1] = (fy - ay) / 11.f;
        obs[tid * 4 + 2] = ax / 10.f;
        obs[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += obs[e * 4 + 0] * wr[0];
        acc += obs[e * 4 + 1] * wr[1];
        acc += obs[e * 4 + 2] * wr[2];
        acc += obs[e * 4 + 3] * wr[3];
        Ha[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = Ha;
      float* Hout = Hb;

      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[8][2][3];
#pragma unroll
          for (int ei = 0; ei < 8; ++ei)
#pragma unroll
            for (int ci = 0; ci < 2; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          // prologue: issue tiles 0..NBUF-2
#pragma unroll
          for (int pf = 0; pf < NBUF - 1; ++pf) {
            for (int i = tid; i < KV_C * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              const float4* src = (const float4*)(Wseg + (size_t)pf * KV_C * 768);
              float4* dst = (float4*)(Wst + pf * KV_C * PASS_ROWS);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < KTOT; ++kt) {
            cp_wait1();
            __syncthreads();
            // now safe to overwrite the ring-oldest tile buffer
            if (kt + NBUF - 1 < KTOT) {
              for (int i = tid; i < KV_C * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                const float4* src = (const float4*)(Wseg + (size_t)(kt + NBUF - 1) * KV_C * 768);
                float4* dst = (float4*)(Wst + ((kt + NBUF - 1) % NBUF) * KV_C * PASS_ROWS);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
              cp_commit();
            }
            const float* wseg_sm = Wst + (kt % NBUF) * KV_C * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < KV_C; ++kk) {
              int k = kt * KV_C + kk;
              const float4 a_lo = *(const float4*)&Hin[k * HP + e0];
              const float4 a_hi = *(const float4*)&Hin[k * HP + e0 + 4];
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float2 b_zh = *(const float2*)(wrow);
              const float2 b_zg = *(const float2*)(wrow + 128);
              const float2 b_zp = *(const float2*)(wrow + 256);
              {
                float avs[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                a_hi.x, a_hi.y, a_hi.z, a_hi.w};
#pragma unroll
                for (int ei = 0; ei < 8; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                }
              }
            }
          }
          __syncthreads();
          // elementwise
#pragma unroll
          for (int ci = 0; ci < 2; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hout[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        const float* tmp = Hin; Hin = Hout; Hout = (float*)tmp;
      }

      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Hin[k * HP + e] * wr[k];
        logits_sm[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = logits_sm[tid * 4], l1 = logits_sm[tid * 4 + 1],
                l2 = logits_sm[tid * 4 + 2], l3 = logits_sm[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          action_sm[tid] = act;
        }
      }
      __syncthreads();

      if (tid == 0) flag_sm[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = agent_sm[tid * 2], ay = agent_sm[tid * 2 + 1];
          int act = action_sm[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
          bool hit = (ax == food_sm[tid * 2]) && (ay == food_sm[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; flag_sm[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && flag_sm[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}

// ============================================================ rollout_c4
// B=64, E_TILE=8, C_TILE=4, TPB=256: 96 gate-FMA per k per thread (~91% issue
// mix ceiling). gmem Hout + restage, staged W KV8, ldcs/stcs state.
template <int B>
__global__ void __launch_bounds__((B / 8) * 32) rollout_c4(
    const float* __restrict__ Wt,
    const float* __restrict__ We,
    const float* __restrict__ be,
    const float* __restrict__ Wa,
    const float* __restrict__ ba,
    float* __restrict__ agent,
    float* __restrict__ food,
    long long* __restrict__ rng,
    float* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ last_logits,
    int* __restrict__ hit_count,
    unsigned char* __restrict__ hits_g,
    float* __restrict__ hscr_g,
    int N, int horizon, int num_chunks) {
  constexpr int HP = HPAD(B);
  constexpr int TPB = (B / 8) * 32;
  constexpr int KV_P = 8;
  constexpr int WSTAGE = 2 * KV_P * PASS_ROWS;
  extern __shared__ float sm[];
  float* Ha = sm;
  float* Wst = Ha + 256 * HP;
  float* obs = Wst + WSTAGE;
  float* agent_sm = obs + B * 4;
  float* food_sm = agent_sm + B * 2;
  float* logits_sm = food_sm + B * 2;
  int* action_sm = (int*)(logits_sm + B * 4);
  int* flag_sm = action_sm + B;
  cg::grid_group grid = cg::this_grid();
  float* Hscr = hscr_g + (size_t)blockIdx.x * 256 * HP;

  const int tid = threadIdx.x;
  const int env_grp = tid / 32;
  const int comp_grp = tid % 32;
  const int e0 = env_grp * 8;
  const int c0 = comp_grp * 4;

  for (int t = 0; t < horizon; ++t) {
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
      const int eb = chunk * B;
      if (tid < B) {
        int ge = eb + tid;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
        food_sm[tid * 2] = fx; food_sm[tid * 2 + 1] = fy;
        obs[tid * 4 + 0] = (fx - ax) / 11.f;
        obs[tid * 4 + 1] = (fy - ay) / 11.f;
        obs[tid * 4 + 2] = ax / 10.f;
        obs[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      for (int idx = tid; idx < 256 * B; idx += TPB) {
        int e = idx % B;
        int comp = idx / B;
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += obs[e * 4 + 0] * wr[0];
        acc += obs[e * 4 + 1] * wr[1];
        acc += obs[e * 4 + 2] * wr[2];
        acc += obs[e * 4 + 3] * wr[3];
        Ha[comp * HP + e] = acc;
      }
      __syncthreads();

      const float* Hin = Ha;

      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[8][4][3];
#pragma unroll
          for (int ei = 0; ei < 8; ++ei)
#pragma unroll
            for (int ci = 0; ci < 4; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          {
            for (int i = tid; i < KV_P * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              const float4* src = (const float4*)(Wseg);
              float4* dst = (float4*)(Wst);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < 256 / KV_P; ++kt) {
            if (kt + 1 < 256 / KV_P) {
              for (int i = tid; i < KV_P * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                const float4* src = (const float4*)(Wseg + (size_t)(kt + 1) * KV_P * 768);
                float4* dst = (float4*)(Wst + ((kt + 1) & 1) * KV_P * PASS_ROWS);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
            }
            cp_commit();
            cp_wait1();
            __syncthreads();
            const float* wseg_sm = Wst + (kt & 1) * KV_P * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < KV_P; ++kk) {
              int k = kt * KV_P + kk;
              const float4 a_lo = *(const float4*)&Hin[k * HP + e0];
              const float4 a_hi = *(const float4*)&Hin[k * HP + e0 + 4];
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float4 b_zh = *(const float4*)(wrow);
              const float4 b_zg = *(const float4*)(wrow + 128);
              const float4 b_zp = *(const float4*)(wrow + 256);
              {
                float avs[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                a_hi.x, a_hi.y, a_hi.z, a_hi.w};
#pragma unroll
                for (int ei = 0; ei < 8; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][2][0] += av * b_zh.z; acc[ei][3][0] += av * b_zh.w;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][2][1] += av * b_zg.z; acc[ei][3][1] += av * b_zg.w;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                  acc[ei][2][2] += av * b_zp.z; acc[ei][3][2] += av * b_zp.w;
                }
              }
            }
            __syncthreads();
          }
          cp_wait0();
          // elementwise
#pragma unroll
          for (int ci = 0; ci < 4; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < 8; ++ei) {
              int ge = eb + e0 + ei;
              float st = 0.f;
              if (ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = Hin[comp * HP + e0 + ei];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              Hscr[comp * HP + e0 + ei] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        __threadfence_block();
        for (int i = tid * 4; i < 256 * HP; i += TPB * 4) {
          cp_async16(&Ha[i], &Hscr[i]);
        }
        cp_commit();
        cp_wait0();
        __syncthreads();
      }

      // heads
      for (int idx = tid; idx < B * 4; idx += TPB) {
        int e = idx % B;
        int a = idx / B;
        const float* wr = Wa + a * 256;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += Ha[k * HP + e] * wr[k];
        logits_sm[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float l0 = logits_sm[tid * 4], l1 = logits_sm[tid * 4 + 1],
                l2 = logits_sm[tid * 4 + 2], l3 = logits_sm[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          action_sm[tid] = act;
        }
      }
      __syncthreads();

      if (tid == 0) flag_sm[0] = 0;
      __syncthreads();
      if (tid < B) {
        int ge = eb + tid;
        if (ge < N) {
          float ax = agent_sm[tid * 2], ay = agent_sm[tid * 2 + 1];
          int act = action_sm[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          agent_sm[tid * 2] = ax; agent_sm[tid * 2 + 1] = ay;
          bool hit = (ax == food_sm[tid * 2]) && (ay == food_sm[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (hit) { rewards[ge] += 1.f; flag_sm[0] = 1; }
        }
      }
      __syncthreads();
      if (tid == 0 && flag_sm[0]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int eb = chunk * B;
        if (tid < B) {
          int ge = eb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }
}

// ============================================================ V8 dual rollout
// Two 32-env chunks per block share one W-stage stream: halves L2/DRAM weight
// traffic per env vs B=32 single-chunk. H inputs (Hin0/Hin1) in smem; H out to
// per-block gmem scratch + restage per layer. state via ldcs/stcs (zero at
// t==0, no pre-zero), rng seeded in-kernel, positions written at end.
template <int ET>
__global__ void __launch_bounds__(1024) rollout_dual(
    const float* __restrict__ Wt,
    const float* __restrict__ We,
    const float* __restrict__ be,
    const float* __restrict__ Wa,
    const float* __restrict__ ba,
    float* __restrict__ agent,
    float* __restrict__ food,
    long long* __restrict__ rng,
    float* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ last_logits,
    int* __restrict__ hit_count,
    unsigned char* __restrict__ hits_g,
    float* __restrict__ hscr_g,
    long long* __restrict__ pos_g,
    long long seed10007,
    int N, int horizon, int num_units) {
  constexpr int BC = 32;
  constexpr int HP = HPAD(BC);
  constexpr int KV = 8;
  constexpr int TPB = 1024;
  constexpr int WSTAGE = 2 * KV * PASS_ROWS;
  extern __shared__ float sm[];
  float* Ha_s = sm;                      // [2][256*HP]
  float* Wst_s = Ha_s + 2 * 256 * HP;
  float* agent_s = Wst_s + WSTAGE;       // [2*BC*2]
  float* food_s = agent_s + 2 * BC * 2;  // [2*BC*2]
  float* obs_s = food_s + 2 * BC * 2;    // [2*BC*4] (aliased for logits later)
  float* logits_s = obs_s;               // same region: obs dead after encoder
  int* action_s = (int*)(obs_s + 2 * BC * 4);  // [2*BC]
  int* flag_s = action_s + 2 * BC;       // [8]
  cg::grid_group grid = cg::this_grid();
  float* Hscr = hscr_g + (size_t)blockIdx.x * 2 * 256 * HP;

  const int tid = threadIdx.x;
  const int cg2 = tid & 63;
  const int eg = (tid >> 6) & 7;
  const int ch = tid >> 9;
  const int e0 = eg * ET;
  const int c0 = cg2 * 2;
  const float* HinC = Ha_s + ch * 256 * HP;
  float* HscrC = Hscr + ch * 256 * HP;

  for (int t = 0; t < horizon; ++t) {
    for (int unit = blockIdx.x; unit < num_units; unit += gridDim.x) {
      const int cb = unit * 2 * BC;
      if (tid < 2 * BC) {
        int cc = tid / BC, el = tid % BC;
        int ge = cb + cc * BC + el;
        float ax = 0.f, ay = 0.f, fx = 0.f, fy = 0.f;
        if (ge < N) {
          ax = agent[ge * 2]; ay = agent[ge * 2 + 1];
          fx = food[ge * 2]; fy = food[ge * 2 + 1];
        }
        agent_s[tid * 2] = ax; agent_s[tid * 2 + 1] = ay;
        food_s[tid * 2] = fx; food_s[tid * 2 + 1] = fy;
        obs_s[tid * 4 + 0] = (fx - ax) / 11.f;
        obs_s[tid * 4 + 1] = (fy - ay) / 11.f;
        obs_s[tid * 4 + 2] = ax / 10.f;
        obs_s[tid * 4 + 3] = ay / 10.f;
      }
      __syncthreads();
      for (int idx = tid; idx < 256 * 2 * BC; idx += TPB) {
        int e = idx % (2 * BC);
        int comp = idx / (2 * BC);
        const float* wr = We + comp * 4;
        float acc = be[comp];
        acc += obs_s[e * 4 + 0] * wr[0];
        acc += obs_s[e * 4 + 1] * wr[1];
        acc += obs_s[e * 4 + 2] * wr[2];
        acc += obs_s[e * 4 + 3] * wr[3];
        Ha_s[(e / BC) * 256 * HP + comp * HP + (e % BC)] = acc;
      }
      __syncthreads();

      for (int l = 0; l < 3; ++l) {
        const float* Wl = Wt + (size_t)l * 256 * 768;
        for (int p = 0; p < 2; ++p) {
          const float* Wseg = Wl + p * PASS_ROWS;
          float acc[ET][2][3];
#pragma unroll
          for (int ei = 0; ei < ET; ++ei)
#pragma unroll
            for (int ci = 0; ci < 2; ++ci)
#pragma unroll
              for (int g = 0; g < 3; ++g) acc[ei][ci][g] = 0.f;

          {
            for (int i = tid; i < KV * (PASS_ROWS / 4); i += TPB) {
              int row = i / (PASS_ROWS / 4);
              int c4 = i % (PASS_ROWS / 4);
              const float4* src = (const float4*)(Wseg);
              float4* dst = (float4*)(Wst_s);
              cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                         src + row * (768 / 4) + c4);
            }
            cp_commit();
          }
#pragma unroll 1
          for (int kt = 0; kt < (256 / KV); ++kt) {
            if (kt + 1 < (256 / KV)) {
              for (int i = tid; i < KV * (PASS_ROWS / 4); i += TPB) {
                int row = i / (PASS_ROWS / 4);
                int c4 = i % (PASS_ROWS / 4);
                const float4* src = (const float4*)(Wseg + (size_t)(kt + 1) * KV * 768);
                float4* dst = (float4*)(Wst_s + ((kt + 1) & 1) * KV * PASS_ROWS);
                cp_async16(dst + row * (PASS_ROWS / 4) + c4,
                           src + row * (768 / 4) + c4);
              }
            }
            cp_commit();
            cp_wait1();
            __syncthreads();
            const float* wseg_sm = Wst_s + (kt & 1) * KV * PASS_ROWS;
#pragma unroll
            for (int kk = 0; kk < KV; ++kk) {
              int k = kt * KV + kk;
              const float* wrow = wseg_sm + kk * PASS_ROWS + c0;
              const float2 b_zh = *(const float2*)(wrow);
              const float2 b_zg = *(const float2*)(wrow + 128);
              const float2 b_zp = *(const float2*)(wrow + 256);
              {
                float avs[ET];
#pragma unroll
                for (int z = 0; z < ET / 4; ++z) {
                  const float4 a4 = *(const float4*)&HinC[k * HP + e0 + z * 4];
                  avs[z * 4] = a4.x; avs[z * 4 + 1] = a4.y;
                  avs[z * 4 + 2] = a4.z; avs[z * 4 + 3] = a4.w;
                }
#pragma unroll
                for (int ei = 0; ei < ET; ++ei) {
                  float av = avs[ei];
                  acc[ei][0][0] += av * b_zh.x; acc[ei][1][0] += av * b_zh.y;
                  acc[ei][0][1] += av * b_zg.x; acc[ei][1][1] += av * b_zg.y;
                  acc[ei][0][2] += av * b_zp.x; acc[ei][1][2] += av * b_zp.y;
                }
              }
            }
            __syncthreads();
          }
          cp_wait0();
          // elementwise
#pragma unroll
          for (int ci = 0; ci < 2; ++ci) {
            int comp = p * 128 + c0 + ci;
#pragma unroll
            for (int ei = 0; ei < ET; ++ei) {
              int ge = cb + ch * BC + e0 + ei;
              float st = 0.f;
              if (t != 0 && ge < N) st = ldcs(&state[((size_t)ge * 3 + l) * 256 + comp]);
              float hin = HinC[comp * HP + (e0 + ei)];
              float out = st + sig(acc[ei][ci][1]) * (tanhf(acc[ei][ci][0]) - st);
              float pp = sig(acc[ei][ci][2]);
              HscrC[comp * HP + (e0 + ei)] = pp * out + (1.0f - pp) * hin;
              if (ge < N) stcs(&state[((size_t)ge * 3 + l) * 256 + comp], out);
            }
          }
          __syncthreads();
        }
        __threadfence_block();
        for (int i = tid * 4; i < 2 * 256 * HP; i += TPB * 4) {
          cp_async16(&Ha_s[i], &Hscr[i]);
        }
        cp_commit();
        cp_wait0();
        __syncthreads();
      }

      // heads for both chunks
      for (int idx = tid; idx < 2 * BC * 4; idx += TPB) {
        int e = idx % (2 * BC);
        int a = idx / (2 * BC);
        const float* wr = Wa + a * 256;
        const float* HinH = Ha_s + (e / BC) * 256 * HP;
        int el = e % BC;
        float acc = ba[a];
        for (int k = 0; k < 256; ++k) acc += HinH[k * HP + el] * wr[k];
        logits_s[e * 4 + a] = acc;
      }
      __syncthreads();
      if (tid < 2 * BC) {
        int ge = cb + tid;
        if (ge < N) {
          float l0 = logits_s[tid * 4], l1 = logits_s[tid * 4 + 1],
                l2 = logits_s[tid * 4 + 2], l3 = logits_s[tid * 4 + 3];
          if (t == horizon - 1) {
            last_logits[ge * 4] = l0; last_logits[ge * 4 + 1] = l1;
            last_logits[ge * 4 + 2] = l2; last_logits[ge * 4 + 3] = l3;
          }
          int act = 0;
          float best = l0;
          if (l1 > best) { best = l1; act = 1; }
          if (l2 > best) { best = l2; act = 2; }
          if (l3 > best) { best = l3; act = 3; }
          action_s[tid] = act;
        }
      }
      __syncthreads();

      if (tid < 2) flag_s[tid] = 0;
      __syncthreads();
      if (tid < 2 * BC) {
        int cc = tid / BC;
        int ge = cb + tid;
        if (ge < N) {
          float ax = agent_s[tid * 2], ay = agent_s[tid * 2 + 1];
          int act = action_s[tid];
          float dx = (act == 2) ? -1.f : (act == 3) ? 1.f : 0.f;
          float dy = (act == 0) ? -1.f : (act == 1) ? 1.f : 0.f;
          ax = fminf(fmaxf(ax + dx, 0.f), 10.f);
          ay = fminf(fmaxf(ay + dy, 0.f), 10.f);
          agent[ge * 2] = ax;
          agent[ge * 2 + 1] = ay;
          agent_s[tid * 2] = ax; agent_s[tid * 2 + 1] = ay;
          bool hit = (ax == food_s[tid * 2]) && (ay == food_s[tid * 2 + 1]);
          hits_g[ge] = hit ? 1 : 0;
          if (t == 0) {
            rewards[ge] = hit ? 1.f : 0.f;
            rng[ge] = (long long)((unsigned long long)ge + (unsigned long long)seed10007);
          } else if (hit) rewards[ge] += 1.f;
          if (hit) flag_s[cc] = 1;
        }
      }
      __syncthreads();
      if (tid < 2 && flag_s[tid]) atomicAdd(&hit_count[t], 1);
    }

    grid.sync();

    bool any = __ldcg(&hit_count[t]) > 0;
    if (any) {
      for (int unit = blockIdx.x; unit < num_units; unit += gridDim.x) {
        int cb = unit * 2 * BC;
        if (tid < 2 * BC) {
          int ge = cb + tid;
          if (ge < N) {
            unsigned long long r = (unsigned long long)rng[ge];
            unsigned long long r1 = lcg_step(r);
            unsigned long long r2 = lcg_step(r1);
            rng[ge] = (long long)r2;
            if (hits_g[ge]) {
              food[ge * 2] = (float)(long long)(r1 % BOARD_D);
              food[ge * 2 + 1] = (float)(long long)(r2 % BOARD_D);
            }
          }
        }
      }
    }
  }

  // final positions
  for (int unit = blockIdx.x; unit < num_units; unit += gridDim.x) {
    int cb = unit * 2 * BC;
    if (tid < 2 * BC) {
      int ge = cb + tid;
      if (ge < N) {
        pos_g[ge * 2] = (long long)rintf(agent[ge * 2]);
        pos_g[ge * 2 + 1] = (long long)rintf(agent[ge * 2 + 1]);
      }
    }
  }
}

// ============================================================ host
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

void policy_forward_cuda(torch::Tensor w_enc, torch::Tensor b_enc,
                         torch::Tensor w_gru,
                         torch::Tensor w_a, torch::Tensor b_a,
                         torch::Tensor w_v, torch::Tensor b_v,
                         torch::Tensor obs, torch::Tensor state,
                         torch::Tensor state_out, torch::Tensor logits,
                         torch::Tensor value) {
  int N = obs.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  int tpb = 128;
  policy_forward_kernel<<<cdiv(N, tpb), tpb, 0, stream>>>(
      w_enc.data_ptr<float>(), b_enc.data_ptr<float>(), w_gru.data_ptr<float>(),
      w_a.data_ptr<float>(), b_a.data_ptr<float>(), w_v.data_ptr<float>(),
      b_v.data_ptr<float>(), obs.data_ptr<float>(), state.data_ptr<float>(),
      state_out.data_ptr<float>(), logits.data_ptr<float>(), value.data_ptr<float>(), N);
}

void env_step_cuda(torch::Tensor agent, torch::Tensor food, torch::Tensor actions,
                   torch::Tensor rng, torch::Tensor reward, torch::Tensor hit,
                   torch::Tensor any_hit) {
  int N = agent.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(any_hit.data_ptr<int>(), 0, sizeof(int), stream);
  int tpb = 256;
  env_move_kernel<<<cdiv(N, tpb), tpb, 0, stream>>>(
      agent.data_ptr<float>(), food.data_ptr<float>(),
      (const long long*)actions.data_ptr<int64_t>(),
      reward.data_ptr<float>(), (unsigned char*)hit.data_ptr<uint8_t>(),
      any_hit.data_ptr<int>(), N);
  env_respawn_kernel<<<cdiv(N, tpb), tpb, 0, stream>>>(
      food.data_ptr<float>(), (unsigned char*)hit.data_ptr<uint8_t>(),
      (long long*)rng.data_ptr<int64_t>(), any_hit.data_ptr<int>(), N);
}

template <int B>
static void launch_rollout(dim3 grid, dim3 block, size_t smem_bytes, cudaStream_t stream,
                           const float* Wt, const float* We, const float* be,
                           const float* Wa, const float* ba,
                           float* agent, float* food, long long* rng, float* state,
                           float* rewards, float* last_logits, int* hit_count,
                           unsigned char* hits_g, int N, int horizon, int num_chunks) {
  void* args[] = {(void*)&Wt, (void*)&We, (void*)&be, (void*)&Wa, (void*)&ba,
                  (void*)&agent, (void*)&food, (void*)&rng, (void*)&state,
                  (void*)&rewards, (void*)&last_logits, (void*)&hit_count,
                  (void*)&hits_g, (void*)&N, (void*)&horizon, (void*)&num_chunks};
  cudaError_t err = cudaLaunchCooperativeKernel((void*)rollout_v2<B>, grid, block,
                                                args, smem_bytes, stream);
  TORCH_CHECK(err == cudaSuccess, "coop launch failed: ", cudaGetErrorString(err));
}

template <int B, int KSP, int HMODE, int WMODE, int ET = 4>
static int v3_max_blocks(size_t smem_bytes) {
  constexpr int TPB = (B / ET) * 64 * KSP;
  cudaFuncSetAttribute((const void*)rollout_v3<B, KSP, HMODE, WMODE, ET>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_v3<B, KSP, HMODE, WMODE, ET>,
                                                TPB, smem_bytes);
  TORCH_CHECK(occ > 0, "rollout_v3<", B, ",", KSP, ",", HMODE, ",", WMODE,
              "> does not fit: occ=0");
  (void)0;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  return occ * sms;
}

template <int B, int KSP, int HMODE, int WMODE, int ET = 4>
static void v3_launch(int grid, size_t smem_bytes, cudaStream_t stream,
                      const float* Wt, const float* We, const float* be,
                      const float* Wa, const float* ba,
                      float* agent, float* food, long long* rng, float* state,
                      float* rewards, float* last_logits, int* hit_count,
                      unsigned char* hits_g, float* hscr, int N, int horizon,
                      int num_chunks) {
  constexpr int TPB = (B / ET) * 64 * KSP;
  void* args[] = {(void*)&Wt, (void*)&We, (void*)&be, (void*)&Wa, (void*)&ba,
                  (void*)&agent, (void*)&food, (void*)&rng, (void*)&state,
                  (void*)&rewards, (void*)&last_logits, (void*)&hit_count,
                  (void*)&hits_g, (void*)&hscr, (void*)&N, (void*)&horizon,
                  (void*)&num_chunks};
  cudaError_t err = cudaLaunchCooperativeKernel((void*)rollout_v3<B, KSP, HMODE, WMODE, ET>,
                                                dim3(grid), dim3(TPB), args,
                                                smem_bytes, stream);
  TORCH_CHECK(err == cudaSuccess, "coop launch v3 failed: ", cudaGetErrorString(err));
}

__host__ __device__ __forceinline__ int smem_floats_v3(int B, int KSP, int HMODE, int WMODE) {
  int HP = HPAD(B);
  int KV = (KSP == 2) ? 4 : 8;
  int fl = ((HMODE == 1) ? 2 : 1) * 256 * HP
         + ((WMODE == 0) ? (KSP * 2 * KV * PASS_ROWS) : 0)
         + B * 4 + B * 2 + B * 2 + B * 4 + B + 8;
  return fl + 32;
}

int rollout_cuda3(int variant, torch::Tensor Wt, torch::Tensor w_enc, torch::Tensor b_enc,
                  torch::Tensor w_a, torch::Tensor b_a,
                  torch::Tensor agent, torch::Tensor food,
                  torch::Tensor rng, torch::Tensor state, torch::Tensor rewards,
                  torch::Tensor last_logits, torch::Tensor hit_count,
                  torch::Tensor hits, torch::Tensor hscr, torch::Tensor pos,
                  int64_t seed10007) {
  int N = agent.size(0);
  if (N == 0) return 0;
  int horizon = hit_count.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(hit_count.data_ptr<int>(), 0, sizeof(int) * horizon, stream);

  static int cap[16];
  static bool cap_init = [](int* c) { for (int i = 0; i < 16; ++i) c[i] = -1; return true; } (cap);
  (void)cap_init;
  const float* Wt_ = Wt.data_ptr<float>();
  const float* We_ = w_enc.data_ptr<float>();
  const float* be_ = b_enc.data_ptr<float>();
  const float* Wa_ = w_a.data_ptr<float>();
  const float* ba_ = b_a.data_ptr<float>();
  float* ag = agent.data_ptr<float>();
  float* fd = food.data_ptr<float>();
  long long* rg = (long long*)rng.data_ptr<int64_t>();
  float* st = state.data_ptr<float>();
  float* rw = rewards.data_ptr<float>();
  float* ll = last_logits.data_ptr<float>();
  int* hc = hit_count.data_ptr<int>();
  unsigned char* hg = (unsigned char*)hits.data_ptr<uint8_t>();
  float* hs = hscr.data_ptr<float>();
  long long* ps = (long long*)pos.data_ptr<int64_t>();
  long long s10007 = (long long)seed10007;

#define V3_CASE(idx, BB, KK, HH, WW, EE)                                              \
  case idx: {                                                                         \
    size_t smem = smem_floats_v3(BB, KK, HH, WW) * sizeof(float);                     \
    if (cap[idx] < 0) cap[idx] = v3_max_blocks<BB, KK, HH, WW, EE>(smem);             \
    int chunks = cdiv(N, BB);                                                         \
    int grid = chunks < cap[idx] ? chunks : cap[idx];                                 \
    v3_launch<BB, KK, HH, WW, EE>(grid, smem, stream.stream(), Wt_, We_, be_, Wa_,    \
                              ba_, ag, fd, rg, st, rw, ll, hc, hg, hs, N, horizon,    \
                              chunks);                                                \
    break;                                                                            \
  }
  switch (variant) {
    V3_CASE(0, 16, 1, 0, 0, 4)   // narrow16 gmem-H
    V3_CASE(1, 24, 2, 1, 1, 4)   // wide24 ping-pong ldg-W
    V3_CASE(2, 32, 1, 1, 1, 4)   // narrow32 ping-pong ldg-W
    V3_CASE(3, 32, 2, 1, 0, 4)   // wide32 ping-pong staged-W
    V3_CASE(4, 32, 2, 1, 1, 4)   // wide32 ping-pong ldg-W
    V3_CASE(5, 64, 1, 0, 0, 4)   // narrow64 gmem-H staged-W
    V3_CASE(6, 32, 2, 1, 0, 8)   // wide32 ping-pong staged-W, E_TILE=8
    V3_CASE(7, 64, 1, 0, 0, 8)   // narrow64 gmem-H staged-W, E_TILE=8
    V3_CASE(10, 32, 1, 1, 1, 8)  // narrow32 ping-pong ldg-W, E_TILE=8
    V3_CASE(11, 64, 1, 0, 1, 8)  // narrow64 gmem-H ldg-W, E_TILE=8
  case 9: {
    constexpr int BC = 32;
    constexpr int TPB9 = (BC / 8) * 64;
    size_t smem = (size_t)(2 * 256 * (BC + 4) + 3 * 4 * PASS_ROWS + 2 * BC * 4 + BC * 12 + 8 + 32) * sizeof(float);
    if (cap[9] < 0) {
      cudaFuncSetAttribute((const void*)rollout_pipe<32>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_pipe<32>, TPB9, smem);
      TORCH_CHECK(occ > 0, "rollout_pipe does not fit: occ=0");
      int sms = 0;
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
      cap[9] = occ * sms;
    }
    int chunks = cdiv(N, BC);
    int grid = chunks < cap[9] ? chunks : cap[9];
    void* args[] = {(void*)&Wt_, (void*)&We_, (void*)&be_, (void*)&Wa_, (void*)&ba_,
                    (void*)&ag, (void*)&fd, (void*)&rg, (void*)&st,
                    (void*)&rw, (void*)&ll, (void*)&hc, (void*)&hg,
                    (void*)&N, (void*)&horizon, (void*)&chunks};
    cudaError_t err = cudaLaunchCooperativeKernel((const void*)rollout_pipe<32>,
                                                  dim3(grid), dim3(TPB9), args, smem, stream);
    TORCH_CHECK(err == cudaSuccess, "pipe launch failed: ", cudaGetErrorString(err));
    break;
  }
  case 15: {
    constexpr int BC = 32;
    constexpr int TPB15 = (BC / 8) * 64;
    size_t smem = (size_t)(2 * 256 * (BC + 4) + 2 * 8 * PASS_ROWS + BC * 12 + 8 + 32) * sizeof(float);
    cudaFuncSetAttribute((const void*)rollout_probe<32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_probe<32>, TPB15, smem);
    TORCH_CHECK(occ > 0, "probe does not fit");
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int chunks = cdiv(N, BC);
    int grid = chunks < occ * sms ? chunks : occ * sms;
    if (cap[15] < 0) cap[15] = occ * sms;
    grid = chunks < cap[15] ? chunks : cap[15];
    void* args[] = {(void*)&Wt_, (void*)&We_, (void*)&be_, (void*)&Wa_, (void*)&ba_,
                    (void*)&ag, (void*)&fd, (void*)&rg, (void*)&st,
                    (void*)&rw, (void*)&ll, (void*)&hc, (void*)&hg,
                    (void*)&N, (void*)&horizon, (void*)&chunks};
    cudaError_t err = cudaLaunchCooperativeKernel((const void*)rollout_probe<32>,
                                                  dim3(grid), dim3(TPB15), args, smem, stream);
    TORCH_CHECK(err == cudaSuccess, "probe launch failed: ", cudaGetErrorString(err));
    break;
  }
  case 14: {
    constexpr int BC = 32;
    constexpr int TPB13 = (BC / 8) * 64;
    const int NBUFi = 3;
    const int KVi = 4;
    size_t smem = (size_t)(2 * 256 * (BC + 4) + NBUFi * KVi * PASS_ROWS + BC * 12 + 8 + 32) * sizeof(float);
    int& capv = cap[variant];
    if (capv < 0) {
      cudaFuncSetAttribute((const void*)rollout_p1<32, 3, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_p1<32, 3, 4>, TPB13, smem);
      TORCH_CHECK(occ > 0, "p1 n3 does not fit");
      int sms = 0;
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
      capv = occ * sms;
    }
    {
      int chunks = cdiv(N, BC);
      int grid = chunks < capv ? chunks : capv;
      void* args[] = {(void*)&Wt_, (void*)&We_, (void*)&be_, (void*)&Wa_, (void*)&ba_,
                      (void*)&ag, (void*)&fd, (void*)&rg, (void*)&st,
                      (void*)&rw, (void*)&ll, (void*)&hc, (void*)&hg,
                      (void*)&N, (void*)&horizon, (void*)&chunks};
      cudaError_t err = cudaLaunchCooperativeKernel((const void*)rollout_p1<32, 3, 4>,
                                                    dim3(grid), dim3(TPB13), args, smem, stream);
      TORCH_CHECK(err == cudaSuccess, "p1 launch failed: ", cudaGetErrorString(err));
    }
    break;
  }
  case 12: {
    constexpr int BC = 64;
    constexpr int TPB12 = (BC / 8) * 32;
    size_t smem = (size_t)(256 * (BC + 4) + 2 * 8 * PASS_ROWS + BC * 12 + BC + 8 + 32) * sizeof(float);
    if (cap[12] < 0) {
      cudaFuncSetAttribute((const void*)rollout_c4<64>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_c4<64>, TPB12, smem);
      TORCH_CHECK(occ > 0, "rollout_c4 does not fit: occ=0");
      int sms = 0;
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
      cap[12] = occ * sms;
    }
    int chunks = cdiv(N, BC);
    int grid = chunks < cap[12] ? chunks : cap[12];
    void* args[] = {(void*)&Wt_, (void*)&We_, (void*)&be_, (void*)&Wa_, (void*)&ba_,
                    (void*)&ag, (void*)&fd, (void*)&rg, (void*)&st,
                    (void*)&rw, (void*)&ll, (void*)&hc, (void*)&hg,
                    (void*)&hs, (void*)&N, (void*)&horizon, (void*)&chunks};
    cudaError_t err = cudaLaunchCooperativeKernel((const void*)rollout_c4<64>,
                                                  dim3(grid), dim3(TPB12), args, smem, stream);
    TORCH_CHECK(err == cudaSuccess, "c4 launch failed: ", cudaGetErrorString(err));
    break;
  }
  case 8: {
    constexpr int BC = 32;
    size_t smem = (size_t)(2 * 256 * (BC + 4) + 2 * 8 * PASS_ROWS + (128 + 128 + 256 + 64 + 8) + 32) * sizeof(float);
    if (cap[8] < 0) {
      cudaFuncSetAttribute((const void*)rollout_dual<4>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_dual<4>, 1024, smem);
      TORCH_CHECK(occ > 0, "rollout_dual does not fit: occ=0");
      int sms = 0;
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
      cap[8] = occ * sms;
    }
    int units = cdiv(N, 2 * BC);
    int grid = units < cap[8] ? units : cap[8];
    void* args[] = {(void*)&Wt_, (void*)&We_, (void*)&be_, (void*)&Wa_, (void*)&ba_,
                    (void*)&ag, (void*)&fd, (void*)&rg, (void*)&st,
                    (void*)&rw, (void*)&ll, (void*)&hc, (void*)&hg,
                    (void*)&hs, (void*)&ps, (void*)&s10007,
                    (void*)&N, (void*)&horizon, (void*)&units};
    cudaError_t err = cudaLaunchCooperativeKernel((const void*)rollout_dual<4>,
                                                  dim3(grid), dim3(1024), args, smem, stream);
    TORCH_CHECK(err == cudaSuccess, "dual launch failed: ", cudaGetErrorString(err));
    break;
  }
    default:
      TORCH_CHECK(false, "bad variant ", variant);
  }
#undef V3_CASE
  return 0;
}

int rollout_cuda(torch::Tensor Wt, torch::Tensor w_enc, torch::Tensor b_enc,
                 torch::Tensor w_a, torch::Tensor b_a,
                 torch::Tensor agent, torch::Tensor food,
                 torch::Tensor rng, torch::Tensor state, torch::Tensor rewards,
                 torch::Tensor last_logits, torch::Tensor hit_count,
                 torch::Tensor hits) {
  int N = agent.size(0);
  if (N == 0) return 0;
  int horizon = hit_count.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(hit_count.data_ptr<int>(), 0, sizeof(int) * horizon, stream);

  constexpr int B = 32;
  constexpr int TPB = (B / E_TILE) * 64;
  int num_chunks = cdiv(N, B);
  size_t smem_bytes = smem_floats(B) * sizeof(float);

  static int max_blocks = -1;
  if (max_blocks < 0) {
    cudaFuncSetAttribute((const void*)rollout_v2<B>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)rollout_v2<B>,
                                                  TPB, smem_bytes);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    max_blocks = occ * sms;
    TORCH_CHECK(occ > 0, "rollout kernel does not fit: occ=0");
  }
  int grid = num_chunks < max_blocks ? num_chunks : max_blocks;

  launch_rollout<B>(dim3(grid), dim3(TPB), smem_bytes, stream.stream(),
                    Wt.data_ptr<float>(), w_enc.data_ptr<float>(),
                    b_enc.data_ptr<float>(), w_a.data_ptr<float>(),
                    b_a.data_ptr<float>(), agent.data_ptr<float>(),
                    food.data_ptr<float>(), (long long*)rng.data_ptr<int64_t>(),
                    state.data_ptr<float>(), rewards.data_ptr<float>(),
                    last_logits.data_ptr<float>(), hit_count.data_ptr<int>(),
                    (unsigned char*)hits.data_ptr<uint8_t>(), N, horizon, num_chunks);
  return 0;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void policy_forward_cuda(torch::Tensor w_enc, torch::Tensor b_enc,
                         torch::Tensor w_gru,
                         torch::Tensor w_a, torch::Tensor b_a,
                         torch::Tensor w_v, torch::Tensor b_v,
                         torch::Tensor obs, torch::Tensor state,
                         torch::Tensor state_out, torch::Tensor logits,
                         torch::Tensor value);
void env_step_cuda(torch::Tensor agent, torch::Tensor food, torch::Tensor actions,
                   torch::Tensor rng, torch::Tensor reward, torch::Tensor hit,
                   torch::Tensor any_hit);
int rollout_cuda(torch::Tensor Wt, torch::Tensor w_enc, torch::Tensor b_enc,
                 torch::Tensor w_a, torch::Tensor b_a,
                 torch::Tensor agent, torch::Tensor food,
                 torch::Tensor rng, torch::Tensor state, torch::Tensor rewards,
                 torch::Tensor last_logits, torch::Tensor hit_count,
                 torch::Tensor hits);
int rollout_cuda3(int variant, torch::Tensor Wt, torch::Tensor w_enc, torch::Tensor b_enc,
                  torch::Tensor w_a, torch::Tensor b_a,
                  torch::Tensor agent, torch::Tensor food,
                  torch::Tensor rng, torch::Tensor state, torch::Tensor rewards,
                  torch::Tensor last_logits, torch::Tensor hit_count,
                  torch::Tensor hits, torch::Tensor hscr, torch::Tensor pos,
                  int64_t seed10007);
"""

_ext = load_inline(
    name="grid_mingru_sps_v3",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["policy_forward_cuda", "env_step_cuda", "rollout_cuda", "rollout_cuda3"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


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


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous()


# --- weight repack: [3][768][256] -> [3][256][768] with rows grouped by
# (pass of 128 comps, gate, comp) so each GEMM pass stages a contiguous 384-row
# segment per k-tile from global memory.
_PERM_IDX_CACHE: dict = {}
# packed-cache entry: weakref to the source param + version + packed tensor.
_PACK: dict = {"ref": None, "ver": -1, "packed": None}


def _perm_idx(device) -> torch.Tensor:
    idx = _PERM_IDX_CACHE.get(device)
    if idx is None:
        p = [g * 256 + pp * 128 + c for pp in range(2) for g in range(3) for c in range(128)]
        idx = torch.tensor(p, dtype=torch.long, device=device)
        _PERM_IDX_CACHE[device] = idx
    return idx


def _packed_weights(model: Model) -> torch.Tensor:
    wg = model.w_gru
    if (
        _PACK["ref"] is not None
        and _PACK["ref"]() is wg
        and _PACK["ver"] == wg._version
        and _PACK["packed"].device == wg.device
    ):
        return _PACK["packed"]
    packed = (
        wg.detach().contiguous().index_select(1, _perm_idx(wg.device)).transpose(1, 2).contiguous()
    )
    _PACK["packed"] = packed
    _PACK["ref"] = weakref.ref(wg)
    _PACK["ver"] = wg._version
    return packed


def policy_forward(model: Model, obs: torch.Tensor, state: torch.Tensor):
    """obs (N,4), state (N,3,256) -> logits, new_state, value. CUDA kernel."""
    N = obs.shape[0]
    device = obs.device
    state_ = state.contiguous()
    obs_ = obs.contiguous()
    state_out = torch.empty_like(state_)
    logits = torch.empty(N, NUM_ACTIONS, device=device, dtype=torch.float32)
    value = torch.empty(N, device=device, dtype=torch.float32)
    _ext.policy_forward_cuda(
        _contig(model.w_enc), _contig(model.b_enc), _contig(model.w_gru),
        _contig(model.w_a), _contig(model.b_a), _contig(model.w_v), _contig(model.b_v),
        obs_, state_, state_out, logits, value,
    )
    return logits, state_out, value


def env_step(agent: torch.Tensor, food: torch.Tensor, actions: torch.Tensor,
             rng_state: torch.Tensor):
    """Match reference.env_step exactly (functional; inputs untouched)."""
    agent_out = agent.contiguous().clone()
    food_out = food.contiguous().clone()
    rng_out = rng_state.contiguous().clone()
    N = agent_out.shape[0]
    device = agent_out.device
    reward = torch.empty(N, device=device, dtype=torch.float32)
    hit = torch.empty(N, device=device, dtype=torch.uint8)
    any_hit = torch.empty(1, device=device, dtype=torch.int32)
    _ext.env_step_cuda(agent_out, food_out, actions.contiguous(), rng_out,
                       reward, hit, any_hit)
    return agent_out, food_out, reward, rng_out


_SCR: dict = {}
_VARIANT_B = {0: 16, 1: 24, 2: 32, 3: 32, 4: 32, 5: 64, 6: 32, 7: 64, 9: 32, 10: 32, 11: 64, 12: 64, 14: 32, 15: 32}


def _hscr_buf(variant: int, device) -> torch.Tensor:
    buf = _SCR.get(variant)
    if buf is None or buf.device != device:
        if variant == 8:  # dual-chunk: 2 regions of [256][36] per block
            buf = torch.empty(376 * 2 * 256 * 36, device=device, dtype=torch.float32)
        else:
            HP = _VARIANT_B[variant] + 4
            buf = torch.empty(376 * 256 * HP, device=device, dtype=torch.float32)
        _SCR[variant] = buf
    return buf


def _pick_variant(num_envs: int) -> int:
    ov = os.environ.get("KBH_ROLLOUT_VARIANT", "")
    if ov:
        return int(ov)
    # Tuned across windows on the busy box: the B=32/E8/ping-pong staged kernel
    # (-2, rollout_v2) won every A/B against K-split, LDG, deep-pipeline, E8C4,
    # dual-chunk and B in {16,24,48,64} variants (see DEVLOG.md). Keep it flat:
    # its 8192-band rival (v11) measured a tie in calm interleaved pairs.
    return -2


_PIN: dict = {}


def _pin_buf(num_envs: int, slot: int) -> torch.Tensor:
    key = (num_envs, slot)
    ent = _PIN.get(key)
    if ent is None:
        ent = [torch.empty((num_envs, 2), dtype=torch.int64, pin_memory=True), None]
        _PIN[key] = ent
    else:
        # a previous run() may still be reading this pinned buffer on-GPU
        if ent[1] is not None:
            ent[1].synchronize()
            ent[1] = None
    return ent


def run(num_envs: int, horizon: int, seed: int, model: Model | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Model()
    model = model.to(device).eval()

    # GPU-side inits first (async; overlaps the CPU randint below)
    variant = _pick_variant(num_envs)
    if variant == 8:
        state = torch.empty(num_envs, GRU_LAYERS, HIDDEN, device=device)
        rewards = torch.empty(num_envs, device=device)
        last_logits = torch.empty(num_envs, NUM_ACTIONS, device=device)
        rng_state = torch.empty(num_envs, device=device, dtype=torch.int64)
    else:
        state = torch.zeros(num_envs, GRU_LAYERS, HIDDEN, device=device)
        rewards = torch.zeros(num_envs, device=device)
        rng_state = torch.arange(num_envs, device=device, dtype=torch.int64) + (seed * 10007)
    last_logits = (torch.empty(num_envs, NUM_ACTIONS, device=device) if variant == 8
                   else torch.zeros(num_envs, NUM_ACTIONS, device=device))
    hit_count = torch.zeros(horizon, device=device, dtype=torch.int32)
    hits = torch.zeros(num_envs, device=device, dtype=torch.uint8)
    positions = torch.empty(num_envs, 2, device=device, dtype=torch.int64)

    # Reference init draws: two sequential CPU MT19937 randint(0,11,(N,2)) calls.
    # Generate into pinned staging, H2D int64, cast to fp32 on GPU (exact for 0..10).
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    ent_a = _pin_buf(num_envs, 0)
    torch.randint(0, BOARD, (num_envs, 2), generator=g, out=ent_a[0])
    agent_i = ent_a[0].to(device, non_blocking=True)
    ent_f = _pin_buf(num_envs, 1)
    torch.randint(0, BOARD, (num_envs, 2), generator=g, out=ent_f[0])
    food_i = ent_f[0].to(device, non_blocking=True)
    ev_a = torch.cuda.Event(); ev_a.record()
    ent_a[1] = ev_a
    ev_f = torch.cuda.Event(); ev_f.record()
    ent_f[1] = ev_f
    agent = agent_i.float()
    food = food_i.float()

    with torch.no_grad():
        if variant < 0:
            _ext.rollout_cuda(
                _packed_weights(model), _contig(model.w_enc), _contig(model.b_enc),
                _contig(model.w_a), _contig(model.b_a), agent, food, rng_state,
                state, rewards, last_logits, hit_count, hits,
            )
        else:
            _ext.rollout_cuda3(
                variant, _packed_weights(model), _contig(model.w_enc), _contig(model.b_enc),
                _contig(model.w_a), _contig(model.b_a), agent, food, rng_state,
                state, rewards, last_logits, hit_count, hits, _hscr_buf(variant, device),
                positions, seed * 10007,
            )

    return {
        "rewards": rewards.detach(),
        "positions": (positions.detach() if variant == 8 else agent.detach().round().long()),
        "last_logits": last_logits.detach(),
        "state": state.detach(),
    }


def get_init_inputs():
    return []


def get_inputs():
    return []
