"""CUDA grid-foraging + 3-layer MinGRU policy (fused cooperative rollout)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

BOARD = 11
OBS_DIM = 4
HIDDEN = 256
GRU_LAYERS = 3
NUM_ACTIONS = 4
GRU_OUT = 3 * HIDDEN

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace cg = cooperative_groups;

constexpr int BOARD = 11;
constexpr int HIDDEN = 256;
constexpr int GRU_LAYERS = 3;
constexpr int NUM_ACTIONS = 4;
constexpr int GRU_OUT = 768;
constexpr int OBS_DIM = 4;

__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ int64_t lcg_step(int64_t rng) {
    uint64_t r = static_cast<uint64_t>(rng);
    r = r * 6364136223846793005ULL + 1ULL;
    return static_cast<int64_t>(r & 0x7FFFFFFFFFFFFFFFULL);
}

// ---------------------------------------------------------------------------
// policy_forward: one block (256 threads) per env
// ---------------------------------------------------------------------------
__global__ void policy_forward_kernel(
    const float* __restrict__ w_enc,   // [256, 4]
    const float* __restrict__ b_enc,   // [256]
    const float* __restrict__ w_gru,   // [3, 768, 256]
    const float* __restrict__ w_a,     // [4, 256]
    const float* __restrict__ b_a,     // [4]
    const float* __restrict__ w_v,     // [1, 256]
    const float* __restrict__ b_v,     // [1]
    const float* __restrict__ obs,     // [N, 4]
    const float* __restrict__ state,   // [N, 3, 256]
    float* __restrict__ logits,        // [N, 4]
    float* __restrict__ new_state,     // [N, 3, 256]
    float* __restrict__ value,         // [N]
    int num_envs
) {
    const int env = blockIdx.x;
    if (env >= num_envs) return;
    const int tid = threadIdx.x; // 0..255

    __shared__ float sh_h[HIDDEN];
    __shared__ float sh_obs[OBS_DIM];
    __shared__ float sh_logits[NUM_ACTIONS];
    __shared__ float sh_val;

    if (tid < OBS_DIM) {
        sh_obs[tid] = obs[env * OBS_DIM + tid];
    }
    __syncthreads();

    // Encoder: h = obs @ W_enc.T + b_enc
    {
        float acc = b_enc[tid];
        #pragma unroll
        for (int j = 0; j < OBS_DIM; ++j) {
            acc += sh_obs[j] * w_enc[tid * OBS_DIM + j];
        }
        sh_h[tid] = acc;
    }
    __syncthreads();

    // 3 MinGRU layers
    for (int layer = 0; layer < GRU_LAYERS; ++layer) {
        const float st = state[(env * GRU_LAYERS + layer) * HIDDEN + tid];
        const float* W = w_gru + layer * GRU_OUT * HIDDEN;

        float zh = 0.f, zg = 0.f, zp = 0.f;
        // GEMV: each thread owns one output of each gate
        #pragma unroll 8
        for (int k = 0; k < HIDDEN; ++k) {
            const float hk = sh_h[k];
            zh += hk * __ldg(W + tid * HIDDEN + k);
            zg += hk * __ldg(W + (tid + HIDDEN) * HIDDEN + k);
            zp += hk * __ldg(W + (tid + 2 * HIDDEN) * HIDDEN + k);
        }

        const float out = st + d_sigmoid(zg) * (tanhf(zh) - st);
        const float p = d_sigmoid(zp);
        const float h_new = p * out + (1.f - p) * sh_h[tid];

        new_state[(env * GRU_LAYERS + layer) * HIDDEN + tid] = out;
        __syncthreads();
        sh_h[tid] = h_new;
        __syncthreads();
    }

    // Action logits (4) and value (1) — reduce on thread 0 for heads
    // Parallel partial dots into shared, then finish.
    __shared__ float sh_partial[NUM_ACTIONS][32]; // 256/32 = 8 warps, pad to 32
    // Use warp-level for 4 logits: each warp computes one? Simpler: all threads
    // contribute to all 4 logits via shared accumulation.

    // Direct: threads 0..3 each compute full logit (read all h)
    if (tid < NUM_ACTIONS) {
        float acc = b_a[tid];
        const float* row = w_a + tid * HIDDEN;
        #pragma unroll 8
        for (int k = 0; k < HIDDEN; ++k) {
            acc += sh_h[k] * __ldg(row + k);
        }
        sh_logits[tid] = acc;
    }
    if (tid == 0) {
        float acc = b_v[0];
        #pragma unroll 8
        for (int k = 0; k < HIDDEN; ++k) {
            acc += sh_h[k] * __ldg(w_v + k);
        }
        sh_val = acc;
    }
    __syncthreads();

    if (tid < NUM_ACTIONS) {
        logits[env * NUM_ACTIONS + tid] = sh_logits[tid];
    }
    if (tid == 0) {
        value[env] = sh_val;
    }
}

// ---------------------------------------------------------------------------
// env_step: move + hit (kernel 1), respawn (kernel 2)
// ---------------------------------------------------------------------------
__global__ void env_move_kernel(
    float* __restrict__ agent,       // [N, 2]
    const float* __restrict__ food,  // [N, 2]
    const int* __restrict__ actions, // [N]
    float* __restrict__ reward,      // [N]
    uint8_t* __restrict__ hit,       // [N]
    int* __restrict__ any_hit,       // scalar
    int num_envs
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;

    float ax = agent[i * 2 + 0];
    float ay = agent[i * 2 + 1];
    const int a = actions[i];
    // 0 up (y-), 1 down (y+), 2 left (x-), 3 right (x+)
    if (a == 0) ay -= 1.f;
    else if (a == 1) ay += 1.f;
    else if (a == 2) ax -= 1.f;
    else if (a == 3) ax += 1.f;
    ax = fminf(fmaxf(ax, 0.f), (float)(BOARD - 1));
    ay = fminf(fmaxf(ay, 0.f), (float)(BOARD - 1));
    agent[i * 2 + 0] = ax;
    agent[i * 2 + 1] = ay;

    const float fx = food[i * 2 + 0];
    const float fy = food[i * 2 + 1];
    const int h = (ax == fx) && (ay == fy);
    hit[i] = (uint8_t)h;
    reward[i] = h ? 1.f : 0.f;
    if (h) atomicOr(any_hit, 1);
}

__global__ void env_respawn_kernel(
    float* __restrict__ food,            // [N, 2]
    const uint8_t* __restrict__ hit,     // [N]
    int64_t* __restrict__ rng_state,     // [N]
    const int* __restrict__ any_hit,     // scalar
    int num_envs
) {
    if (*any_hit == 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;

    int64_t rng = rng_state[i];
    rng = lcg_step(rng);
    const int64_t fx = rng % BOARD;
    rng = lcg_step(rng);
    const int64_t fy = rng % BOARD;
    rng_state[i] = rng;
    if (hit[i]) {
        food[i * 2 + 0] = (float)fx;
        food[i * 2 + 1] = (float)fy;
    }
}

__global__ void clear_flag_kernel(int* flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *flag = 0;
}

// ---------------------------------------------------------------------------
// Fused cooperative megakernel: full horizon rollout, greedy argmax
// ---------------------------------------------------------------------------
__global__ void fused_rollout_kernel(
    const float* __restrict__ w_enc,
    const float* __restrict__ b_enc,
    const float* __restrict__ w_gru,
    const float* __restrict__ w_a,
    const float* __restrict__ b_a,
    float* __restrict__ agent,         // [N, 2]
    float* __restrict__ food,          // [N, 2]
    float* __restrict__ state,         // [N, 3, 256]
    float* __restrict__ rewards,       // [N]
    float* __restrict__ last_logits,   // [N, 4]
    int64_t* __restrict__ rng_state,   // [N]
    int* __restrict__ any_hit,         // scalar device flag
    uint8_t* __restrict__ hit_buf,     // [N]
    int num_envs,
    int horizon
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x; // 0..255
    const int num_blocks = gridDim.x;

    __shared__ float sh_h[HIDDEN];
    __shared__ float sh_obs[OBS_DIM];
    __shared__ float sh_logits[NUM_ACTIONS];
    __shared__ int sh_action;
    __shared__ float sh_ax, sh_ay, sh_fx, sh_fy;
    __shared__ int64_t sh_rng;
    __shared__ float sh_rew;
    __shared__ int sh_hit;

    for (int t = 0; t < horizon; ++t) {
        // ---- Phase 1: policy + move + hit for each env owned by this block ----
        for (int env = blockIdx.x; env < num_envs; env += num_blocks) {
            // Load env positions (thread 0)
            if (tid == 0) {
                sh_ax = agent[env * 2 + 0];
                sh_ay = agent[env * 2 + 1];
                sh_fx = food[env * 2 + 0];
                sh_fy = food[env * 2 + 1];
                sh_rng = rng_state[env];
                sh_rew = rewards[env];
            }
            __syncthreads();

            // Observation
            if (tid == 0) {
                sh_obs[0] = (sh_fx - sh_ax) / (float)BOARD;
                sh_obs[1] = (sh_fy - sh_ay) / (float)BOARD;
                sh_obs[2] = sh_ax / (float)(BOARD - 1);
                sh_obs[3] = sh_ay / (float)(BOARD - 1);
            }
            __syncthreads();

            // Encoder
            {
                float acc = b_enc[tid];
                #pragma unroll
                for (int j = 0; j < OBS_DIM; ++j) {
                    acc += sh_obs[j] * w_enc[tid * OBS_DIM + j];
                }
                sh_h[tid] = acc;
            }
            __syncthreads();

            // MinGRU x3
            for (int layer = 0; layer < GRU_LAYERS; ++layer) {
                const float st = state[(env * GRU_LAYERS + layer) * HIDDEN + tid];
                const float* W = w_gru + layer * GRU_OUT * HIDDEN;

                float zh = 0.f, zg = 0.f, zp = 0.f;
                #pragma unroll 8
                for (int k = 0; k < HIDDEN; ++k) {
                    const float hk = sh_h[k];
                    zh += hk * __ldg(W + tid * HIDDEN + k);
                    zg += hk * __ldg(W + (tid + HIDDEN) * HIDDEN + k);
                    zp += hk * __ldg(W + (tid + 2 * HIDDEN) * HIDDEN + k);
                }

                const float out = st + d_sigmoid(zg) * (tanhf(zh) - st);
                const float p = d_sigmoid(zp);
                const float h_new = p * out + (1.f - p) * sh_h[tid];

                state[(env * GRU_LAYERS + layer) * HIDDEN + tid] = out;
                __syncthreads();
                sh_h[tid] = h_new;
                __syncthreads();
            }

            // Logits
            if (tid < NUM_ACTIONS) {
                float acc = b_a[tid];
                const float* row = w_a + tid * HIDDEN;
                #pragma unroll 8
                for (int k = 0; k < HIDDEN; ++k) {
                    acc += sh_h[k] * __ldg(row + k);
                }
                sh_logits[tid] = acc;
            }
            __syncthreads();

            // Store last logits every step (final step kept)
            if (tid < NUM_ACTIONS) {
                last_logits[env * NUM_ACTIONS + tid] = sh_logits[tid];
            }

            // Argmax
            if (tid == 0) {
                int best = 0;
                float best_v = sh_logits[0];
                #pragma unroll
                for (int a = 1; a < NUM_ACTIONS; ++a) {
                    if (sh_logits[a] > best_v) {
                        best_v = sh_logits[a];
                        best = a;
                    }
                }
                sh_action = best;

                // Env step move
                float ax = sh_ax, ay = sh_ay;
                const int act = best;
                if (act == 0) ay -= 1.f;
                else if (act == 1) ay += 1.f;
                else if (act == 2) ax -= 1.f;
                else if (act == 3) ax += 1.f;
                ax = fminf(fmaxf(ax, 0.f), (float)(BOARD - 1));
                ay = fminf(fmaxf(ay, 0.f), (float)(BOARD - 1));
                sh_ax = ax;
                sh_ay = ay;

                const int h = (ax == sh_fx) && (ay == sh_fy);
                sh_hit = h;
                if (h) {
                    sh_rew += 1.f;
                    atomicOr(any_hit, 1);
                }
                hit_buf[env] = (uint8_t)h;
                agent[env * 2 + 0] = ax;
                agent[env * 2 + 1] = ay;
                rewards[env] = sh_rew;
            }
            __syncthreads();
        }

        grid.sync();

        // ---- Phase 2: respawn if any hit ----
        const int any = *any_hit;
        if (any) {
            for (int env = blockIdx.x; env < num_envs; env += num_blocks) {
                if (tid == 0) {
                    int64_t rng = rng_state[env];
                    rng = lcg_step(rng);
                    const int64_t fx = rng % BOARD;
                    rng = lcg_step(rng);
                    const int64_t fy = rng % BOARD;
                    rng_state[env] = rng;
                    if (hit_buf[env]) {
                        food[env * 2 + 0] = (float)fx;
                        food[env * 2 + 1] = (float)fy;
                    }
                }
            }
        }

        // Clear flag for next step
        if (tid == 0 && blockIdx.x == 0) {
            *any_hit = 0;
        }
        grid.sync();
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
static void launch_policy(
    torch::Tensor w_enc, torch::Tensor b_enc,
    torch::Tensor w_gru, torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor w_v, torch::Tensor b_v,
    torch::Tensor obs, torch::Tensor state,
    torch::Tensor logits, torch::Tensor new_state, torch::Tensor value
) {
    const int N = (int)obs.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    policy_forward_kernel<<<N, HIDDEN, 0, stream>>>(
        w_enc.data_ptr<float>(), b_enc.data_ptr<float>(),
        w_gru.data_ptr<float>(), w_a.data_ptr<float>(), b_a.data_ptr<float>(),
        w_v.data_ptr<float>(), b_v.data_ptr<float>(),
        obs.data_ptr<float>(), state.data_ptr<float>(),
        logits.data_ptr<float>(), new_state.data_ptr<float>(), value.data_ptr<float>(),
        N
    );
}

static std::vector<torch::Tensor> launch_env_step(
    torch::Tensor agent, torch::Tensor food,
    torch::Tensor actions, torch::Tensor rng_state
) {
    const int N = (int)agent.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_f = agent.options();
    auto reward = torch::empty({N}, opts_f);
    auto hit = torch::empty({N}, agent.options().dtype(torch::kUInt8));
    auto any_hit = torch::zeros({1}, agent.options().dtype(torch::kInt32));

    // clone outputs to match reference non-inplace semantics for safety
    auto agent_out = agent.clone();
    auto food_out = food.clone();
    auto rng_out = rng_state.clone();

    auto actions_i = actions.to(torch::kInt32).contiguous();

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    env_move_kernel<<<blocks, threads, 0, stream>>>(
        agent_out.data_ptr<float>(), food_out.data_ptr<float>(),
        actions_i.data_ptr<int>(), reward.data_ptr<float>(),
        hit.data_ptr<uint8_t>(), any_hit.data_ptr<int>(), N
    );
    env_respawn_kernel<<<blocks, threads, 0, stream>>>(
        food_out.data_ptr<float>(), hit.data_ptr<uint8_t>(),
        rng_out.data_ptr<int64_t>(), any_hit.data_ptr<int>(), N
    );
    return {agent_out, food_out, reward, rng_out};
}

static void launch_rollout(
    torch::Tensor w_enc, torch::Tensor b_enc,
    torch::Tensor w_gru, torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor agent, torch::Tensor food, torch::Tensor state,
    torch::Tensor rewards, torch::Tensor last_logits, torch::Tensor rng_state,
    int horizon
) {
    const int N = (int)agent.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto any_hit = torch::zeros({1}, agent.options().dtype(torch::kInt32));
    auto hit_buf = torch::empty({N}, agent.options().dtype(torch::kUInt8));

    int block_size = HIDDEN;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, fused_rollout_kernel, block_size, 0
    );
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, agent.get_device());
    int num_blocks = blocks_per_sm * prop.multiProcessorCount;
    // Leave headroom for cooperative residency
    if (num_blocks > prop.multiProcessorCount * blocks_per_sm)
        num_blocks = prop.multiProcessorCount * blocks_per_sm;
    // Cap at num_envs
    if (num_blocks > N) num_blocks = N;
    if (num_blocks < 1) num_blocks = 1;

    // Use proper pointer locals for cooperative launch args
    const float* p_w_enc = w_enc.data_ptr<float>();
    const float* p_b_enc = b_enc.data_ptr<float>();
    const float* p_w_gru = w_gru.data_ptr<float>();
    const float* p_w_a = w_a.data_ptr<float>();
    const float* p_b_a = b_a.data_ptr<float>();
    float* p_agent = agent.data_ptr<float>();
    float* p_food = food.data_ptr<float>();
    float* p_state = state.data_ptr<float>();
    float* p_rewards = rewards.data_ptr<float>();
    float* p_logits = last_logits.data_ptr<float>();
    int64_t* p_rng = rng_state.data_ptr<int64_t>();
    int* p_any = any_hit.data_ptr<int>();
    uint8_t* p_hit = hit_buf.data_ptr<uint8_t>();
    int n_envs = N;
    int hor = horizon;

    void* kernel_args[] = {
        (void*)&p_w_enc, (void*)&p_b_enc, (void*)&p_w_gru,
        (void*)&p_w_a, (void*)&p_b_a,
        (void*)&p_agent, (void*)&p_food, (void*)&p_state,
        (void*)&p_rewards, (void*)&p_logits, (void*)&p_rng,
        (void*)&p_any, (void*)&p_hit,
        (void*)&n_envs, (void*)&hor
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)fused_rollout_kernel,
        dim3(num_blocks), dim3(block_size),
        kernel_args, 0, stream.stream()
    );
    if (err != cudaSuccess) {
        // Fallback: non-cooperative multi-launch loop (still CUDA kernels)
        // Use policy + env kernels per step
        TORCH_CHECK(false, "cudaLaunchCooperativeKernel failed: ", cudaGetErrorString(err),
                    " blocks=", num_blocks, " blocks_per_sm=", blocks_per_sm);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("policy_forward", &launch_policy, "MinGRU policy forward");
    m.def("env_step", &launch_env_step, "Env step");
    m.def("rollout", &launch_rollout, "Fused cooperative rollout");
}
"""

_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="grid_mingru_sps_cuda",
            cpp_sources=[],
            cuda_sources=[CUDA_SRC],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _mod


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
    """obs (N,4), state (N,L,H) -> logits (N,4), new_state (N,L,H), value (N,)."""
    mod = _get_mod()
    obs = obs.contiguous().float()
    state = state.contiguous().float()
    N = obs.shape[0]
    device = obs.device
    logits = torch.empty(N, NUM_ACTIONS, device=device, dtype=torch.float32)
    new_state = torch.empty(N, GRU_LAYERS, HIDDEN, device=device, dtype=torch.float32)
    value = torch.empty(N, device=device, dtype=torch.float32)
    mod.policy_forward(
        model.w_enc.contiguous(),
        model.b_enc.contiguous(),
        model.w_gru.contiguous(),
        model.w_a.contiguous(),
        model.b_a.contiguous(),
        model.w_v.contiguous(),
        model.b_v.contiguous(),
        obs,
        state,
        logits,
        new_state,
        value,
    )
    return logits, new_state, value


def env_step(
    agent: torch.Tensor,
    food: torch.Tensor,
    actions: torch.Tensor,
    rng_state: torch.Tensor,
):
    mod = _get_mod()
    agent = agent.contiguous().float()
    food = food.contiguous().float()
    actions = actions.contiguous()
    rng_state = rng_state.contiguous().to(torch.int64)
    outs = mod.env_step(agent, food, actions, rng_state)
    return outs[0], outs[1], outs[2], outs[3]


def run(num_envs: int, horizon: int, seed: int, model: Model | None = None) -> dict:
    device = torch.device("cuda:0")
    if model is None:
        model = Model()
    model = model.to(device).eval()
    mod = _get_mod()

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    agent = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    food = torch.randint(0, BOARD, (num_envs, 2), generator=g).float().to(device)
    rng_state = torch.arange(num_envs, device=device, dtype=torch.int64) + (seed * 10007)
    state = torch.zeros(num_envs, GRU_LAYERS, HIDDEN, device=device)
    rewards = torch.zeros(num_envs, device=device)
    last_logits = torch.zeros(num_envs, NUM_ACTIONS, device=device)

    with torch.no_grad():
        mod.rollout(
            model.w_enc.contiguous(),
            model.b_enc.contiguous(),
            model.w_gru.contiguous(),
            model.w_a.contiguous(),
            model.b_a.contiguous(),
            agent,
            food,
            state,
            rewards,
            last_logits,
            rng_state,
            int(horizon),
        )

    return {
        "rewards": rewards.detach(),
        "positions": agent.detach().round().long(),
        "last_logits": last_logits.detach(),
        "state": state.detach(),
    }
