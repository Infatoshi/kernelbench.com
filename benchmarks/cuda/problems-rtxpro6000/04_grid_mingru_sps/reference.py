"""Naive PyTorch reference: vectorized grid foraging + 3-layer MinGRU policy.

Correctness oracle for policy forward + env step. SPS floor via run().

MDP:
  - num_envs independent agents on an 11x11 board
  - actions 0/1/2/3 = up/down/left/right, clamped
  - reward +1 on food; food respawns via a simple LCG (not torch.randint)
  - obs: 4 floats (dx/11, dy/11, x/10, y/10)

Policy (craftax.cu h256/L3 geometry):
  Linear(4->256) -> MinGRU x3 (h=256, highway) -> Linear(256->4) + value head

solution.py must provide:
  class Model  # load_state_dict compatible
  def policy_forward(model, obs, state) -> (logits, new_state, value)
  def env_step(agent, food, actions, rng_state) -> (agent, food, reward, rng_state)
  def run(num_envs, horizon, seed, model=None) -> dict  # for SPS
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            p.data.normal_(0.0, 0.02, generator=g)

    def forward(self, obs: torch.Tensor, state: torch.Tensor):
        return policy_forward(self, obs, state)


def policy_forward(model: Model, obs: torch.Tensor, state: torch.Tensor):
    """obs (N,4), state (N,L,H) -> logits (N,4), new_state (N,L,H), value (N,)."""
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


def obs_from_state(agent: torch.Tensor, food: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            (food[:, 0] - agent[:, 0]) / BOARD,
            (food[:, 1] - agent[:, 1]) / BOARD,
            agent[:, 0] / (BOARD - 1),
            agent[:, 1] / (BOARD - 1),
        ],
        dim=-1,
    )


def _lcg_step(rng: torch.Tensor) -> torch.Tensor:
    """Simple LCG on uint64-ish int64 tensor: rng = rng * A + C (mod 2^63-ish)."""
    # Keep positive int64 range.
    return (rng * 6364136223846793005 + 1) & 0x7FFFFFFFFFFFFFFF


def env_step(
    agent: torch.Tensor,
    food: torch.Tensor,
    actions: torch.Tensor,
    rng_state: torch.Tensor,
):
    """Deterministic env step. rng_state: (N,) int64 for food respawns."""
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
        # Advance RNG for every env (stable), use only hits for food write.
        rng_state = _lcg_step(rng_state)
        fx = (rng_state % BOARD).to(agent.dtype)
        rng_state = _lcg_step(rng_state)
        fy = (rng_state % BOARD).to(agent.dtype)
        new_food = torch.stack([fx, fy], dim=-1)
        food = food.clone()
        food[hit] = new_food[hit]
    return agent, food, reward, rng_state


def run(num_envs: int, horizon: int, seed: int, model: Model | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Model()
    model = model.to(device).eval()

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
            obs = obs_from_state(agent, food)
            logits, state, _value = policy_forward(model, obs, state)
            last_logits = logits
            # Greedy actions for determinism (argmax).
            actions = torch.argmax(logits, dim=-1)
            agent, food, r, rng_state = env_step(agent, food, actions, rng_state)
            rewards = rewards + r

    return {
        "rewards": rewards.detach(),
        "positions": agent.detach().round().long(),
        "last_logits": last_logits.detach(),
        "state": state.detach(),
    }


def get_init_inputs():
    return []


def get_inputs():
    return []
