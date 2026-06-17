"""PyTorch reference for the grid-foraging PPO training megakernel.

This is the correctness oracle AND the throughput floor. It implements the
whole RL loop -- a vectorized grid-foraging environment, a tiny MLP
actor-critic, and a PPO update -- as ordinary (un-fused) PyTorch. Every step
launches many small kernels and round-trips through the Python interpreter;
that launch/dispatch overhead is exactly what a fused training megakernel is
meant to remove.

The grading contract (what check.py and benchmark.py call):

    train(total_env_steps: int, seed: int) -> list[float]

Run a full PPO training run *from scratch* for `total_env_steps` environment
steps, fully determined by `seed`, and return the per-iteration mean episodic
return (the "return curve"). solution.py must expose the same function and
learn the same task to the same return level; it is free to fuse, compile, or
hand-write kernels however it likes, but it must implement the same MDP and the
same PPO update so its return curve matches this reference's.

The environment, network, and PPO hyperparameters below are the canonical
definition of the task. They are fixed: a submission that changes the MDP
(grid size, horizon, reward) or the learning signal is not solving this
problem.
"""
from __future__ import annotations

import torch
import torch.nn as nn

OP_TYPE = "rl_grid_ppo"
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

# --- Environment: vectorized grid foraging ----------------------------------
GRID = 11          # GRID x GRID board
NUM_ENVS = 4096    # parallel environments stepped together
HORIZON = 32       # steps per episode; env auto-resets at the boundary
OBS_DIM = 4        # (food-agent dx, dy, agent x, agent y), normalized
N_ACT = 4          # up, down, left, right

# --- Policy: tiny MLP actor-critic ------------------------------------------
HIDDEN = 64

# --- PPO ---------------------------------------------------------------------
ROLLOUT = HORIZON  # one fresh episode per rollout (aligned with HORIZON)
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCHES = 4
LR = 3.0e-3
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Action deltas indexed by action id: 0=up, 1=down, 2=left, 3=right.
# (dx, dy) on an (x, y) board; positions are clamped to [0, GRID-1].
_ACT_DX = (0, 0, -1, 1)
_ACT_DY = (-1, 1, 0, 0)


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(OBS_DIM, HIDDEN)
        self.pi = nn.Linear(HIDDEN, N_ACT)
        self.vf = nn.Linear(HIDDEN, 1)

    def forward(self, obs: torch.Tensor):
        h = torch.tanh(self.body(obs))
        return self.pi(h), self.vf(h).squeeze(-1)


class GridForage:
    """Vectorized grid-foraging MDP. Reward +1 for stepping onto the food,
    which then respawns at a uniformly random cell. Agent persists; the whole
    board re-randomizes every HORIZON steps."""

    def __init__(self, num_envs: int, device, gen: torch.Generator) -> None:
        self.n = num_envs
        self.device = device
        self.gen = gen
        self.dx = torch.tensor(_ACT_DX, device=device)
        self.dy = torch.tensor(_ACT_DY, device=device)
        self.reset()

    def _rand_cell(self) -> torch.Tensor:
        return torch.randint(0, GRID, (self.n,), device=self.device, generator=self.gen)

    def reset(self) -> torch.Tensor:
        self.ax = self._rand_cell()
        self.ay = self._rand_cell()
        self.fx = self._rand_cell()
        self.fy = self._rand_cell()
        return self._obs()

    def _obs(self) -> torch.Tensor:
        scale = 1.0 / (GRID - 1)
        return torch.stack(
            [
                (self.fx - self.ax).float() / GRID,
                (self.fy - self.ay).float() / GRID,
                self.ax.float() * scale,
                self.ay.float() * scale,
            ],
            dim=-1,
        )

    def step(self, action: torch.Tensor):
        self.ax = (self.ax + self.dx[action]).clamp_(0, GRID - 1)
        self.ay = (self.ay + self.dy[action]).clamp_(0, GRID - 1)
        caught = (self.ax == self.fx) & (self.ay == self.fy)
        reward = caught.float()
        if caught.any():
            nfx = self._rand_cell()
            nfy = self._rand_cell()
            self.fx = torch.where(caught, nfx, self.fx)
            self.fy = torch.where(caught, nfy, self.fy)
        return self._obs(), reward


def _gae(rewards, values, last_value):
    """Generalized advantage estimation over a (T, N) rollout with the episode
    ending at T (no bootstrap past the horizon)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in range(T - 1, -1, -1):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 0.0 if t == T - 1 else 1.0
        delta = rewards[t] + GAMMA * next_v * nonterminal - values[t]
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def train(total_env_steps: int, seed: int) -> list[float]:
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    torch.manual_seed(seed)

    policy = Policy().to(device)
    # Deterministic init from the same generator.
    with torch.no_grad():
        for p in policy.parameters():
            if p.dim() > 1:
                bound = (1.0 / p.shape[1]) ** 0.5
                p.uniform_(-bound, bound, generator=gen)
            else:
                p.zero_()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    env = GridForage(NUM_ENVS, device, gen)
    iters = max(1, total_env_steps // (ROLLOUT * NUM_ENVS))
    curve: list[float] = []

    for _ in range(iters):
        obs = env.reset()
        obs_buf = torch.empty(ROLLOUT, NUM_ENVS, OBS_DIM, device=device)
        act_buf = torch.empty(ROLLOUT, NUM_ENVS, dtype=torch.long, device=device)
        logp_buf = torch.empty(ROLLOUT, NUM_ENVS, device=device)
        val_buf = torch.empty(ROLLOUT, NUM_ENVS, device=device)
        rew_buf = torch.empty(ROLLOUT, NUM_ENVS, device=device)

        for t in range(ROLLOUT):
            with torch.no_grad():
                logits, value = policy(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                logp = torch.log_softmax(logits, dim=-1).gather(-1, action[:, None]).squeeze(-1)
            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value
            obs, reward = env.step(action)
            rew_buf[t] = reward

        with torch.no_grad():
            _, last_value = policy(obs)
            adv, ret = _gae(rew_buf, val_buf, last_value)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        n = b_obs.shape[0]
        mb = n // MINIBATCHES

        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=device, generator=gen)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                logits, value = policy(b_obs[idx])
                logsm = torch.log_softmax(logits, dim=-1)
                new_logp = logsm.gather(-1, b_act[idx][:, None]).squeeze(-1)
                entropy = -(logsm * torch.softmax(logits, dim=-1)).sum(-1).mean()
                ratio = torch.exp(new_logp - b_logp[idx])
                a = b_adv[idx]
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                vloss = (value - b_ret[idx]).pow(2).mean()
                loss = pg + VF_COEF * vloss - ENT_COEF * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

        curve.append(float(rew_buf.sum(0).mean().item()))

    return curve


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    c = train(ROLLOUT * NUM_ENVS * 40, seed=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    steps = ROLLOUT * NUM_ENVS * len(c)
    print(f"iters={len(c)} return_first={c[0]:.3f} return_last={c[-1]:.3f}")
    print(f"elapsed={dt:.2f}s sps={steps / dt:,.0f}")
