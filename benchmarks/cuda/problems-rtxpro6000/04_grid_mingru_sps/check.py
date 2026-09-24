"""Correctness + CUDA language gate for grid+MinGRU SPS.

Checks:
  1. Language gate (CUDA evidence, no Triton/DSL)
  2. policy_forward logits/state match reference on fixed inputs
  3. env_step matches on fixed (agent, food, actions, rng)
  4. Greedy run() positions/rewards/logits match at (128, 8) and at every
     graded shape; a position mismatch is excused only where the reference
     picked its action by a top-2 logit gap under position_tie_margin
"""
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval import cuda_language as cl  # noqa: E402
from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.cuda_language import collect_solution_sources  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)


def main():
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    meta = yaml.safe_load(Path("problem.yaml").read_text()) if Path("problem.yaml").exists() else {}
    sol_src = collect_solution_sources(Path("."))

    for forbidden in meta.get("forbidden", []):
        if re.search(re.escape(forbidden), sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    ok, messages, report = cl.check_cuda_language(sol_src, meta)
    Path("cuda_language.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("framework.txt").write_text(report["framework"] + "\n")
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    print(
        f"cuda_language: ok framework={report['framework']} "
        f"evidence={','.join(report['cuda_evidence']) or 'none'}"
    )

    if not hasattr(solution, "Model"):
        print("FAIL: solution.py must define class Model")
        sys.exit(1)
    if not hasattr(solution, "run"):
        print("FAIL: solution.py must define run(num_envs, horizon, seed, model=None)")
        sys.exit(1)

    device = torch.device("cuda:0")
    tol = float((meta.get("logit_tolerance") or {}).get("float32", 1e-3))
    margin = float(meta.get("position_tie_margin", 1e-5))

    for seed in (42, 123, 456):
        ref_model = reference.Model().to(device).eval()
        ref_model.reset_parameters(seed)
        sol_model = solution.Model().to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch: {e}")
            sys.exit(1)

        n = 256
        torch.manual_seed(seed)
        obs = torch.randn(n, reference.OBS_DIM, device=device)
        state = torch.randn(n, reference.GRU_LAYERS, reference.HIDDEN, device=device) * 0.1

        for case in numeric_stress_cases(meta.get("name", "")):
            with numeric_stress_context(ref_model, sol_model, [obs, state], case) as (c_obs, c_state):
                with torch.no_grad():
                    r_logits, r_state, r_val = reference.policy_forward(ref_model, c_obs, c_state)
                    if hasattr(solution, "policy_forward"):
                        s_logits, s_state, s_val = solution.policy_forward(sol_model, c_obs, c_state)
                    else:
                        s_logits, s_state, s_val = sol_model(c_obs, c_state)

            case_tol = tolerance_for_case({"float32": tol}, case)
            for name, ref_t, sol_t in [
                ("logits", r_logits, s_logits),
                ("state", r_state, s_state),
                ("value", r_val, s_val),
            ]:
                ok, msg = check_correctness(
                    ref_t.float(), sol_t.float(), dtype=torch.float32, override=case_tol
                )
                if not ok:
                    print(f"FAIL: seed {seed} case {case.name} policy_forward {name}: {msg}")
                    sys.exit(1)

        # env_step with fixed actions
        agent = torch.randint(0, reference.BOARD, (n, 2), device=device).float()
        food = torch.randint(0, reference.BOARD, (n, 2), device=device).float()
        actions = torch.randint(0, 4, (n,), device=device)
        rng = torch.arange(n, device=device, dtype=torch.int64) + seed

        r_agent, r_food, r_rew, r_rng = reference.env_step(agent, food, actions, rng)
        if hasattr(solution, "env_step"):
            s_agent, s_food, s_rew, s_rng = solution.env_step(agent, food, actions, rng)
            if not torch.equal(r_agent, s_agent) or not torch.equal(r_food, s_food):
                print(f"FAIL: seed {seed} env_step positions/food mismatch")
                sys.exit(1)
            if not torch.allclose(r_rew, s_rew, atol=0, rtol=0):
                print(f"FAIL: seed {seed} env_step reward mismatch")
                sys.exit(1)
            if not torch.equal(r_rng, s_rng):
                print(f"FAIL: seed {seed} env_step rng_state mismatch")
                sys.exit(1)

        # Full rollouts: the short (128, 8) run plus every graded shape. The
        # check used to stop at (128, 8), which never ran the paths a kernel
        # takes at graded env counts (DEVLOG 2026-09-24).
        rollouts = [(128, 8)] + [(s["num_envs"], s["horizon"]) for s in shapes.SHAPES]
        for n_envs, horizon in rollouts:
            msg = _check_rollout(reference, solution, ref_model, sol_model, n_envs, horizon, seed, tol, margin)
            if msg:
                print(f"FAIL: seed {seed} run({n_envs}, {horizon}) {msg}")
                sys.exit(1)

    print("PASS")


def _ref_rollout(reference, model, n, horizon, seed):
    """reference.run step for step, also keeping per-step positions and the
    top-2 logit gap that decided each greedy action."""
    device = torch.device("cuda:0")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    agent = torch.randint(0, reference.BOARD, (n, 2), generator=g).float().to(device)
    food = torch.randint(0, reference.BOARD, (n, 2), generator=g).float().to(device)
    rng = torch.arange(n, device=device, dtype=torch.int64) + (seed * 10007)
    state = torch.zeros(n, reference.GRU_LAYERS, reference.HIDDEN, device=device)
    rewards = torch.zeros(n, device=device)
    positions, gaps = [], []
    with torch.no_grad():
        for _t in range(horizon):
            obs = reference.obs_from_state(agent, food)
            logits, state, _value = reference.policy_forward(model, obs, state)
            top2 = logits.topk(2, dim=-1).values
            gaps.append(top2[:, 0] - top2[:, 1])
            agent, food, r, rng = reference.env_step(agent, food, torch.argmax(logits, dim=-1), rng)
            rewards = rewards + r
            positions.append(agent.round().long())
    return torch.stack(positions), torch.stack(gaps), rewards, logits


def _sol_run(solution, sol_model, n, horizon, seed):
    try:
        return solution.run(n, horizon, seed, model=sol_model)
    except TypeError:
        return solution.run(n, horizon, seed)


def _check_rollout(reference, solution, ref_model, sol_model, n, horizon, seed, tol, margin):
    """Return a failure message, or None.

    Final positions must equal the reference, except for an env whose first
    divergence is a greedy action the reference itself decided by a top-2
    logit gap below `margin`: a near-tie that any reordering of fp32 sums may
    flip. Such envs are also left out of the reward and last-logit checks,
    since their trajectories legitimately differ from that step on.
    """
    device = torch.device("cuda:0")
    ref_pos, ref_gap, ref_rewards, ref_logits = _ref_rollout(reference, ref_model, n, horizon, seed)
    # Guard against this copy drifting from the oracle it mirrors.
    if not torch.equal(reference.run(n, horizon, seed, model=ref_model)["positions"].to(device), ref_pos[-1]):
        return "internal: logged reference rollout differs from reference.run"
    out = _sol_run(solution, sol_model, n, horizon, seed)
    sol_pos = out["positions"].to(device).long()
    differs = (sol_pos != ref_pos[-1]).any(-1)
    keep = torch.ones(n, dtype=torch.bool, device=device)
    if differs.any():
        # Locate each differing env's first divergence by replaying shorter horizons.
        first = torch.full((n,), -1, dtype=torch.long, device=device)
        for h in range(1, horizon + 1):
            pos_h = _sol_run(solution, sol_model, n, h, seed)["positions"].to(device).long()
            first[((pos_h != ref_pos[h - 1]).any(-1)) & (first < 0)] = h
        idx = torch.nonzero(differs).squeeze(1)
        steps = first[idx]
        if (steps < 1).any():
            return (f"positions: {int((steps < 1).sum())} env(s) differ after {horizon} steps "
                    "but at no shorter horizon (horizon-dependent result)")
        gap = ref_gap[steps - 1, idx]
        real = gap >= margin
        if real.any():
            return (f"positions mismatch on {int(real.sum())} of {n} envs at reference top-2 logit "
                    f"gaps up to {gap[real].max().item():.3g} (tie margin {margin:g})")
        keep[idx] = False
    if not torch.allclose(ref_rewards[keep].float(), out["rewards"].to(device)[keep].float(), atol=1e-5):
        return "rewards mismatch"
    ok, msg = check_correctness(
        ref_logits[keep].float().cpu(),
        out["last_logits"].to(device)[keep].float().cpu(),
        dtype=torch.float32,
        override={"float32": tol},
    )
    return None if ok else f"last_logits: {msg}"


if __name__ == "__main__":
    main()
