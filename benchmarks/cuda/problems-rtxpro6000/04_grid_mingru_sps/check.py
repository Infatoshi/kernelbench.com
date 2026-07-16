"""Correctness + CUDA language gate for grid+MinGRU SPS.

Checks:
  1. Language gate (CUDA evidence, no Triton/DSL)
  2. policy_forward logits/state match reference on fixed inputs
  3. env_step matches on fixed (agent, food, actions, rng)
  4. Short greedy run() positions/rewards/logits match
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

        # Short full run
        ref_out = reference.run(128, 8, seed, model=ref_model)
        try:
            sol_out = solution.run(128, 8, seed, model=sol_model)
        except TypeError:
            sol_out = solution.run(128, 8, seed)

        if not torch.equal(ref_out["positions"].cpu(), sol_out["positions"].cpu()):
            print(f"FAIL: seed {seed} run() positions mismatch")
            sys.exit(1)
        if not torch.allclose(
            ref_out["rewards"].float().cpu(), sol_out["rewards"].float().cpu(), atol=1e-5
        ):
            print(f"FAIL: seed {seed} run() rewards mismatch")
            sys.exit(1)
        ok, msg = check_correctness(
            ref_out["last_logits"].float().cpu(),
            sol_out["last_logits"].float().cpu(),
            dtype=torch.float32,
            override={"float32": tol},
        )
        if not ok:
            print(f"FAIL: seed {seed} run() last_logits: {msg}")
            sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
