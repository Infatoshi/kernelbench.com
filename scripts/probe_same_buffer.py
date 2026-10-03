"""Same-buffer overwrite probe for audits (root AGENTS.md publish rule 3).

A solution that caches anything keyed on tensor identity (data_ptr, _version,
a prepared pointer table, a packed weight copy) can return stale results when
the same buffers are overwritten in place. This replays one archived
solution.py against reference.py twice on the same model objects, overwriting
weights (and, for mega, the input buffer) in place between the calls, and prints
cos(out1, out2) (must be low: the output moved) and cos(ref, sol) after the
overwrite (must be at the gate).

Run from inside a regrade workspace problem dir (cwd holds reference.py and
solution.py, `src/` two levels up), on a quiet GPU, with the bench venv:

  uv run python <repo>/scripts/probe_same_buffer.py cuda03|cuda04|mega02

Overwrites go through both `p.copy_` under no_grad (bumps `_version`) and
`p.data.copy_` (does not), since a version-keyed cache only survives the first.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())  # reference.py / solution.py live in the workspace, not next to this script

import torch
import torch.nn.functional as F

import reference as ref
import solution as sol


def cos(a, b) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def fresh(p: torch.Tensor, g: torch.Generator, std: float) -> torch.Tensor:
    t = torch.empty(p.shape, dtype=torch.float32, device="cpu").normal_(0.0, std, generator=g)
    return t.to(p.dtype)


def overwrite(ref_model, sol_model, seed: int, via_data: bool, std: float = 0.02) -> None:
    """Write identical new values into both models' existing parameter buffers."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sp = dict(sol_model.named_parameters())
    for name, p in ref_model.named_parameters():
        if p.dim() < 2:
            continue
        new = fresh(p, g, std).to(p.device)
        for t in (p, sp[name]):
            if via_data:
                t.data.copy_(new)
            else:
                with torch.no_grad():
                    t.copy_(new)


def cuda03() -> None:
    dev = torch.device("cuda:0")
    ctx, dec, seed = 512, 8, 42
    max_seq = ctx + dec + 8
    rm = ref.Model(ref.NUM_LAYERS, max_seq).to(dev).eval()
    overwrite(rm, rm, 1, via_data=False)
    sm = sol.Model(ref.NUM_LAYERS, max_seq).to(dev).eval()
    sm.load_state_dict(rm.state_dict(), strict=True)
    ptrs = [p.data_ptr() for p in sm.parameters()]
    o1 = sol.run(ctx, dec, seed, model=sm)["last_hidden"]
    for via_data in (False, True):
        overwrite(rm, sm, 7 + via_data, via_data)
        assert ptrs == [p.data_ptr() for p in sm.parameters()], "parameter buffers moved"
        r2 = ref.run(ctx, dec, seed, model=rm, max_seq=max_seq)["last_hidden"]
        o2 = sol.run(ctx, dec, seed, model=sm)["last_hidden"]
        print(f"cuda03 via_data={via_data}: cos(out1,out2)={cos(o1, o2):.7f} "
              f"cos(ref,sol)={cos(r2, o2):.7f} max_abs={(r2.float() - o2.float()).abs().max().item():.3e}")
        o1 = o2


def cuda04() -> None:
    dev = torch.device("cuda:0")
    n, horizon, seed = 4096, 32, 42
    rm = ref.Model().to(dev).eval()
    sm = sol.Model().to(dev).eval()
    sm.load_state_dict(rm.state_dict(), strict=True)
    o1 = sol.run(n, horizon, seed, model=sm)
    for via_data in (False, True):
        overwrite(rm, sm, 11 + via_data, via_data, std=0.05)
        r2 = ref.run(n, horizon, seed, model=rm)
        o2 = sol.run(n, horizon, seed, model=sm)
        pos_match = (r2["positions"] == o2["positions"]).all(dim=-1).float().mean().item()
        print(f"cuda04 via_data={via_data}: cos(logits1,logits2)={cos(o1['last_logits'], o2['last_logits']):.7f} "
              f"cos(ref,sol logits)={cos(r2['last_logits'], o2['last_logits']):.7f} "
              f"positions_match={pos_match:.6f} rewards_equal={torch.equal(r2['rewards'], o2['rewards'])}")
        o1 = o2


def mega02() -> None:
    import shapes
    cfg = ref.build_config(shapes.SHAPES[0])
    dev = torch.device("cuda:0")
    rm = ref.Model(cfg).to(dev).eval()
    sm = sol.Model(cfg).to(dev).eval()
    sm.load_state_dict(rm.state_dict(), strict=True)
    ctx, seed = 2048, 0
    st_r = ref.init_state(cfg, ctx, seed)
    st_s = ref.init_state(cfg, ctx, seed)
    h = ref.init_token(cfg, seed)          # one input buffer, reused for every call
    with torch.no_grad():
        o_r, st_r = rm.step(h.clone(), st_r)
        o1, st_s = sm.step(h, st_s)
        print(f"mega02 baseline: cos(ref,sol)={cos(o_r, o1):.7f}")
        for via_data in (False, True):
            h.copy_(ref.init_token(cfg, seed + 10 + via_data))   # same buffer, new values
            # router is an nn.Linear weight; buffers (w_q/scales/zeros) are overwritten too
            g = torch.Generator(device="cpu").manual_seed(21 + via_data)
            sb = dict(sm.named_buffers())
            for name, br in rm.named_buffers():
                bs = sb[name]
                if name.endswith("scales"):
                    new = (br.float() * (0.5 + torch.rand(br.shape, generator=g).to(br.device))).to(br.dtype)
                    for t in (br, bs):
                        (t.data if via_data else t).copy_(new)
            overwrite(rm, sm, 31 + via_data, via_data)
            o_r, st_r = rm.step(h.clone(), st_r)
            o2, st_s = sm.step(h, st_s)
            print(f"mega02 via_data={via_data}: cos(out1,out2)={cos(o1, o2):.7f} cos(ref,sol)={cos(o_r, o2):.7f} "
                  f"S={cos(st_r[0]['S'], st_s[0]['S']):.7f} c_kv={cos(st_r[3]['c_kv'], st_s[3]['c_kv']):.7f}")
            o1 = o2


if __name__ == "__main__":
    {"cuda03": cuda03, "cuda04": cuda04, "mega02": mega02}[sys.argv[1]]()
