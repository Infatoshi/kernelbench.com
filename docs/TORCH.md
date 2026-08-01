# Torch version policy

Locked torch versions differ across benches (as of 2026-07-31: hard and mega
lock 2.11.0; cuda, mini, and multi lock 2.13.0; all specify `torch>=2.11`).
This is DELIBERATELY not unified, and the split is documented instead of
"fixed", because torch is not the scored surface:

- **Correctness**: `reference.py` in eager torch is the oracle, and per-dtype
  tolerances absorb minor cross-version numeric drift. Any working torch
  produces the same pass/fail verdicts.
- **Performance**: the published number is the agent kernel's measured time
  against a hardware roofline (peak TFLOPS / bandwidth from `src/hardware/`),
  which torch's version cannot move. Even a slow torch reference would not
  change a roofline grade.
- **ms-anchored problems** (cuda bench): graded against the eager anchor
  FROZEN at deck publication. Upgrading torch later cannot re-grade
  historical cells; anchors are never re-measured on a new torch.
- **Provenance**: `environment_notes` in each leaderboard build now records
  the live torch version via `importlib.metadata`, so every published wave
  self-describes its instrument.

What DOES matter about torch, and is enforced elsewhere:

- **Functionality**: the wheel must match the node's CUDA driver (cu128 vs
  cu130 — see the Lambda/Brev bootstrap notes in AGENTS.md). Launch gates
  probe `torch.cuda.is_available()`, not `nvcc`.
- **torch.compile baselines**: opt-in diagnostics only
  (`KBH_BENCHMARK_BASELINES=1`), never the score. torch 2.11 needs
  `./scripts/patch_torch.sh` for the inductor CSE bug.
- **Mid-generation stability**: do not bump a bench's lock in the middle of a
  published wave — cells within one comparison set should share an
  instrument. Bump between waves freely; the version lands in
  `environment_notes`.
