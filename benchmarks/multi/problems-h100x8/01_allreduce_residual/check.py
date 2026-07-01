"""Correctness runner — launches the problem under torchrun and compares per-rank
outputs (reference vs solution) with rank-distinct inputs. Thin wrapper around
the shared launcher; do not edit per-problem logic here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.launcher import run_check  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_check(Path(__file__).resolve().parent))
