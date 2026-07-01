"""Correctness runner (thin wrapper around the shared launcher)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.launcher import run_check  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_check(Path(__file__).resolve().parent))
