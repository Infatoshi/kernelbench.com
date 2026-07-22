"""Busbw benchmark — launches the problem under torchrun, times the solution
(500 warmup / 100 timed), reports achieved NVLink busbw and peak_fraction. Thin
wrapper around the shared launcher; do not edit per-problem logic here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.launcher import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(Path(__file__).resolve().parent))
