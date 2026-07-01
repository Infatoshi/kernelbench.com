#!/usr/bin/env python3
"""CLI wrapper around src/eval/megakernel.py.

Emit the deterministic megakernel-authenticity evidence bundle (and optionally
the rendered judge prompt) for a run archive or a solution directory. This feeds
the authenticity judge in the mandatory audit; it does NOT decide authenticity
by itself.

Usage:
  uv run python scripts/megakernel_evidence.py outputs/runs/<run_dir>
  uv run python scripts/megakernel_evidence.py outputs/runs/<run_dir> --prompt --problem 02_kimi_linear_decode
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.megakernel import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
