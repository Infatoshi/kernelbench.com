"""Browser-based transcript viewer for KernelBench-Hard agent runs.

Generates self-contained static HTML from a run directory's transcript and
artifacts. Supports claude, codex, kimi, cursor, droid harness formats.

Usage:
    uv run python -m src.viewer <run_dir>
    uv run python -m src.viewer compare <run_dir1> <run_dir2> ...
"""
