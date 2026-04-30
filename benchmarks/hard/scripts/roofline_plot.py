"""Generate a roofline plot for a single run.

Reads result.json + benchmark.log from a run directory, plots achieved
throughput against hardware peak, with eager/compile/SOTA reference lines.

Usage:
    uv run python scripts/roofline_plot.py outputs/runs/<run_dir>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_benchmark_log(bench_log: Path) -> dict:
    """Pull per-shape, per-variant throughput from the benchmark.py stdout."""
    if not bench_log.exists():
        return {}
    text = bench_log.read_text()
    # benchmark.py prints lines like:
    #   shape=0 variant=solution tflops=123.4 gbps=678.9
    pattern = re.compile(
        r"shape=(\d+)\s+variant=(\S+)\s+tflops=([0-9.]+)\s+gbps=([0-9.]+)"
    )
    results = {}
    for m in pattern.finditer(text):
        shape_idx = int(m.group(1))
        variant = m.group(2)
        tflops = float(m.group(3))
        gbps = float(m.group(4))
        results.setdefault(shape_idx, {})[variant] = {"tflops": tflops, "gbps": gbps}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    result_path = run_dir / "result.json"
    bench_log = run_dir / "benchmark.log"

    if not result_path.exists():
        print(f"no result.json in {run_dir}", file=sys.stderr)
        sys.exit(1)

    result = json.loads(result_path.read_text())
    per_shape = parse_benchmark_log(bench_log)
    if not per_shape:
        print(f"no parseable benchmark output in {bench_log}", file=sys.stderr)
        sys.exit(1)

    variants = ["eager", "compiled", "sota", "solution"]
    colors = {"eager": "#888", "compiled": "#4a90e2", "sota": "#2ecc71", "solution": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    shape_indices = sorted(per_shape.keys())
    x = list(range(len(shape_indices)))

    for v in variants:
        ys = [per_shape[s].get(v, {}).get("tflops", 0) for s in shape_indices]
        if any(y > 0 for y in ys):
            ax.plot(x, ys, marker="o", label=v, color=colors.get(v, "#000"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"shape {s}" for s in shape_indices])
    ax.set_ylabel("TFLOPS")
    ax.set_title(
        f"{result['problem']} — {result['harness']}/{result['model']}"
        f" — peak_fraction={result['peak_fraction']}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    out = run_dir / "roofline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
