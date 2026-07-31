"""Sequential isolated re-grade — the only source of publishable numbers.

For each run, rebuild a clean workspace from the PRISTINE deck, drop in only the
agent-authored files, and run check.py then benchmark.py one cell at a time on a
quiet node. In-run numbers are never publishable: the agent's session may have
overlapped a co-tenant, and its workspace may contain edited benchmark files.
Rebuilding from the pristine deck means a solution that only passed because it
tampered with the grader fails here.

    python3 scripts/regrade.py                       # every run in outputs/runs
    python3 scripts/regrade.py outputs/runs/<run_id>

Writes results/annotations/<run_id>.yaml with the clean figures.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
DECK = BENCH_ROOT / "problems-h100x4"
RUNS = BENCH_ROOT / "outputs" / "runs"
SCRATCH = BENCH_ROOT / "outputs" / "regrade"
ANNOTATIONS = BENCH_ROOT / "results" / "annotations"

# Rebuilt from the deck every time; an agent copy of these is never trusted.
IMMUTABLE = ["reference.py", "check.py", "benchmark.py", "shapes.py",
             "problem.yaml", "sota.py", "PROMPT.txt"]


def preflight_quiet_node() -> None:
    if os.environ.get("KBM_ALLOW_BUSY") == "1":
        print("WARNING: KBM_ALLOW_BUSY=1 — re-grade may be contended.", flush=True)
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    busy = [f"GPU{i.strip()}={u.strip()}MiB"
            for i, u in (ln.split(",") for ln in out.strip().splitlines())
            if int(u) > 2048]
    if busy:
        raise SystemExit(
            "REFUSING to re-grade: another CUDA tenant on " + ", ".join(busy) + ".\n"
            "Published numbers must come from a quiet node (KBM_ALLOW_BUSY=1 to override).")


def problem_of(run_dir: Path) -> str | None:
    ws = run_dir / "ws" / "problems-h100x4"
    if not ws.is_dir():
        return None
    subs = [d for d in ws.iterdir() if d.is_dir()]
    if len(subs) == 1:
        return subs[0].name
    # agents sometimes leave scratch dirs beside the problem dir; the run id
    # ends with the problem name, so use that to disambiguate
    named = [d for d in subs if run_dir.name.endswith(d.name)]
    return named[0].name if len(named) == 1 else None


def build_workspace(run_dir: Path, problem: str) -> Path | None:
    src = run_dir / "ws" / "problems-h100x4" / problem
    if not (src / "solution.py").is_file():
        return None
    dst = SCRATCH / run_dir.name / "problems-h100x4" / problem
    if dst.exists():
        shutil.rmtree(dst.parent.parent)
    dst.mkdir(parents=True)
    for f in IMMUTABLE:                      # pristine, never the agent's copy
        shutil.copy2(DECK / problem / f, dst / f)
    for f in sorted(src.iterdir()):          # agent-authored files only
        if f.name in IMMUTABLE or f.name == "__pycache__" or not f.is_file():
            continue
        shutil.copy2(f, dst / f.name)
    link = SCRATCH / run_dir.name / "src"
    if not link.exists():
        link.symlink_to(BENCH_ROOT / "src")
    return dst


def run_stage(workspace: Path, stage: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, f"{stage}.py"], cwd=workspace,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    preflight_quiet_node()
    targets = [Path(a).resolve() for a in sys.argv[1:]] or sorted(
        d for d in RUNS.iterdir() if d.is_dir())
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in targets:
        problem = problem_of(run_dir)
        if not problem:
            print(f"skip {run_dir.name}: no single problem workspace")
            continue
        ws = build_workspace(run_dir, problem)
        if ws is None:
            print(f"skip {run_dir.name}: no solution.py")
            rows.append((run_dir.name, "no_solution", ""))
            continue
        print(f"\n=== {run_dir.name}", flush=True)
        rc, out = run_stage(ws, "check")
        passed = bool(re.search(r"^PASS", out, re.M)) and rc == 0
        headline, value = "", ""
        if passed:
            rc_b, out_b = run_stage(ws, "benchmark")
            m = re.search(r"^(speedup|peak_fraction): ([0-9.]+)", out_b, re.M)
            if m:
                headline, value = m.group(1), m.group(2)
            dev = re.search(r"^device: (.+)$", out_b, re.M)
            device = dev.group(1) if dev else "unknown"
        else:
            device = "n/a"
            first = re.search(r"^(FAIL.*)$", out, re.M)
            print("  check FAILED:", first.group(1)[:160] if first else f"rc={rc}")
        print(f"  check={'PASS' if passed else 'FAIL'} {headline}={value} [{device}]")
        rows.append((run_dir.name, "PASS" if passed else "FAIL", f"{headline}={value}"))

        ann = ANNOTATIONS / f"{run_dir.name}.yaml"
        prev = ann.read_text() if ann.exists() else ""
        body = [f"run_id: {run_dir.name}", f"problem: {problem}",
                f"regrade_check: {'PASS' if passed else 'FAIL'}"]
        if headline:
            body.append(f"{headline}_clean: {value}   # sequential isolated re-grade, quiet node")
        body.append(f"regrade_device: {device}")
        body.append("regrade_workspace: rebuilt from pristine deck (agent copies of "
                    "reference/check/benchmark/shapes/problem.yaml/sota discarded)")
        ann.write_text("\n".join(body) + "\n" + prev)

    print("\n=== summary")
    for name, verdict, val in rows:
        print(f"  {name[:60]:<60} {verdict:<5} {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
