"""Reward-hack lint — a TRIPWIRE, not a verdict.

Statically scans solution.py for the hack patterns we have actually caught by
hand (DEVLOG audits) plus the anti-hack rules from competition harnesses. This
command reports findings for audit. Bundle-era publication separately imports
the same policy and vetoes only high-confidence HACK hits; FLAG remains manual
review. "Greps are tripwires, not audits."

Usage:
  python scripts/reward_hack_lint.py <run_id>     # one run
  python scripts/reward_hack_lint.py --all        # every run referenced by the live leaderboard.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))
from scripts.published_submission import (  # noqa: E402
    open_verified_submission,
    read_bounded_text,
    read_json_file,
    trusted_archive_lock,
    validate_run_id,
)
from scripts.reward_hack_tripwires import lint_source  # noqa: E402
from scripts.submission_bundle import BundleError, load_run_result  # noqa: E402


def lint_one(sol: Path):
    src = read_bounded_text(sol, errors="ignore")
    return lint_source(src)


def report(run_id: str):
    run_id = validate_run_id(run_id)
    d = ROOT / "outputs/runs" / run_id
    # grader tampering is caught by the template-mutation guard, not in solution.py
    try:
        if load_run_result(d).get("template_mutated"):
            print(f"  FLAGGED {run_id}")
            print(
                "      [HACK] template mutation: agent edited a grader file "
                "(problem.yaml/check.py/...). See template_mutations.log."
            )
            return 1
    except BundleError as exc:
        print(f"  FLAGGED {run_id}")
        print(f"      [HACK] unsafe result archive: {exc}")
        return 1
    try:
        with open_verified_submission(d) as view:
            has_kernel, hits = lint_one(view.solution)
    except Exception as exc:
        print(f"  FLAGGED {run_id}")
        print(f"      [HACK] unverifiable submission archive: {exc}")
        return 1
    if not hits:
        print(f"  CLEAN  {run_id}  (authored kernel: {has_kernel})")
        return 0
    print(f"  FLAGGED {run_id}  (authored kernel: {has_kernel})")
    for sev, label, line, snip, why in hits:
        print(f"      [{sev}] {label} @L{line}: {snip!r}  — {why}")
    return 1


def main(argv):
    with trusted_archive_lock():
        if not argv:
            print(__doc__)
            return 2
        if argv[0] == "--all":
            # Lint the LIVE published board (what the site serves), not a stale copy.
            lb_path = ROOT / "results/leaderboard.json"
            if not lb_path.exists() and not lb_path.is_symlink():
                lb_path = ROOT / "results/leaderboard_v2.json"
            lb = read_json_file(lb_path)
            rids = sorted(
                {
                    validate_run_id(c["run_id"])
                    for m in lb["models"]
                    for c in m["results"].values()
                    if c.get("run_id")
                }
            )
            n = sum(report(r) for r in rids)
            print(f"\n{n}/{len(rids)} runs flagged for audit.")
        else:
            report(argv[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
