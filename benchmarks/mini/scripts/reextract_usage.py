"""Re-run token extraction over existing runs and patch result.json in place.

Recovers usage for runs whose transcript has a terminal result event but whose
result.json.usage is null/zero (e.g. the *-claude dispatch gap). Does NOT touch
runs that have no recoverable tokens. Usage:

    uv run python scripts/reextract_usage.py [runs_dir ...]

Defaults to outputs/runs. Prints a one-line summary of how many were patched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_usage import extract  # noqa: E402


def _harness_of(run_dir: Path, result: dict) -> str | None:
    h = result.get("harness")
    if h:
        return h
    # fall back to run-id form <ts>_<harness>_<model>_<problem>
    parts = run_dir.name.split("_")
    return parts[2] if len(parts) > 2 else None


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]] or [ROOT / "outputs/runs"]
    patched = scanned = skipped = 0
    for base in dirs:
        if not base.exists():
            continue
        for rj in sorted(base.glob("2026*/result.json")):
            scanned += 1
            try:
                r = json.load(open(rj))
            except Exception:
                continue
            run_dir = rj.parent
            harness = _harness_of(run_dir, r)
            if not harness:
                continue
            new = extract(run_dir, harness)
            old = r.get("usage") or {}
            old_out = old.get("output_tokens")
            new_out = new.get("output_tokens")
            # Only patch when re-extraction recovers a real, larger token count.
            if new_out and (not old_out or new_out > old_out):
                r["usage"] = new
                json.dump(r, open(rj, "w"), indent=2)
                patched += 1
            else:
                skipped += 1
    print(f"reextract: scanned={scanned} patched={patched} unchanged={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
