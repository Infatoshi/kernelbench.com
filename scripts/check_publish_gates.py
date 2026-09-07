#!/usr/bin/env python3
"""Publish gates: fail `kb publish` / `kb deploy` loudly when a finished run
would not actually show up on the site.

Every gate here exists because a real wave shipped without the model on the
homepage (GPT-6 Astra Pro, 2026-09-07: annotations committed, boards rebuilt,
site deployed, and the model still invisible because three hand-maintained
rosters had no entry). Docs alone did not prevent it; this script does.

Gates (all must pass; the message names the exact edit):
  A. roster    every slug in public/data/models.json is in LIVE_MODEL_SLUGS or
               RETIRED_MODEL_SLUGS (app/_lib/models.server.ts) or
               CHART_HIDDEN_SLUGS (app/_lib/models.ts); every LIVE slug and
               its board model ids are in MODEL_NAMES, SHORT_NAMES and
               LIVE_MODEL_IDS (app/_lib/charts.ts) and in MODEL_NAMES of
               scripts/build_model_index.py (else the site shows the raw slug).
  B. mega gpu  every clean/interesting mega annotation whose run dir is
               present carries the `gpu` marker file build_mega_leaderboard.py
               requires (without it the row is silently dropped).
  C. cuda      every clean/interesting, correct, RTX cuda annotation is in
               benchmarks/cuda/results/published_runs.json run_ids, or listed
               under its "excluded" map with a reason.
  D. tracked   no untracked annotation YAML (an untracked one ships the cell
               as unaudited and the homepage chart drops it).

Usage: uv run python scripts/check_publish_gates.py   (exit 1 on any failure)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ts_set(path: Path, name: str) -> set[str]:
    text = path.read_text()
    if name not in text:
        return set()
    body = text.split(name, 1)[1].split("])", 1)[0]
    return set(re.findall(r'"([^"]+)"', body))


def ts_record_keys(path: Path, name: str) -> set[str]:
    text = path.read_text()
    if name not in text:
        return set()
    body = text.split(name, 1)[1].split("\n}", 1)[0]
    keys = set(re.findall(r'^\s*"([^"]+)"\s*:', body, re.M))
    keys |= set(re.findall(r"^\s*([A-Za-z0-9_]+)\s*:", body, re.M))
    return keys


def py_dict_keys(path: Path, name: str) -> set[str]:
    text = path.read_text()
    body = text.split(f"{name} = {{", 1)[1].split("\n}", 1)[0]
    return set(re.findall(r'^\s*"([^"]+)"\s*:', body, re.M))


def yaml_field(text: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
    return (m.group(1).strip().strip('"').strip("'") if m else "")


_TAG_TAIL = re.compile(r"\s*\[[^\]]*\]\s*$")


def slugify(model_field: str) -> str:
    """Mirror of scripts/build_model_index.py slugify (kept import-free so the
    gate needs no pyyaml). If that one changes, change this one."""
    s = (model_field or "").strip().lower()
    s = _TAG_TAIL.sub(lambda m: "-" + re.sub(r"[^a-z0-9]", "", m.group(0)), s)
    s = re.sub(r"_1m_?$", "-1m", s)
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return s


def catalog_model_ids() -> set[str]:
    """Every `model` id the boards ship (what app/_lib/charts.ts keys on)."""
    text = (REPO / "public/data/catalog.json").read_text()
    return set(re.findall(r'"model"\s*:\s*"([^"]+)"', text))


def gate_roster(fail: list[str]) -> None:
    server = REPO / "app/_lib/models.server.ts"
    charts = REPO / "app/_lib/charts.ts"
    live = ts_set(server, "LIVE_MODEL_SLUGS = new Set([")
    retired = ts_set(server, "RETIRED_MODEL_SLUGS = new Set([")
    hidden = ts_set(REPO / "app/_lib/models.ts", "CHART_HIDDEN_SLUGS = new Set([")
    names = ts_record_keys(charts, "MODEL_NAMES: Record<string, string> = {")
    shorts = ts_record_keys(charts, "SHORT_NAMES: Record<string, string> = {")
    live_ids = ts_set(charts, "LIVE_MODEL_IDS = new Set([")
    index_names = py_dict_keys(REPO / "scripts/build_model_index.py", "MODEL_NAMES")
    models = json.loads((REPO / "public/data/models.json").read_text())["models"]
    board_ids = catalog_model_ids()
    for m in models:
        slug = m["slug"]
        if slug not in live | retired | hidden:
            fail.append(
                f"[A roster] new model slug '{slug}' is in models.json but in none of "
                f"LIVE_MODEL_SLUGS / RETIRED_MODEL_SLUGS (app/_lib/models.server.ts) or "
                f"CHART_HIDDEN_SLUGS (app/_lib/models.ts). Add it to LIVE_MODEL_SLUGS to "
                f"show it on the homepage (the normal case for a fresh wave), or to "
                f"RETIRED_MODEL_SLUGS if it is deliberately off the homepage."
            )
        if slug not in live:
            continue
        if slug not in index_names:
            fail.append(
                f"[A name] live slug '{slug}' has no display name in MODEL_NAMES of "
                f"scripts/build_model_index.py (site would show the raw slug). Add "
                f"'\"{slug}\": \"<Display Name>\",' then rerun kb publish."
            )
        ids = sorted(i for i in board_ids if slugify(i) == slug)
        for mid in ids:
            for table, have in (("MODEL_NAMES", names), ("SHORT_NAMES", shorts), ("LIVE_MODEL_IDS", live_ids)):
                if mid not in have:
                    fail.append(
                        f"[A charts] board model id '{mid}' (live slug {slug}) is missing from "
                        f"{table} in app/_lib/charts.ts; the chart would label it '{mid}' "
                        f"and treat it as a removed model. Add the key."
                    )


def annotations(bench: str):
    for f in sorted((REPO / "benchmarks" / bench / "results/annotations").glob("*.yaml")):
        yield f.stem, f.read_text()


def gate_mega_marker(fail: list[str]) -> None:
    run_roots = [REPO / "benchmarks/mega/outputs/runs"] + sorted((REPO / "benchmarks/mega/outputs").glob("runs-*"))
    for rid, text in annotations("mega"):
        if yaml_field(text, "verdict") not in ("clean", "interesting"):
            continue
        for root in run_roots:
            d = root / rid
            if d.is_dir() and not (d / "gpu").exists():
                fail.append(
                    f"[B mega gpu] {d.relative_to(REPO)} has a clean annotation but no `gpu` "
                    f"marker; build_mega_leaderboard.py drops it silently. Write the GPU label "
                    f"(e.g. `RTX PRO 6000 Blackwell`, `H100`, `B200`) to that file and republish."
                )


def gate_cuda_manifest(fail: list[str]) -> None:
    mf = REPO / "benchmarks/cuda/results/published_runs.json"
    data = json.loads(mf.read_text())
    published = set(data.get("run_ids", []))
    excluded = data.get("excluded", {})
    for rid, text in annotations("cuda"):
        if yaml_field(text, "verdict") not in ("clean", "interesting"):
            continue
        if yaml_field(text, "correct").lower() != "true":
            continue
        if "RTX" not in yaml_field(text, "gpu"):
            continue
        if rid in published or rid in excluded:
            continue
        fail.append(
            f"[C cuda manifest] {rid} is audited clean and correct on RTX but absent from "
            f"benchmarks/cuda/results/published_runs.json; the RTX cuda board only publishes "
            f"run_ids listed there. Add it to run_ids (or to \"excluded\" with a reason) and "
            f"rerun kb publish cuda."
        )


def gate_tracked(fail: list[str]) -> None:
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--",
         "benchmarks/cuda/results/annotations", "benchmarks/mega/results/annotations"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.split()
    for f in out:
        fail.append(f"[D tracked] {f} is not git-tracked; models.json ships it as unaudited. `git add` it and rerun kb publish.")


def main() -> int:
    fail: list[str] = []
    gate_roster(fail)
    gate_mega_marker(fail)
    gate_cuda_manifest(fail)
    gate_tracked(fail)
    if fail:
        print("PUBLISH GATES FAILED (the site would not show what you just published):", file=sys.stderr)
        for f in fail:
            print("  - " + f, file=sys.stderr)
        print(f"{len(fail)} gate failure(s). Fix, then rerun kb publish.", file=sys.stderr)
        return 1
    print("kb: publish gates OK (roster, mega gpu marker, cuda manifest, tracked annotations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
