#!/usr/bin/env python3
"""Build public/data/models.json — the model-centric index for kernelbench.com.

Joins every published board (hard v2 + h100/b200 variants, mega CSV, cuda v2)
with the reward-hack annotations from all three benches into one artifact keyed
by canonical model slug. Consumed by app/_lib/models.ts (homepage models
section, /models/[slug] pages, model-list bench pages).

Ground rules (mirrors AGENTS.md):
- Verdicts come ONLY from annotations/*.yaml (audited truth). Lint is a
  tripwire and persists nowhere; do not invent verdicts here.
- A run is a FLAG when its annotation verdict is in FLAG_VERDICTS, or (mega)
  when megakernel_authentic is false. `interesting`, `bug`, `harness_limited`,
  `fail` are NOT hacks.
- Join annotations by the `model:` field (slugified), never by harness strings
  or run_id parsing.
- Legacy v1 hard rows are date-tagged snapshots of an evolving prompt set; they
  become a per-model footnote, never ranking data.

Run: uv run --with pyyaml python scripts/build_model_index.py
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "public" / "data" / "models.json"

FLAG_VERDICTS = {"reward_hack", "contamination", "rubric_leak"}

# HF trace datasets per bench; omit a key (or set None) only when that
# dataset is not published yet — site then drops the trace link.
TRACE_DATASETS = {
    "hard": "kernelbench-hard-traces",
    "mega": "kernelbench-mega-traces",
    "cuda": "kernelbench-cuda-traces",
}

# Pretty names keyed by canonical slug. Keep in sync with MODEL_NAMES in
# app/_lib/charts.ts (that map is keyed by raw chart ids; this one by slug).
MODEL_NAMES = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-fable-5": "Claude Fable 5",
    "glm-5.2": "GLM-5.2",
    "glm-5.1": "GLM-5.1",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "grok-4.5": "Grok 4.5",
    "grok-build": "Grok Build",
    "minimax-m3": "MiniMax-M3",
    "kimi-k2.7-code": "Kimi K2.7-Code",
    "kimi-k2.6": "Kimi K2.6",
    "kinetic-0715": "Kimi K3 (256k)",
    "kinetic-0715-1m": "Kimi K3 (1M)",
    "composer-2.5-fast": "Composer 2.5 Fast",
    "composer-2.5": "Composer 2.5",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "longcat-2.0": "LongCat 2.0",
    "hy3": "Tencent Hy3",
    "qwen3.6-max-preview": "Qwen 3.6 Max Preview",
    "qwen3.6-plus": "Qwen 3.6 Plus",
    "qwen3.6-27b": "Qwen 3.6 27B",
    "qwen3.7-max": "Qwen 3.7 Max",
    "mimo-v2.5-pro": "MiMo V2.5 Pro",
    "minimax-m2.7": "MiniMax M2.7",
    "nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
}

LAB_BY_PREFIX = [
    ("claude", "Anthropic"),
    ("gpt", "OpenAI"),
    ("glm", "Z.ai"),
    ("kimi", "Moonshot AI"),
    ("kinetic", "Moonshot AI"),
    ("grok", "xAI"),
    ("composer", "Cursor"),
    ("gemini", "Google"),
    ("deepseek", "DeepSeek"),
    ("minimax", "MiniMax"),
    ("longcat", "Meituan"),
    ("hy3", "Tencent"),
    ("qwen", "Alibaba"),
    ("mimo", "Xiaomi"),
    ("nemotron", "NVIDIA"),
    ("fugu", "Sakana AI"),
]

_TAG_TAIL = re.compile(r"\s*\[[^\]]*\]\s*$")
_TS_PREFIX = re.compile(r"^\d{8}_\d{6}_")
KNOWN_PROBLEMS: set[str] = set()


def slugify(model_field: str) -> str:
    """Canonical model slug: lowercase, provider-path stripped, label-tag folded.

    A bracket tag is part of the identity (kinetic-0715[1m] is a distinct
    model entry from kinetic-0715), so fold it into the slug: "x[1m]" -> "x-1m".
    Run-id-derived fields carry the same tag as "x_1m_"; normalize identically.
    """
    s = (model_field or "").strip().lower()
    s = _TAG_TAIL.sub(lambda m: "-" + re.sub(r"[^a-z0-9]", "", m.group(0)), s)
    s = re.sub(r"_1m_?$", "-1m", s)
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return s


def display_name(slug: str) -> str:
    return MODEL_NAMES.get(slug, slug)


def lab_for(slug: str) -> str:
    for prefix, lab in LAB_BY_PREFIX:
        if slug.startswith(prefix):
            return lab
    return ""


def trace_url(bench: str, run_id: str) -> str | None:
    ds = TRACE_DATASETS.get(bench)
    if not ds or not run_id:
        return None
    return f"https://huggingface.co/datasets/Infatoshi/{ds}/blob/main/{run_id}.jsonl"


def solution_url(run_id: str) -> str | None:
    if not run_id:
        return None
    p = REPO / "public" / "runs" / f"{run_id}_solution.py.txt"
    return f"/runs/{run_id}_solution.py.txt" if p.exists() else None


class Models:
    def __init__(self) -> None:
        self.by_slug: dict[str, dict] = {}

    def get(self, slug: str) -> dict:
        if slug not in self.by_slug:
            self.by_slug[slug] = {
                "slug": slug,
                "name": display_name(slug),
                "lab": lab_for(slug),
                "benches": {},
                "legacy": {},
            }
        return self.by_slug[slug]


def norm_perf(sum_ratio: float, n: int) -> float | None:
    return round(sum_ratio / n, 4) if n else None


def load_site_board(models: Models, bench: str, path: Path, gpu_key: str | None) -> None:
    """Ingest one site-schema leaderboard file (what the pages render).

    gpu_key None = canonical board (fills benches.<bench>); otherwise fills
    benches.<bench>.gpus.<gpu_key> with compact per-GPU data.
    `passed` mirrors the published file's own pass_count; per-cell validity +
    perf are recomputed after annotations join.
    """
    d = json.loads(path.read_text())
    problems = d.get("problems", [])
    KNOWN_PROBLEMS.update(problems)
    # The 3090 deck drops 01_fp8_gemm but its problems array still lists it;
    # the attempted-problem union across rows is the true board denominator.
    n_attempted = max((len(m.get("results") or {}) for m in d.get("models", [])), default=0)
    total_problems = n_attempted or len(problems)

    for m in d.get("models", []):
        slug = slugify(m.get("model") or m.get("label") or "")
        entry = models.get(slug)
        cells = {}
        for prob, c in (m.get("results") or {}).items():
            verdict = c.get("annotation_verdict") or "unaudited"
            cells[prob] = {
                "run_id": c.get("run_id"),
                "correct": bool(c.get("correct")),
                "has_solution": bool(c.get("has_solution")),
                "score": c.get("peak_fraction"),
                "verdict": verdict,
                "valid": bool(c.get("correct")) and c.get("peak_fraction") is not None and verdict not in FLAG_VERDICTS,
                "elapsed_seconds": c.get("elapsed_seconds"),
                "solution_url": solution_url(c.get("run_id") or ""),
                "trace_url": trace_url(bench, c.get("run_id") or ""),
            }
        block = {
            "label": m.get("label"),
            "harness": m.get("harness"),
            "effort": m.get("effort") or None,
            "passed": m.get("pass_count", 0),
            "total_problems": total_problems,
            "perf": None,
            "cells": cells,
        }
        if gpu_key is None:
            block["gpus"] = {}
            entry["benches"][bench] = block
        else:
            bench_block = entry["benches"].setdefault(
                bench,
                {
                    "label": None, "harness": None, "effort": None,
                    "passed": 0, "total_problems": total_problems, "perf": None,
                    "cells": {}, "gpus": {},
                },
            )
            bench_block.setdefault("gpus", {})[gpu_key] = block


def load_mega(models: Models, csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    KNOWN_PROBLEMS.update(r["problem"] for r in rows)
    canonical = [r for r in rows if "RTX PRO" in (r.get("gpu") or "")]

    def ingest(rowset: list[dict], gpu_key: str | None) -> None:
        # best row per (slug, problem) by score among correct rows
        picked: dict[tuple[str, str], dict] = {}
        for r in rowset:
            slug = slugify(r.get("model") or "")
            key = (slug, r["problem"])
            try:
                s = float(r.get("score") or 0)
            except ValueError:
                s = 0.0
            prev = picked.get(key)
            prev_s = float(prev["score"]) if prev else -1.0
            if prev is None or (r.get("correct") == "true" and s > prev_s):
                picked[key] = r
        per_slug: dict[str, dict] = {}
        for (slug, prob), r in picked.items():
            b = per_slug.setdefault(slug, {"cells": {}})
            b["harnesses"] = b.get("harnesses", {})
            b["harnesses"][r.get("harness") or ""] = b["harnesses"].get(r.get("harness") or "", 0) + 1
            correct = r.get("correct") == "true"
            b["cells"][prob] = {
                "run_id": r.get("run_id"),
                "correct": correct,
                "score": float(r["score"]) if r.get("score") else None,
                "tok_s": int(float(r["tok_s"])) if r.get("tok_s") else None,
                "ctx": {k: float(r[k]) for k in ("ctx2048", "ctx8192", "ctx16384") if r.get(k)},
                "framework": r.get("framework") or None,
                "solution_url": solution_url(r.get("run_id") or ""),
                "trace_url": trace_url("mega", r.get("run_id") or ""),
                # verdict/validity filled in after annotations are joined
                "verdict": "unaudited",
                "valid": correct,
            }
        for slug, b in per_slug.items():
            harness = max(b.get("harnesses", {"": 1}), key=b.get("harnesses", {"": 1}).get) or None
            block = {
                "label": f"{harness}/{slug}" if harness else slug,
                "harness": harness,
                "effort": None,
                "passed": 0,  # recomputed after flags join
                "total_problems": len({r["problem"] for r in rows}),
                "perf": None,
                "cells": b["cells"],
            }
            entry = models.get(slug)
            if gpu_key is None:
                block["gpus"] = {}
                entry["benches"]["mega"] = block
            else:
                mb = entry["benches"].setdefault(
                    "mega",
                    {"label": None, "harness": None, "effort": None, "passed": 0,
                     "total_problems": block["total_problems"], "perf": None,
                     "cells": {}, "gpus": {}},
                )
                mb.setdefault("gpus", {})[gpu_key] = block

    ingest(canonical, None)
    for gpu_key, needle in (("h100", "H100"), ("b200", "B200")):
        sub = [r for r in rows if (r.get("gpu") or "").startswith(needle)]
        if sub:
            ingest(sub, gpu_key)


def load_legacy_v1(models: Models, path: Path) -> None:
    d = json.loads(path.read_text())
    n_problems = len(d.get("problems", []))
    KNOWN_PROBLEMS.update(d.get("problems", []))
    for m in d.get("models", []):
        slug = slugify(m.get("model") or m.get("label") or "")
        entry = models.get(slug)
        leg = entry["legacy"].setdefault("hard_v1", {"best_pass_count": 0, "total_problems": n_problems, "labels": set()})
        leg["labels"].add(m.get("label"))
        leg["best_pass_count"] = max(leg["best_pass_count"], m.get("pass_count", 0))
    for entry in models.by_slug.values():
        leg = entry["legacy"].get("hard_v1")
        if leg and isinstance(leg.get("labels"), set):
            leg["labels"] = sorted(leg["labels"])


def annotation_files(ann_dir: Path) -> list[Path]:
    """Git-TRACKED annotation files only.

    Untracked YAMLs are in-flight audits from another workstream (e.g. a model
    not yet meant to be public) — reading them into the committed models.json
    would publish them early. Falls back to a plain glob outside a git repo.
    """
    try:
        rel = ann_dir.relative_to(REPO)
        out = subprocess.run(
            ["git", "ls-files", str(rel)], capture_output=True, text=True, cwd=REPO, check=False
        )
        if out.returncode == 0:
            return sorted(
                REPO / line
                for line in out.stdout.splitlines()
                if line.strip().endswith(".yaml")
            )
    except (ValueError, OSError):
        pass
    return sorted(ann_dir.glob("*.yaml"))


def regex_annotation(text: str) -> dict:
    """Last-resort extractor for YAMLs pyyaml rejects (verbatim field scrape)."""
    out: dict = {}
    for k in ("run_id", "model", "verdict", "problem", "harness"):
        m = re.search(rf"^{k}:\s*(.+?)\s*$", text, re.M)
        if m:
            out[k] = m.group(1).strip().strip('"').strip("'")
    m = re.search(r"^megakernel_authentic:\s*(\w+)", text, re.M)
    if m:
        out["megakernel_authentic"] = m.group(1).lower() == "true"
    out["summary"] = ""
    return out


def model_from_run_id(run_id: str) -> str | None:
    """Fallback for audit YAMLs with no `model:` field.

    run_id = <ts>_<harness>_<model>_<problem>; harness slugs never contain
    underscores, and <problem> is matched against the union of known deck names
    (longest first) so multi-underscore problem names survive.
    """
    s = _TS_PREFIX.sub("", run_id)
    for p in sorted(KNOWN_PROBLEMS, key=len, reverse=True):
        if s.endswith("_" + p):
            s = s[: -(len(p) + 1)]
            break
    parts = s.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[1] else None


def join_annotations(models: Models, bench: str, ann_dir: Path) -> tuple[list[str], int]:
    """Attach audit counts + flag lists. Returns (annotation-only slugs, n_files)."""
    files = [f for f in annotation_files(ann_dir) if f.exists()]
    annotation_only: set[str] = set()
    for f in files:
        try:
            a = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError:
            print(f"  WARN: yaml parse fallback {f.name}")
            a = regex_annotation(f.read_text())
        run_id = a.get("run_id") or f.stem
        model_field = (a.get("model") or "").strip() if isinstance(a.get("model"), str) else ""
        if not model_field:
            model_field = model_from_run_id(run_id) or ""
        slug = slugify(model_field)
        if not slug or not re.fullmatch(r"[a-z0-9][a-z0-9.\-_]*", slug):
            print(f"  WARN: cannot resolve model for {f.name} (got {slug!r})")
            continue
        verdict = (a.get("verdict") or "").strip()
        authentic = a.get("megakernel_authentic")
        flagged = verdict in FLAG_VERDICTS or authentic is False

        entry = models.get(slug)
        bench_block = entry["benches"].get(bench)
        if bench_block is None:
            annotation_only.add(slug)
            bench_block = {"label": None, "harness": None, "effort": None,
                           "passed": 0, "total_problems": 0, "perf": None,
                           "cells": {}, "gpus": {}}
            entry["benches"][bench] = bench_block
        bench_block["audited"] = bench_block.get("audited", 0) + 1
        blocks = [bench_block, *bench_block.get("gpus", {}).values()]
        if flagged:
            eff = verdict if verdict in FLAG_VERDICTS else "megakernel_not_authentic"
            bench_block.setdefault("flags", []).append(
                {"run_id": run_id, "verdict": eff, "summary": (a.get("summary") or "").strip()}
            )
            # reflect the flag onto the matching cell so the UI chips line up
            for block in blocks:
                for cell in block.get("cells", {}).values():
                    if cell.get("run_id") == run_id:
                        cell["verdict"] = eff
                        cell["valid"] = False
        else:
            for block in blocks:
                for cell in block.get("cells", {}).values():
                    if cell.get("run_id") == run_id and cell.get("verdict") in (None, "unaudited"):
                        cell["verdict"] = verdict or "unaudited"
    return sorted(annotation_only), len(files)


# Problems excluded from the perf aggregate. Cells remain in models.json /
# archives (runs, traces, solutions). Site UI also hides these via
# MEGA_HIDDEN_PROBLEMS in app/_lib/models.ts — Mega public board is
# Kimi-Linear only; RL Grid PPO skill is graded on KernelBench-CUDA.
PERF_EXCLUDE: dict[str, set[str]] = {"mega": {"01_rl_grid_ppo"}}


def join_catalog(models: Models) -> None:
    """Attach outcome codes/labels from public/data/catalog.json by run_id.

    Catalog is the universal sweep table (scripts/build_catalog.py). Missing
    catalog is a soft no-op so model-index builds still work offline.
    """
    path = REPO / "public" / "data" / "catalog.json"
    if not path.exists():
        print("WARN: no catalog.json — outcomes not joined (run scripts/build_catalog.py)")
        return
    cat = json.loads(path.read_text())
    legend = {x["code"]: x["label"] for x in cat.get("legend", [])}
    by_run = {c["run_id"]: c for c in cat.get("cells", []) if c.get("run_id")}
    n = 0
    for e in models.by_slug.values():
        for bench, block in e.get("benches", {}).items():
            blocks = [block, *block.get("gpus", {}).values()]
            for b in blocks:
                for prob, cell in b.get("cells", {}).items():
                    rid = cell.get("run_id")
                    src = by_run.get(rid) if rid else None
                    if not src:
                        # synthesize pass/fail without archive detail
                        if cell.get("valid"):
                            cell["outcome"] = "pass"
                            cell["outcome_label"] = legend.get("pass", "pass")
                        else:
                            cell["outcome"] = "other"
                            cell["outcome_label"] = legend.get("other", "fail")
                        continue
                    cell["outcome"] = src.get("outcome") or "other"
                    cell["outcome_label"] = legend.get(
                        cell["outcome"], src.get("outcome") or "fail"
                    )
                    if src.get("failure_reason") is not None:
                        cell["failure_reason"] = src["failure_reason"]
                    n += 1
    print(f"joined catalog outcomes into {n} cells")


def compute_perf(models: Models) -> None:
    """perf = mean(cell score / board-best score) over the FULL problem deck.

    Failed / invalid / missing cells score 0. Denominator is every problem that
    appears on this board (minus PERF_EXCLUDE), not only the cells a model
    passed — otherwise a 3/6 model can outrank a 6/6 model on mean alone.

    Bests are recomputed from CURRENT board rows post-flag (valid cells only).
    """
    for bench in ("hard", "mega", "cuda"):
        blocks = [e["benches"][bench] for e in models.by_slug.values() if bench in e["benches"]]
        groups: list[list[dict]] = [blocks]
        gpu_keys = sorted({k for b in blocks for k in b.get("gpus", {})})
        for g in gpu_keys:
            groups.append([b["gpus"][g] for b in blocks if g in b.get("gpus", {})])
        exclude = PERF_EXCLUDE.get(bench, set())
        for group in groups:
            if not group:
                continue
            # Deck = union of problems any model on this board has a cell for.
            deck = sorted(
                {
                    prob
                    for b in group
                    for prob in b.get("cells", {})
                    if prob not in exclude
                }
            )
            if not deck:
                for b in group:
                    b["perf"] = None
                continue
            best: dict[str, float] = {}
            for b in group:
                for prob, cell in b.get("cells", {}).items():
                    if prob in exclude:
                        continue
                    s = cell.get("score")
                    if cell.get("valid") and s and s > best.get(prob, 0.0):
                        best[prob] = s
            for b in group:
                # No cells at all → not on this board.
                if not b.get("cells"):
                    b["perf"] = None
                    continue
                rs = 0.0
                for prob in deck:
                    cell = b.get("cells", {}).get(prob) or {}
                    s = cell.get("score")
                    if cell.get("valid") and s and best.get(prob, 0.0) > 0:
                        rs += s / best[prob]
                    # else: 0 contribution (fail / invalid / missing)
                b["perf"] = norm_perf(rs, len(deck))


GPU_LABELS = {
    "rtxpro6000": "RTX PRO 6000",
    "h100": "H100 PCIe",
    "b200": "B200",
    "rtx3090": "RTX 3090",
}


def finalize(models: Models) -> dict:
    bench_meta = {}
    for bench in ("hard", "cuda"):
        p = REPO / "benchmarks" / bench / "results" / "leaderboard_v2.json"
        if p.exists():
            d = json.loads(p.read_text())
            bench_meta[bench] = {"problems": d.get("problems", []), "gpu": d.get("hardware", "")}
    mega_rows = list(csv.DictReader((REPO / "public" / "data" / "mega" / "results.csv").read_text().splitlines()))
    bench_meta["mega"] = {
        "problems": sorted({r["problem"] for r in mega_rows}),
        "gpu": "RTX PRO 6000 Blackwell",
    }
    for bench in bench_meta:
        gpu_keys = sorted({
            k for e in models.by_slug.values()
            for k in e["benches"].get(bench, {}).get("gpus", {})
        })
        bench_meta[bench]["gpus"] = ["rtxpro6000", *gpu_keys]
        bench_meta[bench]["gpu_labels"] = {k: GPU_LABELS.get(k, k) for k in bench_meta[bench]["gpus"]}

    out_models = []
    for slug in sorted(models.by_slug):
        e = models.by_slug[slug]
        total_audited = 0
        total_flagged = 0
        for b, block in e["benches"].items():
            block["flags"] = block.get("flags", [])
            block["audited"] = block.get("audited", 0)
            block["flagged"] = len(block["flags"])
            total_audited += block["audited"]
            total_flagged += block["flagged"]
        e["totals"] = {"audited": total_audited, "flagged": total_flagged}
        if not e.get("legacy"):
            e["legacy"] = {}
        out_models.append(e)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benches": bench_meta,
        "methodology": (
            "Rank per bench: valid passes (audited-clean correct cells / problems) desc, "
            "then mean normalized performance over the FULL problem deck (cell score / "
            "board best per problem; fail/invalid/missing cells count as 0) desc. "
            "Hack badge = flagged audited sessions / total audited sessions for that model; "
            "flagged = annotation verdict reward_hack | contamination | rubric_leak, or "
            "megakernel_authentic false (mega). Verdicts come from per-run audit YAMLs, not "
            "static lint. Hack rate is displayed, never a sort key."
        ),
        "models": out_models,
    }


def main() -> None:
    models = Models()
    load_site_board(models, "hard", REPO / "benchmarks/hard/results/leaderboard.json", None)
    for gpu_key in ("h100", "b200", "rtx3090"):
        p = REPO / f"benchmarks/hard/results/leaderboard.{gpu_key}.json"
        if p.exists():
            load_site_board(models, "hard", p, gpu_key)
    cuda_board = REPO / "benchmarks/cuda/results/leaderboard.json"
    if cuda_board.exists():
        load_site_board(models, "cuda", cuda_board, None)
    load_mega(models, REPO / "public/data/mega/results.csv")
    load_legacy_v1(models, REPO / "benchmarks/hard/results/leaderboard_v1.json")

    annotation_only_all: dict[str, list[str]] = {}
    n_files_total = 0
    for bench in ("hard", "mega", "cuda"):
        ann = REPO / f"benchmarks/{bench}/results/annotations"
        if ann.exists():
            only, n_files = join_annotations(models, bench, ann)
            n_files_total += n_files
            if only:
                annotation_only_all[bench] = only

    compute_perf(models)
    # mega pass counts are derived (no published per-model count); v2 boards
    # carry the builder's valid_pass_count already
    for e in models.by_slug.values():
        mb = e["benches"].get("mega")
        if not mb:
            continue
        for block in [mb, *mb.get("gpus", {}).values()]:
            if block.get("cells"):
                block["passed"] = sum(1 for c in block["cells"].values() if c.get("valid"))

    join_catalog(models)

    out = finalize(models)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    # ---- verification report ----
    n = len(out["models"])
    flagged_models = [m for m in out["models"] if m["totals"]["flagged"]]
    print(f"models.json written: {OUT} ({n} models)")
    for bench in ("hard", "mega", "cuda"):
        present = [m["slug"] for m in out["models"] if bench in m["benches"]]
        print(f"  {bench}: {len(present)} models")
    print(f"  flagged models: {len(flagged_models)} -> {[m['slug'] for m in flagged_models]}")
    for bench, slugs in annotation_only_all.items():
        print(f"  {bench} annotations with no board row (sink section): {slugs}")
    # invariant: every tracked annotation file joined to exactly one model
    total_flags = sum(m["totals"]["flagged"] for m in out["models"])
    total_aud = sum(m["totals"]["audited"] for m in out["models"])
    print(f"  audited total {total_aud} (tracked annotation files: {n_files_total})  flags total {total_flags}")
    assert total_aud == n_files_total, "annotation count mismatch — a yaml failed to join"


if __name__ == "__main__":
    main()
