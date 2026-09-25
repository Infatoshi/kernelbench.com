"""Nearby-field rank chart for the audited GPT-6 Sol Mega post.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_gpt6_sol_mega.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "media/posts/unaudited"))
from make_charts import save_rank  # noqa: E402

RUN_ID = "20260922_123218_codex_gpt-6-sol_02_kimi_linear_decode"
PEERS = {
    "gemini-3.8-flash-high": "Gemini 3.8",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "muse-spark-1.3": "Muse 1.3",
    "grok-4.5": "Grok 4.5",
}


def main() -> None:
    annotation = ROOT / f"benchmarks/mega/results/annotations/{RUN_ID}.yaml"
    audit = yaml.safe_load(annotation.read_text())
    assert audit["run_id"] == RUN_ID and audit["verdict"] in {"clean", "interesting"}
    assert audit["correct"] is True and audit["gpu"] == "RTX_PRO_6000"

    with (ROOT / "public/data/mega/results.csv").open(newline="") as file:
        field = list(csv.DictReader(file))
    rows = []
    for model_id, label in PEERS.items():
        matches = [r for r in field if r["model"] == model_id
                   and r["gpu"] == "RTX PRO 6000 Blackwell"
                   and r["correct"] == "true" and r["megakernel_judged"] == "true"]
        assert len(matches) == 1, (model_id, matches)
        rows.append((label, float(matches[0]["score"]), False))
    rows.append(("GPT-6 Sol", float(audit["peak_fraction"]), True))
    rows.sort(key=lambda row: row[1], reverse=True)

    out = ROOT / "media/posts/audited/gpt6sol-mega/01.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_rank(out, rows, "Kimi-Linear Decode · speedup vs PyTorch · nearby field", "{:.2f}x")


if __name__ == "__main__":
    main()
