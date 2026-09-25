"""Grid + MinGRU rank charts for the GPT-6 Sol and Luna CUDA drafts.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_gpt6_sol_luna_cuda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "media/posts/unaudited"))
from make_charts import save_rank  # noqa: E402

PROBLEM = "04_grid_mingru_sps"
RUNS = {
    "gpt-6-sol": "20260922_123418_codex_gpt-6-sol_04_grid_mingru_sps",
    "gpt-6-luna": "20260922_123423_codex_gpt-6-luna_04_grid_mingru_sps",
}
LABELS = {"gpt-6-sol": "GPT-6 Sol", "gpt-6-luna": "GPT-6 Luna"}
SHORT = {
    "claude-opus-5": "Opus 5",
    "claude-opus-5-5": "Opus 5.5",
    "claude-fable-5-1": "Fable 5.1",
    "gpt-6-astra-pro": "GPT-6 Astra",
    "grok-4.7": "Grok 4.7",
}


def main() -> None:
    index = json.loads((ROOT / "public/data/models.json").read_text())
    field = []
    for model in index["models"]:
        if model["slug"] in RUNS:
            continue
        cell = (model.get("benches", {}).get("cuda") or {}).get("cells", {}).get(PROBLEM) or {}
        if cell.get("valid") and cell.get("score") is not None:
            field.append((SHORT.get(model["slug"], model["name"]), float(cell["score"])))
    field.sort(key=lambda row: row[1], reverse=True)
    field = field[:5]

    for model, run_id in RUNS.items():
        annotation = ROOT / f"benchmarks/cuda/results/annotations/{run_id}.yaml"
        audit = yaml.safe_load(annotation.read_text())
        assert audit["run_id"] == run_id and audit["verdict"] in {"clean", "interesting"}
        assert audit["correct"] is True and audit["gpu"] == "RTX_PRO_6000"
        rows = [(name, score, False) for name, score in field]
        rows.append((LABELS[model], float(audit["peak_fraction"]), True))
        rows.sort(key=lambda row: row[1], reverse=True)
        out = ROOT / f"media/posts/audited/{model.replace('-', '')}-cuda/01.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_rank(out, rows, "Grid + MinGRU · score / 150M steps/s target · RTX PRO 6000", "{:.3f}")


if __name__ == "__main__":
    main()
