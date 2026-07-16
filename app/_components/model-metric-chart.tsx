"use client"

import { useState } from "react"
import { BENCH_LABELS, type BarView, type Bench } from "../_lib/models"
import { ModelBars } from "./model-bars"

// Bench metric toggle (AA-style): switch the model bar chart between
// benchmarks. MULTI renders a coming-soon state until its first sweep lands.

type ChartKey = Bench | "multi"

const ORDER: ChartKey[] = ["mega", "hard", "cuda", "multi"]

const MULTI_PROBLEMS = [
  "AllReduce + Residual",
  "ReduceScatter + RMSNorm",
  "AllGather + fp8 Dequant",
  "MoE All-to-All",
  "Ulysses All-to-All",
  "fp8 ReduceScatter Grad",
]

function MultiComingSoon() {
  return (
    <div className="mbar-coming">
      <p className="mbar-coming-head">
        <span className="mbar-coming-pill">coming soon</span>
        <span>8×H100 SXM · NVSwitch · NVLink4 · ~900 GB/s/GPU</span>
      </p>
      <p className="mbar-coming-blurb">
        Agents rewrite PyTorch + NCCL collectives as fine-grained NVLink
        kernels (CUDA / Triton / NVSHMEM / CUDA symmetric memory), graded on
        busbw — bus-bandwidth efficiency, never TFLOPS.
      </p>
      <div className="chip-row" style={{ justifyContent: "center" }}>
        {MULTI_PROBLEMS.map((p) => (
          <span key={p} className="link-chip link-chip-muted">
            {p}
          </span>
        ))}
      </div>
    </div>
  )
}

export function ModelMetricChart({
  views,
  defaultBench = "hard",
}: {
  views: Partial<Record<Bench, BarView>>
  defaultBench?: Bench
}) {
  const available = ORDER.filter((k) => k === "multi" || (views[k] && views[k]!.rows.length > 0))
  const [sel, setSel] = useState<ChartKey>(
    available.includes(defaultBench) ? defaultBench : (available[0] ?? "hard"),
  )
  const view = sel === "multi" ? null : views[sel]
  return (
    <div className="chart-panel">
      <div className="chart-panel-head">
        <span className="chart-panel-title">
          {sel === "multi" ? "Multi" : `${BENCH_LABELS[sel]} leaderboard`}
        </span>
        <span className="gpu-toggle" role="tablist" aria-label="benchmark">
          {available.map((k) => (
            <button
              key={k}
              className={`gpu-toggle-btn${k === sel ? " active" : ""}`}
              onClick={() => setSel(k)}
              role="tab"
              aria-selected={k === sel}
            >
              {k.toUpperCase()}
            </button>
          ))}
        </span>
      </div>
      {sel === "multi" ? <MultiComingSoon /> : view && <ModelBars view={view} />}
    </div>
  )
}
