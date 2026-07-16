"use client"

import { useState } from "react"
import type { BarView, Bench } from "../_lib/models"
import { ModelBars } from "./model-bars"

// Metric toggle (AA-style): switch the model bar chart between benchmarks.

export function ModelMetricChart({
  views,
  defaultBench = "hard",
}: {
  views: Partial<Record<Bench, BarView>>
  defaultBench?: Bench
}) {
  const order: Bench[] = ["mega", "hard", "cuda"]
  const available = order.filter((b) => views[b] && views[b]!.rows.length > 0)
  const [sel, setSel] = useState<Bench>(
    available.includes(defaultBench) ? defaultBench : (available[0] ?? "hard"),
  )
  const view = views[sel]
  if (!view) return null
  return (
    <>
      <div className="chip-row" style={{ justifyContent: "center", marginBottom: "1.2rem" }}>
        <span className="gpu-toggle" role="tablist" aria-label="benchmark">
          {available.map((b) => (
            <button
              key={b}
              className={`gpu-toggle-btn${b === sel ? " active" : ""}`}
              onClick={() => setSel(b)}
              role="tab"
              aria-selected={b === sel}
            >
              {b.toUpperCase()}
            </button>
          ))}
        </span>
      </div>
      <ModelBars view={view} />
    </>
  )
}
