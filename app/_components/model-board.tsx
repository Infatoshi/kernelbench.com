"use client"

import { useState } from "react"
import type { BarView } from "../_lib/models"
import { ModelBars } from "./model-bars"

// Client wrapper for benches with multiple per-GPU boards (hard: RTX PRO 6000,
// H100, B200, RTX 3090; mega: 3 boards). Data is precomputed server-side per
// GPU; this just switches between the bar-chart views.

export interface GpuView {
  key: string
  label: string
  blurb?: string
  bars: BarView
}

export function ModelGpuBoard({
  views,
  initialGpu,
}: {
  views: GpuView[]
  /** preselect a board (e.g. from a ?gpu= deep link); falls back to the
   *  first non-empty view when absent or unknown */
  initialGpu?: string
}) {
  const nonEmpty = views.filter((v) => v.bars.rows.length > 0)
  const [sel, setSel] = useState(
    (nonEmpty.find((v) => v.key === initialGpu) ?? nonEmpty[0] ?? views[0])?.key ?? "",
  )
  const view = (nonEmpty.find((v) => v.key === sel) ?? nonEmpty[0] ?? views[0])!
  return (
    <div className="chart-panel">
      <div className="chart-panel-head">
        <span className="chart-panel-title">{view.label}</span>
        <span className="gpu-toggle" role="tablist" aria-label="GPU board">
          {views.map((v) => (
            <button
              key={v.key}
              className={`gpu-toggle-btn${v.key === view.key ? " active" : ""}`}
              onClick={() => setSel(v.key)}
              role="tab"
              aria-selected={v.key === view.key}
            >
              {v.label}
            </button>
          ))}
        </span>
      </div>
      {view.blurb && (
        <p className="leaderboard-muted" style={{ fontSize: "0.8rem", marginTop: "-0.3rem" }}>
          {view.blurb}
        </p>
      )}
      <ModelBars view={view.bars} />
    </div>
  )
}
