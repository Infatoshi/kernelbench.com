"use client"

import { useState } from "react"
import type { BarView, ReportView } from "../_lib/models"
import { ModelBars } from "./model-bars"
import { ModelReportCard } from "./model-report-card"

// Client wrapper for multi-GPU boards. Hard uses per-problem report cards;
// mega keeps the single-metric bar chart.

export interface GpuView {
  key: string
  label: string
  blurb?: string
  bars?: BarView
  report?: ReportView
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
  const nonEmpty = views.filter(
    (v) => (v.report?.rows.length ?? 0) > 0 || (v.bars?.rows.length ?? 0) > 0,
  )
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
              type="button"
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
      {view.report ? (
        <ModelReportCard view={view.report} />
      ) : view.bars ? (
        <ModelBars view={view.bars} />
      ) : null}
    </div>
  )
}
