"use client"

import { useState } from "react"
import type { ModelRow } from "../_lib/models"
import { ModelList } from "./model-list"

// Client wrapper for benches with multiple per-GPU boards (hard: RTX PRO 6000,
// H100, B200, RTX 3090; mega: 3 boards). Rows are precomputed server-side per
// GPU; this just switches between them.

export interface GpuView {
  key: string
  label: string
  blurb?: string
  board: ModelRow[]
  sink: ModelRow[]
}

export function ModelGpuBoard({ views }: { views: GpuView[] }) {
  const [sel, setSel] = useState(views[0]?.key ?? "")
  const view = views.find((v) => v.key === sel) ?? views[0]
  if (!view) return null
  return (
    <>
      <div className="chip-row" style={{ justifyContent: "center", marginBottom: "1rem" }}>
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
      <ModelList rows={view.board} />
      {view.sink.length > 0 && (
        <div className="model-sink-section">
          <p className="model-sink-label">
            No valid published results on this board — audited sessions below were flagged or invalid
          </p>
          <ModelList rows={view.sink} sink />
        </div>
      )}
    </>
  )
}
