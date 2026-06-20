"use client"

import { useState } from "react"
import type { ChartData } from "@/lib/charts"

// GPU-toggled chart: pick a GPU and the benchmark renders as a single sorted
// bar per model. The full cross-GPU numbers live in an expandable table.
export function GroupedBars({
  title,
  subtitle,
  data,
}: {
  title: string
  subtitle: string
  data: ChartData
}) {
  const { series, groups, suffix, decimals } = data
  const [gpu, setGpu] = useState(0)
  const fmt = (v: number) => `${v.toFixed(decimals)}${suffix}`

  // Sorted view for the selected GPU: real values first (desc), DNF last.
  const rows = groups
    .map((g) => ({ label: g.label, value: g.values[gpu] }))
    .sort((a, b) => {
      if (a.value == null) return 1
      if (b.value == null) return -1
      return b.value - a.value
    })
  const localMax = Math.max(...rows.map((r) => r.value ?? 0), 1)

  return (
    <div className="chart">
      <div className="chart-head">
        <div>
          <h3 className="chart-title">{title}</h3>
          <p className="chart-subtitle">{subtitle}</p>
        </div>
        <div className="gpu-toggle" role="group" aria-label="Select GPU">
          {series.map((s, i) => (
            <button
              key={s.key}
              type="button"
              className={i === gpu ? "gpu-toggle-btn active" : "gpu-toggle-btn"}
              aria-pressed={i === gpu}
              onClick={() => setGpu(i)}
            >
              {s.key}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-body">
        {rows.map((r) => (
          <div key={r.label} className="chart-row">
            <div className="chart-row-label">{r.label}</div>
            <div className="chart-track">
              <div className="chart-bar-wrap">
                <div
                  className="chart-bar"
                  style={{
                    width:
                      r.value == null
                        ? "0%"
                        : `${Math.max((r.value / localMax) * 100, 1.5)}%`,
                    background: "var(--color-accent)",
                  }}
                />
              </div>
              <span className={r.value == null ? "chart-val chart-val-dnf" : "chart-val"}>
                {r.value == null ? "DNF" : fmt(r.value)}
              </span>
            </div>
          </div>
        ))}
      </div>

      <details className="chart-table-wrap">
        <summary>Show data table</summary>
        <table className="chart-table">
          <thead>
            <tr>
              <th>Model</th>
              {series.map((s) => (
                <th key={s.key}>{s.key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.label}>
                <td>{g.label}</td>
                {g.values.map((v, i) => (
                  <td key={series[i].key} className={v == null ? "chart-val-dnf" : ""}>
                    {v == null ? "DNF" : fmt(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}
