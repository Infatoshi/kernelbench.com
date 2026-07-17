"use client"

import { useState } from "react"
import Link from "next/link"
import { DEFAULT_GPU, type ColChart } from "../_lib/models"

// Homepage rankings: Mega → CUDA → Hard column charts with a single GPU
// tab strip (B200 / RTX PRO 6000 / H100). Charts are precomputed server-side
// per GPU; this only switches which set is shown.

function yLabel(unit: ColChart["unit"], v: number): string {
  if (unit === "pct") return `${v}%`
  if (unit === "%") return `${+(v * 100).toFixed(1)}%`
  return `${v}`
}

function barTextColor(hex: string): string {
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#101010" : "#ffffff"
}

function ColumnChart({ chart }: { chart: ColChart }) {
  const lines = [1, 0.75, 0.5, 0.25].map((f) => chart.maxValue * f)
  return (
    <div className="chart-panel">
      <div className="chart-panel-head">
        <span className="chart-panel-title">{chart.title}</span>
        <span className="colchart-sub">{chart.subtitle}</span>
      </div>
      <div className="colchart">
        <div className="colchart-y" aria-hidden="true">
          {lines.map((v) => (
            <span key={v}>{yLabel(chart.unit, v)}</span>
          ))}
          <span>0</span>
        </div>
        <div className="colchart-main">
          <div className="colchart-plot">
            <div className="colchart-cols">
              {chart.columns.map((c, i) => {
                const pct =
                  c.value == null ? null : Math.min(100, (c.value / chart.maxValue) * 100)
                const inside = pct != null && pct >= 16
                const glow = c.value != null && i < 3 ? ` colchart-glow-${i + 1}` : ""
                return (
                  <Link
                    key={c.slug}
                    href={`/models/${c.slug}`}
                    className="colchart-col no-underline"
                    title={`${c.name} · ${c.lab}${c.display ? ` · ${c.display}` : " · no result"}`}
                  >
                    <span
                      className={`colchart-bar${c.value == null ? " colchart-bar-empty" : ""}${glow}`}
                      style={
                        pct == null
                          ? undefined
                          : ({
                              height: `${pct.toFixed(1)}%`,
                              background: c.brand.color,
                              "--glow": c.brand.color,
                              animationDelay: `${Math.min(i * 55, 550)}ms`,
                            } as React.CSSProperties)
                      }
                    >
                      {c.display != null && (
                        <span
                          className={`colchart-val ${inside ? "colchart-val-in" : "colchart-val-out"}`}
                          style={inside ? { color: barTextColor(c.brand.color) } : undefined}
                        >
                          {c.display}
                        </span>
                      )}
                    </span>
                  </Link>
                )
              })}
            </div>
          </div>
          <div className="colchart-cols colchart-labels">
            {chart.columns.map((c) => (
              <Link
                key={c.slug}
                href={`/models/${c.slug}`}
                className="colchart-col no-underline"
                title={`${c.name} · ${c.lab}${c.display ? ` · ${c.display}` : " · no result"}`}
              >
                <span className="colchart-x">
                  {c.brand.logo ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={c.brand.logo} alt="" className="colchart-logo" loading="lazy" />
                  ) : (
                    <span
                      className="colchart-letter"
                      style={{ borderColor: c.brand.color, color: c.brand.color }}
                    >
                      {c.lab.slice(0, 1)}
                    </span>
                  )}
                  <span className="colchart-name">{c.name}</span>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function HomeRankings({
  gpus,
  chartsByGpu,
}: {
  gpus: { key: string; label: string }[]
  /** precomputed charts per GPU key, each in HOME_BENCH_ORDER */
  chartsByGpu: Record<string, ColChart[]>
}) {
  const defaultKey =
    gpus.find((g) => g.key === DEFAULT_GPU)?.key ?? gpus[0]?.key ?? "h100"
  const [gpu, setGpu] = useState(defaultKey)
  const active = chartsByGpu[gpu] ? gpu : defaultKey
  const charts = chartsByGpu[active] ?? []

  return (
    <section aria-label="Rankings">
      <div className="section-head">
        <h2 className="section-title">Rankings</h2>
        <div className="home-rank-controls">
          <span className="section-note">best left — click a column for cells and audits</span>
          <div className="gpu-toggle" role="tablist" aria-label="GPU board">
            {gpus.map((g) => (
              <button
                key={g.key}
                type="button"
                className={`gpu-toggle-btn${g.key === active ? " active" : ""}`}
                onClick={() => setGpu(g.key)}
                role="tab"
                aria-selected={g.key === active}
              >
                {g.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="colchart-stack">
        {charts.map((chart) => (
          <ColumnChart key={`${active}-${chart.title}`} chart={chart} />
        ))}
      </div>
    </section>
  )
}
