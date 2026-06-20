"use client"

import { useState } from "react"
import type { EffData } from "@/lib/charts"

// Performance vs compute (output tokens) scatter with the efficiency frontier
// highlighted. Inline SVG so it matches the site theme and stays responsive.
const W = 760
const H = 420
const PAD = { l: 56, r: 18, t: 18, b: 44 }

export function EfficiencyChart({ mega, hard }: { mega: EffData; hard: EffData }) {
  const [tab, setTab] = useState<"mega" | "hard">("mega")
  const data = tab === "mega" ? mega : hard
  const fmtY = (v: number) => `${v.toFixed(data.yDecimals)}${data.ySuffix}`

  const xMax = Math.max(...data.points.map((p) => p.x)) * 1.12
  const yMax = Math.max(...data.points.map((p) => p.y)) * 1.16
  const px = (x: number) => PAD.l + (x / xMax) * (W - PAD.l - PAD.r)
  const py = (y: number) => H - PAD.b - (y / yMax) * (H - PAD.t - PAD.b)

  const frontier = data.points
    .filter((p) => p.frontier)
    .sort((a, b) => a.x - b.x)
  const xTicks = niceTicks(xMax, 5)
  const yTicks = niceTicks(yMax, 5)

  return (
    <div>
      <div className="chart-head">
        <div>
          <h3 className="chart-title">Performance vs compute</h3>
          <p className="chart-subtitle">
            Does the win just cost more tokens? Output tokens = the compute each model
            chose to spend. RTX PRO 6000; models with clean token telemetry.
          </p>
        </div>
        <div className="gpu-toggle" role="group" aria-label="Select benchmark">
          {(["mega", "hard"] as const).map((t) => (
            <button
              key={t}
              type="button"
              className={t === tab ? "gpu-toggle-btn active" : "gpu-toggle-btn"}
              aria-pressed={t === tab}
              onClick={() => setTab(t)}
            >
              {t === "mega" ? "Mega" : "Hard"}
            </button>
          ))}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="eff-svg"
        role="img"
        aria-label={`${tab} performance versus output tokens`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={py(t)} y2={py(t)} className="eff-grid" />
            <text x={PAD.l - 8} y={py(t) + 4} className="eff-tick" textAnchor="end">
              {fmtY(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={px(t)} y={H - PAD.b + 18} className="eff-tick" textAnchor="middle">
            {Math.round(t / 1000)}k
          </text>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 6} className="eff-axis" textAnchor="middle">
          output tokens spent
        </text>

        <polyline
          points={frontier.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
          className="eff-frontier"
        />

        {data.points.map((p) => {
          const right = px(p.x) > W * 0.72
          return (
            <g key={p.label}>
              <circle cx={px(p.x)} cy={py(p.y)} r={6} className={p.frontier ? "eff-dot on" : "eff-dot"} />
              <text
                x={px(p.x) + (right ? -10 : 10)}
                y={py(p.y) - 9}
                textAnchor={right ? "end" : "start"}
                className={p.frontier ? "eff-label on" : "eff-label"}
              >
                {p.label}
              </text>
            </g>
          )
        })}
      </svg>

      <p className="eff-legend">
        <span className="eff-key on" /> on the efficiency frontier (most performance per token)
        <span className="eff-key" /> dominated (spent more, delivered less)
      </p>
    </div>
  )
}

function niceTicks(max: number, count: number): number[] {
  const step = max / count
  const mag = Math.pow(10, Math.floor(Math.log10(step)))
  const norm = step / mag
  const nice = norm >= 5 ? 5 : norm >= 2 ? 2 : 1
  const s = nice * mag
  const out: number[] = []
  for (let v = 0; v <= max; v += s) out.push(v)
  return out
}
