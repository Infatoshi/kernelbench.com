"use client"

import { useState } from "react"
import type { EffByGpu, EffPoint } from "@/app/_lib/charts"

// Performance vs compute (output tokens) scatter with the efficiency frontier
// highlighted. Benchmark toggle (Mega/Hard) + GPU toggle (only GPUs with clean
// token telemetry appear). Inline SVG so it matches the theme and stays responsive.
const W = 760
const H = 420
const PAD = { l: 56, r: 18, t: 18, b: 44 }

export function EfficiencyChart({ mega, hard }: { mega: EffByGpu; hard: EffByGpu }) {
  const [tab, setTab] = useState<"mega" | "hard">("mega")
  const byGpu = tab === "mega" ? mega : hard
  const gpus = Object.keys(byGpu)
  // Prefer B200 when present; tab order is H100 → RTX → B200.
  const preferred = gpus.includes("B200") ? "B200" : (gpus[0] ?? "")
  const [gpu, setGpu] = useState(preferred)
  const activeGpu = byGpu[gpu] ? gpu : preferred
  const data = byGpu[activeGpu]

  if (!data || !data.points.length) {
    return (
      <div className="chart-head">
        <h3 className="chart-title">Performance vs compute</h3>
      </div>
    )
  }

  const fmtY = (v: number) => `${v.toFixed(data.yDecimals)}${data.ySuffix}`
  const xMax = Math.max(...data.points.map((p) => p.x), 1) * 1.12
  const yMax = Math.max(...data.points.map((p) => p.y), 1) * 1.16
  const px = (x: number) => PAD.l + (x / xMax) * (W - PAD.l - PAD.r)
  const py = (y: number) => H - PAD.b - (y / yMax) * (H - PAD.t - PAD.b)
  const frontier = data.points.filter((p) => p.frontier).sort((a, b) => a.x - b.x)
  const xTicks = niceTicks(xMax, 5)
  const yTicks = niceTicks(yMax, 5)

  return (
    <div>
      <div className="chart-head">
        <div>
          <h3 className="chart-title">Performance vs compute</h3>
          <p className="chart-subtitle">
            Does the win just cost more tokens? Output tokens = the compute each model
            chose to spend. Models with clean token telemetry.
          </p>
        </div>
        <div className="eff-toggles">
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
          <div className="gpu-toggle" role="group" aria-label="Select GPU">
            {gpus.map((g) => (
              <button
                key={g}
                type="button"
                className={g === activeGpu ? "gpu-toggle-btn active" : "gpu-toggle-btn"}
                aria-pressed={g === activeGpu}
                onClick={() => setGpu(g)}
              >
                {g}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="eff-svg" role="img"
        aria-label={`${tab} performance versus output tokens on ${activeGpu}`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={py(t)} y2={py(t)} className="eff-grid" />
            <text x={PAD.l - 8} y={py(t) + 4} className="eff-tick" textAnchor="end">{fmtY(t)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={px(t)} y={H - PAD.b + 18} className="eff-tick" textAnchor="middle">
            {Math.round(t / 1000)}k
          </text>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 6} className="eff-axis" textAnchor="middle">output tokens spent</text>
        <polyline points={frontier.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")} className="eff-frontier" />
        {placeLabels(data.points, px, py).map(({ p, cx, cy, level, right }) => {
          const lx = cx + (right ? -10 : 10)
          const ly = cy - 9 - level * 15
          return (
            <g key={`${p.label}-${p.x}-${p.y}`}>
              <line x1={cx} y1={cy} x2={lx} y2={ly - 3} className="eff-leader" />
              <circle cx={cx} cy={cy} r={6} className={p.frontier ? "eff-dot on" : "eff-dot"} />
              <text x={lx} y={ly} textAnchor={right ? "end" : "start"}
                className={p.frontier ? "eff-label on" : "eff-label"}>{p.label}</text>
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

// Place labels above each point; lift one row at a time until the box clears
// every already-placed label (leader lines keep each label tied to its dot).
function placeLabels(points: EffPoint[], px: (x: number) => number, py: (y: number) => number) {
  const sorted = [...points].sort((a, b) => a.x - b.x)
  const boxes: { x0: number; x1: number; y0: number; y1: number }[] = []
  return sorted.map((p) => {
    const cx = px(p.x)
    const cy = py(p.y)
    const right = cx > W * 0.72
    const w = p.label.length * 6.6 + 6
    let level = 0
    let box = { x0: 0, x1: 0, y0: 0, y1: 0 }
    while (level < 8) {
      const ly = cy - 9 - level * 15
      const x0 = right ? cx - 10 - w : cx + 10
      box = { x0, x1: x0 + w, y0: ly - 12, y1: ly + 3 }
      if (!boxes.some((b) => box.x0 < b.x1 && box.x1 > b.x0 && box.y0 < b.y1 && box.y1 > b.y0)) break
      level++
    }
    boxes.push(box)
    return { p, cx, cy, level, right }
  })
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
