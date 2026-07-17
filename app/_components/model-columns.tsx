import Link from "next/link"
import type { ColChart } from "../_lib/models"

// AA-style vertical column charts for the model views. Pure server rendering
// (no toggles, no client state): performance is disaggregated into one chart
// per bench (Mega / Hard / CUDA), a slim Multi coming-soon panel, and one
// compiled "Correctness" chart below. Each chart ranks its own metric —
// highest score leftmost — and only scored models get a column (see
// sortColumnsByValue in _lib/models.ts).

const MULTI_PROBLEMS = [
  "AllReduce + Residual",
  "ReduceScatter + RMSNorm",
  "AllGather + fp8 Dequant",
  "MoE All-to-All",
  "Ulysses All-to-All",
  "fp8 ReduceScatter Grad",
]

function yLabel(unit: ColChart["unit"], v: number): string {
  if (unit === "pct") return `${v}%`
  if (unit === "%") return `${+(v * 100).toFixed(1)}%`
  return `${v}`
}

// In-bar value labels sit on the brand color, so pick text color by luma:
// light bars (xAI #e6e6e6, Cursor #c9c9c9, Meituan #ffd60a) need dark text.
function barTextColor(hex: string): string {
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#101010" : "#ffffff"
}

function ColumnChart({ chart }: { chart: ColChart }) {
  // gridline values: maxValue * (1, .75, .5, .25) top to bottom; baseline = 0
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
        {/* bars live alone inside the fixed-height plot (labels in their own
            row below) so bar height % maps 1:1 onto the gridlines and tall
            bars can never flex-shrink into a lie */}
        <div className="colchart-main">
          <div className="colchart-plot">
            <div className="colchart-cols">
              {chart.columns.map((c) => {
                const pct =
                  c.value == null ? null : Math.min(100, (c.value / chart.maxValue) * 100)
                // AA convention: value inside the bar top when it fits (>= ~16%
                // of plot height), floated above the bar otherwise.
                const inside = pct != null && pct >= 16
                return (
                <Link
                  key={c.slug}
                  href={`/models/${c.slug}`}
                  className="colchart-col no-underline"
                  title={`${c.name} · ${c.lab}${c.display ? ` · ${c.display}` : " · no result"}`}
                >
                  <span
                    className={`colchart-bar${c.value == null ? " colchart-bar-empty" : ""}`}
                    style={
                      pct == null
                        ? undefined
                        : {
                            height: `${pct.toFixed(1)}%`,
                            background: c.brand.color,
                          }
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
                    <img src={c.brand.logo} alt="" className="colchart-logo" loading="lazy" />
                  ) : (
                    <span className="colchart-letter" style={{ borderColor: c.brand.color, color: c.brand.color }}>
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

function MultiComingSoon() {
  return (
    <div className="chart-panel">
      <div className="chart-panel-head">
        <span className="chart-panel-title">Multi performance</span>
        <span className="colchart-sub">8×H100 SXM · graded on busbw, never TFLOPS</span>
      </div>
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
    </div>
  )
}

export function ModelScoreboards({
  perf,
  correctness,
}: {
  /** one chart per published bench, in display order (mega, hard, cuda) */
  perf: ColChart[]
  correctness: ColChart
}) {
  return (
    <div className="colchart-stack">
      {perf.map((chart) => (
        <ColumnChart key={chart.title} chart={chart} />
      ))}
      <MultiComingSoon />
      <ColumnChart chart={correctness} />
    </div>
  )
}
