import Link from "next/link"
import type { ColChart } from "../_lib/models"

// AA-style vertical column charts for the model views. Pure server rendering
// (no toggles, no client state): performance is disaggregated into one chart
// per bench (Mega / Hard / CUDA), a slim Multi coming-soon panel, and one
// compiled "Correctness" chart below. All charts share the same column order
// (see columnOrder in _lib/models.ts) so a model sits in the same slot
// everywhere; missing results render an empty slot, not a zero bar.

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
  if (unit === "%") return `${Math.round(v * 100)}%`
  return `${v}`
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
        <div className="colchart-plot">
          <div className="colchart-cols">
            {chart.columns.map((c) => (
              <Link
                key={c.slug}
                href={`/models/${c.slug}`}
                className="colchart-col no-underline"
                title={`${c.name} · ${c.lab}${c.display ? ` · ${c.display}` : " · no result"}`}
              >
                <span className="colchart-val">{c.display ?? ""}</span>
                <span
                  className={`colchart-bar${c.value == null ? " colchart-bar-empty" : ""}`}
                  style={
                    c.value == null
                      ? undefined
                      : {
                          height: `${Math.min(100, (c.value / chart.maxValue) * 100).toFixed(1)}%`,
                          background: c.brand.color,
                        }
                  }
                />
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
