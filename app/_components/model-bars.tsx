import Link from "next/link"
import { BENCH_LABELS, type BarRow, type BarView } from "../_lib/models"

// AA-style horizontal bar chart: one bar per model, y-axis labels are the
// model name + lab logo, bars colored by lab brand, score value at bar end.
// Grouped by pass tier (correctness first), ordered within a tier by score.

function LabMark({ row }: { row: BarRow }) {
  if (row.brand.logo) {
    return (
      <img
        src={row.brand.logo}
        alt={row.lab}
        title={row.lab}
        className="mbar-logo"
        width={18}
        height={18}
        loading="lazy"
      />
    )
  }
  return (
    <span
      className="mbar-logo mbar-logo-letter"
      title={row.lab}
      style={{ color: row.brand.color, borderColor: row.brand.color }}
    >
      {(row.lab || row.name).trim().charAt(0).toUpperCase()}
    </span>
  )
}

function FlagMark({ row }: { row: BarRow }) {
  if (row.audited === 0) return null
  if (row.flagged > 0) {
    return (
      <span
        className="mbar-flag"
        title={`${row.flagged} of ${row.audited} audited sessions flagged (reward_hack / contamination / rubric_leak / non-authentic megakernel)`}
      >
        {row.flagged}/{row.audited} flagged
      </span>
    )
  }
  return (
    <span className="mbar-audited" title={`${row.audited} audited sessions, none flagged`}>
      {row.audited} audited
    </span>
  )
}

// Per-problem correctness pips: one dot per deck problem, filled = that many
// problems passed the correctness check. Reads at a glance without the
// "3/6" fraction; the words live in the tooltip and on the model page.
function CorrectnessPips({ row }: { row: BarRow }) {
  if (row.total <= 1) return null
  return (
    <span
      className="mbar-pips"
      title={`${row.passed} of ${row.total} problems pass the correctness check`}
      aria-label={`${row.passed} of ${row.total} problems pass the correctness check`}
    >
      {Array.from({ length: row.total }, (_, i) => (
        <span key={i} className={i < row.passed ? "mbar-pip mbar-pip-on" : "mbar-pip"} />
      ))}
    </span>
  )
}

export function ModelBars({ view }: { view: BarView }) {
  const max = view.maxValue || 1
  const fmtTick = (t: number) =>
    view.bench === "mega" ? `${(max * t).toFixed(1)}x` : `${Math.round(max * t * 100)}%`
  return (
    <div className="mbar" role="figure" aria-label={view.axis}>
      {view.rows.map((row, ri) => {
        const pct = Math.max(1.2, Math.min(100, (row.value / max) * 100))
        return (
          <div key={row.slug}>
            <Link href={`/models/${row.slug}`} className="mbar-row no-underline">
              <span className="mbar-label">
                <span className="mbar-name-line">
                  <LabMark row={row} />
                  <span className="mbar-name">{row.name}</span>
                </span>
                <span className="mbar-sub">
                  {row.subtitle && <span>{row.subtitle}</span>}
                  <FlagMark row={row} />
                </span>
              </span>
              <span className="mbar-track">
                <span
                  className="mbar-fill"
                  style={{
                    width: `${pct.toFixed(2)}%`,
                    background: row.brand.color,
                    animationDelay: `${Math.min(ri * 50, 500)}ms`,
                  }}
                />
              </span>
              <span className="mbar-right">
                <span className="mbar-val tabular">{row.display}</span>
                <CorrectnessPips row={row} />
              </span>
            </Link>
          </div>
        )
      })}
      <div className="mbar-axis">
        <span />
        <span className="mbar-axis-ticks">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <span key={t} className="tabular">
              {fmtTick(t)}
            </span>
          ))}
        </span>
        <span />
      </div>
      <p className="mbar-axis-label">
        {BENCH_LABELS[view.bench]} · {view.axis}
      </p>
    </div>
  )
}
