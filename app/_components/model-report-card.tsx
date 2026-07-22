import Link from "next/link"
import type { ProblemChip, ReportRow, ReportView } from "../_lib/models"

// Per-problem report card: one row per model, six (or N) chips — pass shows
// peak fraction, fail shows a short reason. Rank is by pass count only; we
// never compress fails into a single "half speed" bar.
// Each chip with a run links to its /runs/<gpu>/<rid> page.

function LabMark({ row }: { row: ReportRow }) {
  if (row.brand.logo) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
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

function FlagMark({ row }: { row: ReportRow }) {
  if (row.audited === 0) return null
  if (row.flagged > 0) {
    return (
      <span
        className="mbar-flag"
        title={`${row.flagged} of ${row.audited} audited sessions flagged`}
      >
        {row.flagged}/{row.audited} flagged
      </span>
    )
  }
  return (
    <span className="mbar-audited" title={`${row.audited} audited, none flagged`}>
      {row.audited} audited
    </span>
  )
}

function Chip({ chip }: { chip: ProblemChip }) {
  const cls =
    chip.kind === "pass"
      ? "rcell rcell-pass"
      : chip.kind === "hack"
        ? "rcell rcell-hack"
        : chip.kind === "numerics"
          ? "rcell rcell-fail"
          : chip.kind === "no_kernel"
            ? "rcell rcell-empty"
            : "rcell rcell-fail"
  const body = (
    <>
      <span className="rcell-prob">{chip.short}</span>
      <span className="rcell-val tabular">{chip.label}</span>
    </>
  )
  if (chip.page_url) {
    return (
      <Link
        href={chip.page_url}
        className={`${cls} rcell-openable no-underline`}
        title={`${chip.problem}: ${chip.title} — open run page`}
      >
        {body}
      </Link>
    )
  }
  return (
    <span className={cls} title={`${chip.problem}: ${chip.title}`}>
      {body}
    </span>
  )
}

export function ModelReportCard({ view }: { view: ReportView }) {
  let lastTier: string | null = null
  return (
    <div className="rcard" role="figure" aria-label={view.axis}>
      <div className="rcard-head" aria-hidden="true">
        <span className="rcard-head-label" />
        <span className="rcard-head-chips">
          {view.problems.map((p) => (
            <span key={p.id} className="rcell rcell-head">
              <span className="rcell-prob">{p.short}</span>
            </span>
          ))}
        </span>
        <span className="rcard-head-right" />
      </div>
      {view.rows.map((row) => {
        const tier = `${row.passed}/${row.total}`
        const showTier = tier !== lastTier
        lastTier = tier
        return (
          <div key={row.slug}>
            {showTier && (
              <p className="mbar-tier">
                {row.passed >= row.total ? `full pass · ${tier}` : `${tier} passed`}
              </p>
            )}
            <div className="rcard-row">
              <span className="mbar-label">
                <Link href={`/models/${row.slug}`} className="mbar-name-line no-underline">
                  <LabMark row={row} />
                  <span className="mbar-name">{row.name}</span>
                </Link>
                <span className="mbar-sub">
                  {row.subtitle && <span>{row.subtitle}</span>}
                  <FlagMark row={row} />
                  {row.meanWhenCorrect != null && (
                    <span
                      className="rcard-when-ok"
                      title="Mean only on problems that passed (not used for rank)"
                    >
                      when ok{" "}
                      {row.meanWhenCorrect > 1.5
                        ? `${row.meanWhenCorrect.toFixed(1)}x`
                        : `${(row.meanWhenCorrect * 100).toFixed(0)}%`}
                    </span>
                  )}
                </span>
              </span>
              <span className="rcard-chips">
                {row.chips.map((c) => (
                  <Chip key={c.problem} chip={c} />
                ))}
              </span>
              <span className="mbar-right">
                <span
                  className={`status-pill ${row.passed >= row.total ? "status-pill-good" : "status-pill-muted"}`}
                >
                  {row.passed}/{row.total}
                </span>
              </span>
            </div>
          </div>
        )
      })}
      <p className="mbar-axis-label">{view.axis}</p>
      <p className="rcard-legend">
        <span className="rcell rcell-pass rcell-legend-swatch">
          <span className="rcell-val">%</span>
        </span>
        peak fraction
        <span className="rcell rcell-fail rcell-legend-swatch">
          <span className="rcell-val">check</span>
        </span>
        wrong / no pass
        <span className="rcell rcell-empty rcell-legend-swatch">
          <span className="rcell-val">empty</span>
        </span>
        no solution
        <span className="rcell rcell-hack rcell-legend-swatch">
          <span className="rcell-val">hack</span>
        </span>
        audit flag
      </p>
    </div>
  )
}
