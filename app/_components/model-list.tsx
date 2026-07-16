import Link from "next/link"
import type { ModelRow } from "../_lib/models"

// Pure presentational rows used by the homepage index, per-bench pages, and
// the client GPU-toggle board. No server-only imports so it can render inside
// a client component too.

function passPill(row: ModelRow) {
  const full = row.total > 0 && row.passed >= row.total
  return (
    <span
      className={`status-pill ${full ? "status-pill-good" : "status-pill-muted"}`}
      title={`${row.passed} of ${row.total} problems correct`}
    >
      {row.passed}/{row.total}
    </span>
  )
}

function hackBadge(row: ModelRow) {
  if (row.audited === 0) return null
  if (row.flagged > 0) {
    return (
      <span
        className="status-pill status-pill-bad"
        title={`${row.flagged} of ${row.audited} audited sessions flagged (reward_hack / contamination / rubric_leak / non-authentic megakernel)`}
      >
        {row.flagged}/{row.audited} flagged
      </span>
    )
  }
  return (
    <span
      className="audit-chip audit-chip-muted"
      title={`${row.audited} audited sessions, none flagged`}
    >
      {row.audited} audited
    </span>
  )
}

function perfBar(row: ModelRow) {
  if (row.perf == null) return null
  const pct = Math.max(0, Math.min(1, row.perf))
  return (
    <span className="model-row-perf" title="mean normalized performance (cell / board best)">
      <span className="speed-bar">
        <span className="speed-fill" style={{ width: `${(pct * 100).toFixed(1)}%` }} />
      </span>
      <span className="model-row-perf-val">{row.perf.toFixed(2)}</span>
    </span>
  )
}

export function ModelList({
  rows,
  showBadges,
  sink,
}: {
  rows: ModelRow[]
  showBadges?: boolean
  sink?: boolean
}) {
  return (
    <div className={`model-list${sink ? " model-sink" : ""}`}>
      {rows.map((row, i) => (
        <Link
          key={row.slug}
          href={`/models/${row.slug}`}
          className="model-row no-underline"
        >
          <span className={`model-row-rank${!sink && i === 0 ? " model-row-rank-1" : ""}`}>
            {sink ? "-" : i + 1}
          </span>
          <span className="model-row-id">
            <span className="model-row-name">{row.name}</span>
            <span className="model-row-sub">
              {[row.lab, row.subtitle].filter(Boolean).join(" · ")}
            </span>
          </span>
          <span className="model-row-metrics">
            {showBadges && row.badges.length > 0 ? (
              <span className="model-badges">
                {row.badges.map((b) => (
                  <span
                    key={b.label}
                    className={`status-pill ${b.tone === "good" ? "status-pill-good" : "status-pill-muted"}`}
                  >
                    {b.label} {b.value}
                  </span>
                ))}
              </span>
            ) : (
              !sink && passPill(row)
            )}
            {!sink && perfBar(row)}
            {hackBadge(row)}
          </span>
        </Link>
      ))}
    </div>
  )
}

export function ModelBoard({
  board,
  sink,
  showBadges,
}: {
  board: ModelRow[]
  sink: ModelRow[]
  showBadges?: boolean
}) {
  return (
    <>
      <ModelList rows={board} showBadges={showBadges} />
      {sink.length > 0 && (
        <div className="model-sink-section">
          <p className="model-sink-label">
            No valid published results — audited sessions below were flagged or invalid
          </p>
          <ModelList rows={sink} showBadges={showBadges} sink />
        </div>
      )}
    </>
  )
}
