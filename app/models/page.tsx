import Link from "next/link"
import { brandFor, rowsForIndex } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { PageHead } from "@/app/_components/page-head"

// The model directory: every model with published cells, ranked, each row
// opening its per-model page (cells, audits, integrity record). The homepage
// column charts stay the comparison view — this is the roster.

export const metadata = { title: "models · kernelbench" }

export default async function ModelsPage() {
  const idx = await loadModelIndex()
  const { board, sink } = rowsForIndex(idx)
  return (
    <div className="space-y-6">
      <PageHead
        kicker="Index"
        title="Models"
        sub={
          <>
            <strong>{board.length}</strong> models on the boards
            {sink.length > 0 && (
              <>
                {" "}
                · <span className="mroster-dim">{sink.length} more ran but published no cells</span>
              </>
            )}
          </>
        }
        notes={
          <>
            <p>
              <strong>Ordering.</strong> Models rank by benches fully passed,
              then mean score relative to each board&apos;s best published
              model (1.00 = board leader). Per-bench chips show valid passes;
              the flag pill counts audited sessions that failed the reward-hack
              review — it is displayed, never a sort key.
            </p>
            <p>
              <strong>Roster vs charts.</strong> This directory lists every
              published model, including superseded siblings the homepage
              charts curate away. Click any row for its per-problem cells,
              audit chips, GPU boards, and integrity record — or open the{" "}
              <Link href="/runs">run index</Link> for raw transcripts.
            </p>
          </>
        }
      />

      <div className="chart-panel">
        <div className="chart-panel-head">
          <span className="chart-panel-title">Ranked</span>
          <span className="panel-note">bars = mean share of each board&apos;s best model</span>
        </div>
        <div className="mbar mroster">
          {board.map((row, i) => {
            const brand = brandFor(row.lab, row.slug)
            const width = Math.min(100, Math.max(0, (row.perf ?? 0) * 100))
            return (
              <Link key={row.slug} href={`/models/${row.slug}`} className="mbar-row mroster-row no-underline">
                <span className="mbar-label">
                  <span className="mbar-name-line">
                    {brand.logo ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={brand.logo} alt="" className="mbar-logo" aria-hidden="true" />
                    ) : (
                      <span className="mbar-logo mbar-logo-letter" style={{ color: brand.color }} aria-hidden="true">
                        {row.lab.slice(0, 1)}
                      </span>
                    )}
                    <span className="mroster-id">
                      <span className="mbar-name">{row.name}</span>
                      <span className="mroster-lab">{row.lab}</span>
                    </span>
                  </span>
                </span>
                <span className="mroster-mid">
                  <span className="mroster-badges">
                    {row.badges.map((b) => (
                      <span
                        key={b.label}
                        className={`mroster-chip${b.tone === "good" ? " mroster-chip-good" : ""}`}
                      >
                        {b.label} {b.value}
                      </span>
                    ))}
                  </span>
                  <span className="mbar-track mroster-track">
                    <span
                      className="mbar-fill"
                      style={{
                        width: `${width.toFixed(1)}%`,
                        background: brand.color,
                        animationDelay: `${Math.min(i * 45, 500)}ms`,
                      }}
                    />
                  </span>
                </span>
                <span className="mbar-right">
                  <span className="mbar-val tabular">
                    {row.perf != null ? row.perf.toFixed(2) : "—"}
                  </span>
                  {row.flagged > 0 ? (
                    <span
                      className="status-pill status-pill-bad"
                      title={`${row.flagged} of ${row.audited} audited sessions flagged`}
                    >
                      {row.flagged}/{row.audited}
                    </span>
                  ) : row.audited > 0 ? (
                    <span className="audit-chip audit-chip-muted">{row.audited} aud</span>
                  ) : null}
                </span>
              </Link>
            )
          })}
        </div>
      </div>

      {sink.length > 0 && (
        <div className="mroster-sink">
          <p className="board-kicker">ran but published no cells</p>
          <div className="mroster-sink-list">
            {sink.map((row) => (
              <Link key={row.slug} href={`/models/${row.slug}`} className="mroster-sink-row no-underline">
                <span className="mroster-sink-name">{row.name}</span>
                <span className="mroster-lab">{row.lab}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
