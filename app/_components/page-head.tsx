// Shared bench-page header: mono kicker + title + one-line sub, left-aligned.
// Verbose methodology folds into an optional <details> slot.

export function PageHead({
  kicker,
  title,
  sub,
  notesTitle = "methodology + notes",
  notes,
}: {
  kicker: string
  title: React.ReactNode
  sub?: React.ReactNode
  notesTitle?: string
  notes?: React.ReactNode
}) {
  return (
    <div className="page-head">
      <p className="page-head-kicker">{kicker}</p>
      <h1 className="page-head-title">{title}</h1>
      {sub && <p className="page-head-sub">{sub}</p>}
      {notes && (
        <details className="notes-details">
          <summary>{notesTitle}</summary>
          <div className="notes-body">{notes}</div>
        </details>
      )}
    </div>
  )
}
