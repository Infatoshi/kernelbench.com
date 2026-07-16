import Link from "next/link"
import { notFound } from "next/navigation"
import {
  BENCH_LABELS,
  FLAG_VERDICTS,
  auditChipClass,
  benchValue,
  problemLabel,
  type Bench,
  type GpuBlock,
  type ModelCell,
} from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"

// One static page per published model (and per audited-but-unpublished model,
// whose page carries the integrity record). Everything renders from
// public/data/models.json at build time.

export const dynamicParams = false

export async function generateStaticParams() {
  const idx = await loadModelIndex()
  return idx.models.map((m) => ({ slug: m.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const idx = await loadModelIndex()
  const model = idx.models.find((m) => m.slug === slug)
  return { title: model ? `${model.name} · kernelbench` : "model · kernelbench" }
}

function fmtDur(sec: number): string {
  const h = Math.floor(sec / 3600)
  const m = Math.round((sec % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m`
  return `${Math.round(sec)}s`
}

function CellCard({
  bench,
  probKey,
  cell,
}: {
  bench: Bench
  probKey: string
  cell: ModelCell | undefined
}) {
  if (!cell) {
    return (
      <div className="cell-card">
        <div className="cell-card-head">
          <span className="cell-card-problem">{problemLabel(probKey)}</span>
          <span className="status-pill status-pill-muted">no run</span>
        </div>
      </div>
    )
  }
  const flagged = FLAG_VERDICTS.has(cell.verdict)
  return (
    <div className="cell-card">
      <div className="cell-card-head">
        <span className="cell-card-problem">{problemLabel(probKey)}</span>
        <span className={`status-pill ${cell.correct ? (flagged ? "status-pill-warn" : "status-pill-good") : "status-pill-bad"}`}>
          {cell.correct ? "pass" : "fail"}
        </span>
      </div>
      <div className="cell-card-metrics">
        {cell.score != null && (
          <span
            className="cell-score tabular"
            title={bench === "mega" ? "best speedup vs torch baseline" : "peak fraction of roofline"}
          >
            {benchValue(bench, cell.score)}
          </span>
        )}
        <span className={auditChipClass(cell.verdict)}>{cell.verdict.replace(/_/g, " ")}</span>
      </div>
      {bench === "mega" && (cell.tok_s != null || cell.ctx) && (
        <div className="leaderboard-muted" style={{ fontSize: "0.72rem" }}>
          {cell.tok_s != null && <span className="tabular">{cell.tok_s} tok/s&nbsp;&nbsp;</span>}
          {cell.ctx &&
            Object.entries(cell.ctx).map(([k, v]) => (
              <span key={k} className="tabular" style={{ marginRight: "0.55rem" }}>
                {k.replace("ctx", "")} ctx {v.toFixed(2)}x
              </span>
            ))}
          {cell.framework && <span> · {cell.framework}</span>}
        </div>
      )}
      {cell.elapsed_seconds != null && (
        <div className="leaderboard-muted" style={{ fontSize: "0.72rem" }}>
          session {fmtDur(cell.elapsed_seconds)}
        </div>
      )}
      <div className="cell-card-links">
        {cell.solution_url && (
          <Link href={cell.solution_url} className="link-chip">
            solution
          </Link>
        )}
        {cell.trace_url && (
          <a href={cell.trace_url} className="link-chip" target="_blank" rel="noreferrer">
            trace
          </a>
        )}
      </div>
    </div>
  )
}

function CellGrid({
  bench,
  problems,
  block,
}: {
  bench: Bench
  problems: string[]
  block: GpuBlock
}) {
  return (
    <div className="cell-grid">
      {problems.map((p) => (
        <CellCard key={p} bench={bench} probKey={p} cell={block.cells[p]} />
      ))}
    </div>
  )
}

export default async function ModelPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const idx = await loadModelIndex()
  const model = idx.models.find((m) => m.slug === slug)
  if (!model) notFound()

  const benchOrder: Bench[] = ["hard", "mega", "cuda"]
  const benches = benchOrder.filter((b) => model.benches[b])
  const legacy = model.legacy?.hard_v1

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          {model.name}
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-2">
          {model.lab}
          {model.totals.audited > 0 && (
            <span className="ml-3">
              {model.totals.flagged > 0 ? (
                <span className="status-pill status-pill-bad">
                  {model.totals.flagged}/{model.totals.audited} flagged
                </span>
              ) : (
                <span className="audit-chip audit-chip-muted">
                  {model.totals.audited} audited · none flagged
                </span>
              )}
            </span>
          )}
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
          One row per published session deck. Score bars are peak fraction of the
          board roofline (Hard / CUDA) or best speedup vs the torch baseline
          (Mega). Audit chips come from the human/subagent reward-hack review of
          every published cell.
        </p>
      </section>

      {benches.map((bench) => {
        const block = model.benches[bench]!
        const meta = idx.benches[bench]
        const canonicalLabel = meta?.gpu_labels?.["rtxpro6000"] ?? "canonical"
        const gpuKeys = (meta?.gpus ?? []).filter((g) => g !== "rtxpro6000" && block.gpus?.[g])
        return (
          <section key={bench}>
            <div className="flex items-center justify-center gap-3 flex-wrap mb-4">
              <h2 className="text-sm font-semibold text-[var(--color-fg-bright)]">
                {BENCH_LABELS[bench].toLowerCase()}
              </h2>
              <span className="leaderboard-muted text-xs">
                {block.label}
              </span>
              <span className={`status-pill ${block.passed >= block.total_problems ? "status-pill-good" : "status-pill-muted"}`}>
                {block.passed}/{block.total_problems}
              </span>
              {block.perf != null && (
                <span className="model-row-perf">
                  <span className="speed-bar">
                    <span className="speed-fill" style={{ width: `${Math.min(100, Math.max(0, block.perf * 100)).toFixed(1)}%` }} />
                  </span>
                  <span className="model-row-perf-val">{block.perf.toFixed(2)}</span>
                </span>
              )}
              {block.flagged > 0 ? (
                <span
                  className="status-pill status-pill-bad"
                  title={`${block.flagged} of ${block.audited} audited sessions flagged`}
                >
                  {block.flagged}/{block.audited} flagged
                </span>
              ) : (
                block.audited > 0 && (
                  <span className="audit-chip audit-chip-muted">{block.audited} audited</span>
                )
              )}
            </div>

            <p className="model-sink-label" style={{ marginBottom: "0.6rem" }}>
              {canonicalLabel}
            </p>
            <CellGrid bench={bench} problems={meta?.problems ?? []} block={block} />

            {gpuKeys.map((g) => (
              <div key={g} style={{ marginTop: "1.4rem" }}>
                <p className="model-sink-label" style={{ marginBottom: "0.6rem" }}>
                  {meta?.gpu_labels?.[g] ?? g}
                </p>
                <CellGrid bench={bench} problems={meta?.problems ?? []} block={block.gpus[g]!} />
              </div>
            ))}

            {block.flags.length > 0 && (
              <div className="flag-list" style={{ marginTop: "1.4rem" }}>
                {block.flags.map((f) => (
                  <div key={f.run_id} className="flag-item">
                    <div className="flag-item-head">
                      <span className={auditChipClass(f.verdict)}>{f.verdict.replace(/_/g, " ")}</span>
                      <span className="flag-item-run">{f.run_id}</span>
                    </div>
                    <p className="flag-item-summary">{f.summary}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        )
      })}

      {benches.length === 0 && (
        <section>
          <p className="text-xs text-[var(--color-fg-muted)]">
            No published board cells for this model — the integrity record above
            comes from audited sessions that did not publish a valid result.
          </p>
        </section>
      )}

      {legacy && (
        <section>
          <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
            Legacy pre-v2 hard board: best {legacy.best_pass_count}/
            {legacy.total_problems} passed across snapshot labels{" "}
            <span className="font-mono">{legacy.labels.join(", ")}</span>.
          </p>
        </section>
      )}

      <section>
        <p className="text-xs text-[var(--color-fg)] max-w-4xl leading-relaxed">
          Methodology: {idx.methodology} Browse the{" "}
          <Link href="/runs" className="underline underline-offset-2">
            run index
          </Link>{" "}
          for transcripts, submitted solutions, checks, timing, and costs.
        </p>
      </section>
    </div>
  )
}
