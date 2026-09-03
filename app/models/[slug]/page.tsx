import Link from "next/link"
import { notFound } from "next/navigation"
import {
  BENCH_LABELS,
  CANONICAL_GPU,
  FLAG_VERDICTS,
  SITE_HIDDEN_GPUS,
  auditChipClass,
  benchValue,
  brandFor,
  problemLabel,
  visibleProblems,
  type AuditOutcome,
  type Bench,
  type GpuBlock,
  type ModelCell,
  type ModelEntry,
  type ModelIndex,
} from "@/app/_lib/models"
import { loadAllModelIndex } from "@/app/_lib/models.server"
import { PageHead } from "@/app/_components/page-head"

// One static page per published model (and per audited-but-unpublished model,
// whose page carries the integrity record). Everything renders from
// public/data/models.json at build time.

export const dynamicParams = false

export async function generateStaticParams() {
  const idx = await loadAllModelIndex()
  return idx.models.map((m) => ({ slug: m.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const idx = await loadAllModelIndex()
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

/** Bench-native headline score over valid cells — same math as the homepage
 *  column charts: mean peak fraction (hard/cuda), best speedup (mega). */
function nativeScore(bench: Bench, block: GpuBlock): number | null {
  const vals = Object.values(block.cells).filter((c) => c.valid && c.score != null)
  if (vals.length === 0) return null
  if (bench === "mega") return Math.max(...vals.map((c) => c.score!))
  return vals.reduce((s, c) => s + c.score!, 0) / vals.length
}

/** A cell whose measurement came off a different GPU SKU than the board it is
 *  filed under (annotation `board_eligible: false`, projected by
 *  build_model_index as outcome "hardware"). It is not a result on this board,
 *  so it never renders as a scored row in a ranked grid — the number would read
 *  as an H100 PCIe (or RTX PRO 6000) score that was never measured there. */
function isOffBoardHardware(cell: ModelCell | undefined): boolean {
  return cell?.outcome === "hardware"
}

/** Split one board view into the cells that belong to it and the off-board
 *  hardware-mismatch cells, which get their own unranked block. */
function splitOffBoard(
  problems: string[],
  block: GpuBlock,
): { ranked: GpuBlock; offBoard: string[] } {
  const offBoard = problems.filter((p) => isOffBoardHardware(block.cells[p]))
  if (offBoard.length === 0) return { ranked: block, offBoard }
  const cells: Record<string, ModelCell> = {}
  for (const [p, c] of Object.entries(block.cells)) {
    if (!isOffBoardHardware(c)) cells[p] = c
  }
  return { ranked: { ...block, cells }, offBoard }
}

/** True when an audited attempt was measured on a different SKU than the board
 *  namespace it maps onto. `board_eligible: false` is the explicit marker; the
 *  SXM/PCIe split is implicit in the label (identify.py keys H100 SXM5 as
 *  H100_SXM, while the site's "h100" tab is the PCIe board). */
function isHardwareMismatch(outcome: AuditOutcome): boolean {
  if (outcome.board_eligible === false) return true
  return /\bSXM/i.test(outcome.gpu ?? "")
}

function CellCard({
  bench,
  gpu,
  probKey,
  cell,
  /** false = this card is outside the ranked grid (wrong GPU SKU): show the
   *  run and its links, never a number that would read as a board score. */
  ranked = true,
  hardwareNote,
}: {
  bench: Bench
  gpu: string
  probKey: string
  cell: ModelCell | undefined
  ranked?: boolean
  hardwareNote?: string | null
}) {
  if (!cell) {
    return (
      <div className="cell-card cell-card-empty">
        <div className="cell-card-head">
          <span className="cell-card-problem">{problemLabel(probKey)}</span>
          <span className="status-pill status-pill-muted">no run</span>
        </div>
      </div>
    )
  }
  const flagged = FLAG_VERDICTS.has(cell.verdict)
  const outcome = ranked
    ? cell.outcome_label ?? cell.failure_reason?.replace(/_/g, " ")
    : "wrong GPU SKU"
  const status = cell.valid && ranked ? "pass" : outcome || "failed"
  const pill = cell.valid && ranked
    ? "status-pill-good"
    : flagged
      ? "status-pill-warn"
      : cell.outcome === "wrong"
        ? "status-pill-bad"
        : cell.outcome === "empty"
          ? "status-pill-muted"
          : "status-pill-warn"
  return (
    <div className="cell-card">
      <div className="cell-card-head">
        <span className="cell-card-problem">{problemLabel(probKey)}</span>
        <span className={`status-pill ${pill}`}>{status}</span>
      </div>
      <div className="cell-card-metrics">
        {ranked && bench !== "mega" && cell.score != null && (
          <span
            className={`cell-card-score tabular${cell.valid ? "" : " cell-card-score-dim"}`}
            title="peak fraction of roofline"
          >
            {benchValue(bench, cell.score)}
          </span>
        )}
        {!ranked && <span className="cell-card-score tabular cell-card-score-dim">—</span>}
        <span className={ranked ? auditChipClass(cell.verdict) : "audit-chip audit-chip-muted"}>
          {ranked ? cell.verdict.replace(/_/g, " ") : "hardware mismatch"}
        </span>
      </div>
      {bench === "mega" && (
        <div className={`mega-result${cell.valid && ranked ? "" : " mega-result-invalid"}`}>
          <span className="mega-result-value tabular">
            {cell.valid && ranked && cell.score != null
              ? `${cell.score.toFixed(2)}x`
              : "no result"}
          </span>
          <span className="mega-result-label">
            {cell.valid && ranked
              ? "full-model speedup vs torch"
              : outcome || "no publishable speedup"}
          </span>
        </div>
      )}
      {(bench === "mega" && (cell.tok_s != null || cell.ctx)) || cell.elapsed_seconds != null ? (
        <div className="cell-card-sub tabular">
          {bench === "mega" && cell.tok_s != null && <span>{cell.tok_s} tok/s</span>}
          {bench === "mega" &&
            cell.ctx &&
            Object.entries(cell.ctx).map(([k, v]) => (
              <span key={k}>
                {k.replace("ctx", "")} ctx {v.toFixed(2)}x
              </span>
            ))}
          {bench === "mega" && cell.framework && <span>{cell.framework}</span>}
          {cell.elapsed_seconds != null && <span>session {fmtDur(cell.elapsed_seconds)}</span>}
        </div>
      ) : null}
      {!ranked ? (
        <p className="cell-card-cause">
          measured on {hardwareNote ?? "another GPU SKU"} — not a result on this board
        </p>
      ) : (
        !cell.valid && outcome && bench !== "mega" && (
          <p className="cell-card-cause">{outcome}</p>
        )
      )}
      <div className="cell-card-links">
        {cell.run_id && (
          <Link href={`/runs/${gpu}/${cell.run_id}`} className="link-chip">
            run page
          </Link>
        )}
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
  gpu,
  problems,
  block,
  ranked = true,
  hardwareNotes,
}: {
  bench: Bench
  gpu: string
  problems: string[]
  block: GpuBlock
  ranked?: boolean
  hardwareNotes?: Record<string, string | null>
}) {
  return (
    <div className="cell-grid">
      {problems.map((p) => (
        <CellCard
          key={p}
          bench={bench}
          gpu={gpu}
          probKey={p}
          cell={block.cells[p]}
          ranked={ranked}
          hardwareNote={hardwareNotes?.[block.cells[p]?.run_id ?? ""] ?? null}
        />
      ))}
    </div>
  )
}

/** Unranked block for cells whose measurement came off another GPU SKU. Run
 *  pages, solutions, and traces stay reachable; the number does not. */
function OffBoardGrid({
  bench,
  gpu,
  problems,
  block,
  hardwareNotes,
}: {
  bench: Bench
  gpu: string
  problems: string[]
  block: GpuBlock
  hardwareNotes: Record<string, string | null>
}) {
  return (
    <div className="board-extra">
      <p className="board-kicker">
        other hardware
        <span className="board-kicker-dim">· not ranked on this board</span>
      </p>
      <CellGrid
        bench={bench}
        gpu={gpu}
        problems={problems}
        block={block}
        ranked={false}
        hardwareNotes={hardwareNotes}
      />
    </div>
  )
}


function AuditOutcomeCard({ outcome, bench }: { outcome: AuditOutcome; bench: Bench }) {
  const gpu =
    outcome.gpu?.toUpperCase().includes("H100")
      ? "h100"
      : outcome.gpu?.toUpperCase().includes("B200")
        ? "b200"
        : CANONICAL_GPU
  const flagged = FLAG_VERDICTS.has(outcome.verdict)
  const hardwareMismatch = isHardwareMismatch(outcome)
  const status = hardwareMismatch
    ? "wrong GPU SKU"
    : flagged
      ? "excluded by audit"
      : outcome.publish_grade
        ? "publishable"
        : outcome.correct === false
          ? outcome.failure_reason?.replace(/_/g, " ") || "correctness failed"
          : outcome.measurement_status?.replace(/_/g, " ") || "audit evidence"
  const metric =
    outcome.score == null
      ? "no score"
      : bench === "mega"
        ? `${outcome.score.toFixed(2)}x`
        : `${(outcome.score * 100).toFixed(2)}%`
  return (
    <div className="mega-outcome-card">
      <div className="mega-outcome-head">
        <span className="mega-outcome-gpu">
          {outcome.gpu ?? "unknown GPU"} · {outcome.problem ? problemLabel(outcome.problem) : "unknown problem"}
        </span>
        <span className={hardwareMismatch ? "audit-chip audit-chip-muted" : auditChipClass(outcome.verdict)}>
          {hardwareMismatch ? "hardware mismatch" : outcome.verdict.replace(/_/g, " ")}
        </span>
      </div>
      <div className="mega-outcome-result">
        <strong>{metric}</strong>
        <span>{status}</span>
      </div>
      {outcome.summary && <p>{outcome.summary}</p>}
      <Link href={`/runs/${gpu}/${outcome.run_id}`} className="link-chip">
        audited run
      </Link>
    </div>
  )
}

function BenchPanel({
  bench,
  model,
  idx,
}: {
  bench: Bench
  model: ModelEntry
  idx: ModelIndex
}) {
  const block = model.benches[bench]!
  const meta = idx.benches[bench]
  const canonicalLabel = meta?.gpu_labels?.["rtxpro6000"] ?? "RTX PRO 6000"
  const gpuKeys = (meta?.gpus ?? []).filter(
    (g) => g !== "rtxpro6000" && !SITE_HIDDEN_GPUS.has(g) && block.gpus?.[g],
  )
  // Visible deck only (e.g. mega drops 01_rl_grid_ppo from the public board).
  const problems = visibleProblems(
    bench,
    meta?.problems?.length ? meta.problems : Object.keys(block.cells),
  )
  const passedVisible = problems.filter((p) => {
    const c = block.cells[p]
    return Boolean(c?.valid && c.score != null)
  }).length
  const totalVisible = problems.length
  const full = totalVisible > 0 && passedVisible >= totalVisible
  const harness = [block.harness, block.effort].filter(Boolean).join(" · ")
  const attempts = (block.outcomes ?? []).filter(
    (outcome) => !block.harness || outcome.harness?.startsWith(block.harness),
  )
  // run_id -> the SKU the annotation recorded, for the off-board note.
  const hardwareNotes: Record<string, string | null> = {}
  for (const outcome of block.outcomes ?? []) {
    hardwareNotes[outcome.run_id] = outcome.gpu ?? null
  }
  const canonical = splitOffBoard(problems, block)
  return (
    <section className="chart-panel board-panel">
      <div className="chart-panel-head">
        <div className="board-head-title">
          <span className="board-name">{BENCH_LABELS[bench]}</span>
          {harness && <span className="board-harness">{harness}</span>}
        </div>
        <div className="board-head-chips">
          <span className={`status-pill ${full ? "status-pill-good" : "status-pill-muted"}`}>
            {passedVisible}/{totalVisible} pass
          </span>
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
      </div>

      <p className="board-kicker">
        {canonicalLabel}
        <span className="board-kicker-dim">· canonical board</span>
      </p>
      <CellGrid
        bench={bench}
        gpu={CANONICAL_GPU}
        problems={problems}
        block={canonical.ranked}
      />
      {canonical.offBoard.length > 0 && (
        <OffBoardGrid
          bench={bench}
          gpu={CANONICAL_GPU}
          problems={canonical.offBoard}
          block={block}
          hardwareNotes={hardwareNotes}
        />
      )}

      {gpuKeys.map((g) => {
        const view = block.gpus[g]!
        const split = splitOffBoard(problems, view)
        return (
          <div key={g} className="board-extra">
            <p className="board-kicker">{meta?.gpu_labels?.[g] ?? g}</p>
            <CellGrid bench={bench} gpu={g} problems={problems} block={split.ranked} />
            {split.offBoard.length > 0 && (
              <OffBoardGrid
                bench={bench}
                gpu={g}
                problems={split.offBoard}
                block={view}
                hardwareNotes={hardwareNotes}
              />
            )}
          </div>
        )
      })}

      {attempts.length > 0 && (
        <div className="mega-outcomes">
          <p className="board-kicker">all audited {block.harness ?? "selected"} attempts</p>
          <p className="mega-outcomes-note">
            Every attempted cell stays visible, including correctness failures, hardware mismatches,
            contaminated runs, and audit rejects. Only publishable results contribute to the board above.
          </p>
          <div className="mega-outcome-grid">
            {attempts.map((outcome) => (
              <AuditOutcomeCard key={outcome.run_id} outcome={outcome} bench={bench} />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default async function ModelPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const idx = await loadAllModelIndex()
  const model = idx.models.find((m) => m.slug === slug)
  if (!model) notFound()

  const benchOrder: Bench[] = ["hard", "mega", "cuda"]
  const benches = benchOrder.filter((b) => model.benches[b])
  const legacy = model.legacy?.hard_v1
  const brand = brandFor(model.lab, model.slug)

  const passed = benches.reduce((s, b) => s + (model.benches[b]?.passed ?? 0), 0)
  const total = benches.reduce((s, b) => s + (model.benches[b]?.total_problems ?? 0), 0)
  const { audited, flagged } = model.totals

  return (
    <div id="top" className="space-y-6">
      <PageHead
        kicker={`Model · ${model.lab}`}
        title={
          <>
            {brand.logo && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={brand.logo} alt="" className="page-head-logo" aria-hidden="true" />
            )}
            {model.name}
          </>
        }
        sub={
          <>
            {benches.length > 0 ? (
              <>
                {benches.length} bench {benches.length === 1 ? "deck" : "decks"} ·{" "}
                <strong>
                  {passed}/{total}
                </strong>{" "}
                problems correct on canonical boards
                {audited > 0 && (
                  <>
                    {" "}
                    · {audited} audited cells
                    {flagged > 0 && (
                      <>
                        {" "}
                        — <span className="mstat-flag-txt">{flagged} flagged</span>
                      </>
                    )}
                  </>
                )}
                .
              </>
            ) : (
              legacy
                ? "No published cells on the current decks — the legacy board record is below."
                : "No published cells on the current decks."
            )}
          </>
        }
        notes={
          <>
            <p>
              <strong>How to read.</strong> Cell scores are peak fraction of the
              board roofline (Hard / CUDA) or best speedup vs the torch baseline
              (Mega), over one unlimited agent session per cell. Audit chips come
              from the human/subagent reward-hack review of every published
              cell; scores from flagged sessions render dimmed — they don&apos;t
              count toward the charts.
            </p>
            <p>
              <strong>Board summary bars</strong> are each score relative to the
              best published model on that board (1.00 = board leader); the
              printed number is the bench-native score.
            </p>
            <p>
              <strong>Methodology.</strong> {idx.methodology} Browse the{" "}
              <Link href="/runs">run index</Link> for transcripts, submitted
              solutions, checks, timing, and costs.
            </p>
          </>
        }
      />

      {benches.length > 0 && (
        <div className="chart-panel">
          <div className="chart-panel-head">
            <span className="chart-panel-title">Board summary</span>
            <span className="panel-note">
              bars = share of each board&apos;s best model · numbers = bench-native score
            </span>
          </div>
          <div className="mbar mstat">
            {benches.map((bench, i) => {
              const block = model.benches[bench]!
              const native = nativeScore(bench, block)
              const width = Math.min(100, Math.max(0, (block.perf ?? 0) * 100))
              const full = block.total_problems > 0 && block.passed >= block.total_problems
              return (
                <div className="mbar-row" key={bench}>
                  <div className="mbar-label">
                    <span className="mstat-bench">{BENCH_LABELS[bench]}</span>
                  </div>
                  <span className="mbar-track">
                    <span
                      className="mbar-fill"
                      style={{
                        width: `${width.toFixed(1)}%`,
                        background: brand.color,
                        animationDelay: `${Math.min(i * 120, 360)}ms`,
                      }}
                    />
                  </span>
                  <span className="mbar-right">
                    <span className="mbar-val tabular">
                      {native != null ? benchValue(bench, native) : "—"}
                    </span>
                    <span className={`status-pill ${full ? "status-pill-good" : "status-pill-muted"}`}>
                      {block.passed}/{block.total_problems}
                    </span>
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {benches.map((bench) => (
        <BenchPanel key={bench} bench={bench} model={model} idx={idx} />
      ))}

      {legacy && (
        <p className="model-legacy">
          legacy pre-v2 hard board — best {legacy.best_pass_count}/{legacy.total_problems} passed
          across snapshots <span className="font-mono">{legacy.labels.join(", ")}</span>
        </p>
      )}
    </div>
  )
}
