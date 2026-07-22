import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { Metadata } from "next"
import Link from "next/link"
import { notFound } from "next/navigation"
import {
  CANONICAL_GPU,
  FLAG_VERDICTS,
  problemLabel,
  type Bench,
  type ModelCell,
} from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"

// One page per published run: the cell you clicked on a board, expanded.
// Fully static — every (gpu, run_id) cell in models.json gets a page at
// build time. Shape/stat data comes from public/data/rundetail (absent for
// mega, which archives no per-shape sweep); the redacted kernel is inlined
// from public/runs at build so the page needs no client fetches.

export const dynamicParams = false

const GPU_NAMES: Record<string, string> = {
  rtxpro6000: "RTX PRO 6000",
  h100: "H100",
  b200: "B200",
}

interface ShapeRow {
  idx: number
  label?: string
  ms?: number
  tflops?: number
  gbps?: number
  frac?: number
  compute_util?: number
  mem_util?: number
  bound?: "compute" | "memory"
  util?: number
}

interface RunDetailData {
  run_id: string
  bench: string
  gpu: string
  problem: string
  regime: string | null
  dtype: string
  peak_tflops: number | null
  peak_bw_gb_s: number | null
  annotation_verdict: string | null
  stats: {
    agent_s: number | null
    total_s: number | null
    check_s: number | null
    benchmark_s: number | null
    output_tokens: number | null
    cost_usd: number | null
  }
  gpu_lock: { wait_s: number; active_s: number; acquisitions: number } | null
  shapes: ShapeRow[]
}

interface FoundCell {
  bench: Bench
  problem: string
  modelName: string
  modelSlug: string
  harness: string | null
  cell: ModelCell
}

async function findCell(gpu: string, rid: string): Promise<FoundCell | null> {
  const idx = await loadModelIndex()
  for (const m of idx.models) {
    for (const [bench, block] of Object.entries(m.benches)) {
      if (!block) continue
      const views: { g: string; cells: Record<string, ModelCell> }[] = [
        { g: CANONICAL_GPU, cells: block.cells ?? {} },
      ]
      for (const [g, v] of Object.entries(block.gpus ?? {}))
        views.push({ g, cells: v.cells ?? {} })
      for (const { g, cells } of views) {
        if (g !== gpu) continue
        for (const [problem, cell] of Object.entries(cells)) {
          if (cell.run_id === rid)
            return {
              bench: bench as Bench,
              problem,
              modelName: m.name,
              modelSlug: m.slug,
              harness: block.harness,
              cell,
            }
        }
      }
    }
  }
  return null
}

async function readPublic(rel: string): Promise<string | null> {
  try {
    return await readFile(join(process.cwd(), "public", rel), "utf8")
  } catch {
    return null
  }
}

async function loadDetail(gpu: string, rid: string): Promise<RunDetailData | null> {
  const prefix = gpu === CANONICAL_GPU ? "" : `${gpu}/`
  const raw = await readPublic(`data/rundetail/${prefix}${rid}.json`)
  return raw ? (JSON.parse(raw) as RunDetailData) : null
}

export async function generateStaticParams() {
  const idx = await loadModelIndex()
  const params: { gpu: string; rid: string }[] = []
  const seen = new Set<string>()
  for (const m of idx.models) {
    for (const block of Object.values(m.benches)) {
      if (!block) continue
      const views: { g: string; cells: Record<string, ModelCell> }[] = [
        { g: CANONICAL_GPU, cells: block.cells ?? {} },
      ]
      for (const [g, v] of Object.entries(block.gpus ?? {}))
        views.push({ g, cells: v.cells ?? {} })
      for (const { g, cells } of views) {
        for (const cell of Object.values(cells)) {
          if (!cell.run_id) continue
          const key = `${g}/${cell.run_id}`
          if (seen.has(key)) continue
          seen.add(key)
          params.push({ gpu: g, rid: cell.run_id })
        }
      }
    }
  }
  return params
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ gpu: string; rid: string }>
}): Promise<Metadata> {
  const { gpu, rid } = await params
  const found = await findCell(gpu, rid)
  if (!found) return {}
  const title = `${found.modelName} · ${problemLabel(found.problem)} · ${GPU_NAMES[gpu] ?? gpu}`
  return { title, description: `KernelBench run ${rid}` }
}

function fmtDuration(s: number | null | undefined): string {
  if (s == null) return "—"
  if (s < 90) return `${s}s`
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

function fmtInt(n: number | null | undefined): string {
  return n == null ? "—" : n.toLocaleString("en-US")
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="rdetail-stat">
      <span className="rdetail-stat-label">{label}</span>
      <span className="rdetail-stat-value tabular">{value}</span>
    </span>
  )
}

/** The fraction that feeds the geomean: benchmark.py's per-shape official
 *  fraction where archived, else the bound-specific utilization. */
function shapeFrac(s: ShapeRow): number {
  return s.frac ?? s.util ?? 0
}

function geomean(vals: number[]): number | null {
  const pos = vals.filter((v) => v > 0)
  if (!pos.length) return null
  return Math.exp(pos.reduce((a, v) => a + Math.log(v), 0) / pos.length)
}

function ShapeStrip({ d, official }: { d: RunDetailData; official: number | null }) {
  if (!d.shapes.length) {
    return (
      <p className="rdetail-muted">No per-shape benchmark data archived for this run.</p>
    )
  }
  const fracs = d.shapes.map(shapeFrac)
  const gm = geomean(fracs)
  return (
    <div className="rdetail-shapes">
      {d.shapes.map((s) => {
        const f = shapeFrac(s)
        const achieved =
          s.bound === "memory"
            ? `${((s.gbps ?? 0) / 1000).toFixed(2)} TB/s`
            : `${fmtInt(Math.round(s.tflops ?? 0))} TFLOPS`
        const ceiling =
          s.bound === "memory"
            ? `${((d.peak_bw_gb_s ?? 0) / 1000).toFixed(1)} TB/s HBM`
            : `${fmtInt(d.peak_tflops)} TF ${d.dtype} peak`
        const other =
          s.bound === "memory"
            ? s.compute_util != null
              ? `also ${fmtInt(Math.round(s.tflops ?? 0))} TFLOPS (${Math.round((s.compute_util ?? 0) * 100)}% of compute)`
              : null
            : s.mem_util != null
              ? `also ${((s.gbps ?? 0) / 1000).toFixed(2)} TB/s (${Math.round((s.mem_util ?? 0) * 100)}% of HBM)`
              : null
        return (
          <div key={s.idx} className="rdetail-shape">
            <span className="rdetail-shape-dims tabular">{s.label ?? `shape ${s.idx}`}</span>
            <span className="rdetail-shape-ms tabular">
              {s.ms != null ? `${s.ms.toFixed(3)} ms` : "—"}
            </span>
            <span className="rdetail-shape-track">
              <span
                className={`rdetail-shape-fill rdetail-${s.bound ?? "compute"}`}
                style={{ width: `${Math.min(f, 1) * 100}%` }}
              />
            </span>
            <span className="rdetail-shape-util tabular">{(f * 100).toFixed(1)}%</span>
            <span className="rdetail-shape-detail">
              {achieved} · {Math.round((s.util ?? 0) * 100)}% of {ceiling}
              {other ? ` · ${other}` : ""}
            </span>
          </div>
        )
      })}
      <p className="rdetail-legend">
        <span className="rdetail-swatch rdetail-compute" /> compute-bound
        <span className="rdetail-swatch rdetail-memory" /> memory-bound · bar +
        right column = official fraction of the ceiling (the geomean input)
      </p>
      {gm != null && (
        <p className="run-page-math tabular">
          geomean({fracs.map((f) => `${(f * 100).toFixed(1)}%`).join(" · ")}) ={" "}
          <strong>{(gm * 100).toFixed(1)}%</strong>
          {official != null && Math.abs(gm - official) > 0.002 && (
            <>
              {" "}
              · published {(official * 100).toFixed(1)}% (lower of repeated
              isolated re-benchmark passes)
            </>
          )}
        </p>
      )}
    </div>
  )
}

export default async function RunPage({
  params,
}: {
  params: Promise<{ gpu: string; rid: string }>
}) {
  const { gpu, rid } = await params
  const found = await findCell(gpu, rid)
  if (!found) notFound()
  const { bench, problem, modelName, modelSlug, harness, cell } = found

  const detail = await loadDetail(gpu, rid)
  const solution = cell.solution_url
    ? await readPublic(cell.solution_url.replace(/^\//, ""))
    : null

  const gpuName = GPU_NAMES[gpu] ?? gpu
  const isSpeedup = (cell.score ?? 0) > 1.5
  const headline =
    cell.valid && cell.score != null
      ? isSpeedup
        ? `${cell.score.toFixed(2)}×`
        : `${(cell.score * 100).toFixed(cell.score >= 0.1 ? 1 : 2)}%`
      : (cell.outcome_label ?? cell.failure_reason ?? "no pass")
  const headlineSub =
    cell.valid && cell.score != null
      ? isSpeedup
        ? "geomean speedup across shapes"
        : "geomean peak fraction across shapes"
      : "did not score"
  const verdict = cell.verdict
  const flagged = verdict && FLAG_VERDICTS.has(verdict)

  return (
    <main className="run-page">
      <p className="page-head-kicker">
        <Link href={`/#${bench}`}>KernelBench {bench}</Link> · {gpuName}
      </p>
      <h1 className="page-head-title">
        {problemLabel(problem)}{" "}
        <span className="rdetail-model">
          <Link href={`/models/${modelSlug}`}>{modelName}</Link>
        </span>
      </h1>
      <div className="rdetail-headline run-page-headline">
        <span className="rdetail-geomean tabular">{headline}</span>
        <span className="rdetail-geomean-label">{headlineSub}</span>
      </div>

      {flagged && <p className="rdetail-flag">audit verdict: {verdict}</p>}
      {verdict === "clean" && (
        <p className="rdetail-muted">manually audited: clean</p>
      )}

      <div className="rdetail-stats">
        {harness && <Stat label="harness" value={harness} />}
        {detail ? (
          <>
            <Stat label="agent session" value={fmtDuration(detail.stats.agent_s)} />
            <Stat label="total wall" value={fmtDuration(detail.stats.total_s)} />
            <Stat label="check" value={fmtDuration(detail.stats.check_s)} />
            <Stat label="benchmark" value={fmtDuration(detail.stats.benchmark_s)} />
            <Stat label="output tokens" value={fmtInt(detail.stats.output_tokens)} />
            {detail.stats.cost_usd != null && (
              <Stat label="cost" value={`$${detail.stats.cost_usd.toFixed(2)}`} />
            )}
            {detail.gpu_lock && (
              <>
                <Stat label="gpu-lock wait" value={fmtDuration(detail.gpu_lock.wait_s)} />
                <Stat label="gpu-lock held" value={fmtDuration(detail.gpu_lock.active_s)} />
              </>
            )}
            <Stat label="regime" value={detail.regime ?? "—"} />
          </>
        ) : (
          cell.elapsed_seconds != null && (
            <Stat label="agent session" value={fmtDuration(cell.elapsed_seconds)} />
          )
        )}
      </div>

      {detail && (
        <>
          <h2 className="rdetail-section">
            Per-shape vs governing ceiling
            <span className="rdetail-section-note">
              each shape graded against whichever binds — {detail.dtype} compute or
              HBM bandwidth
            </span>
          </h2>
          <ShapeStrip
            d={detail}
            official={cell.valid && !isSpeedup ? cell.score : null}
          />
        </>
      )}

      <div className="rdetail-links">
        {cell.trace_url && (
          <a className="rdetail-linkbtn" href={cell.trace_url} target="_blank" rel="noreferrer">
            Agent trace ↗
          </a>
        )}
        {cell.solution_url && (
          <a className="rdetail-linkbtn" href={cell.solution_url} target="_blank" rel="noreferrer">
            Raw solution ↗
          </a>
        )}
      </div>

      {solution && (
        <details className="run-page-solution">
          <summary className="rdetail-linkbtn">Kernel source (redacted)</summary>
          <pre className="rdetail-code">
            <code>{solution}</code>
          </pre>
        </details>
      )}

      <p className="rdetail-muted run-page-rid tabular">{rid}</p>
    </main>
  )
}
