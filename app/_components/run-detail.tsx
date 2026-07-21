"use client"

import { useEffect, useState } from "react"
import type { ProblemChip } from "../_lib/models"

// Run-detail panel: click a board chip → the row expands in place (no modal)
// with the per-shape roofline breakdown, session stats, GPU-lock totals, the
// (redacted) kernel, and the trace link. Data comes from
// /data/rundetail/<run_id>.json (built by scripts/build_run_detail.py);
// solution text from /runs/*_solution.py.txt.

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
  harness: string | null
  model: string | null
  regime: string | null
  dtype: string
  peak_tflops: number | null
  peak_bw_gb_s: number | null
  correct: boolean | null
  failure_reason: string | null
  peak_fraction: number | null
  annotation_verdict: string | null
  stats: {
    agent_s: number | null
    total_s: number | null
    check_s: number | null
    benchmark_s: number | null
    output_tokens: number | null
    cost_usd: number | null
    template_mutated: boolean | null
  }
  gpu_lock: { wait_s: number; active_s: number; acquisitions: number } | null
  shapes: ShapeRow[]
  has_solution_text: boolean
}

const GPU_NAMES: Record<string, string> = {
  rtxpro6000: "RTX PRO 6000",
  h100: "H100 PCIe",
  b200: "B200",
  rtx3090: "RTX 3090",
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

function ShapeStrip({ d }: { d: RunDetailData }) {
  if (!d.shapes.length) {
    return (
      <p className="rdetail-muted">
        No per-shape benchmark data archived for this run.
      </p>
    )
  }
  return (
    <div className="rdetail-shapes">
      {d.shapes.map((s) => {
        const util = s.util ?? 0
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
            <span className="rdetail-shape-dims tabular">
              {s.label ?? `shape ${s.idx}`}
            </span>
            <span className="rdetail-shape-ms tabular">
              {s.ms != null ? `${s.ms.toFixed(3)} ms` : "—"}
            </span>
            <span className="rdetail-shape-track">
              <span
                className={`rdetail-shape-fill rdetail-${s.bound ?? "compute"}`}
                style={{ width: `${Math.min(util, 1) * 100}%` }}
              />
              {s.frac != null && (
                <span
                  className="rdetail-shape-tick"
                  title={`official fraction ${s.frac.toFixed(3)} (geomean input)`}
                  style={{ left: `${Math.min(s.frac, 1) * 100}%` }}
                />
              )}
            </span>
            <span className="rdetail-shape-util tabular">
              {Math.round(util * 100)}%
            </span>
            <span className="rdetail-shape-detail">
              {achieved} · {Math.round(util * 100)}% of {ceiling}
              {other ? ` · ${other}` : ""}
            </span>
          </div>
        )
      })}
      <p className="rdetail-legend">
        <span className="rdetail-swatch rdetail-compute" /> compute-bound
        <span className="rdetail-swatch rdetail-memory" /> memory-bound
        <span className="rdetail-swatch rdetail-tickswatch" /> official fraction
        (feeds the geomean)
      </p>
    </div>
  )
}

export function RunDetailPanel({
  chip,
  modelName,
  onClose,
}: {
  chip: ProblemChip
  modelName: string
  onClose: () => void
}) {
  const [data, setData] = useState<RunDetailData | null>(null)
  const [dataMissing, setDataMissing] = useState(false)
  const [solution, setSolution] = useState<string | null>(null)
  const [showSolution, setShowSolution] = useState(false)

  useEffect(() => {
    if (!chip.run_id) return
    let alive = true
    setData(null)
    setDataMissing(false)
    setSolution(null)
    setShowSolution(false)
    fetch(chip.detail_url ?? `/data/rundetail/${chip.run_id}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d) => alive && setData(d))
      .catch(() => alive && setDataMissing(true))
    return () => {
      alive = false
    }
  }, [chip.run_id, chip.detail_url])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  useEffect(() => {
    if (!showSolution || solution != null || !chip.solution_url) return
    let alive = true
    fetch(chip.solution_url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status}`))))
      .then((t) => alive && setSolution(t))
      .catch(() => alive && setSolution("// solution text unavailable"))
    return () => {
      alive = false
    }
  }, [showSolution, solution, chip.solution_url])

  const verdict = data?.annotation_verdict
  return (
    <section
      className="rdetail-inline"
      aria-label={`${chip.problem} run detail`}
    >
      <header className="rdetail-head">
        <div>
          <p className="rdetail-kicker">
            {data ? `${data.bench} · ${GPU_NAMES[data.gpu] ?? data.gpu}` : "run"}
          </p>
          <h3 className="rdetail-title">
            {chip.problem} <span className="rdetail-model">{modelName}</span>
          </h3>
        </div>
        <div className="rdetail-headline">
          {chip.kind === "pass" ? (
            <>
              <span className="rdetail-geomean tabular">{chip.label}</span>
              <span className="rdetail-geomean-label">geomean of shapes</span>
            </>
          ) : (
            <span className="status-pill status-pill-bad">{chip.label}</span>
          )}
        </div>
        <button
          type="button"
          className="rdetail-close"
          onClick={onClose}
          aria-label="Close run detail"
        >
          ✕
        </button>
      </header>

      {verdict && verdict !== "clean" && (
        <p className="rdetail-flag">audit verdict: {verdict}</p>
      )}

      {dataMissing && (
        <p className="rdetail-muted">
          No archived detail for this run yet (archive not synced). Solution
          and trace links below still work.
        </p>
      )}

      {data && (
        <>
          <div className="rdetail-stats">
            <Stat label="agent session" value={fmtDuration(data.stats.agent_s)} />
            <Stat label="total wall" value={fmtDuration(data.stats.total_s)} />
            <Stat label="check" value={fmtDuration(data.stats.check_s)} />
            <Stat label="benchmark" value={fmtDuration(data.stats.benchmark_s)} />
            <Stat
              label="output tokens"
              value={fmtInt(data.stats.output_tokens)}
            />
            {data.stats.cost_usd != null && (
              <Stat label="cost" value={`$${data.stats.cost_usd.toFixed(2)}`} />
            )}
            {data.gpu_lock && (
              <>
                <Stat
                  label="gpu-lock wait"
                  value={fmtDuration(data.gpu_lock.wait_s)}
                />
                <Stat
                  label="gpu-lock held"
                  value={fmtDuration(data.gpu_lock.active_s)}
                />
              </>
            )}
            <Stat label="regime" value={data.regime ?? "—"} />
          </div>
          <h4 className="rdetail-section">
            Per-shape vs governing ceiling
            <span className="rdetail-section-note">
              each shape graded against whichever binds — {data.dtype} compute
              or HBM bandwidth
            </span>
          </h4>
          <ShapeStrip d={data} />
        </>
      )}

      <div className="rdetail-links">
        {chip.solution_url && (
          <button
            type="button"
            className="rdetail-linkbtn"
            onClick={() => setShowSolution((v) => !v)}
          >
            {showSolution ? "Hide kernel" : "Show kernel"}
          </button>
        )}
        {chip.trace_url && (
          <a
            className="rdetail-linkbtn"
            href={chip.trace_url}
            target="_blank"
            rel="noreferrer"
          >
            Agent trace ↗
          </a>
        )}
        {chip.solution_url && (
          <a
            className="rdetail-linkbtn"
            href={chip.solution_url}
            target="_blank"
            rel="noreferrer"
          >
            Raw solution ↗
          </a>
        )}
      </div>

      {showSolution && (
        <pre className="rdetail-code">
          <code>{solution ?? "loading…"}</code>
        </pre>
      )}
    </section>
  )
}
