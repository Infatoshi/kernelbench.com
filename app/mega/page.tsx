"use client"

import { useState, useEffect, useMemo } from "react"
import Link from "next/link"

interface Row {
  gpu: string
  harness: string
  model: string
  problem: string
  correct: boolean
  score: number | null
  tok_s: number | null
  elapsed_s: number | null
  output_tokens: number | null
  ctx2048: number | null
  ctx8192: number | null
  ctx16384: number | null
  framework: string
  has_viewer: boolean
  run_id: string
}

const GPU_ORDER = ["RTX PRO 6000 Blackwell", "H100", "B200"]

type SortKey = "speed" | "tok_s" | "runtime" | "tokens" | null

export default function MegaPage() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({
    key: null,
    dir: "desc",
  })

  useEffect(() => {
    fetch("/data/mega/results.csv")
      .then((r) => r.text())
      .then((csv) => {
        const lines = csv.trim().split("\n")
        const headers = parseCsv(lines[0]).map((h) => h.trim())
        const numericFields = new Set([
          "score",
          "tok_s",
          "elapsed_s",
          "output_tokens",
          "ctx2048",
          "ctx8192",
          "ctx16384",
        ])
        const rows: Row[] = []
        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue
          const v = parseCsv(lines[i])
          const r: Record<string, unknown> = {}
          headers.forEach((h, idx) => {
            const val = (v[idx] ?? "").trim()
            if (h === "correct") {
              r[h] = val.toLowerCase() === "true"
            } else if (h === "has_viewer") {
              r[h] = val.toLowerCase() === "true"
            } else if (numericFields.has(h)) {
              r[h] =
                val === "" || val.toLowerCase() === "nan" || val === "None"
                  ? null
                  : parseFloat(val)
            } else {
              r[h] = val
            }
          })
          rows.push(r as unknown as Row)
        }
        setData(rows)
        setLoading(false)
      })
  }, [])

  // Max speedup across the whole table, used to normalize the bar widths.
  const maxScore = useMemo(
    () => Math.max(1, ...data.map((r) => r.score ?? 0)),
    [data],
  )

  // Top speedup per GPU (the winner), keyed by run_id.
  const winners = useMemo(() => {
    const best = new Map<string, { runId: string; score: number }>()
    for (const r of data) {
      if (!r.correct || r.score == null) continue
      const cur = best.get(r.gpu)
      if (!cur || r.score > cur.score) {
        best.set(r.gpu, { runId: r.run_id, score: r.score })
      }
    }
    return new Set([...best.values()].map((b) => b.runId))
  }, [data])

  // Rows grouped by GPU (fixed order), default sorted by speedup desc within a
  // group. A clicked column header overrides the within-group sort key while
  // keeping the GPU grouping stable.
  const sorted = useMemo(() => {
    const gpuRank = (g: string) => {
      const i = GPU_ORDER.indexOf(g)
      return i === -1 ? GPU_ORDER.length : i
    }
    const value = (r: Row): number | null => {
      switch (sort.key) {
        case "tok_s":
          return r.tok_s
        case "runtime":
          return r.elapsed_s
        case "tokens":
          return r.output_tokens
        case "speed":
        default:
          return r.score
      }
    }
    const cmp = (a: Row, b: Row) => {
      const av = value(a)
      const bv = value(b)
      const an = av == null
      const bn = bv == null
      if (an && bn) return 0
      if (an) return 1
      if (bn) return -1
      const diff = sort.dir === "asc" ? av - bv : bv - av
      return diff
    }
    return [...data].sort(
      (a, b) =>
        gpuRank(a.gpu) - gpuRank(b.gpu) ||
        a.gpu.localeCompare(b.gpu) ||
        cmp(a, b),
    )
  }, [data, sort])

  const onSort = (key: SortKey) => {
    setSort((cur) =>
      cur.key === key
        ? { key, dir: cur.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" },
    )
  }

  // Track GPU group boundaries so we can render a subtle group label row.
  let lastGpu: string | null = null

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          mega
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          KernelBench-Mega · whole-block megakernels · RTX PRO 6000 Blackwell +
          H100 + B200
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          KernelBench-Mega tests <strong>whole-block megakernels</strong>:
          instead of grading a single isolated op, the agent fuses an entire
          model block into one kernel. Problem{" "}
          <code>03_kimi_linear_decode</code> is a Kimi-Linear W4A16 hybrid
          decode (4-bit weights, bf16 activations). The headline metric{" "}
          <strong>
            score is the decode speedup over an optimized-PyTorch baseline
          </strong>{" "}
          (e.g. <code>19.35x</code> = 19x faster than the reference), not a 0-1
          roofline fraction. <code>tok/s</code> is decode tokens per second.
          Higher is better for both, and results are reported per GPU. The{" "}
          <strong>transcript link is the headline artifact</strong>: it shows
          the model&apos;s full optimization journey from baseline to the final
          megakernel.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          # leaderboard
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4">
          {loading
            ? "loading..."
            : `${sorted.length.toLocaleString()} runs · grouped by GPU, sorted by decode speedup vs optimized-PyTorch baseline`}
        </p>

        <div className="overflow-x-auto box max-h-[70vh]">
          <table className="term tabular text-xs sm:text-sm">
            <thead className="sticky top-0 bg-[var(--color-bg)] z-10">
              <tr>
                <th>gpu</th>
                <th>model</th>
                <th>harness</th>
                <SortableTh
                  label="speedup"
                  sortKey="speed"
                  sort={sort}
                  onSort={onSort}
                  title="decode speedup vs optimized-PyTorch baseline (per-ctx breakdown shown beneath)"
                />
                <SortableTh
                  label="tok/s"
                  sortKey="tok_s"
                  sort={sort}
                  onSort={onSort}
                  title="decode tokens per second"
                />
                <SortableTh
                  label="wall-clock"
                  sortKey="runtime"
                  sort={sort}
                  onSort={onSort}
                  title="agent wall-clock time"
                />
                <SortableTh
                  label="out tok"
                  sortKey="tokens"
                  sort={sort}
                  onSort={onSort}
                  title="agent output tokens"
                />
                <th>framework</th>
                <th>correct</th>
                <th>artifacts</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const isWinner = winners.has(r.run_id)
                const showGpuHeader = r.gpu !== lastGpu
                lastGpu = r.gpu
                return (
                  <tr key={r.run_id || i}>
                    <td className="text-[var(--color-accent)] whitespace-nowrap align-top">
                      {showGpuHeader ? (
                        <span className="font-semibold">{r.gpu}</span>
                      ) : (
                        <span className="text-[var(--color-fg-muted)] opacity-50">
                          {r.gpu}
                        </span>
                      )}
                    </td>
                    <td className="text-[var(--color-fg-bright)] whitespace-nowrap align-top">
                      {r.model}
                    </td>
                    <td className="text-[var(--color-fg-muted)] align-top">
                      {r.harness}
                    </td>
                    <td className="align-top">
                      <SpeedupCell
                        row={r}
                        maxScore={maxScore}
                        isWinner={isWinner}
                      />
                    </td>
                    <td className="text-[var(--color-fg-bright)] align-top whitespace-nowrap">
                      {r.tok_s != null ? r.tok_s.toLocaleString() : "-"}
                    </td>
                    <td className="text-[var(--color-fg-muted)] align-top whitespace-nowrap">
                      {fmtDuration(r.elapsed_s)}
                    </td>
                    <td className="text-[var(--color-fg-muted)] align-top whitespace-nowrap">
                      {fmtCompact(r.output_tokens)}
                    </td>
                    <td className="align-top">
                      {r.framework ? (
                        <span className="link-chip">{r.framework}</span>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td className="align-top">
                      {r.correct ? (
                        <span className="status-pill status-pill-good">
                          PASS
                        </span>
                      ) : (
                        <span className="status-pill status-pill-bad">
                          FAIL
                        </span>
                      )}
                    </td>
                    <td className="align-top">
                      <div className="chip-row">
                        {r.has_viewer ? (
                          <>
                            <a
                              className="link-chip"
                              href={`/runs/${r.run_id}.html`}
                              title="full optimization journey transcript"
                            >
                              transcript
                            </a>
                            <a
                              className="link-chip"
                              href={`/runs/${r.run_id}.html#tab-solution`}
                              title="final megakernel solution"
                            >
                              solution
                            </a>
                          </>
                        ) : (
                          <>
                            <span className="link-chip link-chip-muted">
                              transcript
                            </span>
                            <span className="link-chip link-chip-muted">
                              solution
                            </span>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-[var(--color-fg-muted)] mt-2">
          speedup = decode speedup over an optimized-PyTorch baseline (bar width
          normalized to the fastest run on the board); per-ctx breakdown shows
          speedup at 2k / 8k / 16k decode context. Top speedup per GPU is
          highlighted.
        </p>
      </section>

      <section className="text-sm text-[var(--color-fg)] border-t border-[var(--color-border)] pt-6">
        Source data:{" "}
        <Link href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega">
          github.com/Infatoshi/kernelbench.com
        </Link>
        {" · "}
        <Link href="/runs">runs</Link>
        {" · "}
        <Link href="/#cite">citation</Link>
      </section>
    </div>
  )
}

function SortableTh({
  label,
  sortKey,
  sort,
  onSort,
  title,
}: {
  label: string
  sortKey: SortKey
  sort: { key: SortKey; dir: "asc" | "desc" }
  onSort: (key: SortKey) => void
  title?: string
}) {
  const isActive = sort.key === sortKey
  return (
    <th
      className={isActive ? "sort-column-active sort-header-active" : undefined}
      aria-sort={
        isActive ? (sort.dir === "asc" ? "ascending" : "descending") : "none"
      }
      title={title}
    >
      <button
        type="button"
        className="sort-header-button"
        onClick={() => onSort(sortKey)}
      >
        <span>{label}</span>
        <span
          className={`sort-indicator sort-indicator-${isActive ? sort.dir : "idle"}`}
          aria-hidden="true"
        />
      </button>
    </th>
  )
}

function SpeedupCell({
  row,
  maxScore,
  isWinner,
}: {
  row: Row
  maxScore: number
  isWinner: boolean
}) {
  if (row.score == null) return <span className="cell-missing">-</span>
  const pct = Math.min(100, Math.max(2, (row.score / maxScore) * 100))
  const ctx = [
    { k: "2k", v: row.ctx2048 },
    { k: "8k", v: row.ctx8192 },
    { k: "16k", v: row.ctx16384 },
  ].filter((c) => c.v != null)
  return (
    <div className="speed-cell" style={{ minWidth: "7rem" }}>
      <div className="speed-readout">
        <span className={isWinner ? "cell-score cell-winner" : "cell-score"}>
          {row.score.toFixed(2)}x
        </span>
      </div>
      <div
        className="speed-bar"
        title={`decode speedup ${row.score.toFixed(3)}x`}
      >
        <div
          className={isWinner ? "speed-fill speed-fill-winner" : "speed-fill"}
          style={{ width: `${pct}%` }}
        />
      </div>
      {ctx.length > 0 && (
        <div
          className="text-[0.62rem] text-[var(--color-fg-muted)] mt-1 whitespace-nowrap"
          title="speedup at 2k / 8k / 16k decode context"
        >
          {ctx.map((c, idx) => (
            <span key={c.k}>
              {idx > 0 && " / "}
              {c.k} {c.v!.toFixed(1)}x
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function parseCsv(line: string): string[] {
  const out: string[] = []
  let cur = ""
  let q = false
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (c === '"') q = !q
    else if (c === "," && !q) {
      out.push(cur)
      cur = ""
    } else cur += c
  }
  out.push(cur)
  return out
}

function fmtDuration(seconds: number | null): string {
  if (seconds == null) return "-"
  if (seconds < 60) return `${Math.round(seconds)}s`
  const totalMin = Math.round(seconds / 60)
  const h = Math.floor(totalMin / 60)
  const m = totalMin % 60
  if (h === 0) return `${m}m`
  return m === 0 ? `${h}h` : `${h}h ${m}m`
}

function fmtCompact(value: number | null): string {
  if (value == null) return "-"
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (Math.abs(value) >= 1_000) return `${Math.round(value / 1_000)}k`
  return new Intl.NumberFormat("en-US").format(value)
}
