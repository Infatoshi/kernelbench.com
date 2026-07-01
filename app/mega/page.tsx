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
  kernels: number | null
  megakernel: string
  megakernel_judged: boolean
  has_viewer: boolean
  run_id: string
}

const GPU_ORDER = ["RTX PRO 6000 Blackwell", "H100", "B200"]

const REFERENCE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.reference.py.txt",
)}`
const BASELINE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.baseline.py.txt",
)}`

type SortKey = "speed" | "tok_s" | "runtime" | "tokens" | null

type FilterState = {
  harness: string
  outcome: string
}

export default function MegaPage() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState<FilterState>({
    harness: "all",
    outcome: "all",
  })
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
          "kernels",
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
            } else if (h === "has_viewer" || h === "megakernel_judged") {
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

  const harnesses = useMemo(
    () => unique(data.map((r) => r.harness)),
    [data],
  )

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

  // Filter, then group by GPU (fixed order), default sorted by speedup desc
  // within a group. A clicked column header overrides the within-group sort key
  // while keeping the GPU grouping stable.
  const sorted = useMemo(() => {
    const filtered = data.filter((r) => {
      if (filters.harness !== "all" && r.harness !== filters.harness) {
        return false
      }
      if (filters.outcome !== "all") {
        const bucket = r.correct ? "pass" : "fail"
        if (bucket !== filters.outcome) return false
      }
      return true
    })
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
      return sort.dir === "asc" ? av - bv : bv - av
    }
    return [...filtered].sort(
      (a, b) =>
        gpuRank(a.gpu) - gpuRank(b.gpu) ||
        a.gpu.localeCompare(b.gpu) ||
        cmp(a, b),
    )
  }, [data, filters, sort])

  const onSort = (key: SortKey) => {
    setSort((cur) =>
      cur.key === key
        ? { key, dir: cur.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" },
    )
  }

  const reset = () => {
    setFilters({ harness: "all", outcome: "all" })
    setSort({ key: null, dir: "desc" })
  }

  // Track GPU group boundaries so we can render a subtle group label.
  let lastGpu: string | null = null

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          mega
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          KernelBench-Mega · whole-block megakernels
          <span className="ml-2 text-xs font-semibold text-[var(--color-accent)]">
            ● 3-hour ceiling · RTX PRO 6000 Blackwell + H100 + B200
          </span>
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-2 max-w-4xl leading-relaxed">
          KernelBench-Mega tests whole-block megakernels: instead of grading a
          single isolated op, the agent fuses an entire model block into one
          kernel. Problem <code>02_kimi_linear_decode</code> is a Kimi-Linear
          W4A16 hybrid decode (4-bit weights, bf16 activations). The headline
          metric is the{" "}
          <span className="text-[var(--color-fg)]">
            decode speedup over an optimized-PyTorch baseline
          </span>{" "}
          (e.g. <code>19.35x</code> = 19x faster than the reference), not a 0-1
          roofline fraction; <code>tok/s</code> is decode tokens per second.
          Higher is better for both, and results are reported per GPU. The
          transcript is the headline artifact: it shows the model&apos;s full
          optimization journey from baseline to the final megakernel.
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-6 max-w-4xl leading-relaxed">
          Each run gets a single autonomous session under a 3-hour wall-clock
          ceiling; models self-terminate well under it (the longest run so far
          is ~2.5h). All cells use the same ceiling, so the board is comparable.
          An empty speedup is a 3-hour-timeout DNF.
        </p>
      </section>

      <section>
        <div className="leaderboard-panel">
          <div
            className="leaderboard-controls"
            aria-label="leaderboard filters"
          >
            <FilterSelect
              label="harness"
              value={filters.harness}
              onChange={(harness) => setFilters((f) => ({ ...f, harness }))}
              options={["all", ...harnesses]}
            />
            <FilterSelect
              label="outcome"
              value={filters.outcome}
              onChange={(outcome) => setFilters((f) => ({ ...f, outcome }))}
              options={["all", "pass", "fail"]}
            />
            <button type="button" className="filter-reset" onClick={reset}>
              reset
            </button>
          </div>
          <div className="leaderboard-count">
            {loading
              ? "loading…"
              : `showing ${sorted.length} of ${data.length} rows · grouped by GPU, sorted by decode speedup vs optimized-PyTorch baseline`}
          </div>
          <div className="leaderboard-table-wrap">
            <table className="term leaderboard-runs tabular text-xs">
              <thead>
                <tr>
                  <th>gpu</th>
                  <th>model</th>
                  <th>harness</th>
                  <th>correct</th>
                  <SortableTh
                    label="speedup"
                    sortKey="speed"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableTh
                    label="tok/s"
                    sortKey="tok_s"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableTh
                    label="runtime"
                    sortKey="runtime"
                    sort={sort}
                    onSort={onSort}
                  />
                  <th>framework</th>
                  <th title="custom kernels in the timed path (proxy for launches/step). Green = genuine fused megakernel within the launch budget; red = hides launches (CUDA graph / torch.compile) or is unfused / eager. Hollow dot = provisional (not yet judge-verified).">
                    megakernel
                  </th>
                  <th>files</th>
                  <th>conversation</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r, i) => {
                  const isWinner = winners.has(r.run_id)
                  const showGpuHeader = r.gpu !== lastGpu
                  lastGpu = r.gpu
                  return (
                    <tr key={r.run_id || i}>
                      <td className="leaderboard-model whitespace-nowrap">
                        {showGpuHeader ? (
                          <span className="text-[var(--color-accent)] font-semibold">
                            {r.gpu}
                          </span>
                        ) : (
                          <span className="text-[var(--color-fg-muted)] opacity-50">
                            {r.gpu}
                          </span>
                        )}
                      </td>
                      <td className="leaderboard-model">{r.model}</td>
                      <td className="leaderboard-harness">{r.harness}</td>
                      <td>
                        {r.correct ? (
                          <span className="status-pill status-pill-good">
                            pass
                          </span>
                        ) : (
                          <span className="status-pill status-pill-bad">
                            fail
                          </span>
                        )}
                      </td>
                      <td className={sortCellClass("speed", sort)}>
                        <SpeedupCell
                          row={r}
                          maxScore={maxScore}
                          isWinner={isWinner}
                        />
                      </td>
                      <td className={sortCellClass("tok_s", sort)}>
                        {r.tok_s != null ? (
                          <span className="cell-score">
                            {r.tok_s.toLocaleString()}
                          </span>
                        ) : (
                          <span className="cell-missing">-</span>
                        )}
                      </td>
                      <td className={sortCellClass("runtime", sort)}>
                        <RuntimeCell row={r} />
                      </td>
                      <td>
                        {r.framework ? (
                          <span className="link-chip link-chip-muted">
                            {r.framework}
                          </span>
                        ) : (
                          <span className="cell-missing">-</span>
                        )}
                      </td>
                      <td>
                        <MegakernelCell row={r} />
                      </td>
                      <td>
                        <div className="chip-row">
                          <a className="link-chip" href={REFERENCE_URL}>
                            reference
                          </a>
                          <a className="link-chip" href={BASELINE_URL}>
                            baseline
                          </a>
                          {r.has_viewer ? (
                            <a
                              className="link-chip"
                              href={`/runs/${r.run_id}_solution.py.txt`}
                              title="final megakernel solution"
                            >
                              solution
                            </a>
                          ) : (
                            <span className="link-chip link-chip-muted">
                              solution
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <div className="conversation-cell">
                          <div className="chip-row">
                            <a
                              className="link-chip"
                              href={`https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces/blob/main/${r.run_id}.jsonl`}
                              target="_blank"
                              rel="noopener"
                              title="full agent trace on HuggingFace"
                            >
                              trace ↗
                            </a>
                          </div>
                          <div className="conversation-note">
                            {r.correct
                              ? `${fmtCompact(r.output_tokens)} out tok`
                              : "3h timeout"}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          speedup = decode speedup over an optimized-PyTorch baseline (bar width
          normalized to the fastest run on the board); the per-ctx breakdown
          shows speedup at 2k / 8k / 16k decode context. Top speedup per GPU is
          highlighted. The <span className="text-[var(--color-fg)]">megakernel</span>{" "}
          column is the count of custom kernels in the timed path (the
          launches-per-step proxy) with a{" "}
          <span className="text-[var(--color-accent)]">green</span> /{" "}
          <span className="text-[var(--color-bad)]">red</span> marker for whether
          it is a genuine single fused megakernel or hides launches (CUDA graph /
          torch.compile) / stays unfused; a hollow dot with a trailing{" "}
          <code>?</code> means the verdict is provisional (not yet judge-verified).
          Browse the{" "}
          <Link href="/runs" className="underline underline-offset-2">
            run index
          </Link>{" "}
          for transcripts and solutions, or the{" "}
          <Link
            href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega"
            className="underline underline-offset-2"
          >
            mega benchmark source
          </Link>
          .
        </p>
      </section>
    </div>
  )
}

function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (value: string) => void
}) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

function SortableTh({
  label,
  sortKey,
  sort,
  onSort,
}: {
  label: string
  sortKey: SortKey
  sort: { key: SortKey; dir: "asc" | "desc" }
  onSort: (key: SortKey) => void
}) {
  const isActive = sort.key === sortKey
  return (
    <th
      className={isActive ? "sort-column-active sort-header-active" : undefined}
      aria-sort={
        isActive ? (sort.dir === "asc" ? "ascending" : "descending") : "none"
      }
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

function sortCellClass(key: SortKey, sort: { key: SortKey }) {
  return sort.key === key ? "sort-column-active" : undefined
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
          className="stacked-cell mt-1"
          title="speedup at 2k / 8k / 16k decode context"
        >
          {ctx.map((c) => (
            <span key={c.k}>
              {c.k} {c.v!.toFixed(1)}x
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function MegakernelCell({ row }: { row: Row }) {
  if (!row.megakernel || row.megakernel === "unknown") {
    return <span className="cell-missing">-</span>
  }
  const pass = row.megakernel === "pass"
  const judged = row.megakernel_judged
  const color = pass ? "var(--color-accent)" : "var(--color-bad)"
  const num = row.kernels != null ? row.kernels : "?"
  const title =
    `${pass ? "genuine fused megakernel" : "not a single fused megakernel"} · ` +
    `${num} custom kernel(s) in timed path · ` +
    `${judged ? "judge-verified" : "provisional (not yet judge-verified)"}`
  return (
    <span
      title={title}
      style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: judged ? color : "transparent",
          border: `1.5px solid ${color}`,
        }}
      />
      <span style={{ color: pass ? "var(--color-fg)" : "var(--color-fg-muted)" }}>
        {num}
        {judged ? "" : "?"}
      </span>
    </span>
  )
}

function RuntimeCell({ row }: { row: Row }) {
  if (row.elapsed_s == null && row.output_tokens == null) {
    return <span className="cell-missing">-</span>
  }
  return (
    <div className="stacked-cell">
      <span>wall {fmtDuration(row.elapsed_s)}</span>
      <span>out {fmtCompact(row.output_tokens)}</span>
    </div>
  )
}

function unique(values: string[]) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b))
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
