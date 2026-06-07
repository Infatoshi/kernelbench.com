"use client"

import { useMemo, useState } from "react"

export type HardRunStatus = {
  label: string
  tone: "good" | "bad" | "warn" | "muted"
  annotationSeverity?: "bad" | "warn"
  annotationLabel?: string
}

export type HardRunRecord = {
  key: string
  runId: string | null
  model: string
  harness: string
  problem: string
  problemKey: string
  date: string | null
  time: string | null
  compiled: HardRunStatus
  correct: HardRunStatus
  rewardHack: boolean
  explanation: string | null
  peakFraction: number | null
  speedPct: number | null
  isWinner: boolean
  outputTokens: number | null
  reasoningTokens: number | null
  cacheTokens: number | null
  inputTokens: number | null
  costUsd: number | null
  outputTokensPerSecond: number | null
  elapsedSeconds: number | null
  checkSeconds: number | null
  benchmarkSeconds: number | null
  totalSeconds: number | null
  gpuWaitSeconds: number | null
  gpuActiveSeconds: number | null
  referenceUrl: string
  solutionUrl: string | null
  transcriptUrl: string | null
  scored: string
  note: string
  title: string
  tokenTitle: string
  runtimeTitle: string
  searchText: string
}

type FilterState = {
  problem: string
  harness: string
  outcome: string
}

type SortKey =
  | "model"
  | "harness"
  | "problem"
  | "date"
  | "compiled"
  | "correct"
  | "rewardHack"
  | "explanation"
  | "speed"
  | "tokens"
  | "runtime"

type SortState = {
  key: SortKey
  dir: "asc" | "desc"
}

export function LeaderboardTable({ rows }: { rows: HardRunRecord[] }) {
  const [filters, setFilters] = useState<FilterState>({
    problem: "all",
    harness: "all",
    outcome: "all",
  })
  const [sort, setSort] = useState<SortState>({ key: "problem", dir: "asc" })

  const problems = useMemo(() => unique(rows.map((r) => r.problem)), [rows])
  const harnesses = useMemo(() => unique(rows.map((r) => r.harness)), [rows])
  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (filters.problem !== "all" && row.problem !== filters.problem) return false
      if (filters.harness !== "all" && row.harness !== filters.harness) return false
      if (filters.outcome !== "all" && outcomeBucket(row) !== filters.outcome) {
        return false
      }
      return true
    }).sort((a, b) => compareRunRows(a, b, sort))
  }, [filters, rows, sort])

  const reset = () => {
    setFilters({ problem: "all", harness: "all", outcome: "all" })
    setSort({ key: "problem", dir: "asc" })
  }

  const setSortKey = (key: SortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "asc" },
    )
  }

  return (
    <div className="leaderboard-panel">
      <div className="leaderboard-controls" aria-label="leaderboard filters">
        <FilterSelect
          label="problem"
          value={filters.problem}
          onChange={(problem) => setFilters((f) => ({ ...f, problem }))}
          options={["all", ...problems]}
        />
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
          options={["all", "pass", "fail", "infra", "no run"]}
        />
        <button type="button" className="filter-reset" onClick={reset}>
          reset
        </button>
      </div>
      <div className="leaderboard-count">
        showing {filteredRows.length} of {rows.length} rows
      </div>
      <div className="leaderboard-table-wrap">
        <table className="term leaderboard-runs tabular text-xs">
          <thead>
            <tr>
              <SortableTh label="model" sortKey="model" active={sort} onSort={setSortKey} />
              <SortableTh label="harness" sortKey="harness" active={sort} onSort={setSortKey} />
              <SortableTh label="problem" sortKey="problem" active={sort} onSort={setSortKey} />
              <SortableTh label="date" sortKey="date" active={sort} onSort={setSortKey} />
              <SortableTh label="compiled" sortKey="compiled" active={sort} onSort={setSortKey} />
              <SortableTh label="correct" sortKey="correct" active={sort} onSort={setSortKey} />
              <SortableTh label="reward hacking" sortKey="rewardHack" active={sort} onSort={setSortKey} />
              <SortableTh label="explanation" sortKey="explanation" active={sort} onSort={setSortKey} />
              <SortableTh label="speed of light" sortKey="speed" active={sort} onSort={setSortKey} />
              <SortableTh label="tokens" sortKey="tokens" active={sort} onSort={setSortKey} />
              <SortableTh label="runtime" sortKey="runtime" active={sort} onSort={setSortKey} />
              <th>files</th>
              <th>conversation</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <tr key={row.key}>
                <td className={sortCellClass("model", sort, "leaderboard-model")}>{row.model}</td>
                <td className={sortCellClass("harness", sort, "leaderboard-harness")}>{row.harness}</td>
                <td className={sortCellClass("problem", sort, "leaderboard-problem")}>{row.problem}</td>
                <td className={sortCellClass("date", sort)}>
                  {row.date ? (
                    <div className="stacked-cell">
                      <span>{row.date}</span>
                      <span>{row.time}</span>
                    </div>
                  ) : (
                    <span className="cell-missing">-</span>
                  )}
                </td>
                <td className={sortCellClass("compiled", sort)}>
                  <StatusPill status={row.compiled} />
                </td>
                <td className={sortCellClass("correct", sort)}>
                  <StatusPill status={row.correct} />
                </td>
                <td className={sortCellClass("rewardHack", sort)}>
                  <span
                    className={
                      row.rewardHack
                        ? "status-pill status-pill-bad"
                        : "status-pill status-pill-muted"
                    }
                  >
                    {row.rewardHack ? "yes" : "no"}
                  </span>
                </td>
                <td className={sortCellClass("explanation", sort)}>
                  {row.explanation ? (
                    <div className="leaderboard-explanation" title={row.explanation}>
                      {row.explanation}
                    </div>
                  ) : (
                    <span className="cell-missing">-</span>
                  )}
                </td>
                <td className={sortCellClass("speed", sort)}>
                  <SpeedCell row={row} />
                </td>
                <td className={sortCellClass("tokens", sort)}>
                  <TokenCell row={row} />
                </td>
                <td className={sortCellClass("runtime", sort)}>
                  <RuntimeCell row={row} />
                </td>
                <td>
                  <div className="chip-row">
                    <a className="link-chip" href={row.referenceUrl}>
                      reference
                    </a>
                    {row.solutionUrl ? (
                      <a className="link-chip" href={row.solutionUrl} title={row.title}>
                        solution
                      </a>
                    ) : (
                      <span className="link-chip link-chip-muted">solution</span>
                    )}
                  </div>
                </td>
                <td>
                  <div className="conversation-cell">
                    <div className="chip-row">
                      {row.transcriptUrl ? (
                        <a className="link-chip" href={row.transcriptUrl} title={row.title}>
                          transcript
                        </a>
                      ) : (
                        <span className="link-chip link-chip-muted">transcript</span>
                      )}
                      <span className="link-chip link-chip-muted">{row.scored}</span>
                    </div>
                    <div className="conversation-note" title={row.note}>
                      {row.note}
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
  active,
  onSort,
}: {
  label: string
  sortKey: SortKey
  active: SortState
  onSort: (key: SortKey) => void
}) {
  const isActive = active.key === sortKey
  return (
    <th
      className={isActive ? "sort-column-active sort-header-active" : undefined}
      aria-sort={isActive ? (active.dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className="sort-header-button"
        onClick={() => onSort(sortKey)}
      >
        <span>{label}</span>
        <span className="sort-indicator" aria-hidden="true">
          {isActive ? (active.dir === "asc" ? "↑" : "↓") : "↕"}
        </span>
      </button>
    </th>
  )
}

function sortCellClass(key: SortKey, active: SortState, extra = "") {
  return [extra, active.key === key ? "sort-column-active" : null]
    .filter(Boolean)
    .join(" ")
}

function StatusPill({ status }: { status: HardRunStatus }) {
  return (
    <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
  )
}

function SpeedCell({ row }: { row: HardRunRecord }) {
  if (row.speedPct == null || row.peakFraction == null) {
    return <span className="cell-missing">-</span>
  }
  return (
    <div className="speed-cell">
      <div className="speed-readout">
        <span className={row.isWinner ? "cell-score cell-winner" : "cell-score"}>
          {row.speedPct.toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

function TokenCell({ row }: { row: HardRunRecord }) {
  if (
    row.outputTokens == null &&
    row.reasoningTokens == null &&
    row.cacheTokens == null
  ) {
    return <span className="cell-missing">-</span>
  }
  return (
    <div className="stacked-cell" title={row.tokenTitle}>
      <span>out {fmtCompact(row.outputTokens)}</span>
      <span>think {fmtCompact(row.reasoningTokens)}</span>
      <span>cache {fmtCompact(row.cacheTokens)}</span>
    </div>
  )
}

function RuntimeCell({ row }: { row: HardRunRecord }) {
  if (
    row.elapsedSeconds == null &&
    row.checkSeconds == null &&
    row.benchmarkSeconds == null
  ) {
    return <span className="cell-missing">-</span>
  }
  return (
    <div className="stacked-cell" title={row.runtimeTitle}>
      <span>agent {fmtDurationMaybe(row.elapsedSeconds)}</span>
      <span>check {fmtDurationMaybe(row.checkSeconds)}</span>
      <span>bench {fmtDurationMaybe(row.benchmarkSeconds)}</span>
    </div>
  )
}

function outcomeBucket(row: HardRunRecord) {
  if (row.correct.label === "pass") return "pass"
  if (row.correct.label === "no run") return "no run"
  if (["rate", "early", "time", "err"].includes(row.correct.label)) return "infra"
  return "fail"
}

function compareRunRows(a: HardRunRecord, b: HardRunRecord, sort: SortState) {
  const primary = compareSortValue(sortValue(a, sort.key), sortValue(b, sort.key))
  if (primary !== 0) return sort.dir === "asc" ? primary : -primary
  const problemDiff = a.problemKey.localeCompare(b.problemKey)
  if (problemDiff !== 0) return problemDiff
  const modelDiff = a.model.localeCompare(b.model)
  if (modelDiff !== 0) return modelDiff
  return a.harness.localeCompare(b.harness)
}

function sortValue(row: HardRunRecord, key: SortKey) {
  switch (key) {
    case "model":
      return row.model
    case "harness":
      return row.harness
    case "problem":
      return row.problemKey
    case "date":
      return row.date && row.time ? `${row.date} ${row.time}` : null
    case "compiled":
      return row.compiled.label
    case "correct":
      return row.correct.label
    case "rewardHack":
      return row.rewardHack ? 1 : 0
    case "explanation":
      return row.explanation
    case "speed":
      return row.speedPct
    case "tokens":
      return (row.outputTokens ?? 0) + (row.reasoningTokens ?? 0)
    case "runtime":
      return row.elapsedSeconds
  }
}

function compareSortValue(a: string | number | null, b: string | number | null) {
  if (a == null && b == null) return 0
  if (a == null) return 1
  if (b == null) return -1
  if (typeof a === "number" && typeof b === "number") return a - b
  return String(a).localeCompare(String(b))
}

function unique(values: string[]) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b))
}

function fmtCompact(value: number | null | undefined) {
  if (value == null) return "-"
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return new Intl.NumberFormat("en-US").format(value)
}

function fmtDurationMaybe(value: number | null | undefined) {
  if (value == null) return "-"
  if (value < 60) return `${value}s`
  const min = Math.floor(value / 60)
  const sec = value % 60
  return `${min}m ${sec}s`
}
