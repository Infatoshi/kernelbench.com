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

export function LeaderboardTable({ rows }: { rows: HardRunRecord[] }) {
  const [query, setQuery] = useState("")
  const [filters, setFilters] = useState<FilterState>({
    problem: "all",
    harness: "all",
    outcome: "all",
  })

  const problems = useMemo(() => unique(rows.map((r) => r.problem)), [rows])
  const harnesses = useMemo(() => unique(rows.map((r) => r.harness)), [rows])
  const normalizedQuery = query.trim().toLowerCase()
  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (filters.problem !== "all" && row.problem !== filters.problem) return false
      if (filters.harness !== "all" && row.harness !== filters.harness) return false
      if (filters.outcome !== "all" && outcomeBucket(row) !== filters.outcome) {
        return false
      }
      return fuzzyMatch(normalizedQuery, row.searchText)
    })
  }, [filters, normalizedQuery, rows])

  const reset = () => {
    setQuery("")
    setFilters({ problem: "all", harness: "all", outcome: "all" })
  }

  return (
    <div className="leaderboard-panel">
      <div className="leaderboard-controls" aria-label="leaderboard filters">
        <label className="finder-field">
          <span>find</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="fuzzy find runs, models, problems..."
            type="search"
          />
        </label>
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
              <th>model</th>
              <th>harness</th>
              <th>problem</th>
              <th>date</th>
              <th>compiled</th>
              <th>correct</th>
              <th>speed of light</th>
              <th>tokens</th>
              <th>runtime</th>
              <th>files</th>
              <th>conversation</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <tr key={row.key}>
                <td className="leaderboard-model">{row.model}</td>
                <td className="leaderboard-harness">{row.harness}</td>
                <td className="leaderboard-problem">{row.problem}</td>
                <td>
                  {row.date ? (
                    <div className="stacked-cell">
                      <span>{row.date}</span>
                      <span>{row.time}</span>
                    </div>
                  ) : (
                    <span className="cell-missing">-</span>
                  )}
                </td>
                <td>
                  <StatusPill status={row.compiled} />
                </td>
                <td>
                  <StatusPill status={row.correct} />
                </td>
                <td>
                  <SpeedCell row={row} />
                </td>
                <td>
                  <TokenCell row={row} />
                </td>
                <td>
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

function StatusPill({ status }: { status: HardRunStatus }) {
  return (
    <>
      <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
      {status.annotationSeverity ? (
        <span
          className={`annotation-badge annotation-badge-${status.annotationSeverity}`}
          title={status.annotationLabel}
          aria-label={status.annotationLabel}
        >
          !
        </span>
      ) : null}
    </>
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

function fuzzyMatch(query: string, value: string) {
  if (!query) return true
  const haystack = value.toLowerCase()
  if (haystack.includes(query)) return true
  const parts = query.split(/\s+/).filter(Boolean)
  return parts.every((part) => fuzzyPart(part, haystack))
}

function fuzzyPart(query: string, haystack: string) {
  let cursor = 0
  for (const char of query) {
    cursor = haystack.indexOf(char, cursor)
    if (cursor === -1) return false
    cursor += 1
  }
  return true
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
