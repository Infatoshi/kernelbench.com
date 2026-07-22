"use client"

import Link from "next/link"
import { useEffect, useMemo, useState, type CSSProperties } from "react"
import type { ProblemChip, ReportRow, ReportView } from "../_lib/models"
import { problemLabel } from "../_lib/models"

// Homepage: stacked KernelBench sections. Each score is a mini bar vs the
// best result on that problem (100% = winner).
// Desktop: model × problem matrix.
// Mobile multi-problem: per-problem leaderboards (best → worst).
// Mobile single-problem: model cards.
// No-run cells are blank (not red). Judged fails keep a short color label.
// Every cell with a run is a link to its dedicated /runs/<gpu>/<rid> page.

export interface HomeDeck {
  key: string
  title: string
  accent: string
  byGpu: Record<string, ReportView> | null
  gpus: { key: string; label: string }[]
  defaultGpu: string
}

function bestByProblem(view: ReportView): Map<string, number> {
  const best = new Map<string, number>()
  for (const row of view.rows) {
    for (const c of row.chips) {
      if (c.kind !== "pass" || c.score == null) continue
      const prev = best.get(c.problem)
      if (prev == null || c.score > prev) best.set(c.problem, c.score)
    }
  }
  return best
}

function LabMark({ row }: { row: ReportRow }) {
  if (row.brand.logo) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={row.brand.logo}
        alt=""
        title={row.lab}
        className="hd-logo"
        width={15}
        height={15}
        loading="lazy"
      />
    )
  }
  return (
    <span className="hd-logo hd-logo-letter" title={row.lab}>
      {(row.lab || row.name).trim().charAt(0).toUpperCase()}
    </span>
  )
}

/** Visual tone for non-pass chips.
 *  blank = no run / no kernel (say nothing)
 *  miss  = red — wrong answer (we graded it and it failed)
 *  warn  = amber — ran but process failed (build / slow / cut / infra / OOM)
 *  hack  = amber+ — audit reject
 */
function failTone(chip: ProblemChip): "blank" | "miss" | "warn" | "hack" {
  if (chip.kind === "no_kernel" || chip.label === "empty") return "blank"
  if (chip.kind === "hack" || chip.label === "flag") return "hack"
  // wrong answers are the hard fail; everything else that actually ran is warn
  if (chip.label === "wrong") return "miss"
  return "warn"
}

/** Rank bucket for per-problem mobile boards (lower = higher on the list). */
function rankBucket(chip: ProblemChip): number {
  if (chip.kind === "pass" && chip.score != null) return 0
  const tone = failTone(chip)
  if (tone === "hack") return 1
  if (tone === "miss") return 2
  if (tone === "warn") return 3
  return 4 // blank / empty
}

const BAR_GROW_MS = 1000

/** Format a count-up value to match the chip's published label shape. */
function formatCountUp(t: number, finalLabel: string, score: number): string {
  if (t >= 1) return finalLabel
  const v = score * t
  // Speedup-style labels: "23", "5.1" (score usually > 1.5)
  if (score > 1.5) {
    if (score >= 10) return Math.round(v).toString()
    // one decimal like the static label
    return (Math.round(v * 10) / 10).toFixed(1)
  }
  // Peak-fraction labels as percent points of score (label is e.g. "30" for 0.30)
  const pctPts = score * 100 * t
  if (score >= 0.1) return Math.round(pctPts).toString()
  return (Math.round(pctPts * 10) / 10).toFixed(1)
}

function Chip({
  chip,
  best,
  animDelayMs = 0,
  /** Side label left of the bar (mobile cards). null = no side label. */
  sideLabel = null,
}: {
  chip: ProblemChip
  best: number | undefined
  /** Stagger entrance slightly down the grid */
  animDelayMs?: number
  sideLabel?: string | null
}) {
  const pass = chip.kind === "pass" && chip.score != null && best != null && best > 0
  const pct = pass ? Math.min(100, (chip.score! / best) * 100) : 0
  const fillW = pass ? Math.max(pct, 4) : 0
  const win = pass && chip.score! >= best! - 1e-12
  const blank = !pass && failTone(chip) === "blank"

  // pre → grow (line + count-up) → static
  const [phase, setPhase] = useState<"pre" | "grow" | "static">("pre")
  const [numLabel, setNumLabel] = useState(pass ? "0" : chip.label)

  useEffect(() => {
    if (!pass || chip.score == null) {
      setPhase("static")
      setNumLabel(chip.label)
      return
    }
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    if (reduced) {
      setPhase("static")
      setNumLabel(chip.label)
      return
    }

    setPhase("pre")
    setNumLabel("0")
    let raf = 0
    let growTimer = 0
    let staticTimer = 0
    let growStarted = 0
    const score = chip.score
    const label = chip.label

    const tick = (now: number) => {
      const t = Math.min(1, (now - growStarted) / BAR_GROW_MS)
      const e = 1 - (1 - t) ** 3 // ease-out cubic
      setNumLabel(formatCountUp(e, label, score))
      if (t < 1) raf = requestAnimationFrame(tick)
      else setNumLabel(label)
    }

    growTimer = window.setTimeout(() => {
      setPhase("grow")
      growStarted = performance.now()
      raf = requestAnimationFrame(tick)
    }, animDelayMs)

    staticTimer = window.setTimeout(() => {
      setPhase("static")
      setNumLabel(label)
    }, animDelayMs + BAR_GROW_MS)

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(growTimer)
      clearTimeout(staticTimer)
    }
  }, [pass, chip.score, chip.label, chip.problem, best, animDelayMs])

  const tone = pass
    ? win
      ? "hd-bar-win"
      : "hd-bar-pass"
    : `hd-bar-${failTone(chip)}`

  const fillStyle = {
    ["--hd-w" as string]: `${fillW.toFixed(1)}%`,
  } as CSSProperties

  const barBody = pass ? (
    <>
      <span
        className={`hd-bar-fill${phase === "pre" ? "" : " hd-bar-fill-on"}`}
        style={fillStyle}
      />
      <span className="hd-bar-num tabular">
        {phase === "static" ? chip.label : numLabel}
      </span>
      {win && phase !== "pre" && (
        <span
          className={`hd-bar-star${phase === "static" ? " hd-bar-star-in" : ""}`}
          aria-label="best"
        >
          ★
        </span>
      )}
    </>
  ) : blank ? null : (
    <span className="hd-bar-fail-label" aria-label={chip.title || chip.label || "fail"}>
      {chip.label || "fail"}
    </span>
  )

  const barClass = `hd-bar ${tone}${chip.page_url ? " hd-bar-hot" : ""}`
  return (
    <span className={`hd-cell${sideLabel == null ? " hd-cell-bare" : ""}`}>
      {sideLabel != null && (
        <span className="hd-cell-label" title={chip.problem}>
          {sideLabel}
        </span>
      )}
      {chip.page_url ? (
        <Link
          href={chip.page_url}
          className={`${barClass} no-underline`}
          title={`${problemLabel(chip.problem)}: ${chip.title || chip.label} — open run page`}
          aria-label={`${problemLabel(chip.problem)}: ${chip.title || chip.label}. Open run page`}
        >
          {barBody}
        </Link>
      ) : (
        <span
          className={barClass}
          title={blank ? chip.title || "no run" : chip.title || undefined}
          aria-label={
            blank ? chip.title || "no run" : chip.title || chip.label || undefined
          }
        >
          {barBody}
        </span>
      )}
    </span>
  )
}

function gpuTabLabel(key: string, full: string) {
  if (key === "rtxpro6000") {
    return (
      <>
        <span className="gpu-label-full">{full}</span>
        <span className="gpu-label-short">PRO 6000</span>
      </>
    )
  }
  return full
}

type RankEntry = { row: ReportRow; chip: ProblemChip }

function rankEntriesForProblem(
  view: ReportView,
  problemId: string,
): RankEntry[] {
  const entries: RankEntry[] = []
  for (const row of view.rows) {
    const chip = row.chips.find((c) => c.problem === problemId)
    if (!chip) continue
    entries.push({ row, chip })
  }
  entries.sort((a, b) => {
    const ba = rankBucket(a.chip)
    const bb = rankBucket(b.chip)
    if (ba !== bb) return ba - bb
    if (ba === 0) {
      // Higher score first among passes.
      const sa = a.chip.score ?? 0
      const sb = b.chip.score ?? 0
      if (sb !== sa) return sb - sa
    }
    return a.row.name.localeCompare(b.row.name)
  })
  return entries
}

function DeckPanel({ deck }: { deck: HomeDeck }) {
  const [gpu, setGpu] = useState(deck.defaultGpu)
  const active = deck.byGpu && deck.byGpu[gpu] ? gpu : deck.defaultGpu
  const view = deck.byGpu?.[active] ?? null
  const best = useMemo(() => (view ? bestByProblem(view) : new Map<string, number>()), [view])

  if (!deck.byGpu) {
    return (
      <section
        className="hd-deck hd-deck-soon"
        id={deck.key}
        style={{ ["--deck-accent" as string]: deck.accent }}
      >
        <div className="hd-section-label">
          <span className="hd-k">KernelBench</span>
          <span className="hd-name">{deck.title}</span>
          <span className="hd-soon">soon</span>
        </div>
      </section>
    )
  }
  if (!view) return null
  const nCols = Math.max(view.problems.length, 1)
  const multiProblem = nCols > 1
  let lastTier: string | null = null

  const selectGpu = (key: string) => {
    setGpu(key)
  }

  return (
    <section
      className={`hd-deck${multiProblem ? " hd-deck-multi" : " hd-deck-single"}`}
      id={deck.key}
      style={
        {
          ["--deck-accent" as string]: deck.accent,
          ["--hd-cols" as string]: String(nCols),
        } as CSSProperties
      }
    >
      <div className="hd-section-label">
        <div className="hd-section-title">
          <span className="hd-k">KernelBench</span>
          <span className="hd-name">{deck.title}</span>
        </div>
        {deck.gpus.length > 1 && (
          <div className="gpu-toggle hd-gpu-toggle" role="tablist" aria-label={`${deck.title} GPU`}>
            {deck.gpus.map((g) => (
              <button
                key={g.key}
                type="button"
                className={`gpu-toggle-btn${g.key === active ? " active" : ""}`}
                onClick={() => selectGpu(g.key)}
                role="tab"
                aria-selected={g.key === active}
              >
                {gpuTabLabel(g.key, g.label)}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Desktop matrix (and mobile single-problem model cards) */}
      <div className="hd-matrix">
        <div className="hd-grid-head">
          <span className="hd-head-gutter" aria-hidden />
          <div className="hd-cells">
            {view.problems.map((p) => (
              <span key={p.id} className="hd-col-label" title={p.id}>
                {p.short}
              </span>
            ))}
          </div>
          <span className="hd-head-score" aria-hidden />
        </div>

        {view.rows.map((row, rowIdx) => {
          const tier = `${row.passed}/${row.total}`
          const showTier = tier !== lastTier
          lastTier = tier
          return (
            <div key={row.slug} className="hd-row-block">
              {showTier && <div className="hd-tier">{tier}</div>}
              <div className="hd-row">
                <span className="hd-model">
                  <LabMark row={row} />
                  <span className="hd-model-name">{row.name}</span>
                </span>
                <div className="hd-cells">
                  {row.chips.map((c, i) => {
                    const animDelayMs = Math.min(i * 40 + rowIdx * 28, 420)
                    return (
                      <Chip
                        key={c.problem}
                        chip={c}
                        best={best.get(c.problem)}
                        animDelayMs={animDelayMs}
                        sideLabel={c.short}
                      />
                    )
                  })}
                </div>
                <span className="hd-pass tabular">
                  {row.passed}
                  <span className="hd-pass-den">/{row.total}</span>
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Mobile multi-problem: per-problem leaderboards, best → worst */}
      {multiProblem && (
        <div className="hd-by-problem">
          {view.problems.map((p, pIdx) => {
            const ranked = rankEntriesForProblem(view, p.id)
            const problemBest = best.get(p.id)
            return (
              <div key={p.id} className="hd-problem-board">
                <div className="hd-problem-head">
                  <span className="hd-problem-name" title={p.id}>
                    {p.short}
                  </span>
                  <span className="hd-problem-meta">best → worst</span>
                </div>
                <div className="hd-rank-list">
                  {ranked.map(({ row, chip }, i) => {
                    const animDelayMs = Math.min(pIdx * 80 + i * 28, 520)
                    return (
                      <div key={`${row.slug}::${chip.problem}`} className="hd-rank-row">
                        <span className="hd-rank-model">
                          <LabMark row={row} />
                          <span className="hd-model-name">{row.name}</span>
                        </span>
                        <Chip
                          chip={chip}
                          best={problemBest}
                          animDelayMs={animDelayMs}
                          sideLabel={null}
                        />
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

export function HomeDecks({ decks }: { decks: HomeDeck[] }) {
  return (
    <div className="hd-stack">
      {decks.map((d) => (
        <DeckPanel key={d.key} deck={d} />
      ))}
    </div>
  )
}
