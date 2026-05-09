import { readdir } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard } from "@/lib/data"

type RunRow = {
  run_id: string
  problem: string
  harness: string
  model: string
  effort: string
  correct: boolean
  has_solution: boolean
  peak_fraction: number | null
  elapsed_seconds: number | null
}

type RunsIndexProps = {
  searchParams?: Promise<{
    harness?: string | string[]
  }>
}

async function loadAllRuns(): Promise<RunRow[]> {
  const lb = await loadLeaderboard()
  const viewerEntries = new Set<string>()
  try {
    const entries = await readdir(join(process.cwd(), "public/runs"))
    for (const n of entries) {
      if (n.endsWith(".html")) viewerEntries.add(n.slice(0, -5))
    }
  } catch {
    // no public/runs dir, no viewers — page still works, just empty
  }

  const out: RunRow[] = []
  for (const m of lb.models) {
    for (const [problem, cell] of Object.entries(m.results)) {
      if (!viewerEntries.has(cell.run_id)) continue
      out.push({
        run_id: cell.run_id,
        problem,
        harness: m.harness,
        model: m.model,
        effort: m.effort,
        correct: cell.correct,
        has_solution: cell.has_solution,
        peak_fraction: cell.peak_fraction,
        elapsed_seconds: cell.elapsed_seconds ?? null,
      })
    }
  }
  // Sort by peak_fraction desc, with PASS-without-peak before FAIL/ERR
  out.sort((a, b) => {
    const ap = a.correct ? a.peak_fraction ?? -0.5 : a.has_solution ? -1 : -2
    const bp = b.correct ? b.peak_fraction ?? -0.5 : b.has_solution ? -1 : -2
    return bp - ap
  })
  return out
}

function shortModel(harness: string, model: string, effort: string) {
  let m = model.replace("openrouter-pinned/", "or/")
  if (effort) m += ` [${effort}]`
  return m
}

function statusCell(r: RunRow) {
  if (r.correct) {
    return r.peak_fraction !== null ? (
      <span className="text-[var(--color-fg-bright)] tabular">
        {r.peak_fraction.toFixed(3)}
      </span>
    ) : (
      <span className="text-[var(--color-fg-bright)]">PASS</span>
    )
  }
  if (r.has_solution)
    return <span className="text-[var(--color-fg-dim)]">FAIL</span>
  return <span className="text-[var(--color-bad)]">ERR</span>
}

function harnessClass(harness: string) {
  return harness === "opencode"
    ? "text-[var(--color-bad)]"
    : "text-[var(--color-fg-muted)]"
}

export default async function RunsIndex({ searchParams }: RunsIndexProps) {
  const allRuns = await loadAllRuns()
  const params = searchParams ? await searchParams : {}
  const rawHarness = Array.isArray(params.harness)
    ? params.harness[0]
    : params.harness
  const harnessFilter = rawHarness?.trim()
  const runs = harnessFilter
    ? allRuns.filter((r) => r.harness === harnessFilter)
    : allRuns

  const passes = runs.filter((r) => r.correct).length
  const fails = runs.filter((r) => !r.correct && r.has_solution).length
  const errs = runs.filter((r) => !r.correct && !r.has_solution).length
  const title = harnessFilter ? `${harnessFilter} runs` : "all runs"

  return (
    <div className="space-y-8">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-3">
          {title}
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-4">
          {runs.length} runs · {passes} pass · {fails} fail · {errs} err · sorted by peak_fraction desc
        </p>
        {harnessFilter ? (
          <p className="text-xs text-[var(--color-fg-muted)] mb-4">
            active filter:{" "}
            <span className={harnessClass(harnessFilter)}>
              harness={harnessFilter}
            </span>{" "}
            · <a href="/runs">clear</a>
          </p>
        ) : null}
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl text-sm">
          One row per (model, problem) cell. Click any row to open the full transcript viewer — every tool call, every reasoning step, the model&apos;s solution.py, the check.log, the result.json. The viewer is the same one we use locally to audit runs, just themed for the site.
        </p>
      </section>

      <section className="box overflow-x-auto">
        <table className="term tabular text-xs sm:text-sm">
          <thead>
            <tr>
              <th>peak</th>
              <th>problem</th>
              <th>model</th>
              <th>harness</th>
              <th>elapsed</th>
              <th>run id</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td className="text-right pr-4">{statusCell(r)}</td>
                <td className="text-[var(--color-fg)] whitespace-nowrap">
                  <a
                    href={`/runs/${r.run_id}.html`}
                    className="no-underline hover:text-[var(--color-accent)]"
                  >
                    {r.problem}
                  </a>
                </td>
                <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
                  <a
                    href={`/runs/${r.run_id}.html`}
                    className="no-underline hover:text-[var(--color-accent)]"
                  >
                    {shortModel(r.harness, r.model, r.effort)}
                  </a>
                </td>
                <td className={harnessClass(r.harness)}>{r.harness}</td>
                <td className="text-[var(--color-fg-muted)]">
                  {r.elapsed_seconds !== null
                    ? `${Math.round(r.elapsed_seconds / 60)}m`
                    : "-"}
                </td>
                <td className="text-[var(--color-fg-muted)] text-[10px]">
                  <a
                    href={`/runs/${r.run_id}.html`}
                    className="no-underline hover:text-[var(--color-accent)]"
                  >
                    {r.run_id}
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
