import Link from "next/link"
import {
  ACTIVE_BENCHES,
  BENCH_LABELS,
  CANONICAL_GPU,
  FLAG_VERDICTS,
  SITE_HIDDEN_GPUS,
  problemLabel,
  visibleProblems,
  type Bench,
  type ModelCell,
} from "@/app/_lib/models"
import { loadAllModelIndex } from "@/app/_lib/models.server"
import { PageHead } from "@/app/_components/page-head"

// Active run ledger across Mega and CUDA (Hard was deprecated on 2026-11-23
// and stays archived at /hard and Infatoshi/kernelbench-hard-traces).

export const metadata = { title: "runs · kernelbench" }

const MEGA_TRACES_HF =
  "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces"
const CUDA_TRACES_HF =
  "https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces"
const HARD_TRACES_HF =
  "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces"

const GPU_LABELS: Record<string, string> = {
  rtxpro6000: "RTX PRO 6000",
  h100: "H100 PCIe",
  b200: "B200",
}

const GPU_ORDER: Record<string, number> = {
  rtxpro6000: 0,
  h100: 1,
  b200: 2,
}

type RunRow = {
  key: string
  run_id: string
  bench: Bench
  gpu: string
  problem: string
  modelSlug: string
  modelName: string
  harness: string
  effort: string | null
  score: number | null
  latency_ms: number | null
  valid: boolean
  correct: boolean
  verdict: string | null
  outcome_label: string | null
  failure_reason: string | null
  elapsed_seconds: number | null
  trace_url: string | null
}

type RunsIndexProps = {
  searchParams?: Promise<{
    harness?: string | string[]
    bench?: string | string[]
  }>
}

async function loadActiveRuns(): Promise<RunRow[]> {
  const idx = await loadAllModelIndex()
  const out: RunRow[] = []
  const seen = new Set<string>()

  for (const m of idx.models) {
    for (const bench of ACTIVE_BENCHES) {
      const block = m.benches[bench]
      if (!block) continue
      const meta = idx.benches[bench]
      const allowed = new Set(
        visibleProblems(
          bench,
          meta?.problems?.length ? meta.problems : Object.keys(block.cells),
        ),
      )
      const views: { gpu: string; cells: Record<string, ModelCell> }[] = [
        { gpu: CANONICAL_GPU, cells: block.cells ?? {} },
      ]
      for (const [g, v] of Object.entries(block.gpus ?? {})) {
        if (SITE_HIDDEN_GPUS.has(g)) continue
        views.push({ gpu: g, cells: v.cells ?? {} })
      }
      for (const { gpu, cells } of views) {
        for (const [problem, cell] of Object.entries(cells)) {
          if (!allowed.has(problem) || !cell.run_id) continue
          const key = `${bench}:${gpu}:${cell.run_id}:${problem}`
          if (seen.has(key)) continue
          seen.add(key)
          out.push({
            key,
            run_id: cell.run_id,
            bench,
            gpu,
            problem,
            modelSlug: m.slug,
            modelName: m.name,
            harness: block.harness ?? "unknown",
            effort: block.effort ?? null,
            score: cell.score ?? null,
            latency_ms: cell.latency_ms ?? null,
            valid: Boolean(cell.valid),
            correct: Boolean(cell.correct),
            verdict: cell.verdict ?? null,
            outcome_label: cell.outcome_label ?? null,
            failure_reason: cell.failure_reason ?? null,
            elapsed_seconds: cell.elapsed_seconds ?? null,
            trace_url: cell.trace_url ?? null,
          })
        }
      }
    }
  }

  out.sort((a, b) => {
    const ba = ACTIVE_BENCHES.indexOf(a.bench)
    const bb = ACTIVE_BENCHES.indexOf(b.bench)
    if (ba !== bb) return ba - bb
    const ga = GPU_ORDER[a.gpu] ?? 9
    const gb = GPU_ORDER[b.gpu] ?? 9
    if (ga !== gb) return ga - gb
    if (a.problem !== b.problem) return a.problem.localeCompare(b.problem)
    const av = a.valid && a.score != null ? a.score : -1
    const bv = b.valid && b.score != null ? b.score : -1
    if (bv !== av) return bv - av
    return a.modelName.localeCompare(b.modelName)
  })

  return out
}

function scoreCell(r: RunRow) {
  if (r.valid && r.score != null) {
    if (r.bench === "mega") {
      return (
        <span className="text-[var(--color-fg-bright)] tabular">
          {r.score.toFixed(2)}×
        </span>
      )
    }
    if (r.bench === "cuda" && r.problem === "02_deepseek_nsa" && r.latency_ms != null) {
      return (
        <span className="text-[var(--color-fg-bright)] tabular">
          {r.latency_ms.toFixed(3)} ms
        </span>
      )
    }
    return (
      <span className="text-[var(--color-fg-bright)] tabular">
        {(r.score * 100).toFixed(r.score >= 0.1 ? 1 : 2)}%
      </span>
    )
  }
  const flagged = r.verdict && FLAG_VERDICTS.has(r.verdict)
  if (flagged) {
    return <span className="text-[var(--color-bad)]">{r.verdict}</span>
  }
  const label =
    r.outcome_label ??
    r.failure_reason?.replace(/_/g, " ") ??
    (r.correct ? "unaudited" : "fail")
  return <span className="text-[var(--color-fg-dim)]">{label}</span>
}

function harnessLabel(harness: string) {
  const labels: Record<string, string> = {
    claude: "Claude Code",
    codex: "Codex",
    opencode: "Opencode",
    droid: "droid",
    kimi: "kimi",
    cursor: "Cursor",
    gemini: "Gemini CLI",
    agy: "Antigravity",
    grok: "Grok Build",
  }
  if (harness.endsWith("-claude")) return "Claude Code"
  return labels[harness] ?? harness
}

function harnessClass(harness: string) {
  return harness === "opencode"
    ? "text-[var(--color-bad)]"
    : "text-[var(--color-fg-muted)]"
}

export default async function RunsIndex({ searchParams }: RunsIndexProps) {
  const allRuns = await loadActiveRuns()
  const params = searchParams ? await searchParams : {}
  const rawHarness = Array.isArray(params.harness)
    ? params.harness[0]
    : params.harness
  const rawBench = Array.isArray(params.bench) ? params.bench[0] : params.bench
  const harnessFilter = rawHarness?.trim()
  const benchFilter =
    rawBench === "mega" || rawBench === "cuda" ? rawBench : undefined

  const runs = allRuns.filter((r) => {
    if (harnessFilter && r.harness !== harnessFilter) return false
    if (benchFilter && r.bench !== benchFilter) return false
    return true
  })

  const passed = runs.filter((r) => r.valid && r.score != null).length
  const flagged = runs.filter(
    (r) => r.verdict && FLAG_VERDICTS.has(r.verdict),
  ).length
  const other = runs.length - passed - flagged
  const titleParts = ["Run index"]
  if (benchFilter) titleParts.push(BENCH_LABELS[benchFilter])
  if (harnessFilter) titleParts.push(harnessLabel(harnessFilter))

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Index"
        title={titleParts.join(" · ")}
        sub={
          <>
            <strong>{runs.length}</strong> active Mega / CUDA cells · {passed}{" "}
            audited pass · {flagged} flagged · {other} other
            {(harnessFilter || benchFilter) && (
              <>
                {" "}
                · (<Link href="/runs">clear filters</Link>)
              </>
            )}
          </>
        }
        notes={
          <p>
            Active published cells across KernelBench Mega and CUDA. Click a
            run to inspect per-shape timing, audit notes, and the redacted
            kernel; full JSONL agent transcripts live on HuggingFace in{" "}
            <a href={MEGA_TRACES_HF} target="_blank" rel="noreferrer">
              kernelbench-mega-traces
            </a>{" "}
            and{" "}
            <a href={CUDA_TRACES_HF} target="_blank" rel="noreferrer">
              kernelbench-cuda-traces
            </a>
            . Archived runs from the deprecated{" "}
            <Link href="/hard">Hard</Link> deck stay in{" "}
            <a href={HARD_TRACES_HF} target="_blank" rel="noreferrer">
              kernelbench-hard-traces
            </a>
            .
          </p>
        }
      />

      <section className="box overflow-x-auto">
        <table className="term tabular text-xs sm:text-sm">
          <thead>
            <tr>
              <th>score</th>
              <th>bench</th>
              <th>gpu</th>
              <th>problem</th>
              <th>model</th>
              <th>harness</th>
              <th>run</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const runHref = `/runs/${r.gpu}/${r.run_id}`
              return (
                <tr key={r.key}>
                  <td className="text-right pr-4 whitespace-nowrap">
                    {scoreCell(r)}
                  </td>
                  <td className="text-[var(--color-fg-muted)] whitespace-nowrap uppercase text-[11px]">
                    {BENCH_LABELS[r.bench]}
                  </td>
                  <td className="text-[var(--color-fg-muted)] whitespace-nowrap">
                    {GPU_LABELS[r.gpu] ?? r.gpu}
                  </td>
                  <td className="text-[var(--color-fg)] whitespace-nowrap">
                    <Link
                      href={runHref}
                      className="no-underline hover:text-[var(--color-accent)]"
                    >
                      {problemLabel(r.problem)}
                    </Link>
                  </td>
                  <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
                    <Link
                      href={`/models/${r.modelSlug}`}
                      className="no-underline hover:text-[var(--color-accent)]"
                    >
                      {r.modelName}
                    </Link>
                  </td>
                  <td className={`${harnessClass(r.harness)} whitespace-nowrap`}>
                    {harnessLabel(r.harness)}
                  </td>
                  <td className="text-[var(--color-fg-muted)] text-[10px] whitespace-nowrap">
                    <Link
                      href={runHref}
                      className="no-underline hover:text-[var(--color-accent)]"
                    >
                      {r.run_id}
                    </Link>
                    {r.trace_url && (
                      <>
                        {" "}
                        <a
                          href={r.trace_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[var(--color-fg-dim)] hover:text-[var(--color-accent)]"
                          title="Open agent transcript on HuggingFace"
                        >
                          trace↗
                        </a>
                      </>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
