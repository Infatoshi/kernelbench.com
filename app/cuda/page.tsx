import Link from "next/link"
import { readdir } from "node:fs/promises"
import { join } from "node:path"
import {
  loadLeaderboard,
  loadAnnotations,
  type Cell,
  type Leaderboard,
  type Model,
} from "@/app/_lib/data"
import { LeaderboardTable, type HardRunRecord } from "../hard/leaderboard-table"

// KernelBench-CUDA: CUDA-only sibling of hard. Triton/DSL fail the language
// gate. Four-problem deck on RTX PRO 6000.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

// Full agent transcripts live on HuggingFace (pushed by `kb push-runs cuda`).
const CUDA_TRACES_HF =
  "https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces"

function cudaTraceUrl(runId: string): string {
  return `${CUDA_TRACES_HF}/blob/main/${runId}.jsonl`
}

const CUDA_PROBLEMS = [
  { key: "01_glm52_fused_moe", label: "GLM-5.2 Fused MoE" },
  { key: "02_deepseek_nsa", label: "DeepSeek NSA" },
  { key: "03_megaqwen_decode", label: "MegaQwen Decode" },
  { key: "04_grid_mingru_sps", label: "Grid + MinGRU SPS" },
] as const

const NO_RUN = { label: "no run", tone: "muted" as const }

async function loadAvailableSolutions(): Promise<Set<string>> {
  try {
    const entries = await readdir(join(process.cwd(), "public/runs"))
    return new Set(
      entries
        .filter((n) => n.endsWith("_solution.py.txt"))
        .map((n) => n.slice(0, -"_solution.py.txt".length)),
    )
  } catch {
    return new Set()
  }
}

function runDateParts(runId: string): { date: string | null; time: string | null } {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/)
  if (!m) return { date: null, time: null }
  return {
    date: `${m[1]}-${m[2]}-${m[3]}`,
    time: `${m[4]}:${m[5]}:${m[6]}`,
  }
}

function placeholderRow(problem: { key: string; label: string }): HardRunRecord {
  return {
    key: `${problem.key}:pending`,
    runId: null,
    model: "—",
    harness: "—",
    gpu: "RTX PRO 6000",
    problem: problem.label,
    problemKey: problem.key,
    date: null,
    time: null,
    compiled: NO_RUN,
    correct: NO_RUN,
    auditFlags: [],
    explanation: null,
    peakFraction: null,
    speedPct: null,
    isWinner: false,
    referenceUrl: `${REPO_TREE}/problems-rtxpro6000/${problem.key}/reference.py`,
    solutionUrl: null,
    transcriptUrl: null,
    scored: "0/0",
    note: "awaiting first run",
    title: "no run yet",
    searchText: `${problem.label} ${problem.key} no run`,
  }
}

function cellRow(
  m: Model,
  problem: { key: string; label: string },
  cell: Cell | undefined,
  verdicts: Map<string, { verdict: string; summary?: string }>,
  hasSolution: Set<string>,
  winners: Map<string, string>,
  problemsTotal: number,
): HardRunRecord {
  const base = {
    model: m.label,
    harness: m.harness,
    gpu: "RTX PRO 6000",
    problem: problem.label,
    problemKey: problem.key,
    referenceUrl: `${REPO_TREE}/problems-rtxpro6000/${problem.key}/reference.py`,
    scored: `${m.pass_count}/${problemsTotal}`,
  }
  if (!cell) {
    return {
      ...base,
      key: `${m.label}:${problem.key}:missing`,
      runId: null,
      date: null,
      time: null,
      compiled: NO_RUN,
      correct: NO_RUN,
      auditFlags: [],
      explanation: null,
      peakFraction: null,
      speedPct: null,
      isWinner: false,
      solutionUrl: null,
      transcriptUrl: null,
      note: "no run",
      title: "not attempted",
      searchText: `${m.label} ${m.harness} ${problem.label} ${problem.key} no run`,
    }
  }
  const rid = cell.run_id
  const annot = verdicts.get(rid)
  const isHack = cell.invalid_reason === "reward_hack" || annot?.verdict === "reward_hack"
  const { date, time } = runDateParts(rid)
  const compiled = cell.has_solution
    ? { label: "solution", tone: "good" as const }
    : { label: "no solution", tone: "muted" as const }
  const correct = isHack
    ? { label: "hack", tone: "bad" as const, annotationSeverity: "bad" as const, annotationLabel: "reward hack" }
    : cell.correct
      ? { label: "pass", tone: "good" as const }
      : { label: "fail", tone: "bad" as const }
  const note = annot?.summary || cell.failure_reason || "run details"
  return {
    ...base,
    key: rid,
    runId: rid,
    date,
    time,
    compiled,
    correct,
    auditFlags: isHack ? ["reward_hack"] : [],
    explanation: annot?.summary || cell.invalid_reason || cell.failure_reason || null,
    peakFraction: cell.peak_fraction,
    speedPct: cell.peak_fraction != null ? cell.peak_fraction * 100 : null,
    isWinner: winners.get(problem.key) === rid,
    solutionUrl: hasSolution.has(rid) ? `/runs/${rid}_solution.py.txt` : null,
    transcriptUrl: cudaTraceUrl(rid),
    note,
    title: note,
    searchText: `${m.label} ${m.harness} ${problem.label} ${problem.key} ${rid} ${correct.label}`,
  }
}

export default async function CudaPage() {
  let lb: Leaderboard | null = null
  try {
    lb = await loadLeaderboard("benchmarks/cuda/results/leaderboard.json")
  } catch {
    lb = null
  }
  const models = lb?.models ?? []
  const hasRuns = models.length > 0

  const [annotations, hasSolution] = hasRuns
    ? await Promise.all([
        loadAnnotations("benchmarks/cuda/results/annotations"),
        loadAvailableSolutions(),
      ])
    : [new Map<string, { verdict: string; summary?: string }>(), new Set<string>()]

  const problemsTotal = CUDA_PROBLEMS.length
  let rows: HardRunRecord[]
  if (hasRuns) {
    // winner per problem = best valid (non-hack) passing cell by peak fraction
    const winners = new Map<string, string>()
    for (const p of CUDA_PROBLEMS) {
      let best: { rid: string; pf: number } | null = null
      for (const m of models) {
        const c = m.results[p.key]
        const verdict = c ? annotations.get(c.run_id)?.verdict : undefined
        if (!c || !c.correct || c.peak_fraction == null) continue
        if (c.invalid_reason === "reward_hack" || verdict === "reward_hack") continue
        if (!best || c.peak_fraction > best.pf) best = { rid: c.run_id, pf: c.peak_fraction }
      }
      if (best) winners.set(p.key, best.rid)
    }
    rows = models.flatMap((m) =>
      CUDA_PROBLEMS.map((p) =>
        cellRow(m, p, m.results[p.key], annotations, hasSolution, winners, problemsTotal),
      ),
    )
  } else {
    rows = CUDA_PROBLEMS.map(placeholderRow)
  }

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          cuda
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          RTX PRO 6000 Blackwell (SM120 · GDDR7 · 1.8 TB/s)
          <span className="ml-2 text-xs font-semibold text-[var(--color-accent)]">
            ● CUDA-only language gate · Triton / DSL = fail
          </span>
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-6 max-w-4xl leading-relaxed">
          Isolated CUDA-writing sibling of{" "}
          <Link href="/hard" className="underline underline-offset-2">
            hard
          </Link>
          . Four hard problems: GLM-5.2 fused MoE (256 routed + 1 shared,
          top-8), DeepSeek NSA sparse attention, MegaQwen decode (improve
          baseline; decode-only at 2k–128k), and grid+MinGRU SPS. Hard and Mega
          prompts stay frozen.{" "}
          {!hasRuns && <span className="text-[var(--color-fg)]">No runs yet.</span>}
        </p>
      </section>

      <section>
        <LeaderboardTable rows={rows} />
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Methodology and the full problem deck live in the{" "}
          <Link
            href={`${REPO_TREE}/SPEC.md`}
            className="underline underline-offset-2"
          >
            spec
          </Link>
          .{" "}
          {hasRuns
            ? "Every published cell is contamination-checked and reward-hack audited; full agent traces are on HuggingFace."
            : "Results will populate here as sweeps complete."}
        </p>
      </section>
    </div>
  )
}
