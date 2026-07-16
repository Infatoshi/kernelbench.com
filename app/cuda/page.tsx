import Link from "next/link"
import { LeaderboardTable, type HardRunRecord } from "../hard/leaderboard-table"

// KernelBench-CUDA: CUDA-only sibling of hard. Triton/DSL fail the language
// gate. Four-problem deck on RTX PRO 6000. No published runs yet.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

const CUDA_PROBLEMS = [
  { key: "01_glm52_fused_moe", label: "GLM-5.2 Fused MoE" },
  { key: "02_deepseek_nsa", label: "DeepSeek NSA" },
  { key: "03_megaqwen_decode", label: "MegaQwen Decode" },
  { key: "04_grid_mingru_sps", label: "Grid + MinGRU SPS" },
] as const

const NO_RUN = { label: "no run", tone: "muted" as const }

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

export default function CudaPage() {
  const rows = CUDA_PROBLEMS.map(placeholderRow)

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
          <span className="text-[var(--color-fg)]">No runs yet.</span>
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
          . Results will populate here as sweeps complete.
        </p>
      </section>
    </div>
  )
}
