import Link from "next/link"
import { LeaderboardTable, type HardRunRecord } from "../hard/leaderboard-table"

// KernelBench-Multi: the multi-GPU (NVLink) sibling of the hard bench. Graded on
// busbw (achieved NVLink bus bandwidth / NVLink peak), never TFLOPS. The deck is
// six communication-dominated problems on an 8xH100 SXM (NVSwitch) node.
// No runs yet — this page stands up the table + deck so results can drop in.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/multi"

const MULTI_PROBLEMS = [
  { key: "01_allreduce_residual", label: "AllReduce + Residual" },
  { key: "02_reducescatter_rmsnorm", label: "ReduceScatter + RMSNorm" },
  { key: "03_allgather_fp8", label: "AllGather + fp8 Dequant" },
  { key: "04_moe_all2all", label: "MoE All-to-All" },
  { key: "05_ulysses_all2all", label: "Ulysses All-to-All" },
  { key: "06_fp8_reducescatter_grad", label: "fp8 ReduceScatter Grad" },
] as const

const NO_RUN = { label: "no run", tone: "muted" as const }

function placeholderRow(problem: { key: string; label: string }): HardRunRecord {
  return {
    key: `${problem.key}:pending`,
    runId: null,
    model: "—",
    harness: "—",
    gpu: "8×H100",
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
    referenceUrl: `${REPO_TREE}/problems-h100x8/${problem.key}/reference.py`,
    solutionUrl: null,
    transcriptUrl: null,
    scored: "0/0",
    note: "awaiting first run",
    title: "no run yet",
    searchText: `${problem.label} ${problem.key} no run`,
  }
}

export default function MultiPage() {
  const rows = MULTI_PROBLEMS.map(placeholderRow)

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          multi
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          8×H100 SXM (NVSwitch, NVLink4 · ~900 GB/s/GPU)
          <span className="ml-2 text-xs font-semibold text-[var(--color-accent)]">
            ● graded on busbw (NVLink bus-bandwidth efficiency)
          </span>
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-6 max-w-4xl leading-relaxed">
          The multi-GPU sibling of{" "}
          <Link href="/hard" className="underline underline-offset-2">
            hard
          </Link>
          . A coding agent takes a PyTorch + NCCL reference for a distributed op
          and rewrites it as a fast, fine-grained NVLink kernel (CUDA / Triton /
          NVSHMEM / CUDA symmetric memory / ParallelKittens) that beats the NCCL
          baseline. The graded quantity is{" "}
          <span className="text-[var(--color-fg)]">busbw</span> — achieved NVLink
          bus bandwidth ÷ NVLink peak, never TFLOPS — so every problem is
          deliberately communication-dominated. Six-problem deck, all
          busbw-graded, each forbidding its bare collective.{" "}
          <span className="text-[var(--color-fg)]">No runs yet.</span>
        </p>
      </section>

      <section>
        <LeaderboardTable rows={rows} />
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Methodology and the full problem deck live in the{" "}
          <Link href={`${REPO_TREE}/SPEC.md`} className="underline underline-offset-2">
            spec
          </Link>
          . Results will populate here as sweeps complete on the 8×H100 node.
        </p>
      </section>
    </div>
  )
}
