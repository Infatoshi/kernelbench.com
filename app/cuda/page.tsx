import Link from "next/link"
import { barsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelBars } from "@/app/_components/model-bars"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-CUDA: CUDA-only sibling of hard. Triton/DSL fail the language
// gate. Four-problem deck on RTX PRO 6000.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

export default async function CudaPage() {
  const idx = await loadModelIndex()
  const bars = barsForBench(idx, "cuda")
  const hasRuns = bars.rows.length > 0

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · CUDA"
        title="The CUDA-only deck"
        sub={
          <>
            Hard&apos;s isolated sibling with a language gate: Triton and
            kernel DSLs fail. Four problems — GLM-5.2 fused MoE, DeepSeek NSA,
            MegaQwen decode at 2k–128k, grid+MinGRU SPS — on RTX PRO 6000
            Blackwell (SM120 · GDDR7 · 1.8 TB/s).
            {!hasRuns && <strong> No runs yet.</strong>}
          </>
        }
        notes={
          <>
            <p>
              <strong>Why separate.</strong> Hard and Mega prompts are frozen
              lab boards. CUDA exists to force CUDA evidence and grade
              Triton/DSL cheats without moving their goalposts. Methodology and
              the full problem deck live in the{" "}
              <Link href={`${REPO_TREE}/SPEC.md`}>spec</Link>.
            </p>
            {hasRuns && (
              <p>
                <strong>Artifacts.</strong> Every published cell is
                contamination-checked and reward-hack audited; full agent
                traces are on HuggingFace. Browse the{" "}
                <Link href="/runs">run index</Link> for transcripts, submitted
                solutions, checks, and timing.
              </p>
            )}
          </>
        }
      />
      {hasRuns ? (
        <div className="chart-panel">
          <div className="chart-panel-head">
            <span className="chart-panel-title">RTX PRO 6000</span>
          </div>
          <ModelBars view={bars} />
        </div>
      ) : (
        <p className="text-xs text-[var(--color-fg-muted)]">
          Results will populate here as sweeps complete.
        </p>
      )}
    </div>
  )
}
