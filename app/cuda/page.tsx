import Link from "next/link"
import { barsForBench, rowsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelBars } from "@/app/_components/model-bars"
import { ModelList } from "@/app/_components/model-list"

// KernelBench-CUDA: CUDA-only sibling of hard. Triton/DSL fail the language
// gate. Four-problem deck on RTX PRO 6000.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

export default async function CudaPage() {
  const idx = await loadModelIndex()
  const bars = barsForBench(idx, "cuda")
  const { sink } = rowsForBench(idx, "cuda")
  const hasRuns = bars.rows.length > 0 || sink.length > 0

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
        {hasRuns ? (
          <>
            <ModelBars view={bars} />
            {sink.length > 0 && (
              <div className="model-sink-section">
                <p className="model-sink-label">
                  No valid published results — audited sessions below were flagged or invalid
                </p>
                <ModelList rows={sink} sink />
              </div>
            )}
          </>
        ) : (
          <p className="text-xs text-[var(--color-fg-muted)]">
            Results will populate here as sweeps complete.
          </p>
        )}
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Methodology and the full problem deck live in the{" "}
          <Link
            href={`${REPO_TREE}/SPEC.md`}
            className="underline underline-offset-2"
          >
            spec
          </Link>
          .{" "}
          {hasRuns &&
            "Every published cell is contamination-checked and reward-hack audited; full agent traces are on HuggingFace. Browse the "}
          {hasRuns && (
            <Link href="/runs" className="underline underline-offset-2">
              run index
            </Link>
          )}
          {hasRuns && " for transcripts, submitted solutions, checks, and timing."}
        </p>
      </section>
    </div>
  )
}
